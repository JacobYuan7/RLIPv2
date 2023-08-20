# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from scipy.optimize import linear_sum_assignment
import time

import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from torchvision.ops import RoIAlign
import math
from .transformer import TransformerEncoder, TransformerEncoderLayer
from .deformable_transformer import DeformableTransformerDecoderLayer, DeformableTransformerDecoder
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def xyxy_to_feature_scale(bboxes: torch.Tensor, features: NestedTensor):
    b, nq, ncoor = bboxes.shape # xy xy
    for i in range(b):
        ft_mask = features[-1].mask[i] # [ft_w, ft_h]
        # print(ft_mask.shape)
        valid_w = torch.nonzero(ft_mask[:,0], as_tuple = True)[0] 
        # the first 0 is the index for the tuple and the second 0 is the index for the first nonzero term
        valid_h = torch.nonzero(ft_mask[0,:], as_tuple = True)[0]
        valid_w = ft_mask[:,0].shape[0] if valid_w.shape[0] == 0 else valid_w[0]
        valid_h = ft_mask[0,:].shape[0] if valid_h.shape[0] == 0 else valid_h[0]

       
        # print(bboxes[i, :, 0].shape)
        bboxes[i, :, 0] = bboxes[i, :, 0] * valid_w
        bboxes[i, :, 1] = bboxes[i, :, 1] * valid_h
        bboxes[i, :, 2] = bboxes[i, :, 2] * valid_w
        bboxes[i, :, 3] = bboxes[i, :, 3] * valid_h

    bboxes_roi_align = torch.stack((bboxes[:,:,1], bboxes[:,:,0],
                                    bboxes[:,:,3], bboxes[:,:,2]), dim = -1)

    return bboxes_roi_align #bboxes




class VanillaStochasticDETRHOIauxkl(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.latent_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        self.verb_class_embed = nn.Linear(self.latent_dim, num_verb_classes)
        # self.verb_class_embed = MLP(self.latent_dim, self.latent_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def cal_kl(self, aux_tensor, tensor):
        # ensure the input hasn't used softmax processing
        # tensor1, tensor2 shape [bs, nq, num_class]
        assert aux_tensor.shape == tensor.shape
        bs, nq, num_class = tensor.shape
        aux_tensor = F.softmax(aux_tensor, dim = -1)
        tensor = F.softmax(tensor, dim = -1)
        kl = (tensor * torch.log(tensor/aux_tensor)).sum(dim = -1)
        return kl    # [bs, nq]

        
    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print(len(pos))
        # print(pos[-1].shape)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # self.transformer return decoder info and memory
        # hs: tensor [6, 2, 100, 256]  6 is the #decoder_layers

        outputs_obj_class = self.obj_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
        outputs_verb_class = self.verb_class_embed(hs)
        
        # calculate uncertainty via kl divergence between 4th output and 6th output
        obj_class_kl = self.cal_kl(outputs_obj_class[5,:], outputs_obj_class[4,:]) 
        verb_class_kl = self.cal_kl(outputs_verb_class[5,:], outputs_verb_class[4,:])
        # aux_kl = torch.stack((obj_class_kl, verb_class_kl), dim = 0)


        # verb_mu = self.latent_mu(hs)  # [6, b, nq, latent_dim]
        # verb_log_var = self.latent_log_var(hs)  # [6, b, nq, latent_dim]
        # if self.training:
        #     sampling_num = 5
        # else:
        #     sampling_num = 5
        # stochastic_prior = torch.randn([sampling_num, ] + list(verb_mu.shape), 
        #                                dtype = verb_mu.dtype, 
        #                                device = verb_mu.device)
        # verb_std = torch.exp(0.5*verb_log_var)
        # verb_latent = verb_mu + verb_std * stochastic_prior # [6, b, nq, latent_dim] * [sampling_num, 6, b, nq, latent_dim]
        # outputs_verb_class = self.verb_class_embed(verb_latent)
        # outputs_verb_class = outputs_verb_class.mean(dim = 0)
        # outputs_verb_class += res_outputs_verb_class

        # shared latent variable
        # outputs_obj_class = self.obj_class_embed(verb_latent)
        # outputs_obj_class = outputs_obj_class.mean(dim = 0)
        # outputs_obj_class += res_outputs_obj_class

        # obj_class_mu = self.obj_class_mu(hs)
        # obj_class_log_var = self.obj_class_log_var(hs)
        # obj_class_stochastic_prior = torch.randn([sampling_num, ] + list(obj_class_mu.shape),
        #                                          dtype = obj_class_mu.dtype,
        #                                          device = obj_class_mu.device)
        # obj_class_std = torch.exp(0.5*obj_class_log_var)
        # obj_class_latent = obj_class_mu + obj_class_std * obj_class_stochastic_prior
        # outputs_obj_class = self.obj_class_embed(obj_class_latent).mean(dim = 0)
        # outputs_obj_class += res_outputs_obj_class
        # gaussian_constraint = torch.cat((verb_log_var, obj_class_log_var), dim = -1) # verb_log_var
        
        
        # return KL divergence parameters
        # gaussian_constraint = torch.cat((verb_mu, verb_log_var), dim = -1)
        # out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
        #        'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
        #        'verb_kl_divergence': gaussian_constraint[-1]}

        # return verb_log_var for calculating entropy bound
        
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
               'aux_kl': {'obj_class_kl':obj_class_kl, 'verb_class_kl':verb_class_kl}} 

        if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        # return KL divergence parameters
        # return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d, 'verb_kl_divergence': e}
        #         for a, b, c, d, e in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
        #                               outputs_sub_coord[:-1], outputs_obj_coord[:-1], gaussian_constraint[:-1])]

        # return verb_log_var for calculating entropy bound
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]



class VanillaStochasticDETRHOI(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.latent_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        self.verb_class_embed = nn.Linear(self.latent_dim, num_verb_classes)
        # self.verb_class_embed = MLP(self.latent_dim, self.latent_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # Stochastic 
        self.latent_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, self.latent_dim)
        self.obj_class_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.obj_class_log_var = nn.Linear(hidden_dim, self.latent_dim)

        
    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print(len(pos))
        # print(pos[-1].shape)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # self.transformer return decoder info and memory
        # hs: tensor [6, 2, 100, 256]  6 is the #decoder_layers

        res_outputs_obj_class = self.obj_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
        res_outputs_verb_class = self.verb_class_embed(hs)

        verb_mu = self.latent_mu(hs)  # [6, b, nq, latent_dim]
        verb_log_var = self.latent_log_var(hs)  # [6, b, nq, latent_dim]
        if self.training:
            sampling_num = 5
        else:
            sampling_num = 5
        stochastic_prior = torch.randn([sampling_num, ] + list(verb_mu.shape), 
                                       dtype = verb_mu.dtype, 
                                       device = verb_mu.device)
        verb_std = torch.exp(0.5*verb_log_var)
        verb_latent = verb_mu + verb_std * stochastic_prior # [6, b, nq, latent_dim] * [sampling_num, 6, b, nq, latent_dim]
        outputs_verb_class = self.verb_class_embed(verb_latent)
        outputs_verb_class = outputs_verb_class.mean(dim = 0)
        outputs_verb_class += res_outputs_verb_class

        # shared latent variable
        # outputs_obj_class = self.obj_class_embed(verb_latent)
        # outputs_obj_class = outputs_obj_class.mean(dim = 0)
        # outputs_obj_class += res_outputs_obj_class

        obj_class_mu = self.obj_class_mu(hs)
        obj_class_log_var = self.obj_class_log_var(hs)
        obj_class_stochastic_prior = torch.randn([sampling_num, ] + list(obj_class_mu.shape),
                                                 dtype = obj_class_mu.dtype,
                                                 device = obj_class_mu.device)
        obj_class_std = torch.exp(0.5*obj_class_log_var)
        obj_class_latent = obj_class_mu + obj_class_std * obj_class_stochastic_prior
        outputs_obj_class = self.obj_class_embed(obj_class_latent).mean(dim = 0)
        outputs_obj_class += res_outputs_obj_class
        gaussian_constraint = torch.cat((verb_log_var, obj_class_log_var), dim = -1) # verb_log_var
        
        # return KL divergence parameters
        # gaussian_constraint = torch.cat((verb_mu, verb_log_var), dim = -1)
        # out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
        #        'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
        #        'verb_kl_divergence': gaussian_constraint[-1]}

        # return verb_log_var for calculating entropy bound
        
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
               'verb_log_var': gaussian_constraint[-1]} 

        if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord, gaussian_constraint)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, gaussian_constraint):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        # return KL divergence parameters
        # return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d, 'verb_kl_divergence': e}
        #         for a, b, c, d, e in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
        #                               outputs_sub_coord[:-1], outputs_obj_coord[:-1], gaussian_constraint[:-1])]

        # return verb_log_var for calculating entropy bound
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d, 'verb_log_var': e}
                for a, b, c, d, e in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1], gaussian_constraint[:-1])]



def norm_tensor(tensor):
    norm = torch.norm(tensor, p = 'fro', dim = -1).unsqueeze(dim = -1).expand_as(tensor)
    return tensor/norm


def count_fusion(x, y):
    return F.relu(x + y) - (x - y)*(x - y)

class SemanticGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, attention_type='embedded_dot_pro', head_num = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_type = attention_type

        if attention_type == 'embedded_dot_pro':
            self.relation_dim = hidden_dim
            self.semantic_q = [nn.Linear(input_dim, self.relation_dim),]
            self.semantic_k = [nn.Linear(input_dim, self.relation_dim),]
            self.semantic_v = [nn.Linear(input_dim, hidden_dim),]
            self.semantic_proj_res = nn.Linear(input_dim, hidden_dim)
            for _ in range(num_layers-1):
                self.semantic_q.append(nn.Linear(hidden_dim, hidden_dim))
                self.semantic_k.append(nn.Linear(hidden_dim, hidden_dim))
                self.semantic_v.append(nn.Linear(hidden_dim, hidden_dim))
            self.semantic_q = nn.ModuleList(self.semantic_q)
            self.semantic_k = nn.ModuleList(self.semantic_k)
            self.semantic_v = nn.ModuleList(self.semantic_v)

        elif attention_type == 'multihead_transformer':
            assert self.num_layers == 1
            self.head_num = head_num
            self.bottleneck_dim = int(self.hidden_dim//0.5)
            self.relation_dim = hidden_dim//self.head_num
            self.semantic_q = nn.Linear(input_dim, self.relation_dim)
            self.semantic_k = nn.Linear(input_dim, self.relation_dim)
            self.semantic_q = _get_clones(self.semantic_q, self.head_num)
            self.semantic_k = _get_clones(self.semantic_k, self.head_num)
            self.semantic_v = nn.Linear(input_dim, self.relation_dim)
            # self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(self.head_num)])
            self.semantic_proj_res = nn.Linear(input_dim, hidden_dim)

            self.W_t2 = nn.Linear(hidden_dim, self.bottleneck_dim)
            self.dropout2 = nn.Dropout(0.1)
            self.W_t1 = nn.Linear(self.bottleneck_dim, hidden_dim)
            self.LayerNorm = nn.LayerNorm([self.bottleneck_dim,])

        elif attention_type == 'MLP':
            self.mlp_layers = 3
            self.mlp = nn.ModuleList([nn.Linear(input_dim, hidden_dim) if i==0 else nn.Linear(hidden_dim, hidden_dim) for i in range(self.mlp_layers)])
            # self.nonlinearity = nn.ModuleList([nn.LeakyReLU(negative_slope=0.2, inplace=False) for i in range(self.mlp_layers-1)])
            self.nonlinearity = nn.ModuleList([nn.ReLU() for i in range(self.mlp_layers)])
            self.mlp_ln = nn.LayerNorm([hidden_dim,])
        
        elif attention_type == 'MLP_GNN':
            self.mlp_layers = 2
            self.mlp = nn.ModuleList([nn.Linear(input_dim, hidden_dim) if i==0 else nn.Linear(hidden_dim, hidden_dim) for i in range(self.mlp_layers)])
            # self.nonlinearity = nn.ModuleList([nn.LeakyReLU(negative_slope=0.2, inplace=False) for i in range(self.mlp_layers-1)])
            self.nonlinearity = nn.ModuleList([nn.ReLU() for i in range(self.mlp_layers)])
            self.mlp_ln = nn.ModuleList([nn.LayerNorm([hidden_dim,]) for i in range(self.mlp_layers)]) 

            self.relation_dim = hidden_dim
            self.semantic_ln = nn.ModuleList([nn.LayerNorm([hidden_dim,]) for _ in range(num_layers)])
            self.semantic_nonlinear = nn.ModuleList([nn.ReLU() for _ in range(num_layers)])
            self.semantic_q = nn.ModuleList([nn.Linear(hidden_dim, self.relation_dim) for _ in range(num_layers)])
            self.semantic_k = nn.ModuleList([nn.Linear(hidden_dim, self.relation_dim) for _ in range(num_layers)])
            self.semantic_v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

            # Bilinear Pooling
            # self.nheads = nheads
            # self.bilinear1 = nn.Linear(hidden_dim, hidden_dim)
            # self.bilinear2 = nn.Linear(hidden_dim, hidden_dim)
            # self.bilinear1 = _get_clones(self.bilinear1, nheads)
            # self.bilinear2 = _get_clones(self.bilinear2, nheads)
            # self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
        
            # hid_hid_dim = hidden_dim//nheads
            # self.W3 = nn.Linear(hidden_dim, hid_hid_dim)
            # self.W3 = _get_clones(self.W3, nheads)
            # self.W2 = nn.Linear(hidden_dim, hidden_dim)
            # self.W1 = nn.Linear(hidden_dim, hidden_dim)
            # self.nonlinear = nn.ReLU(inplace = True)
            # self.LayerNorm = nn.LayerNorm([hidden_dim,])
        
        
    def forward(self, x, cooccur_prior = None):
        assert len(x.shape) == 2
        if self.attention_type == 'embedded_dot_pro':
            for i in range(self.num_layers):
                x_q = self.semantic_q[i](x)
                x_k = self.semantic_k[i](x)
                x_v = self.semantic_v[i](x)
                # x_att = torch.einsum('ac,bc->ab', x_q, x_k)
                x_att = torch.einsum('ac,bc->ab', x_q, x_k) / math.sqrt(self.relation_dim)
                x_att = F.softmax(x_att, dim = -1)
                if cooccur_prior is not None:
                    x_att = x_att + cooccur_prior
                    print('cooccur prior')

                if i == 0:
                    x = F.relu(torch.matmul(x_att, x_v)) + self.semantic_proj_res(x) # self.verb_calibration_embedding
                else:
                    x = F.relu(torch.matmul(x_att, x_v)) + x
            trans_x = x
            # trans_x = norm_tensor(x)
        
        if self.attention_type == 'multihead_transformer':
            len_x = len(x.shape)
            if len_x == 2:
                x = x.unsqueeze(dim = 0)
            elif len_x == 4:
                x.shape = l, bs, q, hiddim
                x = x.view((l*bs, q, hiddim))
            elif len_x == 3:
                None
            else:
                print("Shape is not compatible")
                assert False

            x_v = self.semantic_v(x)
            multihead_ft = []
            for i in range(self.head_num):
                x_q_i = self.semantic_q[i](x)  # lbs, q, hiddim
                x_k_i = self.semantic_k[i](x) # * self.coef[i].expand_as(x_q_i)  # lbs, q, hiddim

                x_att_i = torch.einsum('abc,adc->abd', x_q_i, x_k_i) / math.sqrt(self.relation_dim)
                x_att_i = F.softmax(x_att_i, dim = -1)
                att_ft_i = torch.bmm(x_att_i, x_v)
                multihead_ft.append(att_ft_i)

            multihead_ft = torch.cat(multihead_ft, dim = -1)
            trans_ft = self.W_t1(F.relu(self.LayerNorm(self.W_t2(multihead_ft)), inplace = True))
            trans_x = trans_ft + self.semantic_proj_res(x)

            if len_x == 2:
                trans_x = trans_x.squeeze(dim = 0)
            elif len_x == 4:
                trans_x = trans_x.view((l, bs, q, hiddim))
            elif len_x == 3:
                None
        
        if self.attention_type == 'MLP':
            for i in range(self.mlp_layers):
                x = self.mlp[i](x)
                if i == self.mlp_layers-1:
                    x = self.mlp_ln(x)
                    x = self.nonlinearity[i](x)
                else:
                    x = self.nonlinearity[i](x)
            
            # trans_x = norm_tensor(x)
            trans_x = x
        
        if self.attention_type == 'MLP_GNN':
            for i in range(self.mlp_layers):
                x = self.mlp[i](x)
                x = self.mlp_ln[i](x)
                x = self.nonlinearity[i](x)
            
            for i in range(self.num_layers):
                x_q = self.semantic_q[i](x)
                x_k = self.semantic_k[i](x)
                x_v = self.semantic_v[i](x)
                x_att = torch.einsum('ac,bc->ab', x_q, x_k) / math.sqrt(self.relation_dim)
                x_att = F.softmax(x_att, dim = -1)
                x = self.semantic_nonlinear[i](self.semantic_ln[i](torch.matmul(x_att, x_v))) + x
            
            trans_x = x

        return trans_x


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, relation = 'bilinear', dropout = 0.1):
        super().__init__()
        self.relation = relation
        self.hidden_dim = hidden_dim
        if self.relation == 'bilinear':
            self.nheads = nheads
            self.bilinear1 = nn.Linear(hidden_dim, hidden_dim)
            self.bilinear2 = nn.Linear(hidden_dim, hidden_dim)
            self.bilinear1 = _get_clones(self.bilinear1, nheads)
            self.bilinear2 = _get_clones(self.bilinear2, nheads)
            self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
        
            hid_hid_dim = hidden_dim//nheads
            self.W3 = nn.Linear(hidden_dim, hid_hid_dim)
            self.W3 = _get_clones(self.W3, nheads)
            self.W2 = nn.Linear(hidden_dim, hidden_dim)
            self.W1 = nn.Linear(hidden_dim, hidden_dim)
            self.nonlinear = nn.ReLU(inplace = True)
            self.LayerNorm = nn.LayerNorm([hidden_dim,])
        
        if self.relation == 'embedded_dot_pro':
            self.nheads = nheads
            self.hidden_dim = hidden_dim
            self.hid_hid_dim = hidden_dim//nheads
            self.relation_q = nn.Linear(self.hidden_dim, self.hid_hid_dim)
            self.relation_k = nn.Linear(self.hidden_dim, self.hid_hid_dim)
            self.relation_q = _get_clones(self.relation_q, nheads)
            self.relation_k = _get_clones(self.relation_k, nheads)
            # self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
        
            self.W3 = nn.Linear(self.hidden_dim, self.hid_hid_dim)
            self.W3 = _get_clones(self.W3, nheads)
            self.bottleneck_dim = int(self.hidden_dim//0.5)
            self.W2 = nn.Linear(self.hidden_dim, self.bottleneck_dim)
            self.W1 = nn.Linear(self.bottleneck_dim, self.hidden_dim)
            self.nonlinear = nn.ReLU(inplace = True)
            self.LayerNorm = nn.LayerNorm([self.bottleneck_dim,])

        if self.relation == 'VanillaTrans':
            self.nheads = nheads
            self.hidden_dim = hidden_dim
            self.hid_hid_dim = hidden_dim//nheads
            self.relation_q = nn.Linear(self.hidden_dim, self.hid_hid_dim)
            self.relation_k = nn.Linear(self.hidden_dim, self.hid_hid_dim)
            self.relation_q = _get_clones(self.relation_q, nheads)
            self.relation_k = _get_clones(self.relation_k, nheads)
            # self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
        
            self.W3 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.W3 = _get_clones(self.W3, nheads)
            self.bottleneck_dim = int(self.hidden_dim)
            self.W2 = nn.Linear(self.hidden_dim, self.bottleneck_dim)
            self.W1 = nn.Linear(self.bottleneck_dim, self.hidden_dim)
            self.nonlinear = nn.ReLU(inplace = True)
            self.norm2 = nn.LayerNorm(self.hidden_dim)
            self.norm1 = nn.LayerNorm(self.hidden_dim)
            self.dropout3 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x: shape [6,2,100,256]
        '''
        # cal multi-head attention
        if self.relation == 'bilinear':
            x_trans = []
            for i in range(self.nheads):
                x_b1 = self.bilinear1[i](x) # [6,2,100,256]
                x_b2 = self.bilinear2[i](x)
                # x_b1 = torch.sigmoid(self.bilinear1[i](x)) # [6,2,100,256]
                # x_b2 = torch.sigmoid(self.bilinear2[i](x))

                x_b1 = x_b1 * self.coef[i]
                x_att = torch.einsum('abcd,abed->abce', x_b1, x_b2)
                x_att = torch.softmax(x_att, dim = -1)
                x_emb = self.W3[i](x)
                x_i = torch.einsum('abce,abef->abcf', x_att, x_emb)
                x_trans.append(x_i)  # [6,2,100,256/nheads]
            x_trans = torch.cat(x_trans, dim = -1)
            x_trans = self.W1(self.nonlinear(self.LayerNorm(self.W2(x_trans))))
            x_trans = x + x_trans
        
        if self.relation == 'embedded_dot_pro':
            x_trans = []
            for i in range(self.nheads):
                x_r1 = self.relation_q[i](x) # [6,2,100,256]
                x_r2 = self.relation_k[i](x)
                x_att = torch.einsum('abcd,abed->abce', x_r1, x_r2) / math.sqrt(self.hid_hid_dim)
                x_att = torch.softmax(x_att, dim = -1)
                x_emb = self.W3[i](x)
                x_i = torch.einsum('abce,abef->abcf', x_att, x_emb)
                x_trans.append(x_i)  # [6, 2, 100, 256/nheads]
            x_trans = torch.cat(x_trans, dim = -1)
            x_trans = self.W1(self.nonlinear(self.LayerNorm(self.W2(x_trans))))
            x_trans = x + x_trans
        
        if self.relation == 'VanillaTrans':
            x_n = self.norm2(x)
            x_trans = []
            for i in range(self.nheads):
                x_r1 = self.relation_q[i](x_n) # [6,2,100,256]
                x_r2 = self.relation_k[i](x_n)
                x_att = torch.einsum('abcd,abed->abce', x_r1, x_r2) / math.sqrt(self.hid_hid_dim)
                x_att = torch.softmax(x_att, dim = -1)
                x_emb = self.W3[i](x_n)
                x_i = torch.einsum('abce,abef->abcf', x_att, x_emb)
                x_trans.append(x_i)  # [6,2,100,256/nheads]
            x_trans = torch.stack(x_trans, dim = -1).sum(dim = -1)
            x_trans = x + self.dropout3(x_trans)
            x_trans2 = self.norm1(x_trans)
            x_trans2 = self.W1(self.dropout2(self.nonlinear((self.W2(x_trans2)))))
            x_trans = x_trans + self.dropout1(x_trans2)
            
        return x_trans


class InterTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, nheads):
        super().__init__()
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.bilinear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear1 = _get_clones(self.bilinear1, nheads)
        self.bilinear2 = _get_clones(self.bilinear2, nheads)
        self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
    
        hid_hid_dim = hidden_dim//nheads
        self.W3 = nn.Linear(hidden_dim, hid_hid_dim)
        self.W3 = _get_clones(self.W3, nheads)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlinear = nn.ReLU(inplace = True)
        self.LayerNorm = nn.LayerNorm([hidden_dim,])
    
    def forward(self, x, y):
        '''
        Gather y features to x.
        x: [6,2,100,256]
        y: [6,2,100,256]
        '''
        x_trans = []
        for i in range(self.nheads):
            x_b1 = self.bilinear1[i](x) # [6,2,100,256]
            y_b2 = self.bilinear2[i](y)
            x_b1 = x_b1 * self.coef[i]
            x_att = torch.einsum('abcd,abed->abce', x_b1, y_b2)
            x_att = torch.softmax(x_att, dim = -1)
            y_emb = self.W3[i](y)
            x_i = torch.einsum('abce,abef->abcf', x_att, y_emb)
            x_trans.append(x_i)  # [6,2,100,256/nheads]
        x_trans = torch.cat(x_trans, dim = -1)
        x_trans = self.W1(self.nonlinear(self.LayerNorm(self.W2(x_trans))))
        x_trans = x + x_trans
        return x_trans


class InterLambdaLayer(nn.Module):
    def __init__(self, hidden_dim, nheads):
        super().__init__()
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.bilinear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear1 = _get_clones(self.bilinear1, nheads)
        self.bilinear2 = _get_clones(self.bilinear2, nheads)
        self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])
    
        hid_hid_dim = hidden_dim//nheads
        self.W3 = nn.Linear(hidden_dim, hid_hid_dim)
        self.W3 = _get_clones(self.W3, nheads)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlinear = nn.ReLU(inplace = True)
        self.LayerNorm = nn.LayerNorm([hidden_dim,])
    
    def forward(self, x, y):
        '''
        Gather y features to x.
        x: [6,2,100,256]
        y: [6,2,100,256]
        '''
        x_trans = []
        for i in range(self.nheads):
            x_b1 = self.bilinear1[i](x) # [6,2,100,256]
            y_b2 = self.bilinear2[i](y)
            x_b1 = x_b1 * self.coef[i]
            x_att = torch.einsum('abcd,abed->abce', x_b1, y_b2)
            x_att = torch.softmax(x_att, dim = -1)
            y_emb = self.W3[i](y)
            x_i = torch.einsum('abce,abef->abcf', x_att, y_emb)
            x_trans.append(x_i)  # [6,2,100,256/nheads]
        x_trans = torch.cat(x_trans, dim = -1)
        x_trans = self.W1(self.nonlinear(self.LayerNorm(self.W2(x_trans))))
        x_trans = x + x_trans
        return x_trans



class MHCrossAttLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, relation = 'GClike', dropout = 0.1):
        super().__init__()
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        hid_hid_dim = hidden_dim//nheads
        self.bottleneck_dim = int(self.hidden_dim)
        self.relation = relation

        if self.relation == 'GClike':
            self.vision_W3 = nn.Linear(hidden_dim, hid_hid_dim)
            self.vision_sq = nn.Linear(hidden_dim, hid_hid_dim)
            self.vision_ex = nn.Linear(hid_hid_dim, hid_hid_dim)
            self.vision_W3 = _get_clones(self.vision_W3, nheads)
            self.vision_sq = _get_clones(self.vision_sq, nheads)
            self.vision_ex = _get_clones(self.vision_ex, nheads)
            self.vision_W2 = nn.Linear(hidden_dim, self.bottleneck_dim)
            self.vision_W1 = nn.Linear(self.bottleneck_dim, hidden_dim)
            self.vision_LayerNorm = nn.LayerNorm([self.bottleneck_dim,])

            self.semantic_W3 = nn.Linear(hidden_dim, hid_hid_dim)
            self.semantic_sq = nn.Linear(hidden_dim, hid_hid_dim)
            self.semantic_ex = nn.Linear(hid_hid_dim, hid_hid_dim)
            self.semantic_W3 = _get_clones(self.semantic_W3, nheads)
            self.semantic_sq = _get_clones(self.semantic_sq, nheads)
            self.semantic_ex = _get_clones(self.semantic_ex, nheads)
            self.semantic_W2 = nn.Linear(hidden_dim, self.bottleneck_dim)
            self.semantic_W1 = nn.Linear(self.bottleneck_dim, hidden_dim)
            self.semantic_LayerNorm = nn.LayerNorm([self.bottleneck_dim,])
        
        if self.relation == 'VanillaTrans':
            self.vision_W3 = nn.Linear(hidden_dim, hidden_dim)
            self.vision_sq = nn.Linear(hidden_dim, hid_hid_dim)
            self.vision_ex = nn.Linear(hid_hid_dim, hidden_dim)
            self.vision_W3 = _get_clones(self.vision_W3, nheads)
            self.vision_sq = _get_clones(self.vision_sq, nheads)
            self.vision_ex = _get_clones(self.vision_ex, nheads)
            self.vision_W2 = nn.Linear(hidden_dim, self.bottleneck_dim)
            self.vision_W1 = nn.Linear(self.bottleneck_dim, hidden_dim)
            self.vision_LayerNorm2 = nn.LayerNorm(self.hidden_dim)
            self.vision_LayerNorm1 = nn.LayerNorm(self.hidden_dim)
            self.vision_dropout3 = nn.Dropout(dropout)
            self.vision_dropout2 = nn.Dropout(dropout)
            self.vision_dropout1 = nn.Dropout(dropout)

            self.semantic_W3 = nn.Linear(hidden_dim, hidden_dim)
            self.semantic_sq = nn.Linear(hidden_dim, hid_hid_dim)
            self.semantic_ex = nn.Linear(hid_hid_dim, hidden_dim)
            self.semantic_W3 = _get_clones(self.semantic_W3, nheads)
            self.semantic_sq = _get_clones(self.semantic_sq, nheads)
            self.semantic_ex = _get_clones(self.semantic_ex, nheads)
            self.semantic_W2 = nn.Linear(hidden_dim, self.bottleneck_dim)
            self.semantic_W1 = nn.Linear(self.bottleneck_dim, hidden_dim)
            self.semantic_LayerNorm2 = nn.LayerNorm(self.hidden_dim)
            self.semantic_LayerNorm1 = nn.LayerNorm(self.hidden_dim)
            self.semantic_dropout3 = nn.Dropout(dropout)
            self.semantic_dropout2 = nn.Dropout(dropout)
            self.semantic_dropout1 = nn.Dropout(dropout)

    
    def forward(self, vx, sx):
        if self.relation == 'GClike':
            vx_enhance = []
            for i in range(self.nheads):
                vx_att = torch.sigmoid(self.vision_ex[i](torch.relu(self.vision_sq[i](sx))))
                vx_emb = vx_att * self.vision_W3[i](vx) # Self Aggregation (Initial)
                # vx_emb = vx_att * self.vision_W3[i](sx) # Cross Aggregation
                vx_enhance.append(vx_emb)
            vx_enhance = torch.cat(vx_enhance, dim = -1)
            vx_enhance = vx + self.vision_W1(torch.relu(self.vision_LayerNorm(self.vision_W2(vx_enhance))))

            sx_enhance = []
            for i in range(self.nheads):
                sx_att = torch.sigmoid(self.semantic_ex[i](torch.relu(self.semantic_sq[i](vx))))
                sx_emb = sx_att * self.semantic_W3[i](sx) # Self Aggregation (Initial)
                # sx_emb = sx_att * self.semantic_W3[i](vx) # Cross Aggregation
                sx_enhance.append(sx_emb)
            sx_enhance = torch.cat(sx_enhance, dim = -1)
            sx_enhance = sx + self.semantic_W1(torch.relu(self.semantic_LayerNorm(self.semantic_W2(sx_enhance))))
        

        if self.relation == 'VanillaTrans':
            vx_n = self.vision_LayerNorm2(vx)
            sx_n = self.semantic_LayerNorm2(sx)

            vx_enhance = []
            for i in range(self.nheads):
                vx_att = torch.sigmoid(self.vision_ex[i](torch.relu(self.vision_sq[i](sx_n))))
                vx_emb = vx_att * self.vision_W3[i](vx_n) # Self Aggregation (Initial)
                # vx_emb = vx_att * self.vision_W3[i](sx) # Cross Aggregation
                vx_enhance.append(vx_emb)
            vx_enhance = torch.stack(vx_enhance, dim = -1).sum(dim = -1)
            vx = vx + self.vision_dropout3(vx_enhance)
            vx2 = self.vision_LayerNorm1(vx)
            # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            # src = src + self.dropout2(src2)
            vx2 = self.vision_W1(self.vision_dropout2(torch.relu(self.vision_W2(vx2))))
            vx = vx + self.vision_dropout1(vx2)

            sx_enhance = []
            for i in range(self.nheads):
                sx_att = torch.sigmoid(self.semantic_ex[i](torch.relu(self.semantic_sq[i](vx_n))))
                sx_emb = sx_att * self.semantic_W3[i](sx_n) # Self Aggregation (Initial)
                # sx_emb = sx_att * self.semantic_W3[i](vx) # Cross Aggregation
                sx_enhance.append(sx_emb)
            sx_enhance = torch.stack(sx_enhance, dim = -1).sum(dim = -1)
            sx = sx + self.semantic_dropout3(sx_enhance)
            sx2 = self.semantic_LayerNorm1(sx)
            sx2 = self.semantic_W1(self.semantic_dropout2(torch.relu((self.semantic_W2(sx2)))))
            sx = sx + self.semantic_dropout1(sx)

        return vx_enhance, sx_enhance


class MHSelfAttLayer(nn.Module):
    def __init__(self, hidden_dim, nheads):
        super().__init__()
        self.nheads = nheads
        hid_hid_dim = hidden_dim//nheads
        self.vision_W3 = nn.Linear(hidden_dim, hid_hid_dim)
        self.vision_sq = nn.Linear(hidden_dim, hid_hid_dim)
        self.vision_ex = nn.Linear(hid_hid_dim, hid_hid_dim)
        self.vision_W3 = _get_clones(self.vision_W3, nheads)
        self.vision_sq = _get_clones(self.vision_sq, nheads)
        self.vision_ex = _get_clones(self.vision_ex, nheads)
        self.vision_W2 = nn.Linear(hidden_dim, hidden_dim)
        self.vision_W1 = nn.Linear(hidden_dim, hidden_dim)
        self.vision_LayerNorm = nn.LayerNorm([hidden_dim,])

        self.semantic_W3 = nn.Linear(hidden_dim, hid_hid_dim)
        self.semantic_sq = nn.Linear(hidden_dim, hid_hid_dim)
        self.semantic_ex = nn.Linear(hid_hid_dim, hid_hid_dim)
        self.semantic_W3 = _get_clones(self.semantic_W3, nheads)
        self.semantic_sq = _get_clones(self.semantic_sq, nheads)
        self.semantic_ex = _get_clones(self.semantic_ex, nheads)
        self.semantic_W2 = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_W1 = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_LayerNorm = nn.LayerNorm([hidden_dim,])
    
    def forward(self, vx, sx):
        vx_enhance = []
        for i in range(self.nheads):
            vx_att = torch.sigmoid(self.vision_ex[i](torch.relu(self.vision_sq[i](vx))))
            vx_emb = vx_att * self.vision_W3[i](vx)
            vx_enhance.append(vx_emb)
        vx_enhance = torch.cat(vx_enhance, dim = -1)
        vx_enhance = vx + self.vision_W1(torch.relu(self.vision_LayerNorm(self.vision_W2(vx_enhance))))

        sx_enhance = []
        for i in range(self.nheads):
            sx_att = torch.sigmoid(self.semantic_ex[i](torch.relu(self.semantic_sq[i](sx))))
            sx_emb = sx_att * self.semantic_W3[i](sx)
            sx_enhance.append(sx_emb)
        sx_enhance = torch.cat(sx_enhance, dim = -1)
        sx_enhance = sx + self.semantic_W1(torch.relu(self.semantic_LayerNorm(self.semantic_W2(sx_enhance))))
        
        return vx_enhance, sx_enhance


class VanillaCrossAttLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.vision_W1 = nn.Linear(hidden_dim, hidden_dim)
        self.vision_res = nn.Linear(hidden_dim, hidden_dim)
        self.vision_W2 = nn.Linear(hidden_dim, hidden_dim)

        self.semantic_W1 = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_res = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_W2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, vx, sx):
        '''
        vx: vision features [6,2,100,256]
        sx: semantic features [6,2,100,256]
        '''
        # Inter
        res_vx = self.vision_res(vx)
        att_vx = res_vx + res_vx * torch.sigmoid(self.vision_W1(sx))
        # att_vx = res_vx + res_vx * torch.sigmoid(self.vision_W2(torch.relu(self.vision_W1(sx))))
        res_sx = self.semantic_res(sx)
        att_sx = res_sx + res_sx * torch.sigmoid(self.semantic_W1(vx))
        # att_sx = res_sx + res_sx * torch.sigmoid(self.semantic_W2(torch.relu(self.semantic_W1(vx))))

        return att_vx, att_sx



class CrossModalCalibration(nn.Module):
    def __init__(self, hidden_dim, nlayers = 1):
        super().__init__()
        # Inter
        # self.vision_W1 = nn.Linear(hidden_dim, hidden_dim)
        # self.vision_res = nn.Linear(hidden_dim, hidden_dim)
        # self.vision_W2 = nn.Linear(hidden_dim, hidden_dim)
        # self.semantic_W1 = nn.Linear(hidden_dim, hidden_dim)
        # self.semantic_res = nn.Linear(hidden_dim, hidden_dim)
        # self.semantic_W2 = nn.Linear(hidden_dim, hidden_dim)
        # Intra
        # self.vision_intra_trans = TransformerLayer(hidden_dim, nheads = 2)
        # self.semantic_intra_trans = TransformerLayer(hidden_dim, nheads = 2)

        self.nlayers = nlayers
        # Inter-Modal Calibration (InterC)
        # self.CrossAtt = VanillaCrossAttLayer(hidden_dim)
        # self.CrossAtt = MHCrossAttLayer(hidden_dim, nheads = 4, relation = 'VanillaTrans')
        self.CrossAtt = MHCrossAttLayer(hidden_dim, nheads = 2)
        self.CrossAtt = _get_clones(self.CrossAtt, nlayers)
        # self.SelfAtt = MHSelfAttLayer(hidden_dim, nheads=2)
        # self.SelfAtt = _get_clones(self.SelfAtt, nlayers)

        # Inter-Transformer
        # self.vision_inter_trans = InterTransformerLayer(hidden_dim, nheads=2)
        # self.semantic_inter_trans = InterTransformerLayer(hidden_dim, nheads=2)
        # self.vision_inter_trans = _get_clones(self.vision_inter_trans, nlayers)
        # self.semantic_inter_trans = _get_clones(self.semantic_inter_trans, nlayers)

        # Intra-Modal Enhance Calibration (IntraEC)
        # self.vision_intra_trans = TransformerLayer(hidden_dim, nheads = 2, relation = 'embedded_dot_pro')
        # self.semantic_intra_trans = TransformerLayer(hidden_dim, nheads = 2, relation = 'embedded_dot_pro')
        # self.vision_intra_trans = TransformerLayer(hidden_dim, nheads = 4, relation = 'VanillaTrans')
        # self.semantic_intra_trans = TransformerLayer(hidden_dim, nheads = 4, relation = 'VanillaTrans')
        self.vision_intra_trans = TransformerLayer(hidden_dim, nheads = 2)
        self.semantic_intra_trans = TransformerLayer(hidden_dim, nheads = 2)
        self.vision_intra_trans = _get_clones(self.vision_intra_trans, nlayers)
        self.semantic_intra_trans = _get_clones(self.semantic_intra_trans, nlayers)

    def forward(self, vx, sx):
        '''
        vx: vision features [6,2,100,256]
        sx: semantic features [6,2,100,256]
        '''

        for l in range(self.nlayers):
            # MH2CrossAttLayer_intraTrans2_nlayers1, Highest
            # Inter
            att_vx, att_sx = self.CrossAtt[l](vx, sx)
            # Intra
            vx = self.vision_intra_trans[l](att_vx)
            sx = self.semantic_intra_trans[l](att_sx)
            # vx, sx = att_vx, att_sx
            # vx = self.vision_intra_trans[l](vx)
            # sx = self.semantic_intra_trans[l](sx)

            # # SelfAtt
            # # Inter
            # att_vx, att_sx = self.SelfAtt[l](vx, sx)
            # att_vx, att_sx = self.CrossAtt[l](att_vx, att_sx)
            # # Intra
            # vx = self.vision_intra_trans[l](att_vx)
            # sx = self.semantic_intra_trans[l](att_sx)

            # # Inter-Intra
            # # Inter
            # att_vx = self.vision_inter_trans[l](vx, sx)
            # att_sx = self.semantic_inter_trans[l](sx, vx)
            # # Intra
            # vx = self.vision_intra_trans[l](att_vx)
            # sx = self.semantic_intra_trans[l](att_sx)

            # # CrossEnhance-Inter-Intra
            # # Inter
            # att_vx, att_sx = self.CrossAtt[l](vx, sx)
            # att_vx = self.vision_inter_trans[l](att_vx, att_sx)
            # att_sx = self.semantic_inter_trans[l](att_sx, att_vx)
            # # Intra
            # vx = self.vision_intra_trans[l](att_vx)
            # sx = self.semantic_intra_trans[l](att_sx)

            # Inter-Intra-CrossEnhance
            # # Inter
            # att_vx = self.vision_inter_trans[l](vx, sx)
            # att_sx = self.semantic_inter_trans[l](sx, vx)
            # # Intra
            # att_vx = self.vision_intra_trans[l](att_vx)
            # att_sx = self.semantic_intra_trans[l](att_sx)
            # # Cross-Enhance
            # vx, sx = self.CrossAtt[l](att_vx, att_sx)

        return vx, sx



class CrossModalityGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, attention_type='multihead_transformer'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_type = attention_type

        if attention_type == 'embedded_dot_pro':
            self.semantic_q = [nn.Linear(input_dim, hidden_dim),]
            self.semantic_k = [nn.Linear(input_dim, hidden_dim),]
            self.semantic_v = [nn.Linear(input_dim, hidden_dim),]
            # self.semantic_proj_res = nn.Linear(input_dim, hidden_dim)
            for _ in range(num_layers-1):
                self.semantic_q.append(nn.Linear(hidden_dim, hidden_dim))
                self.semantic_k.append(nn.Linear(hidden_dim, hidden_dim))
                self.semantic_v.append(nn.Linear(hidden_dim, hidden_dim))
            self.semantic_q = nn.ModuleList(self.semantic_q)
            self.semantic_k = nn.ModuleList(self.semantic_k)
            self.semantic_v = nn.ModuleList(self.semantic_v)
        elif attention_type == 'multihead_transformer':
            assert self.num_layers == 1
            self.head_num = 4
            self.semantic_q = nn.Linear(input_dim, hidden_dim)
            self.semantic_k = nn.Linear(input_dim, hidden_dim)
            self.semantic_q = _get_clones(self.semantic_q, self.head_num)
            self.semantic_k = _get_clones(self.semantic_k, self.head_num)
            self.semantic_v = nn.Linear(input_dim, hidden_dim)
            self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(self.head_num)])

            self.LayerNorm = nn.LayerNorm([hidden_dim,])
            self.W_t1 = nn.Linear(hidden_dim*self.head_num, hidden_dim)
            self.W_t2 = nn.Linear(hidden_dim, hidden_dim)
        

        self.fusion_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_2 = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_gate = nn.Linear(hidden_dim, hidden_dim)
            
    
    def forward(self, x, y, cooccur_prior = None):
        '''
        x : vision features [6, 2, 100, 256]
        y : language features [117, 256]
        '''
        if self.attention_type == 'embedded_dot_pro':
            assert self.num_layers == 1
            for i in range(self.num_layers):
                x_q = self.semantic_q[i](x)
                y_k = self.semantic_k[i](y)
                y_v = self.semantic_v[i](y)
                # x_att = torch.einsum('ac,bc->ab', x_q, x_k)
                x_att = torch.einsum('abce,de->abcd', x_q, y_k) / math.sqrt(self.hidden_dim) # [6, 2, 100, 117]
                x_att = F.softmax(x_att, dim = -1)
                if cooccur_prior is not None:
                    x_att = x_att + cooccur_prior
                    # print('cooccur')
                semantic_agg = torch.einsum('abcd,de->abce', x_att, y_v)
            return semantic_agg
            
                
                # if i == 0:
                #     x = F.relu(torch.matmul(x_att, x_v)) + self.semantic_proj_res(x) # self.verb_calibration_embedding
                # else:
                #     x = F.relu(torch.matmul(x_att, x_v)) + x
        
        if self.attention_type == 'multihead_transformer':
            assert len(x.shape) == 4
            l, bs, q, hiddim = x.shape
            x = x.reshape((l*bs, q, hiddim))

            assert len(y.shape) == 2
            y = y.unsqueeze(dim = 0)

            y_v = self.semantic_v(y).expand(l*bs, -1, -1)
            multihead_ft = []
            for i in range(self.head_num):
                x_q = self.semantic_q[i](x)  # lbs, q, hiddim
                y_k = self.semantic_k[i](y).expand(l*bs, -1, -1)  # lbs, q, hiddim
                y_k = y_k * self.coef[i].expand_as(y_k)

                x_att = torch.einsum('abc,adc->abd', x_q, y_k)
                x_att = F.softmax(x_att, dim = -1)
                att_ft = torch.bmm(x_att, y_v)
                multihead_ft.append(att_ft)

            multihead_ft = torch.cat(multihead_ft, dim = -1)
            semantic_aug = self.W_t2(F.relu(self.LayerNorm(self.W_t1(multihead_ft)), inplace = True))
            semantic_aug = semantic_aug.view((l, bs, q, hiddim))

            modality_fus = count_fusion(self.fusion_1(semantic_aug), self.fusion_2(x.view((l, bs, q, hiddim))))
            # semantic_gate2 = torch.sigmoid(self.semantic_gate2(x))
            # modality_fus = count_fusion(self.fusion_1(semantic_gate2 * trans_ft), self.fusion_2(x))
            return modality_fus



# inf_time_list = [[],[],[],[],[]]
class OCN(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, dataset = 'hico', aux_loss = False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_verb_classes = num_verb_classes
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.transformer.decoder.obj_class_embed = self.obj_class_embed
        # self.transformer.decoder.verb_class_embed = self.verb_class_embed
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # Initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.obj_class_embed.bias.data = torch.ones(num_obj_classes+1) * bias_value
        self.verb_class_embed.bias.data = torch.ones(num_verb_classes) * bias_value
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)
        # torch.nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        # torch.nn.init.constant_(self.reference_points.bias.data, 0.)

        
        # obj_verb_co with smoothing v2 (81 is uniform)
        if dataset == 'hico':
            # Laplacian Smoothing (Also dubbed Additive Smoothing)
            obj_verb_co = np.load('datasets/priors/obj_verb_cooccurrence.npz')['cond_prob_co_matrices']
            obj_verb_co = torch.cat((torch.tensor(obj_verb_co).float(), torch.zeros((1, num_verb_classes))), dim = 0)
            obj_verb_co = obj_verb_co + 0.1/obj_verb_co.shape[1]
            # obj_verb_co = torch.ones(obj_verb_co.shape)  # beta = infinity
            # obj_verb_co = torch.cat((torch.tensor(obj_verb_co).float(), torch.ones((1, num_verb_classes))), dim = 0)
                # obj_verb_co = obj_verb_co / np.expand_dims(obj_verb_co.sum(axis=1), axis = 1)
            obj_verb_co = obj_verb_co / obj_verb_co.sum(dim=1).unsqueeze(dim = 1)
            self.register_buffer('obj_verb_co', obj_verb_co)
            print('obj_verb_co has nan ? ' + str(np.isnan(obj_verb_co).sum()))

            # Jelinek-Mercer Method
            # obj_verb_co = np.load('datasets/priors/obj_verb_cooccurrence.npz')['cond_prob_co_matrices']
            # obj_verb_co = torch.cat((torch.tensor(obj_verb_co).float(), torch.zeros((1, num_verb_classes))), dim = 0)
            # obj_verb_co = obj_verb_co * (1 - 0.7) + 0.7 / obj_verb_co.shape[1]
            # obj_verb_co = obj_verb_co / obj_verb_co.sum(dim=1).unsqueeze(dim = 1)
            # self.register_buffer('obj_verb_co', obj_verb_co)
            # print('obj_verb_co has nan ? ' + str(np.isnan(obj_verb_co).sum()))

        elif dataset == 'vcoco':
            obj_verb_co = np.load('datasets/priors/obj_verb_cooccurrence_vcoco.npz')['joint_prob_co_matrices']
            print('obj_verb_co has nan ? ' + str(np.isnan(obj_verb_co).sum()))
            obj_verb_co[np.isnan(obj_verb_co)] = 0.1/obj_verb_co.shape[1]  # Eliminate nan entries in the matrix
            obj_verb_co = torch.cat((torch.tensor(obj_verb_co).float(), torch.zeros((1, num_verb_classes))), dim = 0)
            obj_verb_co = obj_verb_co + 0.1/obj_verb_co.shape[1] 
            obj_verb_co = obj_verb_co / obj_verb_co.sum(dim=1).unsqueeze(dim = 1)
            self.register_buffer('obj_verb_co', obj_verb_co)
            print('obj_verb_co has nan ? ' + str(np.isnan(obj_verb_co).sum()))



        # verb_verb_co with smoothing
        if dataset == 'hico':
            verb_verb_co = np.load('datasets/priors/verb_verb_cooccurrence.npz')['cond_prob_co_matrices']  # 实际上用的是Joint Probability
            verb_verb_co = verb_verb_co / np.expand_dims(verb_verb_co.sum(axis=1), axis = 1)
            verb_verb_co[np.isnan(verb_verb_co)] = 0  # add to prevent nan
            self.register_buffer('verb_verb_co', torch.tensor(verb_verb_co).float())
            print('verb_verb_co has nan ? ' + str(np.isnan(verb_verb_co).sum()))
            print('verb_verb_co sum: ' + str(verb_verb_co.sum()))

        elif dataset == 'vcoco':
            verb_verb_co = np.load('datasets/priors/verb_verb_cooccurrence_vcoco.npz')['cond_prob_co_matrices']  # 实际上用的是Joint Probability
            verb_verb_co = verb_verb_co / np.expand_dims(verb_verb_co.sum(axis=1), axis = 1)
            verb_verb_co[np.isnan(verb_verb_co)] = 0  # add to prevent nan
            self.register_buffer('verb_verb_co', torch.tensor(verb_verb_co).float())
            print('verb_verb_co sum: ' + str(verb_verb_co.sum()))


        # verb word embedding
        if dataset == 'hico':
            # Prerained Model embedding
            verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/hico_verb_glove-wiki-gigaword-300.npz')['embedding_list']) # [:,None]
            # verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/hico_verb_glove-wiki-gigaword-50.npz')['embedding_list']) # [:,None]
            # verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/hico_verb_fasttext-wiki-news-subwords-300.npz')['embedding_list'])# [:,None]
            # verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/hico_verb_word2vec-google-news-300.npz')['embedding_list'])# [:,None]
            verb_word_embedding = norm_tensor(verb_word_embedding)
            self.register_buffer('verb_word_embedding', verb_word_embedding)

            # # one_hot verb embedding
            # verb_word_embedding = torch.eye(num_verb_classes)
            # self.register_buffer('verb_word_embedding', verb_word_embedding)
        elif dataset == 'vcoco':
            verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/vcoco_verb_glove-wiki-gigaword-300.npz')['embedding_list'])# [:,None]
            # verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/vcoco_verb_fasttext-wiki-news-subwords-300.npz')['embedding_list'])# [:,None]
            # verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/vcoco_verb_word2vec-google-news-300.npz')['embedding_list'])# [:,None]
            verb_word_embedding = norm_tensor(verb_word_embedding)
            self.register_buffer('verb_word_embedding', verb_word_embedding)
        
        # Semantic Reasoning
        self.semantic_graph = SemanticGraph(300, 256, 1, attention_type='embedded_dot_pro')
        # self.semantic_graph = SemanticGraph(117, 256, 1, attention_type='MLP_GNN')
        # self.semantic_graph = SemanticGraph(300, 256, 1, attention_type='multihead_transformer', head_num = 2)
        # self.semantic_obj_graph = SemanticGraph(300, 256, 1, attention_type='embedded_dot_pro')
        # self.semantic_graph = SemanticGraph(117, 256, 1, attention_type='MLP')
        

        # Cross modality operation
        # self.cross_modality_graph = CrossModalityGraph(hidden_dim, hidden_dim, 1, attention_type='multihead_transformer')
        # self.cross_modality_graph = CrossModalityGraph(hidden_dim, hidden_dim, 1, attention_type='embedded_dot_pro')
        # self.semantic_gate1 = nn.Linear(hidden_dim, num_verb_classes)
        # self.semantic_gate2 = nn.Linear(hidden_dim, hidden_dim)
        # self.hs_gate = nn.Linear(hidden_dim, hidden_dim)
        # self.semantic_gate2_1 = nn.Linear(hidden_dim, hidden_dim//16)
        # self.semantic_gate2_2 = nn.Linear(hidden_dim//16, hidden_dim)
        self.fusion_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_2 = nn.Linear(hidden_dim, hidden_dim)
        self.cross_modal_calibration = CrossModalCalibration(hidden_dim, nlayers = 1)

    def forward(self, samples: NestedTensor, **kwargs):
        inf_time = time.time()
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # inf_time_list[0].append(time.time()-inf_time)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None
        # self.transformer return decoder info and memory
        # hs: tensor [6, 2, 100, 256]  6 is the #decoder_layers

        # New semantic implementation
        # semantic = self.semantic_graph(self.verb_word_embedding, self.verb_verb_co)
        semantic = self.semantic_graph(self.verb_word_embedding)

        # inf_time_list[1].append(time.time()-inf_time)
        # Save semantic embedding
        # np.savez_compressed('GloVe_semantic_embeddings.npz', 
        #                      semantic = np.array(semantic.cpu()))
        
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # self.transformer return decoder info and memory
        # hs: tensor [6, 2, 100, 256]  6 is the #decoder_layers

        outputs_obj_class = self.obj_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
        # inf_time_list[3].append(time.time()-inf_time)

        
        # cross_enhance
        # Attention aggregation
        # semantic_aug = self.cross_modality_graph(hs, semantic)
        # Statistical Prior Aggregation
        outputs_obj_81 = outputs_obj_class.argmax(dim =-1).unsqueeze(-1).expand(-1,-1,-1,self.num_verb_classes) # [6,2,100]
        obj_verb_co = self.obj_verb_co.expand(outputs_obj_81.shape[:-2]+(-1,-1))
        outputs_obj_co = torch.gather(obj_verb_co, dim =2, index = outputs_obj_81) # [6, 2, 100, 117]
        semantic_aug = torch.einsum('abcd,de->abce', outputs_obj_co, semantic)
        cross_hs, cross_semantic_aug = self.cross_modal_calibration(hs, semantic_aug)
        hs_aug = count_fusion(self.fusion_1(cross_hs), self.fusion_2(cross_semantic_aug))


        # Verb Model
            # vanilla
        outputs_verb_class = self.verb_class_embed(hs_aug)


        # Original
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1], 'semantic':semantic, 'verb_verb_co':self.verb_verb_co,}# 'joint_verb_verb_co':self.joint_verb_verb_co,} # 'semantic_low':semantic_low}
        if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)
        return out

    @torch.jit.unused   # Original
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class SeqDETRHOI(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        # self.verb_class_embed = MLP(hidden_dim, hidden_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, **kwargs):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None
        h_decoder_out, obj_decoder_out, verb_decoder_out, hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        
        # self.transformer return decoder info and memory
        # hs: tensor [6, 2, 100, 256]  6 is the #decoder_layers

        outputs_obj_class = self.obj_class_embed(obj_decoder_out)
        outputs_verb_class = self.verb_class_embed(verb_decoder_out)
        outputs_sub_coord = self.sub_bbox_embed(h_decoder_out).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(obj_decoder_out).sigmoid()
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]

class SepDETRHOI(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries*2, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        # self.verb_class_embed = MLP(hidden_dim, hidden_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, **kwargs):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None


        # SepDETRHOI without sharing decoding
        h_decoder_out, obj_decoder_out, verb_decoder_out, hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        outputs_obj_class = self.obj_class_embed(obj_decoder_out)
        outputs_verb_class = self.verb_class_embed(verb_decoder_out)
        outputs_sub_coord = self.sub_bbox_embed(h_decoder_out).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(obj_decoder_out).sigmoid()
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.aux_loss:  
            # Using aux loss means that you will add loss to every intermidiate layer.
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord) 
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class ParSe(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False, subject_class = False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries*2, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        # self.verb_class_embed = MLP(hidden_dim, hidden_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.subject_class = subject_class

    def forward(self, samples: NestedTensor, **kwargs):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print(len(pos))
        # print(pos[-1].shape)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None

        h_decoder_out, obj_decoder_out, verb_decoder_out, hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        outputs_obj_class = self.obj_class_embed(obj_decoder_out)
        outputs_verb_class = self.verb_class_embed(verb_decoder_out)
        outputs_sub_coord = self.sub_bbox_embed(h_decoder_out).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(obj_decoder_out).sigmoid()
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        
        if self.subject_class:
            outputs_sub_class = self.obj_class_embed(h_decoder_out)
            out.update({'pred_sub_logits': outputs_sub_class[-1]})
            if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord, outputs_sub_class) 
        else:
            if self.aux_loss: 
                # Using aux loss means that you will add loss to every intermidiate layer.
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord) 
        return out

    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_sub_class = None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_sub_class is None:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                        outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        else:
            return [{'pred_sub_logits':a, 'pred_obj_logits': b, 'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_obj_boxes': e}
                for a, b, c, d, e in zip(outputs_sub_class[:-1], outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class ParSeDABDETR(nn.Module):
    """ This is the DAB-DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, 
                    aux_loss=False, 
                    iter_update=True,
                    query_dim=4, 
                    bbox_embed_diff_each_layer=False,
                    random_refpoints_xy=False,
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            iter_update: iterative update of boxes
            query_dim: query dimension. 2 for point and 4 for box.
            bbox_embed_diff_each_layer: dont share weights of prediction heads. Default for True.(shared weights.)
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
            
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)

        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        if bbox_embed_diff_each_layer:
            # self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(6)])
            self.sub_bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(6)])
            self.obj_bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(6)])
        else:
            # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        

        # setting query dim
        self.query_dim = query_dim
        assert query_dim in [2, 4]

        self.refpoint_embed = nn.Embedding(num_queries * 2, query_dim)
        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.iter_update = iter_update

        if self.iter_update:
            # self.transformer.decoder.bbox_embed = self.bbox_embed
            self.transformer.ho_decoder.sub_bbox_embed = self.sub_bbox_embed
            self.transformer.ho_decoder.obj_bbox_embed = self.obj_bbox_embed
            self.transformer.verb_decoder.sub_bbox_embed = self.sub_bbox_embed
            self.transformer.verb_decoder.obj_bbox_embed = self.obj_bbox_embed


        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.verb_class_embed.bias.data = torch.ones(num_verb_classes) * bias_value
        self.obj_class_embed.bias.data = torch.ones(num_obj_classes + 1) * bias_value
        # self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # import ipdb; ipdb.set_trace()
        # init bbox_embed
        if bbox_embed_diff_each_layer:
            for sub_bbox_embed in self.sub_bbox_embed:
                nn.init.constant_(sub_bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(sub_bbox_embed.layers[-1].bias.data, 0)
            for obj_bbox_embed in self.obj_bbox_embed:
                nn.init.constant_(obj_bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(obj_bbox_embed.layers[-1].bias.data, 0)
        else:
            # nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            # nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
            nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
            nn.init.constant_(self.obj_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data, 0)

        

    def forward(self, samples: NestedTensor, **kwargs):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        # default pipeline
        embedweight = self.refpoint_embed.weight
        h_hs, o_hs, verb_hs, h_references, o_references = self.transformer(self.input_proj(src), mask, embedweight, pos[-1])
        # hs, reference = self.transformer(self.input_proj(src), mask, embedweight, pos[-1])
        
        if not self.bbox_embed_diff_each_layer:
            # Default
            h_reference_before_sigmoid = inverse_sigmoid(h_references)
            h_tmp = self.sub_bbox_embed(h_hs)
            h_tmp[..., :self.query_dim] += h_reference_before_sigmoid
            outputs_sub_coord = h_tmp.sigmoid()

            o_reference_before_sigmoid = inverse_sigmoid(o_references)
            o_tmp = self.obj_bbox_embed(o_hs)
            o_tmp[..., :self.query_dim] += o_reference_before_sigmoid
            outputs_obj_coord = o_tmp.sigmoid()
        else:
            h_reference_before_sigmoid = inverse_sigmoid(h_references)
            o_reference_before_sigmoid = inverse_sigmoid(o_references)
            outputs_sub_coords = []
            outputs_obj_coords = []
            for lvl in range(h_hs.shape[0]):
                tmp = self.sub_bbox_embed[lvl](h_hs[lvl])
                tmp[..., :self.query_dim] += h_reference_before_sigmoid[lvl]
                outputs_coord = tmp.sigmoid()
                outputs_sub_coords.append(outputs_coord)

                tmp = self.obj_bbox_embed[lvl](o_hs[lvl])
                tmp[..., :self.query_dim] += o_reference_before_sigmoid[lvl]
                outputs_coord = tmp.sigmoid()
                outputs_obj_coords.append(outputs_coord)
            outputs_sub_coord = torch.stack(outputs_sub_coords)
            outputs_obj_coord = torch.stack(outputs_obj_coords)

        # outputs_class = self.class_embed(hs)
        outputs_obj_class = self.obj_class_embed(o_hs)
        outputs_verb_class = self.verb_class_embed(verb_hs)
        # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        
        if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord) 
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]



class ParSeDABDDETR(nn.Module):
    """ This is the DAB-Deformable-DETR for object detection """
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=True, two_stage=False,
                 use_dab=True, 
                 num_patterns=0,
                 random_refpoints_xy=False,
                 subject_class = False
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            use_dab: using dynamic anchor boxes formulation
            num_patterns: number of pattern embeddings
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_dab = use_dab
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
                self.verb_tgt_embed = nn.Embedding(num_queries, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries, 4)
                if random_refpoints_xy:
                    # import ipdb; ipdb.set_trace()
                    self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False
        
        assert self.num_patterns == 0 
        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.verb_class_embed.bias.data = torch.ones(num_verb_classes) * bias_value
        self.obj_class_embed.bias.data = torch.ones(num_obj_classes + 1) * bias_value
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.ho_decoder.num_layers
        if with_box_refine:
            self.verb_class_embed = _get_clones(self.verb_class_embed, num_pred)
            self.obj_class_embed = _get_clones(self.obj_class_embed, num_pred)

            self.sub_bbox_embed = _get_clones(self.sub_bbox_embed, num_pred * 2)
            nn.init.constant_(self.sub_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.ho_decoder.sub_bbox_embed = self.sub_bbox_embed[:num_pred]
            self.transformer.verb_decoder.sub_bbox_embed = self.sub_bbox_embed[num_pred:]

            self.obj_bbox_embed = _get_clones(self.obj_bbox_embed, num_pred * 2)
            nn.init.constant_(self.obj_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.ho_decoder.obj_bbox_embed = self.obj_bbox_embed[:num_pred]
            self.transformer.verb_decoder.obj_bbox_embed = self.obj_bbox_embed[num_pred:]
        else:
            nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.verb_class_embed = nn.ModuleList([self.verb_class_embed for _ in range(num_pred)])
            self.obj_class_embed = nn.ModuleList([self.obj_class_embed for _ in range(num_pred)])
            self.sub_bbox_embed = nn.ModuleList([self.sub_bbox_embed for _ in range(num_pred)])
            self.obj_bbox_embed = nn.ModuleList([self.obj_bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.sub_bbox_embed = None
            self.transformer.decoder.obj_bbox_embed = None
        

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        
        self.subject_class = subject_class

    def forward(self, samples: NestedTensor, **kwargs):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # import ipdb; ipdb.set_trace()

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.num_patterns == 0:
                # Currently, we only support cases when self.num_patterns == 0.
                tgt_embed = self.tgt_embed.weight           # nq, 256
                verb_tgt_embed = self.verb_tgt_embed.weight # nq, 256
                refanchor = self.refpoint_embed.weight      # nq, 4
                query_embeds = torch.cat((tgt_embed, verb_tgt_embed, refanchor), dim=1)
            else:
                # TODO
                # multi patterns
                tgt_embed = self.tgt_embed.weight           # nq, 256
                pat_embed = self.patterns_embed.weight      # num_pat, 256
                tgt_embed = tgt_embed.repeat(self.num_patterns, 1) # nq*num_pat, 256
                pat_embed = pat_embed[:, None, :].repeat(1, self.num_queries, 1).flatten(0, 1) # nq*num_pat, 256
                tgt_all_embed = tgt_embed + pat_embed
                refanchor = self.refpoint_embed.weight.repeat(self.num_patterns, 1)      # nq*num_pat, 4
                query_embeds = torch.cat((tgt_all_embed, refanchor), dim=1)
        else:
            query_embeds = self.query_embed.weight
        # hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
        hs_ho, hs_verb, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
        hs_h, hs_o = hs_ho[:,:,:self.num_queries//2], hs_ho[:,:,self.num_queries//2:]

        outputs_verb_classes = []
        outputs_obj_classes = []
        outputs_sub_coords = []
        outputs_obj_coords = []
        for lvl in range(hs_ho.shape[0]):
            if lvl == 0:
                sub_reference, obj_reference = init_reference  # ref_points = 4
                # sub_reference = obj_reference = init_reference # ref_points = 2
            else:
                sub_reference, obj_reference = inter_references[lvl - 1]
            sub_reference = inverse_sigmoid(sub_reference)
            obj_reference = inverse_sigmoid(obj_reference)

            sub_tmp = self.sub_bbox_embed[lvl](hs_h[lvl])
            if sub_reference.shape[-1] == 4:
                sub_tmp += sub_reference
            else:
                assert sub_reference.shape[-1] == 2
                sub_tmp[..., :2] += sub_reference
            outputs_sub_coord = sub_tmp.sigmoid()
            outputs_sub_coords.append(outputs_sub_coord)

            obj_tmp = self.obj_bbox_embed[lvl](hs_o[lvl])
            if obj_reference.shape[-1] == 4:
                obj_tmp += obj_reference
            else:
                assert obj_reference.shape[-1] == 2
                obj_tmp[..., :2] += obj_reference
            outputs_obj_coord = obj_tmp.sigmoid()
            outputs_obj_coords.append(outputs_obj_coord)

            outputs_verb_class = self.verb_class_embed[lvl](hs_verb[lvl])
            outputs_obj_class = self.obj_class_embed[lvl](hs_o[lvl])
            outputs_verb_classes.append(outputs_verb_class)
            outputs_obj_classes.append(outputs_obj_class)

        # outputs_class = torch.stack(outputs_classes)
        # outputs_coord = torch.stack(outputs_coords)
        outputs_verb_class = torch.stack(outputs_verb_classes)
        outputs_obj_class = torch.stack(outputs_obj_classes)
        outputs_sub_coord = torch.stack(outputs_sub_coords)
        outputs_obj_coord = torch.stack(outputs_obj_coords)


        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.subject_class:
            outputs_sub_class = []
            for lvl in range(hs_h.shape[0]):
                outputs_sub_class.append(self.obj_class_embed[lvl](hs_h[lvl]))
            outputs_sub_class = torch.stack(outputs_sub_class, dim = 0)

            out.update({'pred_sub_logits': outputs_sub_class[-1]})
            if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord, outputs_sub_class) 
        else:
            if self.aux_loss: 
                # Using aux loss means that you will add loss to every intermidiate layer.
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord) 

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_sub_class = None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_sub_class is None:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                        outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        else:
            return [{'pred_sub_logits':a, 'pred_obj_logits': b, 'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_obj_boxes': e}
                for a, b, c, d, e in zip(outputs_sub_class[:-1], outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class RLIP_ParSeDA(nn.Module):
    """ This is the DAB-Deformable-DETR for object detection """
    def __init__(self, backbone, transformer, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=True, two_stage=False,
                 use_dab=True, 
                 num_patterns=0,
                 random_refpoints_xy=False,
                 subject_class = False,
                 pseudo_verb=False,
                 args=None
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            use_dab: using dynamic anchor boxes formulation
            num_patterns: number of pattern embeddings
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # self.class_embed = nn.Linear(hidden_dim, num_classes)
        # self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        # self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_dab = use_dab
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy

        ### version 5 for classification
        self.projection_text = nn.Linear(hidden_dim, hidden_dim)
        prior_prob = 0.01
        self.bias_c = -math.log((1 - prior_prob) / prior_prob)
        self.bias_obj_a = nn.Parameter(torch.zeros((256,), dtype = torch.float32), requires_grad = True)
        self.bias_pred_a = nn.Parameter(torch.zeros((256,), dtype = torch.float32), requires_grad = True)

        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
                self.verb_tgt_embed = nn.Embedding(num_queries, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries, 4)
                if random_refpoints_xy:
                    # import ipdb; ipdb.set_trace()
                    self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False
        
        assert self.num_patterns == 0 
        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.verb_class_embed.bias.data = torch.ones(num_verb_classes) * bias_value
        # self.obj_class_embed.bias.data = torch.ones(num_obj_classes + 1) * bias_value
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.ho_decoder.num_layers
        if with_box_refine:
            # self.verb_class_embed = _get_clones(self.verb_class_embed, num_pred)
            # self.obj_class_embed = _get_clones(self.obj_class_embed, num_pred)

            self.sub_bbox_embed = _get_clones(self.sub_bbox_embed, num_pred * 2)
            nn.init.constant_(self.sub_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.ho_decoder.sub_bbox_embed = self.sub_bbox_embed[:num_pred]
            self.transformer.verb_decoder.sub_bbox_embed = self.sub_bbox_embed[num_pred:]

            self.obj_bbox_embed = _get_clones(self.obj_bbox_embed, num_pred * 2)
            nn.init.constant_(self.obj_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.ho_decoder.obj_bbox_embed = self.obj_bbox_embed[:num_pred]
            self.transformer.verb_decoder.obj_bbox_embed = self.obj_bbox_embed[num_pred:]
        else:
            nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data[2:], -2.0)
            # self.verb_class_embed = nn.ModuleList([self.verb_class_embed for _ in range(num_pred)])
            # self.obj_class_embed = nn.ModuleList([self.obj_class_embed for _ in range(num_pred)])
            self.sub_bbox_embed = nn.ModuleList([self.sub_bbox_embed for _ in range(num_pred)])
            self.obj_bbox_embed = nn.ModuleList([self.obj_bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.sub_bbox_embed = None
            self.transformer.decoder.obj_bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        
        self.subject_class = subject_class
        self.pseudo_verb = pseudo_verb
        self.pseudo_verb_mode = "online"    # "offline"



    def forward(self, samples: NestedTensor,
                      encode_and_save=True, 
                      memory_cache=None,
                      **kwargs):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        if encode_and_save:
            features, pos = self.backbone(samples)

            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)
                assert mask is not None
            # import ipdb; ipdb.set_trace()

            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)

            if self.two_stage:
                query_embeds = None
            elif self.use_dab:
                if self.num_patterns == 0:
                    # Currently, we only support cases when self.num_patterns == 0.
                    tgt_embed = self.tgt_embed.weight           # nq, 256
                    verb_tgt_embed = self.verb_tgt_embed.weight # nq, 256
                    refanchor = self.refpoint_embed.weight      # nq, 4
                    query_embeds = torch.cat((tgt_embed, verb_tgt_embed, refanchor), dim=1)
                else:
                    # TODO
                    # multi patterns
                    tgt_embed = self.tgt_embed.weight           # nq, 256
                    pat_embed = self.patterns_embed.weight      # num_pat, 256
                    tgt_embed = tgt_embed.repeat(self.num_patterns, 1) # nq*num_pat, 256
                    pat_embed = pat_embed[:, None, :].repeat(1, self.num_queries, 1).flatten(0, 1) # nq*num_pat, 256
                    tgt_all_embed = tgt_embed + pat_embed
                    refanchor = self.refpoint_embed.weight.repeat(self.num_patterns, 1)      # nq*num_pat, 4
                    query_embeds = torch.cat((tgt_all_embed, refanchor), dim=1)
            else:
                query_embeds = self.query_embed.weight
            
            # targets = None # kwargs['targets'] if self.verb_tagger else None

            memory_cache = self.transformer(srcs = srcs, 
                                            masks = masks,
                                            pos_embeds = pos, 
                                            query_embed = query_embeds,
                                            text = kwargs['text'],
                                            encode_and_save = True,)
                                            # targets = targets)
            return memory_cache
        else:
            hs_ho_dec, hs_verb_dec, text_dec, init_reference, inter_references, hs_ho_bf_fusion, hs_verb_bf_fusion, enc_outputs_class, enc_outputs_coord_unact = \
                        self.transformer(masks = memory_cache["masks"],
                                         query_embed = memory_cache["ho_query_embed"],
                                         encode_and_save = False,
                                         text_memory=memory_cache["text_memory_resized"],
                                         img_memory=memory_cache["img_memory"],
                                         text_attention_mask=memory_cache["text_attention_mask"],
                                         obj_pred_names_sums=memory_cache["obj_pred_names_sums"],
                                         spatial_shapes=memory_cache["spatial_shapes"],
                                         level_start_index=memory_cache["level_start_index"],
                                         valid_ratios=memory_cache["valid_ratios"])

            hs_h, hs_o = hs_ho_dec[:,:,:self.num_queries//2], hs_ho_dec[:,:,self.num_queries//2:]
            obj_pred_names_sums = memory_cache["obj_pred_names_sums"]
            max_obj_text_len = torch.max(obj_pred_names_sums[:,0])
            max_pred_text_len = torch.max(obj_pred_names_sums[:,1])
            text_len = max_obj_text_len + max_pred_text_len 

            outputs_verb_classes = []
            outputs_obj_classes = []
            outputs_sub_classes = []
            outputs_sub_coords = []
            outputs_obj_coords = []
            for lvl in range(hs_h.shape[0]):
                if lvl == 0:
                    sub_reference, obj_reference = init_reference  # ref_points = 4
                    # sub_reference = obj_reference = init_reference # ref_points = 2
                else:
                    sub_reference, obj_reference = inter_references[lvl - 1]
                sub_reference = inverse_sigmoid(sub_reference)
                obj_reference = inverse_sigmoid(obj_reference)

                sub_tmp = self.sub_bbox_embed[lvl](hs_h[lvl])
                if sub_reference.shape[-1] == 4:
                    sub_tmp += sub_reference
                else:
                    assert sub_reference.shape[-1] == 2
                    sub_tmp[..., :2] += sub_reference
                outputs_sub_coord = sub_tmp.sigmoid()
                outputs_sub_coords.append(outputs_sub_coord)

                obj_tmp = self.obj_bbox_embed[lvl](hs_o[lvl])
                if obj_reference.shape[-1] == 4:
                    obj_tmp += obj_reference
                else:
                    assert obj_reference.shape[-1] == 2
                    obj_tmp[..., :2] += obj_reference
                outputs_obj_coord = obj_tmp.sigmoid()
                outputs_obj_coords.append(outputs_obj_coord)

                ### Version 5 for cross-entropy and focal BCE with the bias trick
                text_memory = F.normalize(text_dec[lvl].transpose(0, 1), p=2, dim=-1)
                proj_text_memory = self.projection_text(text_memory / 2.0)
                # print(proj_text_memory.shape)
                obj_text = proj_text_memory[:,:max_obj_text_len]
                pred_text = proj_text_memory[:,max_obj_text_len:(max_obj_text_len + max_pred_text_len)]
                assert (max_obj_text_len + max_pred_text_len) == proj_text_memory.shape[1]

                outputs_obj_class = torch.einsum('bcd,bed->bce', hs_o[lvl] + self.bias_obj_a, obj_text) + self.bias_c
                outputs_obj_classes.append(outputs_obj_class)
                outputs_verb_class = torch.einsum('bcd,bed->bce', hs_verb_dec[lvl]+ self.bias_pred_a, pred_text) + self.bias_c
                outputs_verb_classes.append(outputs_verb_class)
                if self.subject_class:
                    outputs_sub_class = torch.einsum('bcd,bed->bce', hs_h[lvl] + self.bias_obj_a, obj_text) + self.bias_c
                    outputs_sub_classes.append(outputs_sub_class)

            if self.subject_class:
                outputs_sub_class = torch.stack(outputs_sub_classes)
            outputs_verb_class = torch.stack(outputs_verb_classes)
            outputs_obj_class = torch.stack(outputs_obj_classes)
            outputs_sub_coord = torch.stack(outputs_sub_coords)
            outputs_obj_coord = torch.stack(outputs_obj_coords)

            out = {}
            if self.subject_class:
                out.update(
                    {
                        'pred_sub_logits': outputs_sub_class[-1], 'pred_obj_logits': outputs_obj_class[-1], 
                        'pred_verb_logits': outputs_verb_class[-1], 'pred_sub_boxes': outputs_sub_coord[-1], 
                        'pred_obj_boxes': outputs_obj_coord[-1],
                        # 'verb_decoder_out': hs_verb_dec[-1], # this is to calculate uniformity and alignment
                    }
                )
                if self.aux_loss: 
                    # Using aux loss means that you will add loss to every intermidiate layer.
                    out['aux_outputs'] = self._set_aux_loss(outputs_obj_class = outputs_obj_class, 
                                                            outputs_verb_class = outputs_verb_class,
                                                            outputs_sub_coord = outputs_sub_coord,
                                                            outputs_obj_coord = outputs_obj_coord,
                                                            outputs_sub_class = outputs_sub_class)
                # return out
            else:
                out.update(
                    {
                        'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                        'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]
                    }
                )
                if self.aux_loss: 
                    # Using aux loss means that you will add loss to every intermidiate layer.
                    out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                            outputs_sub_coord, outputs_obj_coord) 
                # return out
            
            if self.pseudo_verb:
                if self.pseudo_verb_mode == "online":
                    ## Using online text_bf_fusion
                    # text_bf_fusion = memory_cache["text_memory_resized"]
                    text_bf_fusion = memory_cache["text_memory_bf_resize"]
                    verb_text_bf_fusion = text_bf_fusion[:,0][max_obj_text_len:(max_obj_text_len + max_pred_text_len)]
                    ## Distance metric 2: Euclidean sim
                    verb_text_len, text_dim = verb_text_bf_fusion.shape
                    text_bf_fusion_1 = verb_text_bf_fusion.repeat(1, verb_text_len).view(-1, text_dim)
                    text_bf_fusion_2 = verb_text_bf_fusion.repeat(verb_text_len, 1).view(-1, text_dim)
                    verb_sim = F.pairwise_distance(text_bf_fusion_1, text_bf_fusion_2, p = 2).view(verb_text_len, verb_text_len) # [num_verbs, hid_dim]
                    verb_sim = verb_sim.max(-1)[0].unsqueeze(dim = -1) - verb_sim
                elif self.pseudo_verb_mode == "offline":
                    ## Using offline text(768dim or 1024) from Roberta
                    rel_text = kwargs['text'][0][1]
                    verb_text_bf_fusion = torch.stack([torch.from_numpy(self.rel_feature[rt]).to(memory_cache["text_memory_resized"].device) for rt in rel_text], dim = 0)
                    ## Distance metric 1: Cosine sim
                    verb_text_bf_fusion = F.normalize(verb_text_bf_fusion, p=2, dim=-1)
                    verb_sim = torch.einsum('ab,cb->ac', verb_text_bf_fusion, verb_text_bf_fusion)
                    # sim_thre = 0.9

                target_classes_verb = torch.cat([t['verb_labels'] for t in kwargs['targets']]) # [num_triplets, num_verbs]
                verb_sim_mask = target_classes_verb.unsqueeze(-1).repeat(1, 1, verb_sim.shape[-1]) # [num_triplets, num_verbs, num_verbs]
                target_verb_sim = (verb_sim_mask * verb_sim).sum(dim = 1) # [num_triplets, num_verbs]
                if target_classes_verb.shape[0] > 0:
                    target_verb_sim = target_verb_sim / target_verb_sim.max(-1)[0].unsqueeze(dim = -1)
                target_verb_sim[target_classes_verb.bool()] = 0 # set gt verb = 0
                 
                ## Pseudo label selection by a threshold
                sim_thre = 0.3
                target_verb_sim = target_verb_sim * (target_verb_sim > sim_thre)
                ## Pseudo label selection by gaussian
                # One thre for one rel
                # sim_thre = target_verb_sim.mean(-1) + target_verb_sim.std(-1) * 2.0
                # target_verb_sim = target_verb_sim * (target_verb_sim > sim_thre.unsqueeze(-1))
                # One thre for all rels
                # sim_thre = target_verb_sim.mean() + target_verb_sim.std() * 1.0
                # target_verb_sim = target_verb_sim * (target_verb_sim > sim_thre)
                
                out.update({'target_verb_sim': target_verb_sim})
                if self.aux_loss:
                    for aux_idx, aux in enumerate(out['aux_outputs']):
                        aux.update({'target_verb_sim': target_verb_sim})
            
            return out

    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_sub_class = None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_sub_class is None:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                        outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        else:
            return [{'pred_sub_logits':a, 'pred_obj_logits': b, 'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_obj_boxes': e}
                for a, b, c, d, e in zip(outputs_sub_class[:-1], outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class RLIP_ParSe(nn.Module):
    def __init__(self, backbone, transformer, num_queries, contrastive_align_loss = False, 
                       contrastive_hdim = 64, aux_loss = False, subject_class = False,
                       use_no_verb_token = False, pseudo_verb = False, args = None):        
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries*2, hidden_dim)
        # self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        # self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        # self.verb_class_embed = MLP(hidden_dim, hidden_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.contrastive_align_loss = contrastive_align_loss
        print("RLIP_ParSe uses contrastive_align_loss: ", self.contrastive_align_loss)
        self.subject_class = subject_class
        self.use_no_verb_token = use_no_verb_token
        self.pseudo_verb = pseudo_verb
        # self.TEST_embedding = nn.Embedding(1, contrastive_hdim)

        # TO DO: Merge ?
        if self.contrastive_align_loss:
            # self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_image_obj = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_image_pred = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim)
            # self.no_obj_embedding = nn.Embedding(1, contrastive_hdim)
            if self.use_no_verb_token:
                self.no_pred_embedding = nn.Embedding(1, contrastive_hdim)
        else:

            ### Version 5
            self.projection_text = nn.Linear(hidden_dim, hidden_dim)
            prior_prob = 0.01
            self.bias_c = -math.log((1 - prior_prob) / prior_prob)
            self.bias_obj_a = nn.Parameter(torch.zeros((256,), dtype = torch.float32), requires_grad = True)
            self.bias_pred_a = nn.Parameter(torch.zeros((256,), dtype = torch.float32), requires_grad = True)
            
        
        # Initialization
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.obj_class_embed.bias.data = torch.ones(num_obj_classes+1) * bias_value
        # self.verb_class_embed.bias.data = torch.ones(num_verb_classes) * bias_value
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)
        # torch.nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        # torch.nn.init.constant_(self.reference_points.bias.data, 0.)


        self.verb_tagger = args.verb_tagger

    def forward(self, samples: NestedTensor, 
                      encode_and_save=True, 
                      memory_cache=None, 
                      **kwargs):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        if encode_and_save:
            assert memory_cache is None
            features, pos = self.backbone(samples)
            # features: [features_tensor from layer4,]  
            # pos: [pos embedding for features from layer4, ]  
            #      layer4 pos embedding shape like: [2, 256, 18, 25]

            src, mask = features[-1].decompose()
            assert mask is not None

            memory_cache = self.transformer(src = self.input_proj(src), 
                                mask = mask, 
                                query_embed = self.query_embed.weight, 
                                pos_embed = pos[-1],
                                text = kwargs['text'], 
                                encode_and_save = True)
            return memory_cache
        else:
            assert memory_cache is not None
            h_decoder_out, obj_decoder_out, verb_decoder_out = self.transformer(
                mask=memory_cache["mask"],
                query_embed=memory_cache["ho_query_embed"],
                pos_embed=memory_cache["pos_embed"],
                encode_and_save=False,
                text_memory=memory_cache["text_memory"][-1],# memory_cache["text_memory"][-1], memory_cache["text_memory_resized"],
                img_memory=memory_cache["img_memory"],
                text_attention_mask=memory_cache["text_attention_mask"],
            )
            # print(memory_cache["text_attention_mask"].sum())

            # Calculate cross-modal score, used for classification
            obj_pred_names_sums = memory_cache["obj_pred_names_sums"]
            max_obj_text_len = torch.max(obj_pred_names_sums[:,0])
            max_pred_text_len = torch.max(obj_pred_names_sums[:,1])
            text_len = max_obj_text_len + max_pred_text_len # memory_cache["text_attention_mask"].shape[1] # shape: [batch_num, max_token_num]
            # norm_img_memory = norm_tensor(memory_cache["img_memory"])
            # obj_text_memory = norm_img_memory[- text_len: - text_len + max_obj_text_len].transpose(0, 1)
            # pred_text_memory =  norm_img_memory[- text_len + max_obj_text_len:].transpose(0, 1)
            if self.contrastive_align_loss:
                proj_h_decoder_out = F.normalize(
                    self.contrastive_align_projection_image_obj(h_decoder_out), p=2, dim=-1)
                proj_obj_decoder_out = F.normalize(
                    self.contrastive_align_projection_image_obj(obj_decoder_out), p=2, dim=-1)
                proj_verb_decoder_out = F.normalize(
                    self.contrastive_align_projection_image_pred(verb_decoder_out), p=2, dim=-1)
                proj_text = F.normalize(
                    self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1), p=2, dim=-1)
                
                proj_obj_text = proj_text[:,:max_obj_text_len]
                proj_pred_text = proj_text[:,max_obj_text_len:(max_obj_text_len + max_pred_text_len)]
                if (max_obj_text_len + max_pred_text_len) != proj_text.shape[1]:
                    print('(max_obj_text_len + max_pred_text_len) != proj_text.shape[0]')
                
                # cat no_obj or no_verb embedding
                bs = proj_obj_text.shape[0]
                # no_obj_embedding = self.no_obj_embedding.weight.unsqueeze(dim = 0).repeat(bs, 1, 1)
                # proj_obj_text = torch.cat((proj_obj_text, no_obj_embedding), dim = 1)
                if self.use_no_verb_token:
                    no_pred_embedding = self.no_pred_embedding.weight.unsqueeze(dim = 0).repeat(bs, 1, 1)
                    proj_pred_text = torch.cat((proj_pred_text, no_pred_embedding), dim = 1)

                outputs_sub_class = torch.einsum('abcd,bed->abce', proj_h_decoder_out, proj_obj_text)
                outputs_obj_class = torch.einsum('abcd,bed->abce', proj_obj_decoder_out, proj_obj_text)
                outputs_verb_class = torch.einsum('abcd,bed->abce', proj_verb_decoder_out, proj_pred_text)
                outputs_sub_coord = self.sub_bbox_embed(h_decoder_out).sigmoid()
                outputs_obj_coord = self.obj_bbox_embed(obj_decoder_out).sigmoid()
            
            else:
                ### Version 5 for cross-entropy and focal BCE with the bias trick
                outputs_sub_coord = self.sub_bbox_embed(h_decoder_out).sigmoid()
                outputs_obj_coord = self.obj_bbox_embed(obj_decoder_out).sigmoid()
                outputs_obj_class = []
                outputs_verb_class = []
                outputs_sub_class = []
                for i in range(obj_decoder_out.shape[0]):
                    i_n = i - obj_decoder_out.shape[0] 
                    # The above line is to defend against the situation when we have more than three layers of language features.
                    # We can use the last three layers of language features.
                    text_memory = F.normalize(memory_cache["text_memory"][i_n].transpose(0, 1), p=2, dim=-1)
                    proj_text_memory = self.projection_text(text_memory / 2.0)
                    obj_text = proj_text_memory[:,:max_obj_text_len]
                    pred_text = proj_text_memory[:,max_obj_text_len:(max_obj_text_len + max_pred_text_len)]
                    assert (max_obj_text_len + max_pred_text_len) == proj_text_memory.shape[1]

                    outputs_obj_class.append(torch.einsum('bcd,bed->bce', obj_decoder_out[i_n] + self.bias_obj_a, obj_text) + self.bias_c)
                    outputs_verb_class.append(torch.einsum('bcd,bed->bce', verb_decoder_out[i_n] + self.bias_pred_a, pred_text) + self.bias_c)
                    if self.subject_class:
                        outputs_sub_class.append(torch.einsum('bcd,bed->bce', h_decoder_out[i_n]  + self.bias_obj_a, obj_text) + self.bias_c)
                outputs_obj_class = torch.stack(outputs_obj_class, dim = 0)
                outputs_verb_class = torch.stack(outputs_verb_class, dim = 0)
                if self.subject_class:
                    outputs_sub_class = torch.stack(outputs_sub_class, dim = 0)


            out = {}
            if self.subject_class:
                out.update(
                    {
                        'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                        'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1], 
                        'pred_sub_logits': outputs_sub_class[-1],
                    }
                )
                if self.aux_loss:
                    # Using aux loss means that you will add loss to every intermidiate layer.
                    out['aux_outputs'] = self._set_aux_loss(outputs_obj_class = outputs_obj_class, 
                                                        outputs_verb_class = outputs_verb_class,
                                                        outputs_sub_coord = outputs_sub_coord,
                                                        outputs_obj_coord = outputs_obj_coord,
                                                        outputs_sub_class = outputs_sub_class) 
                # return out
            else:
                out.update(
                    {
                        'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                        'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
                    }
                )
                if self.aux_loss: 
                    # Using aux loss means that you will add loss to every intermidiate layer.
                    out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                            outputs_sub_coord, outputs_obj_coord) 
                # return out
            
            if self.pseudo_verb:
                ## Using online text_bf_fusion
                # text_bf_fusion = memory_cache["text_memory_resized"]
                text_bf_fusion = memory_cache["text_memory_bf_resized"]
                verb_text_bf_fusion = text_bf_fusion[:,0][max_obj_text_len:(max_obj_text_len + max_pred_text_len)]
                ## Using offline text(768dim) from Roberta
                # rel_text = kwargs['text'][0][1]
                # verb_text_bf_fusion = torch.stack([torch.from_numpy(self.rel_feature[rt]).to(memory_cache["text_memory_resized"].device) for rt in rel_text], dim = 0)

                ## Distance metric 1: Cosine sim
                # verb_text_bf_fusion = F.normalize(verb_text_bf_fusion, p=2, dim=-1)
                # verb_sim = torch.einsum('ab,cb->ac', verb_text_bf_fusion, verb_text_bf_fusion)
                # sim_thre = 0.9
                ## Distance metric 2: Euclidean sim
                verb_text_len, text_dim = verb_text_bf_fusion.shape
                text_bf_fusion_1 = verb_text_bf_fusion.repeat(1, verb_text_len).view(-1, text_dim)
                text_bf_fusion_2 = verb_text_bf_fusion.repeat(verb_text_len, 1).view(-1, text_dim)
                verb_sim = F.pairwise_distance(text_bf_fusion_1, text_bf_fusion_2, p = 2).view(verb_text_len, verb_text_len) # [num_verbs, hid_dim]
                verb_sim = verb_sim.max(-1)[0].unsqueeze(dim = -1) - verb_sim
                
                target_classes_verb = torch.cat([t['verb_labels'] for t in kwargs['targets']]) # [num_triplets, num_verbs]
                verb_sim_mask = target_classes_verb.unsqueeze(-1).repeat(1, 1, verb_sim.shape[-1]) # [num_triplets, num_verbs, num_verbs]
                # print((verb_sim_mask * verb_sim)[-1].sum(-1))
                target_verb_sim = (verb_sim_mask * verb_sim).sum(dim = 1) # [num_triplets, num_verbs]
                if target_classes_verb.shape[0] > 0:
                    target_verb_sim = target_verb_sim / target_verb_sim.max(-1)[0].unsqueeze(dim = -1)
                target_verb_sim[target_classes_verb.bool()] = 0 # set gt verb = 0
                
                ## Pseudo label selection by a threshold
                sim_thre = 0.3
                target_verb_sim = target_verb_sim * (target_verb_sim > sim_thre)
                ## Pseudo label selection by gaussian
                # One thre for one rel
                # sim_thre = target_verb_sim.mean(-1) + target_verb_sim.std(-1) * 2.0
                # target_verb_sim = target_verb_sim * (target_verb_sim > sim_thre.unsqueeze(-1))
                # One thre for all rels
                # sim_thre = target_verb_sim.mean() + target_verb_sim.std() * 1.0
                # target_verb_sim = target_verb_sim * (target_verb_sim > sim_thre)
                
                out.update({'target_verb_sim': target_verb_sim})
                if self.aux_loss:
                    for aux_idx, aux in enumerate(out['aux_outputs']):
                        aux.update({'target_verb_sim': target_verb_sim})
            
            return out

     
    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_sub_class = None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_sub_class is None:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                        outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        else:
            return [{'pred_sub_logits':a, 'pred_obj_logits': b, 'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_obj_boxes': e}
                for a, b, c, d, e in zip(outputs_sub_class[:-1], outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]



class SepDETRHOIv3(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False, subject_class = False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries*2, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        # self.verb_class_embed = MLP(hidden_dim, hidden_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.subject_class = subject_class
        

    def forward(self, samples: NestedTensor, **kwargs):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print(len(pos))
        # print(pos[-1].shape)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None

        h_decoder_out, obj_decoder_out, verb_decoder_out, hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        outputs_obj_class = self.obj_class_embed(obj_decoder_out)
        outputs_verb_class = self.verb_class_embed(verb_decoder_out)
        outputs_sub_coord = self.sub_bbox_embed(h_decoder_out).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(obj_decoder_out).sigmoid()
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.subject_class:
            outputs_sub_class = self.obj_class_embed(h_decoder_out)
            out.update({'pred_sub_logits': outputs_sub_class[-1]})
            if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord, outputs_sub_class) 
        else:
            if self.aux_loss: 
                # Using aux loss means that you will add loss to every intermidiate layer.
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord) 
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_sub_class = None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_sub_class is None:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                        outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        else:
            return [{'pred_sub_logits':a, 'pred_obj_logits': b, 'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_obj_boxes': e}
                for a, b, c, d, e in zip(outputs_sub_class[:-1], outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        
        


class CDNHOI(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False, subject_class = False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        # self.verb_class_embed = MLP(hidden_dim, hidden_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.subject_class = subject_class
    
    def forward(self, samples: NestedTensor, **kwargs):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None
        h_obj_decoder_out, verb_decoder_out, hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        outputs_obj_class = self.obj_class_embed(h_obj_decoder_out)
        outputs_verb_class = self.verb_class_embed(verb_decoder_out)
        outputs_sub_coord = self.sub_bbox_embed(h_obj_decoder_out).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(h_obj_decoder_out).sigmoid()
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.subject_class:
            outputs_sub_class = self.obj_class_embed(h_obj_decoder_out)
            out.update({'pred_sub_logits': outputs_sub_class[-1]})
            if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord, outputs_sub_class) 
        else:
            if self.aux_loss: 
                # Using aux loss means that you will add loss to every intermidiate layer.
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord) 
        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_sub_class = None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_sub_class is None:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                        outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        else:
            return [{'pred_sub_logits':a, 'pred_obj_logits': b, 'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_obj_boxes': e}
                for a, b, c, d, e in zip(outputs_sub_class[:-1], outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]



class DDETRHOI(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, 
                       num_feature_levels, aux_loss=True, with_box_refine=False, two_stage=False):
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            print('self.query_embed.shape', self.query_embed.weight.shape)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.verb_class_embed.bias.data = torch.ones(num_verb_classes) * bias_value
        self.obj_class_embed.bias.data = torch.ones(num_obj_classes + 1) * bias_value
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.verb_class_embed = _get_clones(self.verb_class_embed, num_pred)
            self.obj_class_embed = _get_clones(self.obj_class_embed, num_pred)

            self.sub_bbox_embed = _get_clones(self.sub_bbox_embed, num_pred)
            nn.init.constant_(self.sub_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.sub_bbox_embed = self.sub_bbox_embed

            self.obj_bbox_embed = _get_clones(self.obj_bbox_embed, num_pred)
            nn.init.constant_(self.obj_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.obj_bbox_embed = self.obj_bbox_embed
        else:
            nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.verb_class_embed = nn.ModuleList([self.verb_class_embed for _ in range(num_pred)])
            self.obj_class_embed = nn.ModuleList([self.obj_class_embed for _ in range(num_pred)])
            self.sub_bbox_embed = nn.ModuleList([self.sub_bbox_embed for _ in range(num_pred)])
            self.obj_bbox_embed = nn.ModuleList([self.obj_bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.sub_bbox_embed = None
            self.transformer.decoder.obj_bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.sub_class_embed = self.sub_class_embed
            self.transformer.decoder.obj_class_embed = self.obj_class_embed
            for box_embed in self.sub_bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
            for box_embed in self.obj_bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)


    def forward(self, samples: NestedTensor,  **kwargs):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)


        outputs_verb_classes = []
        outputs_obj_classes = []
        outputs_sub_coords = []
        outputs_obj_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                sub_reference, obj_reference = init_reference  # ref_points = 4
                # sub_reference = obj_reference = init_reference # ref_points = 2
            else:
                sub_reference, obj_reference = inter_references[lvl - 1]
            sub_reference = inverse_sigmoid(sub_reference)
            obj_reference = inverse_sigmoid(obj_reference)

            sub_tmp = self.sub_bbox_embed[lvl](hs[lvl])
            if sub_reference.shape[-1] == 4:
                sub_tmp += sub_reference
            else:
                assert sub_reference.shape[-1] == 2
                sub_tmp[..., :2] += sub_reference
            outputs_sub_coord = sub_tmp.sigmoid()
            outputs_sub_coords.append(outputs_sub_coord)

            obj_tmp = self.obj_bbox_embed[lvl](hs[lvl])
            if obj_reference.shape[-1] == 4:
                obj_tmp += obj_reference
            else:
                assert obj_reference.shape[-1] == 2
                obj_tmp[..., :2] += obj_reference
            outputs_obj_coord = obj_tmp.sigmoid()
            outputs_obj_coords.append(outputs_obj_coord)

            # outputs_sub_coords.append(self.sub_bbox_embed[lvl](hs[lvl]).sigmoid())
            # outputs_obj_coords.append(self.obj_bbox_embed[lvl](hs[lvl]).sigmoid())
            outputs_verb_class = self.verb_class_embed[lvl](hs[lvl])
            outputs_obj_class = self.obj_class_embed[lvl](hs[lvl])
            outputs_verb_classes.append(outputs_verb_class)
            outputs_obj_classes.append(outputs_obj_class)

        outputs_verb_class = torch.stack(outputs_verb_classes)
        outputs_obj_class = torch.stack(outputs_obj_classes)
        outputs_sub_coord = torch.stack(outputs_sub_coords)
        outputs_obj_coord = torch.stack(outputs_obj_coords)

        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class RLIP_ParSeD(nn.Module):
    def __init__(self, backbone, transformer, num_queries, num_feature_levels, 
                       aux_loss=True, with_box_refine=False, two_stage=False, 
                       subject_class=False, verb_curing=False, masked_entity_modeling=None, 
                       pseudo_verb=False, matcher = None, verb_tagger=False, args=None):
        super().__init__()
        # num_obj_classes, num_verb_classes

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        # self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        ### Version 4
        # self.projection_image_obj = nn.Linear(hidden_dim, hidden_dim)
        ### version 5
        self.projection_text = nn.Linear(hidden_dim, hidden_dim)
        prior_prob = 0.01
        self.bias_c = -math.log((1 - prior_prob) / prior_prob)
        self.bias_obj_a = nn.Parameter(torch.zeros((256,), dtype = torch.float32), requires_grad = True)
        self.bias_pred_a = nn.Parameter(torch.zeros((256,), dtype = torch.float32), requires_grad = True)
        
        
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*3)
            print('self.query_embed.shape', self.query_embed.weight.shape)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.subject_class = subject_class

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.verb_class_embed.bias.data = torch.ones(num_verb_classes) * bias_value
        # self.obj_class_embed.bias.data = torch.ones(num_obj_classes + 1) * bias_value
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.ho_decoder.num_layers
        if with_box_refine:
            # self.verb_class_embed = _get_clones(self.verb_class_embed, num_pred)
            # self.obj_class_embed = _get_clones(self.obj_class_embed, num_pred)

            self.sub_bbox_embed = _get_clones(self.sub_bbox_embed, num_pred * 2)
            nn.init.constant_(self.sub_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.ho_decoder.sub_bbox_embed = self.sub_bbox_embed[:num_pred]
            self.transformer.verb_decoder.sub_bbox_embed = self.sub_bbox_embed[num_pred:]

            self.obj_bbox_embed = _get_clones(self.obj_bbox_embed, num_pred * 2)
            nn.init.constant_(self.obj_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.ho_decoder.obj_bbox_embed = self.obj_bbox_embed[:num_pred]
            self.transformer.verb_decoder.obj_bbox_embed = self.obj_bbox_embed[num_pred:]
        else:
            nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data[2:], -2.0)
            # self.verb_class_embed = nn.ModuleList([self.verb_class_embed for _ in range(num_pred)])
            # self.obj_class_embed = nn.ModuleList([self.obj_class_embed for _ in range(num_pred)])
            self.sub_bbox_embed = nn.ModuleList([self.sub_bbox_embed for _ in range(num_pred)])
            self.obj_bbox_embed = nn.ModuleList([self.obj_bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.sub_bbox_embed = None
            self.transformer.decoder.obj_bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            # self.transformer.decoder.sub_class_embed = self.sub_class_embed
            # self.transformer.decoder.obj_class_embed = self.obj_class_embed
            for box_embed in self.sub_bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
            for box_embed in self.obj_bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        
        self.pseudo_verb = pseudo_verb
        self.pseudo_verb_mode = "online" # "offline"
        if self.pseudo_verb:
            self.rel_feature = np.load('/mnt/data-nas/peizhi/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_sup-simcse-roberta-large.npz', allow_pickle = True)['rel_feature'].item()
            # self.rel_feature = np.load('/mnt/data-nas/peizhi/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_text_feature.npz', allow_pickle = True)['rel_feature'].item()
            # self.obj_feature = np.load('datasets/word_embedding/vg_keep_names_v1_no_lias_freq_text_feature.npz')['obj_feature']

        self.verb_curing = verb_curing
        if self.verb_curing:
            self.sub_curing = nn.Linear(hidden_dim, 1)
            self.obj_curing = nn.Linear(hidden_dim, 1)

        self.masked_entity_modeling = masked_entity_modeling
        if self.masked_entity_modeling:
            self.matcher = matcher
            self.masked_ref_points = nn.Linear(hidden_dim, 2)
            self.coord_proj = nn.Linear(4, hidden_dim)
            self.noise_mu = nn.Linear(hidden_dim, hidden_dim)
            self.noise_log_var = nn.Linear(hidden_dim, hidden_dim)

            # Recon modeling
            self.recon_linear = nn.Linear(hidden_dim*2, hidden_dim)
    
        self.verb_tagger = args.verb_tagger

    def forward(self, samples: NestedTensor, 
                      encode_and_save=True, 
                      memory_cache=None, 
                      **kwargs):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        if encode_and_save:
            features, pos = self.backbone(samples)
            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)
                assert mask is not None
            
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)
            
            query_embeds = None
            if not self.two_stage:
                query_embeds = self.query_embed.weight
            
            targets = kwargs['targets'] if self.verb_tagger else None
            memory_cache = self.transformer(srcs = srcs, 
                                            masks = masks,
                                            pos_embeds = pos, 
                                            query_embed = query_embeds,
                                            text = kwargs['text'],
                                            encode_and_save = True,
                                            targets = targets)
            return memory_cache

        else:
            hs_ho_dec, hs_verb_dec, text_dec, init_reference, inter_references, hs_ho_bf_fusion, hs_verb_bf_fusion, enc_outputs_class, enc_outputs_coord_unact = \
                        self.transformer(masks = memory_cache["masks"],
                                         query_embed = memory_cache["ho_query_embed"],
                                         encode_and_save = False,
                                         text_memory=memory_cache["text_memory_resized"],
                                         img_memory=memory_cache["img_memory"],
                                         text_attention_mask=memory_cache["text_attention_mask"],
                                         obj_pred_names_sums=memory_cache["obj_pred_names_sums"],
                                         spatial_shapes=memory_cache["spatial_shapes"],
                                         level_start_index=memory_cache["level_start_index"],
                                         valid_ratios=memory_cache["valid_ratios"],
                                         key_padding_mask=memory_cache["key_padding_mask"],
                                         attn_mask=memory_cache["attn_mask"])

            hs_h, hs_o = hs_ho_dec[:,:,:self.num_queries//2], hs_ho_dec[:,:,self.num_queries//2:]
            obj_pred_names_sums = memory_cache["obj_pred_names_sums"]
            max_obj_text_len = torch.max(obj_pred_names_sums[:,0])
            max_pred_text_len = torch.max(obj_pred_names_sums[:,1])
            text_len = max_obj_text_len + max_pred_text_len 
            
            if self.verb_curing:
                curing_score = self.sub_curing(hs_h[-1]).sigmoid() * self.obj_curing(hs_o[-1]).sigmoid()
            outputs_verb_classes = []
            outputs_obj_classes = []
            outputs_sub_classes = []
            outputs_sub_coords = []
            outputs_obj_coords = []
            for lvl in range(hs_h.shape[0]):
                if lvl == 0:
                    sub_reference, obj_reference = init_reference  # ref_points = 4
                    # sub_reference = obj_reference = init_reference # ref_points = 2
                else:
                    sub_reference, obj_reference = inter_references[lvl - 1]
                sub_reference = inverse_sigmoid(sub_reference)
                obj_reference = inverse_sigmoid(obj_reference)

                sub_tmp = self.sub_bbox_embed[lvl](hs_h[lvl])
                if sub_reference.shape[-1] == 4:
                    sub_tmp += sub_reference
                else:
                    assert sub_reference.shape[-1] == 2
                    sub_tmp[..., :2] += sub_reference
                outputs_sub_coord = sub_tmp.sigmoid()
                outputs_sub_coords.append(outputs_sub_coord)

                obj_tmp = self.obj_bbox_embed[lvl](hs_o[lvl])
                if obj_reference.shape[-1] == 4:
                    obj_tmp += obj_reference
                else:
                    assert obj_reference.shape[-1] == 2
                    obj_tmp[..., :2] += obj_reference
                outputs_obj_coord = obj_tmp.sigmoid()
                outputs_obj_coords.append(outputs_obj_coord)

                ### Version 5 for cross-entropy and focal BCE with the bias trick
                text_memory = F.normalize(text_dec[lvl].transpose(0, 1), p=2, dim=-1)
                proj_text_memory = self.projection_text(text_memory / 2.0)
                # print(proj_text_memory.shape)
                obj_text = proj_text_memory[:,:max_obj_text_len]
                pred_text = proj_text_memory[:,max_obj_text_len:(max_obj_text_len + max_pred_text_len)]
                assert (max_obj_text_len + max_pred_text_len) == proj_text_memory.shape[1]

                outputs_obj_class = torch.einsum('bcd,bed->bce', hs_o[lvl] + self.bias_obj_a, obj_text) + self.bias_c
                outputs_obj_classes.append(outputs_obj_class)
                outputs_verb_class = torch.einsum('bcd,bed->bce', hs_verb_dec[lvl]+ self.bias_pred_a, pred_text) + self.bias_c
                outputs_verb_classes.append(outputs_verb_class)
                if self.subject_class:
                    outputs_sub_class = torch.einsum('bcd,bed->bce', hs_h[lvl] + self.bias_obj_a, obj_text) + self.bias_c
                    outputs_sub_classes.append(outputs_sub_class)

            if self.subject_class:
                outputs_sub_class = torch.stack(outputs_sub_classes)
            outputs_verb_class = torch.stack(outputs_verb_classes)
            outputs_obj_class = torch.stack(outputs_obj_classes)
            outputs_sub_coord = torch.stack(outputs_sub_coords)
            outputs_obj_coord = torch.stack(outputs_obj_coords)

            out = {}
            if self.subject_class:
                if self.verb_curing:
                    out.update(
                        {
                            'pred_sub_logits': outputs_sub_class[-1], 'pred_obj_logits': outputs_obj_class[-1], 
                            'pred_verb_logits': outputs_verb_class[-1], 'pred_sub_boxes': outputs_sub_coord[-1],
                            'pred_obj_boxes': outputs_obj_coord[-1], 'curing_score':curing_score,
                        }
                    )
                    if self.aux_loss: 
                        # Using aux loss means that you will add loss to every intermidiate layer.
                        out['aux_outputs'] = self._set_aux_loss(outputs_obj_class = outputs_obj_class, 
                                                                outputs_verb_class = outputs_verb_class,
                                                                outputs_sub_coord = outputs_sub_coord,
                                                                outputs_obj_coord = outputs_obj_coord,
                                                                outputs_sub_class = outputs_sub_class,
                                                                curing_score = curing_score)
                    # return out
                else:
                    out.update(
                        {
                            'pred_sub_logits': outputs_sub_class[-1], 'pred_obj_logits': outputs_obj_class[-1], 
                            'pred_verb_logits': outputs_verb_class[-1], 'pred_sub_boxes': outputs_sub_coord[-1], 
                            'pred_obj_boxes': outputs_obj_coord[-1],
                            # 'verb_decoder_out': hs_verb_dec[-1], # this is to calculate uniformity and alignment
                        }
                    )
                    if self.aux_loss: 
                        # Using aux loss means that you will add loss to every intermidiate layer.
                        out['aux_outputs'] = self._set_aux_loss(outputs_obj_class = outputs_obj_class, 
                                                                outputs_verb_class = outputs_verb_class,
                                                                outputs_sub_coord = outputs_sub_coord,
                                                                outputs_obj_coord = outputs_obj_coord,
                                                                outputs_sub_class = outputs_sub_class)
                    # return out
            else:
                if self.verb_curing:
                    out.update(
                        {
                            'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                            'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1], 
                            'curing_score':curing_score,
                        }
                    )
                    if self.aux_loss: 
                        # Using aux loss means that you will add loss to every intermidiate layer.
                        out['aux_outputs'] = self._set_aux_loss(outputs_obj_class = outputs_obj_class, 
                                                                outputs_verb_class = outputs_verb_class,
                                                                outputs_sub_coord = outputs_sub_coord,
                                                                outputs_obj_coord = outputs_obj_coord,
                                                                curing_score = curing_score)
                    # return out
                else:
                    out.update(
                        {
                            'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                            'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]
                        }
                    )
                    if self.aux_loss: 
                        # Using aux loss means that you will add loss to every intermidiate layer.
                        out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                                outputs_sub_coord, outputs_obj_coord) 
                    # return out
            
            if self.pseudo_verb:
                if self.pseudo_verb_mode == "online":
                    ## Using online text_bf_fusion
                    # text_bf_fusion = memory_cache["text_memory_resized"]
                    text_bf_fusion = memory_cache["text_memory_bf_resize"]
                    verb_text_bf_fusion = text_bf_fusion[:,0][max_obj_text_len:(max_obj_text_len + max_pred_text_len)]
                    ## Distance metric 2: Euclidean sim
                    verb_text_len, text_dim = verb_text_bf_fusion.shape
                    text_bf_fusion_1 = verb_text_bf_fusion.repeat(1, verb_text_len).view(-1, text_dim)
                    text_bf_fusion_2 = verb_text_bf_fusion.repeat(verb_text_len, 1).view(-1, text_dim)
                    verb_sim = F.pairwise_distance(text_bf_fusion_1, text_bf_fusion_2, p = 2).view(verb_text_len, verb_text_len) # [num_verbs, hid_dim]
                    verb_sim = verb_sim.max(-1)[0].unsqueeze(dim = -1) - verb_sim
                elif self.pseudo_verb_mode == "offline":
                    ## Using offline text(768dim or 1024) from Roberta
                    rel_text = kwargs['text'][0][1]
                    verb_text_bf_fusion = torch.stack([torch.from_numpy(self.rel_feature[rt]).to(memory_cache["text_memory_resized"].device) for rt in rel_text], dim = 0)
                    ## Distance metric 1: Cosine sim
                    verb_text_bf_fusion = F.normalize(verb_text_bf_fusion, p=2, dim=-1)
                    verb_sim = torch.einsum('ab,cb->ac', verb_text_bf_fusion, verb_text_bf_fusion)
                    # sim_thre = 0.9

                target_classes_verb = torch.cat([t['verb_labels'] for t in kwargs['targets']]) # [num_triplets, num_verbs]
                verb_sim_mask = target_classes_verb.unsqueeze(-1).repeat(1, 1, verb_sim.shape[-1]) # [num_triplets, num_verbs, num_verbs]
                target_verb_sim = (verb_sim_mask * verb_sim).sum(dim = 1) # [num_triplets, num_verbs]
                if target_classes_verb.shape[0] > 0:
                    target_verb_sim = target_verb_sim / target_verb_sim.max(-1)[0].unsqueeze(dim = -1)
                target_verb_sim[target_classes_verb.bool()] = 0 # set gt verb = 0

                ## Pseudo label selection by a threshold
                sim_thre = 0.3
                target_verb_sim = target_verb_sim * (target_verb_sim > sim_thre)
                ## Pseudo label selection by gaussian
                # One thre for one rel
                # sim_thre = target_verb_sim.mean(-1) + target_verb_sim.std(-1) * 2.0
                # target_verb_sim = target_verb_sim * (target_verb_sim > sim_thre.unsqueeze(-1))
                # One thre for all rels
                # sim_thre = target_verb_sim.mean() + target_verb_sim.std() * 1.0
                # target_verb_sim = target_verb_sim * (target_verb_sim > sim_thre)
                
                out.update({'target_verb_sim': target_verb_sim})
                if self.aux_loss:
                    for aux_idx, aux in enumerate(out['aux_outputs']):
                        aux.update({'target_verb_sim': target_verb_sim})
                

            if self.masked_entity_modeling:
                ### version 4, Recon modeling, proposed by Jianwen
                ## Vison Masked 3 ALL before Fusion, last sub&obj query (VMAllbfFus2)
                hs_h_bf_fusion, hs_o_bf_fusion = hs_ho_bf_fusion[-1,:,:self.num_queries//2], hs_ho_bf_fusion[-1,:,self.num_queries//2:]
                hs_h_bf_fusion = hs_h_bf_fusion.unsqueeze(0).repeat(3, 1, 1, 1)
                hs_o_bf_fusion = hs_o_bf_fusion.unsqueeze(0).repeat(3, 1, 1, 1)
                hs_h_recon = self.recon_linear(torch.cat((hs_o_bf_fusion, hs_verb_bf_fusion), dim = -1))
                hs_o_recon = self.recon_linear(torch.cat((hs_h_bf_fusion, hs_verb_bf_fusion), dim = -1))
                ## Vison Masked 2 ALL before Fusion (VMAllbfFus)
                # hs_h_bf_fusion, hs_o_bf_fusion = hs_ho_bf_fusion[:,:,:self.num_queries//2], hs_ho_bf_fusion[:,:,self.num_queries//2:]
                # hs_h_recon = self.recon_linear(torch.cat((hs_o_bf_fusion, hs_verb_bf_fusion), dim = -1))
                # hs_o_recon = self.recon_linear(torch.cat((hs_h_bf_fusion, hs_verb_bf_fusion), dim = -1))
                ## Vison Masked 1 (VM)
                # hs_h_recon = self.recon_linear(torch.cat((hs_o, hs_verb_dec), dim = -1))
                # hs_o_recon = self.recon_linear(torch.cat((hs_h, hs_verb_dec), dim = -1))

                recon_obj_classes = []
                recon_sub_classes = []
                recon_sub_coords = []
                recon_obj_coords = []
                for lvl in range(hs_h_recon.shape[0]):
                    if lvl == 0:
                        sub_reference, obj_reference = init_reference  # ref_points = 4
                        # sub_reference = obj_reference = init_reference # ref_points = 2
                    else:
                        sub_reference, obj_reference = inter_references[lvl - 1]
                    sub_reference = inverse_sigmoid(sub_reference)
                    obj_reference = inverse_sigmoid(obj_reference)

                    sub_tmp = self.sub_bbox_embed[lvl](hs_h_recon[lvl])
                    if sub_reference.shape[-1] == 4:
                        sub_tmp += sub_reference
                    else:
                        assert sub_reference.shape[-1] == 2
                        sub_tmp[..., :2] += sub_reference
                    recon_sub_coord = sub_tmp.sigmoid()
                    recon_sub_coords.append(recon_sub_coord)

                    obj_tmp = self.obj_bbox_embed[lvl](hs_o_recon[lvl])
                    if obj_reference.shape[-1] == 4:
                        obj_tmp += obj_reference
                    else:
                        assert obj_reference.shape[-1] == 2
                        obj_tmp[..., :2] += obj_reference
                    recon_obj_coord = obj_tmp.sigmoid()
                    recon_obj_coords.append(recon_obj_coord)

                    text_memory = F.normalize(text_dec[lvl].transpose(0, 1), p=2, dim=-1)
                    proj_text_memory = self.projection_text(text_memory / 2.0)
                    obj_text = proj_text_memory[:,:max_obj_text_len]
                    pred_text = proj_text_memory[:,max_obj_text_len:(max_obj_text_len + max_pred_text_len)]
                    assert (max_obj_text_len + max_pred_text_len) == proj_text_memory.shape[1]

                    recon_obj_class = torch.einsum('bcd,bed->bce', hs_o_recon[lvl] + self.bias_obj_a, obj_text) + self.bias_c
                    recon_obj_classes.append(recon_obj_class)
                    if self.subject_class:
                        recon_sub_class = torch.einsum('bcd,bed->bce', hs_h_recon[lvl] + self.bias_obj_a, obj_text) + self.bias_c
                        recon_sub_classes.append(recon_sub_class)
                if self.subject_class:
                    recon_sub_class = torch.stack(recon_sub_classes)
                recon_obj_class = torch.stack(recon_obj_classes)
                recon_sub_coord = torch.stack(recon_sub_coords)
                recon_obj_coord = torch.stack(recon_obj_coords)

                out.update({'recon_stat': {'pred_sub_logits': recon_sub_class[-1], 
                                            'pred_obj_logits': recon_obj_class[-1], 
                                            'pred_sub_boxes': recon_sub_coord[-1],
                                            'pred_obj_boxes': recon_obj_coord[-1]}})
                if self.aux_loss:
                    for aux_idx, aux in enumerate(out['aux_outputs']):
                        aux.update({'recon_stat': {'pred_sub_logits': recon_sub_class[aux_idx], 
                                                    'pred_obj_logits': recon_obj_class[aux_idx], 
                                                    'pred_sub_boxes': recon_sub_coord[aux_idx],
                                                    'pred_obj_boxes': recon_obj_coord[aux_idx]}})

                          
            return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, 
                            outputs_sub_class = None, curing_score = None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if curing_score is None:
            if outputs_sub_class is None:
                return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                        for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                            outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
            else:
                return [{'pred_sub_logits':a, 'pred_obj_logits': b, 'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_obj_boxes': e}
                        for a, b, c, d, e in zip(outputs_sub_class[:-1], outputs_obj_class[:-1], outputs_verb_class[:-1],
                                            outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        else:
            if outputs_sub_class is None:
                return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d, 'curing_score':curing_score}
                        for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                            outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
            else:
                return [{'pred_sub_logits':a, 'pred_obj_logits': b, 'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_obj_boxes': e, 'curing_score':curing_score}
                        for a, b, c, d, e in zip(outputs_sub_class[:-1], outputs_obj_class[:-1], outputs_verb_class[:-1],
                                            outputs_sub_coord[:-1], outputs_obj_coord[:-1])]

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # indices: list of tensor tuples
        # like [(tensor([ 5, 42, 51, 61]), tensor([2, 3, 0, 1])), (tensor([20]), tensor([0]))]
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

class ParSeD(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, 
                       num_feature_levels, aux_loss=True, with_box_refine=False, two_stage=False, 
                       verb_curing = False, subject_class = False):
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*3)
            print('self.query_embed.shape', self.query_embed.weight.shape)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.verb_class_embed.bias.data = torch.ones(num_verb_classes) * bias_value
        self.obj_class_embed.bias.data = torch.ones(num_obj_classes + 1) * bias_value
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.ho_decoder.num_layers
        if with_box_refine:
            self.verb_class_embed = _get_clones(self.verb_class_embed, num_pred)
            self.obj_class_embed = _get_clones(self.obj_class_embed, num_pred)

            self.sub_bbox_embed = _get_clones(self.sub_bbox_embed, num_pred * 2)
            nn.init.constant_(self.sub_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.ho_decoder.sub_bbox_embed = self.sub_bbox_embed[:num_pred]
            self.transformer.verb_decoder.sub_bbox_embed = self.sub_bbox_embed[num_pred:]

            self.obj_bbox_embed = _get_clones(self.obj_bbox_embed, num_pred * 2)
            nn.init.constant_(self.obj_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.ho_decoder.obj_bbox_embed = self.obj_bbox_embed[:num_pred]
            self.transformer.verb_decoder.obj_bbox_embed = self.obj_bbox_embed[num_pred:]
        else:
            nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.verb_class_embed = nn.ModuleList([self.verb_class_embed for _ in range(num_pred)])
            self.obj_class_embed = nn.ModuleList([self.obj_class_embed for _ in range(num_pred)])
            self.sub_bbox_embed = nn.ModuleList([self.sub_bbox_embed for _ in range(num_pred)])
            self.obj_bbox_embed = nn.ModuleList([self.obj_bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.sub_bbox_embed = None
            self.transformer.decoder.obj_bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.sub_class_embed = self.sub_class_embed
            self.transformer.decoder.obj_class_embed = self.obj_class_embed
            for box_embed in self.sub_bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
            for box_embed in self.obj_bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        
        self.verb_curing = verb_curing
        if self.verb_curing:
            self.sub_curing = nn.Linear(hidden_dim, 1)
            self.obj_curing = nn.Linear(hidden_dim, 1)
        
        self.subject_class = subject_class



    def forward(self, samples: NestedTensor,  **kwargs):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        # print(masks[0].shape, masks[0].sum())

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs_ho, hs_verb, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
        hs_h, hs_o = hs_ho[:,:,:self.num_queries//2], hs_ho[:,:,self.num_queries//2:]

        if self.verb_curing:
            curing_score = self.sub_curing(hs_h[-1]).sigmoid() * self.obj_curing(hs_o[-1]).sigmoid()
        outputs_verb_classes = []
        outputs_obj_classes = []
        outputs_sub_coords = []
        outputs_obj_coords = []
        for lvl in range(hs_ho.shape[0]):
            if lvl == 0:
                sub_reference, obj_reference = init_reference  # ref_points = 4
                # sub_reference = obj_reference = init_reference # ref_points = 2
            else:
                sub_reference, obj_reference = inter_references[lvl - 1]
            sub_reference = inverse_sigmoid(sub_reference)
            obj_reference = inverse_sigmoid(obj_reference)

            sub_tmp = self.sub_bbox_embed[lvl](hs_h[lvl])
            if sub_reference.shape[-1] == 4:
                sub_tmp += sub_reference
            else:
                assert sub_reference.shape[-1] == 2
                sub_tmp[..., :2] += sub_reference
            outputs_sub_coord = sub_tmp.sigmoid()
            outputs_sub_coords.append(outputs_sub_coord)

            obj_tmp = self.obj_bbox_embed[lvl](hs_o[lvl])
            if obj_reference.shape[-1] == 4:
                obj_tmp += obj_reference
            else:
                assert obj_reference.shape[-1] == 2
                obj_tmp[..., :2] += obj_reference
            outputs_obj_coord = obj_tmp.sigmoid()
            outputs_obj_coords.append(outputs_obj_coord)

            # outputs_sub_coords.append(self.sub_bbox_embed[lvl](hs[lvl]).sigmoid())
            # outputs_obj_coords.append(self.obj_bbox_embed[lvl](hs[lvl]).sigmoid())
            outputs_verb_class = self.verb_class_embed[lvl](hs_verb[lvl])
            outputs_obj_class = self.obj_class_embed[lvl](hs_o[lvl])
            outputs_verb_classes.append(outputs_verb_class)
            outputs_obj_classes.append(outputs_obj_class)

        outputs_verb_class = torch.stack(outputs_verb_classes)
        outputs_obj_class = torch.stack(outputs_obj_classes)
        outputs_sub_coord = torch.stack(outputs_sub_coords)
        outputs_obj_coord = torch.stack(outputs_obj_coords)

        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}

        if self.subject_class:
            outputs_sub_class = []
            for lvl in range(hs_h.shape[0]):
                outputs_sub_class.append(self.obj_class_embed[lvl](hs_h[lvl]))
            outputs_sub_class = torch.stack(outputs_sub_class, dim = 0)

            out.update({'pred_sub_logits': outputs_sub_class[-1]})
            if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord, outputs_sub_class) 
        else:
            if self.aux_loss: 
                # Using aux loss means that you will add loss to every intermidiate layer.
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord) 

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_sub_class = None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_sub_class is None:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                        outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        else:
            return [{'pred_sub_logits':a, 'pred_obj_logits': b, 'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_obj_boxes': e}
                for a, b, c, d, e in zip(outputs_sub_class[:-1], outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]



class DETRHOI(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        # self.obj_class_embed = MLP(hidden_dim, hidden_dim, num_obj_classes + 1, 3)
        # self.verb_class_embed = MLP(hidden_dim, hidden_dim, num_verb_classes, 3)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, **kwargs):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # features: [features_tensor from layer4,]  
        # pos: [pos embedding for features from layer4, ]  
        #      layer4 pos embedding shape like: [2, 256, 18, 25]

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # self.transformer return decoder info and memory
        # hs: tensor [6, 2, 100, 256]  6 is the #decoder_layers

        outputs_obj_class = self.obj_class_embed(hs)
        outputs_verb_class = self.verb_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.aux_loss: 
            # Using aux loss means that you will add loss to every intermidiate layer.
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def generate_soft_label_cross_entropy(target_classes, target_classes_o, soft_gt_value, src_logits, idx):
    '''
    This function creates soft_labels for cross entropy loss.
    default shape:
        target_classes: [batch_size, samples], e.g., [2, 100]
        target_classes_o: [num_positive_samples, ], e.g., [15,]
        soft_gt_value: [num_positive_samples, ], e.g., [15,]
        src_logits: [batch_size, samples, num_classes], e.g., [2, 100, 334]
        idx: ([num_positive_samples, ], [num_positive_samples, ])
    '''
    num_classes = src_logits.shape[-1]
    num_positive_samples = soft_gt_value.shape[0]
    target_logits = torch.nn.functional.one_hot(target_classes, src_logits.shape[-1]).float().to(src_logits.device)
    soft_tensor = torch.zeros((num_positive_samples, num_classes)).to(src_logits.device)
    for i, (tensor, soft_gt) in enumerate(zip(soft_tensor, soft_gt_value)):
        tensor.fill_((1 - soft_gt)/num_classes)
        tensor[target_classes_o[i]] += soft_gt
        # tensor.fill_((1 - soft_gt)/(num_classes - 1))
        # tensor[target_classes_o[i]] = soft_gt
    target_logits[idx] = soft_tensor

    return target_logits

class SetCriterionHOI(nn.Module):
    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, 
                       eos_coef, losses, verb_loss_type, obj_loss_type = 'cross_entropy', temperature = 0.07, 
                       matching_symmetric = True, RLIP_ParSe = False, subject_class = False, use_no_verb_token = False, 
                       giou_verb_label = False, verb_curing = False, pseudo_verb = False, triplet_filtering = False,
                       naive_obj_smooth = 0, naive_verb_smooth = 0, args=None):
        super().__init__()

        assert verb_loss_type in ['weighted_bce', 'focal', 'focal_without_sigmoid', 'focal_bce', 'asymmetric_bce', 'CB_focal_bce','bce', 'cross_modal_matching']
        assert obj_loss_type in ['cross_entropy', 'cross_modal_matching', 'cross_entropy_with_tem', 'cross_entropy_with_tem_focal',
                                 'cross_entropy_symmetric']
        print('verb_loss_type:', verb_loss_type, ';', 'obj_loss_type:', obj_loss_type)
        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.verb_loss_type = verb_loss_type
        self.obj_loss_type = obj_loss_type
        self.temperature = temperature
        self.matching_symmetric = matching_symmetric
        self.RLIP_ParSe = RLIP_ParSe
        self.subject_class = subject_class
        self.use_no_verb_token = use_no_verb_token
        self.giou_verb_label = giou_verb_label
        if self.giou_verb_label:
            assert verb_loss_type == 'focal' and obj_loss_type == 'cross_entropy'
        self.pseudo_verb = pseudo_verb
        self.verb_curing = verb_curing
        if self.giou_verb_label:
            assert verb_loss_type == 'focal'
        if self.verb_curing:
            assert verb_loss_type == 'focal'
        self.triplet_filtering = triplet_filtering
        if self.triplet_filtering:
            assert self.subject_class # To make sure that we are using VG with subject classes
        self.naive_verb_smooth = naive_verb_smooth
        self.naive_obj_smooth = naive_obj_smooth
        assert ((self.naive_verb_smooth > 0) and self.giou_verb_label) is not True
        print('Use naive_obj_smooth?', self.naive_obj_smooth)
        print('Use naive_verb_smooth?', self.naive_verb_smooth)
        print('Use pseudo_verb?', self.pseudo_verb)
        print('Use verb_curing?', self.verb_curing)
        print('Use triplet_filtering?', self.triplet_filtering)

        # For CB focal
        samples = np.load('datasets/priors/hico_verb_samples.npz')['matrices']
        samples = torch.tensor(samples).float()
        self.register_buffer('samples', samples)
        self.img_num_hico = 37536
        self.img_num_vcoco = 5400
        self.query_num = 100
        self.register_buffer('bce_weight', self.BCE_weight())

        self.verb_tagger = args.verb_tagger
    
    def BCE_weight(self,):
        total_num = self.img_num_hico * self.query_num
        pos_verb_samples = self.samples
        neg_verb_samples = total_num - pos_verb_samples
        pos_verb_w = torch.ones(self.samples.shape)
        neg_verb_w = torch.sqrt(pos_verb_samples) / torch.sqrt(neg_verb_samples)
        return torch.stack((pos_verb_w, neg_verb_w), dim = 1)

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        if self.obj_loss_type in ['cross_entropy', 'cross_entropy_with_tem', 'cross_entropy_with_tem_focal', \
                                  'cross_entropy_symmetric']:
            # This means that we are performing the pretraining.
            if self.subject_class:
                ### Calculate loss for objects
                if "with_tem" in self.obj_loss_type:
                    src_logits_obj = outputs['pred_obj_logits']/self.temperature
                else:
                    src_logits_obj = outputs['pred_obj_logits']
                obj_weight = torch.ones(src_logits_obj.shape[-1], device = src_logits_obj.device)
                obj_weight[-1] = self.eos_coef
                idx = self._get_src_permutation_idx(indices)
                # idx: a tuple (batch_idx, src_idx)
                target_classes_o_obj = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
                target_classes = torch.full(src_logits_obj.shape[:2], src_logits_obj.shape[-1] - 1,
                                            dtype=torch.int64, device=src_logits_obj.device)
                # target_classes: init with a tensor of size src_logits_obj.shape[:2]
                #                 and filled with self.num_obj_classes (no object class)
                if target_classes_o_obj.shape[0] >= 0:
                    target_classes[idx] = target_classes_o_obj
                # fill the target_classes with the gt object classes
                if "focal" in self.obj_loss_type:
                    gamma = 2
                    # alpha = 0.5 # It won't work because every term is multiplied by an alpha.
                    logprobs = F.cross_entropy(src_logits_obj.transpose(1, 2), target_classes, reduction = 'none')
                    logprobs_weights = obj_weight[target_classes]
                    pt = torch.exp(-logprobs)
                    focal_loss = (1 - pt)**gamma * logprobs * logprobs_weights
                    loss_obj_ce = focal_loss.mean()
                elif "symmetric" in self.obj_loss_type:
                    loss_obj_ce = F.cross_entropy(src_logits_obj.transpose(1, 2), target_classes, obj_weight)
                    # RCE
                    pred = F.softmax(src_logits_obj, dim = -1)
                    pred = torch.clamp(pred, min = 1e-7, max = 1.0)
                    label_one_hot = torch.nn.functional.one_hot(target_classes, src_logits_obj.shape[-1]).float().to(src_logits_obj.device)
                    label_one_hot = torch.clamp(label_one_hot, min = 1e-4, max = 1.0)
                    loss_obj_rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim = 1))

                    # Loss
                    # loss_obj_ce = 0.1 * loss_obj_ce + 1.0 * loss_obj_rce.mean()
                    loss_obj_ce = 6.0 * loss_obj_ce + 0.1 * loss_obj_rce.mean()

                else:
                    # if self.giou_verb_label:
                    #     _, cost_list = self.matcher(outputs, targets, return_cost = True) # cost_giou shape: [bs*num_queries, num_hoi]
                    #     cost_sub_giou, cost_obj_giou = cost_list[1]
                    #     cost_obj_giou = - cost_obj_giou
                    #     query_global = 0
                    #     target_global = 0
                    #     soft_gt_value = []
                    #     for t, (I, J) in zip(targets, indices):
                    #         soft_obj = ((cost_obj_giou[query_global + I, target_global + J]) + 1) / 2 # scale giou to the range from 0 to 1
                    #         assert ((soft_obj >= 0)&(soft_obj <= 1)).all()
                    #         soft_gt_value.append(soft_obj)
                    #         query_global += src_logits_obj.shape[1]
                    #         target_global += J.shape[0]
                    #     soft_gt_value = torch.cat(soft_gt_value)

                    #     target_logits_obj = generate_soft_label_cross_entropy(target_classes, target_classes_o_obj, 
                    #                                                           soft_gt_value, src_logits_obj, idx)
                    #     loss_obj_ce = - (F.log_softmax(src_logits_obj, dim = -1) * target_logits_obj * obj_weight).sum(dim = -1).mean()
                    # else:
                    if self.naive_obj_smooth > 0:
                        target_logits_obj = F.one_hot(target_classes, src_logits_obj.shape[-1]).float().to(src_logits_obj.device)
                        target_logits_obj = target_logits_obj * (1 - self.naive_obj_smooth + self.naive_obj_smooth/src_logits_obj.shape[-1]) + \
                                            (1 - target_logits_obj) * self.naive_obj_smooth/src_logits_obj.shape[-1]
                        # print(target_logits_obj.max(-1)[1], target_logits_obj.sum())
                        loss_obj_ce = - (F.log_softmax(src_logits_obj, dim = -1) * target_logits_obj * obj_weight).sum(dim = -1)
                        loss_obj_ce = loss_obj_ce.sum() / obj_weight[target_classes].sum()
                        # This is significant, we do not use .mean(). Instead, we aggregate weights for all gt labels.
                        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss 
                    else:
                        loss_obj_ce = F.cross_entropy(src_logits_obj.transpose(1, 2), target_classes, obj_weight)
                

                ### Calculate loss for subjects
                if "with_tem" in self.obj_loss_type:
                    src_logits_sub = outputs['pred_sub_logits']/self.temperature
                else:
                    src_logits_sub = outputs['pred_sub_logits']
                sub_weight = torch.ones(src_logits_sub.shape[-1], device = src_logits_sub.device)
                sub_weight[-1] = self.eos_coef
                target_classes_o_sub = torch.cat([t['sub_labels'][J] for t, (_, J) in zip(targets, indices)])
                target_classes = torch.full(src_logits_sub.shape[:2], src_logits_sub.shape[-1] - 1,
                                            dtype=torch.int64, device=src_logits_sub.device)
                if target_classes_o_sub.shape[0] >= 0:
                    target_classes[idx] = target_classes_o_sub
                
                if "focal" in self.obj_loss_type:
                    gamma = 2
                    # alpha = 0.5 # It won't work because every term is multiplied by an alpha.
                    logprobs = F.cross_entropy(src_logits_sub.transpose(1, 2), target_classes, reduction = 'none')
                    logprobs_weights = sub_weight[target_classes]
                    pt = torch.exp(-logprobs)
                    focal_loss = (1 - pt)**gamma * logprobs * logprobs_weights
                    loss_sub_ce = focal_loss.mean()
                else:
                    # if self.giou_verb_label:
                    #     cost_sub_giou = - cost_sub_giou
                    #     query_global = 0
                    #     target_global = 0
                    #     soft_gt_value = []
                    #     for t, (I, J) in zip(targets, indices):
                    #         soft_sub = ((cost_sub_giou[query_global + I, target_global + J]) + 1) / 2 # scale giou to the range from 0 to 1
                    #         assert ((soft_sub >= 0)&(soft_sub <= 1)).all()
                    #         soft_gt_value.append(soft_sub)
                    #         query_global += src_logits_sub.shape[1]
                    #         target_global += J.shape[0]
                    #     soft_gt_value = torch.cat(soft_gt_value)
                    #     target_logits_sub = generate_soft_label_cross_entropy(target_classes, target_classes_o_sub, 
                    #                                                           soft_gt_value, src_logits_sub, idx)
                    #     loss_sub_ce = - (F.log_softmax(src_logits_sub, dim = -1) * target_logits_sub * sub_weight).sum(dim = -1).mean()
                    
                    # else:
                    if self.naive_obj_smooth > 0:
                        target_logits_sub = F.one_hot(target_classes, src_logits_sub.shape[-1]).float().to(src_logits_sub.device)
                        target_logits_sub = target_logits_sub * (1 - self.naive_obj_smooth + self.naive_obj_smooth/src_logits_sub.shape[-1]) + \
                                            (1 - target_logits_sub) * self.naive_obj_smooth/src_logits_sub.shape[-1]
                        # print(target_logits_sub.max(-1)[0], target_logits_sub.sum())
                        loss_sub_ce = - (F.log_softmax(src_logits_sub, dim = -1) * target_logits_sub * sub_weight).sum(dim = -1)
                        loss_sub_ce = loss_sub_ce.sum() / sub_weight[target_classes].sum() 
                        # This is significant, we do not use .mean(). Instead, we aggregate weights for all gt labels.
                        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss 
                    else:
                        loss_sub_ce = F.cross_entropy(src_logits_sub.transpose(1, 2), target_classes, sub_weight)
                
                losses = {'loss_obj_ce': loss_obj_ce + loss_sub_ce}
                # losses = {'loss_obj_ce': torch.tensor(0., device = src_logits_sub.device)}
                if log:
                    losses['obj_class_error'] = 100 - accuracy(src_logits_obj[idx], target_classes_o_obj)[0]
                    losses['sub_class_error'] = 100 - accuracy(src_logits_sub[idx], target_classes_o_sub)[0]
                return losses

            else:
                # hack implementation about 'cross_entropy_with_tem' and 'cross_entropy_with_tem_focal'
                assert self.obj_loss_type in ['cross_entropy', 'cross_entropy_with_tem']
                assert 'pred_obj_logits' in outputs
                if "with_tem" in self.obj_loss_type:
                    src_logits = outputs['pred_obj_logits'] / self.temperature
                else:
                    src_logits = outputs['pred_obj_logits']

                idx = self._get_src_permutation_idx(indices)
                # idx: a tuple (batch_idx, src_idx)
                target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
                target_classes = torch.full(src_logits.shape[:2], src_logits.shape[-1] - 1, # self.num_obj_classes,
                                            dtype=torch.int64, device=src_logits.device)
                # target_classes: init with a tensor of size src_logits.shape[:2]
                #                 and filled with self.num_obj_classes (no object class)
                target_classes[idx] = target_classes_o
                # fill the target_classes with the gt object classes

                # print("src_logits " + str(src_logits.shape)) # [2, 100, 81]
                # print("target_classes " + str(target_classes.shape)) # [2, 100]
                # if self.giou_verb_label:
                #     _, cost_list = self.matcher(outputs, targets, return_cost = True) # cost_giou shape: [bs*num_queries, num_hoi]
                #     # cost_sub_giou, cost_obj_giou = cost_list[1]
                #     # cost_obj_giou = - cost_obj_giou
                #     cost_verb_class = - cost_list[4]
                #     query_global = 0
                #     target_global = 0
                #     soft_gt_value = []
                #     for t, (I, J) in zip(targets, indices):
                #         # soft_obj = ((cost_obj_giou[query_global + I, target_global + J]) + 1) / 2 # scale giou to the range from 0 to 1
                #         soft_obj = cost_verb_class[query_global + I, target_global + J] # scale giou to the range from 0 to 1
                #         assert ((soft_obj >= 0)&(soft_obj <= 1)).all()
                #         soft_gt_value.append(soft_obj)
                #         query_global += src_logits.shape[1]
                #         target_global += J.shape[0]
                #     soft_gt_value = torch.cat(soft_gt_value)

                #     target_logits_obj = generate_soft_label_cross_entropy(target_classes, target_classes_o, 
                #                                                           soft_gt_value, src_logits, idx)
                #     loss_obj_ce = - (F.log_softmax(src_logits, dim = -1) * target_logits_obj * self.empty_weight).sum(dim = -1).mean()
                # else:
                if self.naive_obj_smooth > 0:
                    target_logits = F.one_hot(target_classes, src_logits.shape[-1]).float().to(src_logits.device)
                    target_logits = target_logits * (1 - self.naive_obj_smooth + self.naive_obj_smooth/src_logits.shape[-1]) + \
                                        (1 - target_logits) * self.naive_obj_smooth/src_logits.shape[-1]
                    loss_obj_ce = - (F.log_softmax(src_logits, dim = -1) * target_logits * self.empty_weight).sum(dim = -1)
                    loss_obj_ce = loss_obj_ce.sum() / self.empty_weight[target_classes].sum() 
                    # This is significant, we do not use .mean(). Instead, we aggregate weights for all gt labels.
                    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss 
                else:
                    loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
                losses = {'loss_obj_ce': loss_obj_ce}

                if log:
                    losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
                return losses

        elif self.obj_loss_type == 'cross_modal_matching':
            if self.subject_class:
                loss_sub_matching = self._contrastive_align(outputs, targets, indices, text_type = 'sub')
                loss_obj_matching = self._contrastive_align(outputs, targets, indices, text_type = 'obj')
                losses = {'loss_sub_matching': loss_sub_matching,\
                        'loss_obj_matching': loss_obj_matching}
                if log:
                    idx = self._get_src_permutation_idx(indices)
                    # idx: a tuple (batch_idx, src_idx)
                    target_classes_obj = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
                    target_classes_sub = torch.cat([t['sub_labels'][J] for t, (_, J) in zip(targets, indices)])
                    losses['obj_class_error'] = 100 - accuracy(outputs['pred_obj_logits'][idx], target_classes_obj)[0]
                    losses['sub_class_error'] = 100 - accuracy(outputs['pred_sub_logits'][idx], target_classes_sub)[0]
                return losses
            else:
                assert False
        else:
            assert False


    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits'] # [2, 100, 81]
        # print('pred_logits' + str(pred_logits.shape))
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        # tgt_lengths: number of predicted objects 
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # card_pred: number of true objects 
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        # l1_loss that takes the mean element-wise absolute value difference.
        
        losses = {'obj_cardinality_error': card_err}
        # print('tgt_lengths:'+str(tgt_lengths.shape)) # [2]
        # print('card_pred:'+str(card_pred.shape)) # [2]
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        if self.verb_loss_type in ['weighted_bce', 'focal', 'focal_bce', 'focal_without_sigmoid', 
                                   'asymmetric_bce', 'CB_focal_bce','bce']:
            assert 'pred_verb_logits' in outputs
            src_logits = outputs['pred_verb_logits'] # [2, 100, 117]

            idx = self._get_src_permutation_idx(indices)
            if self.giou_verb_label:
                _, cost_list = self.matcher(outputs, targets, return_cost = True) # cost_giou shape: [bs*num_queries, num_hoi]
                cost_giou = cost_list[0]
                cost_giou = - cost_giou # the matching giou used in QPIC is negative
                # cost_bbox = cost_list[2]
                # cost_bbox = 4 - cost_bbox
                query_global = 0
                target_global = 0
                target_soft = []
                for t, (I, J) in zip(targets, indices):
                    ## relation label v1: only GIoU is considered.
                    soft_verb = ((cost_giou[query_global + I, target_global + J]) + 1) / 2 # scale giou to the range from 0 to 1
                    ## relation label v2: GIoU and l1 bbox are jointly considered.
                    # soft_verb = ((cost_bbox[query_global + I, target_global + J])/4 + \
                    #             ((cost_giou[query_global + I, target_global + J]) + 1) / 2) / 2
                    ## relation label v3: GIoU and obj_cls are jointly considered.
                    # if len(cost_list) == 7:
                    #     cost_sub_cls = - cost_list[5]
                    #     cost_obj_cls = - cost_list[6]
                    #     if cost_obj_cls.shape[1] == 0:
                    #         soft_verb = torch.ones_like(J, device = src_logits.device)
                    #     else:
                    #         cost_cls = torch.stack((cost_sub_cls, cost_obj_cls), dim = 0).min(0)[0]
                    #         soft_verb = (((cost_giou[query_global + I, target_global + J]) + 1)/2 + \
                    #                     cost_cls[query_global + I, target_global + J])/2
                    # elif len(cost_list) == 6:
                    #     cost_obj_cls = - cost_list[5]
                    #     if cost_obj_cls.shape[1] == 0:
                    #         soft_verb = torch.ones_like(J, device = src_logits.device)
                    #     else:
                    #         soft_verb = (((cost_giou[query_global + I, target_global + J]) + 1)/2 + \
                    #                     cost_obj_cls[query_global + I, target_global + J])/2

                    if ((soft_verb >= 0)&(soft_verb <= 1)).all().item() is False:
                        print(soft_verb)
                    assert ((soft_verb >= 0)&(soft_verb <= 1)).all()
                    if self.pseudo_verb:
                        assert 'target_verb_sim' in outputs.keys()
                        target_verb_sim = outputs['target_verb_sim']
                        # print((t['verb_labels'][J]>0).sum(), (target_verb_sim[target_global + J]>0).sum())
                        target_soft.append((t['verb_labels'][J] + target_verb_sim[target_global + J]) * soft_verb.unsqueeze(dim = -1))
                    else:
                        target_soft.append(t['verb_labels'][J] * soft_verb.unsqueeze(dim = -1))
                    query_global += src_logits.shape[1]
                    target_global += J.shape[0]
                target_classes_o = torch.cat(target_soft)
            elif self.naive_verb_smooth > 0:
                target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
                # target_classes_o = target_classes_o * (1 - self.naive_verb_smooth + self.naive_verb_smooth/src_logits.shape[-1])
                target_classes_o = target_classes_o * (1 - self.naive_verb_smooth + self.naive_verb_smooth/src_logits.shape[-1]) + \
                                (1 - target_classes_o) * self.naive_verb_smooth / src_logits.shape[-1]
            else:
                target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
            # [num_of_verbs, 117]
            if self.use_no_verb_token:
                src_logits = src_logits[:, :, :src_logits.shape[2]-1]
            target_classes = torch.zeros_like(src_logits)
            target_classes[idx] = target_classes_o
            # target_classes = torch.ones_like(src_logits)*57
            # target_classes = torch.full(src_logits, self.num_verb_classes,
            #                             dtype=torch.int64, device=src_logits.device)

            if self.verb_loss_type == 'bce':
                loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
            elif self.verb_loss_type == 'focal':
                src_logits = src_logits.sigmoid()
                if self.verb_curing:
                    src_logits = src_logits * outputs['curing_score']
                if self.giou_verb_label or self.naive_verb_smooth > 0:
                    loss_verb_ce = self._soft_neg_loss(src_logits, target_classes)
                else:
                    loss_verb_ce = self._neg_loss(src_logits, target_classes)
            elif self.verb_loss_type == 'focal_without_sigmoid':
                loss_verb_ce = self._neg_loss(src_logits, target_classes)
            elif self.verb_loss_type == 'focal_bce':
                src_logits = src_logits.sigmoid()
                loss_verb_ce = self._focal_bce(src_logits, target_classes)
            elif self.verb_loss_type == 'asymmetric_bce':
                src_logits = src_logits.sigmoid()
                loss_verb_ce = self._asymmetric_bce(src_logits, target_classes)
            elif self.verb_loss_type == 'CB_focal_bce':
                src_logits = src_logits.sigmoid()
                loss_verb_ce = self._CB_focal_bce(src_logits, target_classes)
            elif self.verb_loss_type == 'weighted_bce':
                src_logits = src_logits.sigmoid()
                loss_verb_ce = self._weighted_bce(src_logits, target_classes)

            if 'pri_pred_verb_logits' in outputs:
                pri_src_logits = outputs['pri_pred_verb_logits']
                if self.verb_loss_type == 'bce':
                    loss_verb_ce += F.binary_cross_entropy_with_logits(pri_src_logits, target_classes)
                elif self.verb_loss_type == 'focal':
                    pri_src_logits = pri_src_logits.sigmoid()
                    loss_verb_ce += self._neg_loss(pri_src_logits, target_classes)

            losses = {'loss_verb_ce': loss_verb_ce}
            return losses
        elif self.verb_loss_type in ['cross_modal_matching']:
            loss_verb_matching = self._contrastive_align(outputs, targets, indices, text_type = 'verb')
            losses = {'loss_verb_matching': loss_verb_matching}
            return losses
        else:
            assert False
    
    def loss_verb_tagger(self, outputs, targets, indices, num_interactions, log=True):
        '''
        This is the reconstruction loss for RLIP-ParSe Verb Tagger.
        :param indices: by default, it is 'None'.
        '''
        assert indices == None
        device = outputs['pred_verb_logits'].device

        pair_num_list = [v["sub_boxes"].shape[0] for v in targets]
        pair_idx_list = np.cumsum(pair_num_list)
        pair_idx_list = [0,] + list(pair_idx_list)
        tgt_sub_bbox = torch.cat([v["sub_boxes"] for v in targets])  # x1, y1, x2, y2
        tgt_obj_bbox = torch.cat([v["obj_boxes"] for v in targets])  # x1, y1, x2, y2

        label_bs_idx = torch.cat([torch.tensor([i,]*v["sub_labels"].shape[0], dtype=torch.int64) for i,v in enumerate(targets)]).to(device=device)
        label_query_idx = torch.cat([torch.tensor(range(0, v["sub_labels"].shape[0]), dtype=torch.int64) for v in targets]).to(device=device)
        assert label_bs_idx.shape == label_query_idx.shape
        assert 'pred_verb_logits' in outputs and 'pred_sub_logits' in outputs and 'pred_obj_logits' in outputs and 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        src_logits = outputs['pred_verb_logits'][label_bs_idx, label_query_idx]
        src_logits_sub = outputs['pred_sub_logits'][label_bs_idx, label_query_idx]
        src_logits_obj = outputs['pred_obj_logits'][label_bs_idx, label_query_idx]
        src_sub_boxes = outputs['pred_sub_boxes'][label_bs_idx, label_query_idx]
        src_obj_boxes = outputs['pred_obj_boxes'][label_bs_idx, label_query_idx]
        
        target_classes = torch.cat([t['verb_labels'] for t in targets])
        target_classes_s = torch.cat([t['sub_labels'] for t in targets])
        target_classes_o = torch.cat([t['obj_labels'] for t in targets])
        target_sub_boxes = torch.cat([t['sub_boxes'] for t in targets])
        target_obj_boxes = torch.cat([t['obj_boxes'] for t in targets])

        ### Loss for subjects and objects
        losses = {}
        # Loss for bbox localization
        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            # print(src_sub_boxes)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        
        # Loss for bbox classes
        if src_logits_obj.shape[0] > 0:
            obj_weight = torch.ones(src_logits_obj.shape[-1], device = src_logits_obj.device)
            obj_weight[-1] = self.eos_coef
            loss_obj_ce = F.cross_entropy(src_logits_obj, target_classes_o, obj_weight)

            sub_weight = torch.ones(src_logits_sub.shape[-1], device = src_logits_sub.device)
            sub_weight[-1] = self.eos_coef
            loss_sub_ce = F.cross_entropy(src_logits_sub, target_classes_s, sub_weight)

            losses['loss_obj_ce'] = loss_obj_ce + loss_sub_ce
        else:
            losses['loss_obj_ce'] = torch.tensor(0).to(device = src_logits_obj.device)

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits_obj, target_classes_o)[0]
            losses['sub_class_error'] = 100 - accuracy(src_logits_sub, target_classes_s)[0]
        # losses['loss_obj_ce'] = torch.tensor(0).to(device=src_logits.deivce)
        

        ### Loss for verbs
        if self.verb_loss_type == 'bce':
            loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        elif self.verb_loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._neg_loss(src_logits, target_classes)

        losses['loss_verb_ce'] = loss_verb_ce

        return losses

    def loss_masked_entity_modeling(self, outputs, targets, indices, num_interactions):
        ### version 4, Recon modeling, proposed by Jianwen
        out_dict = {}
        loss_recon_labels = self.loss_obj_labels(outputs['recon_stat'], targets, indices, num_interactions)
        loss_recon_boxes = self.loss_sub_obj_boxes(outputs['recon_stat'], targets, indices, num_interactions)
        out_dict.update(loss_recon_labels)
        out_dict.update(loss_recon_boxes)
        new_out_dict = {i + '_recon': j for i,j in out_dict.items()}
        return new_out_dict
        

    def loss_gt_verb_recon(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits'] # [2, 100, 117]
        semantic = outputs['semantic']
        hs = outputs['hs']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        # [num_of_verbs, 117]
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        
        if self.verb_loss_type == 'bce':
            cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        elif self.verb_loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            # self.check_0_1(verb_gt_recon, 'src_logits')
            cls_loss = self._neg_loss(src_logits, target_classes)
        
        # Loss for All queries
        loss_recon = torch.tensor(0., device = target_classes.device)
        semantic_norm = norm_tensor(semantic)
        hs_norm = norm_tensor(hs)
        cos_sim = torch.einsum('abd,cd->abc', hs_norm, semantic_norm)
        pos_loss = 1 - cos_sim
        neg_loss = torch.clamp(cos_sim - 0.1, min = 0)
        recon_loss = (pos_loss * target_classes + neg_loss * (1 - target_classes)).sum() / target_classes.sum()

        loss = cls_loss + recon_loss

        return {'loss_verb_gt_recon': loss}


    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_sub_boxes = outputs['pred_sub_boxes'][idx] # shape like [5, 4] [6, 4]...
        # print(src_sub_boxes)
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # print(target_sub_boxes)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        
        return losses
    
    def loss_kl_divergence(self, outputs, targets, indices, num_interactions):
        if 'verb_kl_divergence' in outputs:
            kl_param = outputs['verb_kl_divergence']
            bs, num_queries, latentdim2 = kl_param.shape # 2, 100, 256*2
            verb_mu, verb_log_var = kl_param[:,:,:latentdim2//2], kl_param[:,:,latentdim2//2:]
            verb_var = torch.exp(verb_log_var)
            loss = -0.5 * (1 + verb_log_var - verb_mu*verb_mu - verb_var)
            loss = torch.mean(loss)
            
        else:
            assert False

        return {'loss_kl_divergence': loss}

    def cal_entropy_loss(self, log_var, latentdim, bound):
        cons_term = latentdim/2.*(math.log(2*math.pi) + 1.)
        ### The following line seems to be a mistake in my previous implementation
        # var_term = 0.5*torch.sum(log_var, dim = 1) 
        var_term = 0.5*torch.sum(log_var, dim = -1)
        avg_entropy = torch.mean(cons_term + var_term) 
        loss = torch.max(torch.Tensor((0, bound - avg_entropy)).to(avg_entropy.device))
        return loss

    def loss_entropy_bound(self, outputs, targets, indices, num_interactions):
        if 'verb_log_var' in outputs:
            log_var = outputs['verb_log_var']
            b, nq, latentdim = log_var.shape
            latentdim = latentdim//2
            verb_log_var, obj_class_log_var = log_var[...,:latentdim], log_var[...,latentdim:]
            loss = self.cal_entropy_loss(verb_log_var, latentdim, bound = 256) +\
                   self.cal_entropy_loss(obj_class_log_var, latentdim, bound = 256)
 
        elif 'masked_context_log_var' in outputs:
            masked_memory_log_var = outputs['masked_context_log_var']
            _, latentdim = masked_memory_log_var.shape

            # Entropy bound
            cons_term = latentdim/2.*(math.log(2*math.pi) + 1.)
            var_term = 0.5*torch.sum(masked_memory_log_var, dim = 1) # [all pixels with false masks in all batches,]
            pixel_avg_entropy = torch.mean(cons_term + var_term)
            
            loss = torch.max(torch.Tensor((0, 256 - pixel_avg_entropy))).to(pixel_avg_entropy.device)

        else:
            assert False

        return {'loss_entropy_bound': loss}

    
    def loss_verb_hm(self, outputs, targets, indices, num_interactions):
        pred_verb_hm, mask = outputs['verb_hm']
        neg_loss = 0.
        # mask shape [bs,c,h,w]
        for ind, t in enumerate(targets):
            gt_verb_hm = t['verb_hm']
            valid_1 = torch.sum(~mask[ind][:,:,0])
            valid_2 = torch.sum(~mask[ind][:,0,:])
            # interpolate input [bs,c,h,w]
            gt_verb_hm = F.interpolate(gt_verb_hm.unsqueeze(0), size = (valid_1, valid_2)).squeeze(0)

            neg_loss += self._neg_loss(pred_verb_hm[ind][:,:valid_1,:valid_2], gt_verb_hm)


        return {'loss_verb_hm': neg_loss}
    
    def loss_verb_threshold(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        assert 'pred_verb_thr' in outputs
        src_logits = outputs['pred_verb_logits'] # [2, 100, 117]
        thr = outputs['pred_verb_thr'] # [2, 100, 117]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        # [num_of_verbs, 117]
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o  # [2, 100, 117]
        # target_classes = torch.ones_like(src_logits)*57
        # target_classes = torch.full(src_logits, self.num_verb_classes,
        #                             dtype=torch.int64, device=src_logits.device)

        sigma = torch.sigmoid(src_logits - thr)
        loss_verb_thr = self._neg_loss(sigma, target_classes, eps = 1e-6)

        return {'loss_verb_threshold': loss_verb_thr}


    def loss_semantic_similar(self, outputs, targets, indices, num_interactions):
        temperature = 0.05
        if 'semantic' in outputs and 'verb_verb_co' in outputs:
            # semantic = outputs['semantic_low'] # 117, 256
            semantic = outputs['semantic'] # 117, 256
            # verb_verb_co = outputs['verb_verb_co'] # 117, 117
            # verb_verb_co = outputs['joint_verb_verb_co'] # 117, 117
            # Symmetric cond prob
            verb_verb_co = outputs['verb_verb_co'] 
            verb_verb_co = verb_verb_co + verb_verb_co.T
            verb_verb_co = verb_verb_co / verb_verb_co.sum()

            norm_semantic = norm_tensor(semantic)
            # norm_semantic = semantic
            semantic_sim = torch.einsum('ab,cb->ac', norm_semantic, norm_semantic)  # 117, 117
            eye_mask = ~(torch.eye(verb_verb_co.shape[0], device = verb_verb_co.device) == 1)
            semantic_sim = semantic_sim[eye_mask]
            

            # normalize: make semantic_sim sum to 1
            # semantic_sim = semantic_sim

            # MSE loss
            # semantic_mse = F.mse_loss(semantic_sim, verb_verb_co, reduce = False)
            #     # w/ Relaxation
            # # relax_flag = verb_verb_co > 0.01   # 0.01:308; 0.02:97; 0.05:36; 0.1:13; 
            # # loss_sim = torch.sum(semantic_mse[relax_flag])
            #     # w/o Relaxation
            # loss_sim = torch.sum(semantic_mse)

            # KL loss
            # # verb_verb_co = verb_verb_co + verb_verb_co.T
            # semantic_sim = F.log_softmax(semantic_sim, dim = -1)
            # loss_sim = F.kl_div(semantic_sim, verb_verb_co, reduction = 'sum')
            # sum normalization KL
            # semantic_sim = torch.log(semantic_sim / semantic_sim.sum(dim = -1).unsqueeze(dim = -1))
            # loss_sim = F.kl_div(semantic_sim, verb_verb_co, reduction = 'sum')
            # KL loss with relaxation
            # semantic_sim = F.log_softmax(semantic_sim, dim = -1)
            # loss_sim = F.kl_div(semantic_sim, verb_verb_co, reduce = False)
            # loss_sim = (loss_sim[verb_verb_co>0.3]).sum()
            
            # KL loss for joint probability distribution
            # semantic_sim = F.log_softmax(semantic_sim.flatten())
            # loss_sim = F.kl_div(semantic_sim, verb_verb_co.flatten(), reduction = 'sum')
            # KL loss for joint probability distribution with eye mask
            semantic_sim = F.log_softmax(semantic_sim / temperature)
            loss_sim = F.kl_div(semantic_sim, verb_verb_co[eye_mask], reduction = 'sum')

            # semantic_sim_soft = F.softmax(semantic_sim / temperature)
            # semantic_sim_logsoft = F.log_softmax(semantic_sim / temperature)
            # loss_sim = (semantic_sim_soft * (semantic_sim_logsoft - (verb_verb_co[eye_mask]).log())).sum()

            #################### AAAI 2020 implementatioin ##################
            # semantic = outputs['semantic'] # 117, 256
            # verb_verb_co = outputs['verb_verb_co'] 
            # verb_verb_co = verb_verb_co + verb_verb_co.T
            # norm_semantic = norm_tensor(semantic)
            # semantic_sim = torch.einsum('ab,cb->ac', norm_semantic, norm_semantic) # # 117, 117
            # # MSE loss
            # semantic_mse = F.mse_loss(semantic_sim, verb_verb_co, reduce = False)
            #     # w/ Relaxation
            # relax_flag = verb_verb_co > 0.1   # 0.01:308; 0.02:97; 0.05:36; 0.1:13; 
            # loss_sim = torch.sum(semantic_mse[relax_flag])
            #     # w/o Relaxation
            # # loss_sim = torch.sum(semantic_mse)

        else:
            loss_sim = torch.tensor([0.], device = outputs['pred_obj_logits'].device).sum()

        return {'loss_semantic_similar': loss_sim}
    
    def _weighted_bce(self, pred, gt, eps = 1e-6):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = self.bce_weight[:,1]

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * pos_inds
        neg_loss = torch.log(1 - pred) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    
    def _CB_focal_bce(self, pred, gt, eps = 1e-6, gamma = 2, alpha = 0.5, vol = 2, beta = 0.9999):
        beta_weight = (1-beta) / (1 - torch.pow(beta, self.samples)) 
        beta_weight = beta_weight.unsqueeze(dim = 0).unsqueeze(dim = 0)

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * alpha * vol * pos_inds * beta_weight
        neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * (1 - alpha) * vol * neg_inds * beta_weight

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss



    def _asymmetric_bce(self, pred, gt, eps = 1e-6, gamma_pos = 0, gamma_neg = 3, m = 0.01, vol = 1):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred_p = torch.clamp(pred, min = eps, max = 1.)
        pos_loss = torch.log(pred_p) * torch.pow(1 - pred_p, gamma_pos) * vol * pos_inds
        pred_m = torch.clamp(pred - m, min = 0, max = 1. - eps)
        neg_loss = torch.log(1 - pred_m) * torch.pow(pred_m, gamma_neg) * neg_weights * vol * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    
    def _focal_bce(self, pred, gt, eps = 1e-6, gamma = 2, alpha = 0.5, vol = 4):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * alpha * vol * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * (1 - alpha) * vol * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss



    def _neg_loss(self, pred, gt, eps = 1e-6):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss
    
    def _soft_neg_loss(self, pred, gt, eps = 1e-6, beta = 2):
        ''' Modified focal loss. Exactly the same as QFL.
        '''
        pos_inds = gt.gt(0).float()
        neg_inds = gt.eq(0).float()
        # print(pos_inds.sum(), neg_inds.sum())
        pred = torch.clamp(pred, eps, 1.-eps)
        loss = torch.pow(torch.abs(gt - pred), beta) * ((1 - gt) * torch.log(1 - pred) + gt * torch.log(pred))
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = - loss.sum()
        else:
            loss = - loss.sum() / num_pos

        return loss

    def _contrastive_align(self, outputs, targets, indices, text_type):
        '''
        indices: list of tensor tuples
        like [(tensor([ 5, 42, 51, 61]), tensor([2, 3, 0, 1])), (tensor([20]), tensor([0]))]
        '''
        assert self.use_no_verb_token == True  # The following implementation is based on this setting
        assert text_type in ['obj', 'sub', 'verb']
        logits_key = {'obj':'pred_obj_logits', 'sub':'pred_sub_logits', 'verb':'pred_verb_logits'}
        src_logits = outputs[logits_key[text_type]] / self.temperature

        if text_type in ['verb']:
            if sum([j.shape[0] for (_ , j) in indices]) > 0:
                idx = self._get_src_permutation_idx(indices)
                verb_labels = ()
                offset = 0
                global_idx = []
                max_text_len = max([v['verb_labels'].shape[1] for v in targets])
                max_len_tensor = None
                for t, (_, J) in zip(targets, indices):
                    # guard against squeeze_(dim = 0).unsqueeze_(dim = -1) operation
                    if t['verb_labels'].shape[0] > 0:
                        verb_labels += t['verb_labels'].split(1, dim = 0)
                        global_idx.append(J + offset)
                        offset += J.shape[0]
                    elif t['verb_labels'].shape[0] == 0 and t['verb_labels'].shape[1] == max_text_len:
                        # print('The sample with most verbs has zero triplets.')
                        max_len_tensor = torch.zeros((1, t['verb_labels'].shape[1]), device = t['verb_labels'].device)
                global_idx = torch.cat(global_idx)
                if max_len_tensor is not None:
                    verb_labels += (max_len_tensor,)
                for v in verb_labels:
                    v.squeeze_(dim = 0).unsqueeze_(dim = -1)
                tgt_verb_labels = pad_sequence(verb_labels).squeeze_(dim = -1).transpose(0, 1)
                if max_len_tensor is not None:
                    tgt_verb_labels = tgt_verb_labels[:tgt_verb_labels.shape[0]-1,:]
                # Pad the no_pred_token position of tgt_verb_labels to be 0, if self.no_pred_embedding is used in ParSeDETR
                zero_tensor = torch.zeros((tgt_verb_labels.shape[0], 1), device = tgt_verb_labels.device)
                tgt_verb_labels = torch.cat((tgt_verb_labels, zero_tensor), dim = 1)

                ############ The following setp is of GREAT importance ############
                ############ because we need to rearrange the order of the target labels before using positive_map[idx] = tgt_verb_labels.bool() ############
                tgt_verb_labels = tgt_verb_labels[global_idx]
                positive_map = torch.zeros(src_logits.shape, dtype=torch.bool).to(src_logits.device)
                # Replace the no_pred_token position of positive_map to be 1, if self.no_pred_embedding is used in ParSeDETR
                one_tensor = torch.ones((positive_map.shape[0], positive_map.shape[1]), device = positive_map.device)
                positive_map[:, :, positive_map.shape[2]-1] = one_tensor
                positive_map[idx] = tgt_verb_labels.bool()

            else:
                print('This batch (all samples) has zero triplets.')
                for t in targets:
                    if 'image_id' in t.keys():
                        print(f"image_id: {t['image_id']}")
                positive_map = torch.zeros(src_logits.shape, dtype=torch.bool).to(src_logits.device)

        elif text_type in ['obj', 'sub']:
            if sum([j.shape[0] for (_ , j) in indices]) > 0:
                idx = self._get_src_permutation_idx(indices)
                label_key = text_type + '_labels'  # 'obj_labels' or 'sub_labels'
                # idx: a tuple (batch_idx, src_idx)
                text_len = src_logits.shape[-1]
                target_classes_o = []
                for t, (_, J) in zip(targets, indices):
                    for j in J:
                        t_tensor = torch.zeros((text_len,))
                        t_tensor[t[label_key][j]] = 1
                        target_classes_o.append(t_tensor)
                        assert t_tensor.sum() == 1
                # Guard against no objects in all samples in this batch
                target_classes_o = torch.stack(target_classes_o, dim = 0).to(src_logits.device)
                # target_classes_o = torch.cat([t[label_key][J] for t, (_, J) in zip(targets, indices)])
                positive_map = torch.zeros(src_logits.shape, dtype = torch.bool).to(src_logits.device)
                # Replace the no_obj_token position of positive_map to be 1, if self.no_obj_embedding is used in ParSeDETR
                one_tensor = torch.ones((positive_map.shape[0], positive_map.shape[1]), device = positive_map.device)
                positive_map[:, :, positive_map.shape[2]-1] = one_tensor

                # target_classes: init with a tensor of size src_logits.shape[:2]
                #                 and filled with self.num_obj_classes (no object class)
                positive_map[idx] = target_classes_o.bool()
            else:
                print('This batch (all samples) has zero ' + text_type + 's.')
                for t in targets:
                    if 'image_id' in t.keys():
                        print(f"image_id: {t['image_id']}")
                positive_map = torch.zeros(src_logits.shape, dtype=torch.bool).to(src_logits.device)
        
        positive_logits = -src_logits.masked_fill(~positive_map, 0)
        negative_logits = src_logits

        if self.matching_symmetric:
            # calculation of vis-to-text loss
            vis_with_pos = positive_map.any(dim = 2)
            pos_term = positive_logits.sum(dim = 2)
            neg_term = negative_logits.logsumexp(dim = 2)

            num_positive = positive_map.sum(dim = 2) + 1e-6

            vis_to_text_loss = (pos_term / num_positive + neg_term).masked_fill(~vis_with_pos, 0).sum()
            
            # calculation of text-to-vis loss
            text_with_pos = positive_map.any(dim = 1)
            pos_term = positive_logits.sum(dim = 1)
            neg_term = negative_logits.logsumexp(dim = 1)

            num_positive = positive_map.sum(dim = 1) + 1e-6

            text_to_vis_loss = (pos_term / num_positive + neg_term).masked_fill(~text_with_pos, 0).sum()
            
            return (vis_to_text_loss + text_to_vis_loss) / 2
        else:
            # print('None-symmetric')
            # calculation of vis-to-text loss
            vis_with_pos = positive_map.any(dim = 2)
            pos_term = positive_logits.sum(dim = 2)
            neg_term = negative_logits.logsumexp(dim = 2)

            num_positive = positive_map.sum(dim = 2) + 1e-6

            vis_to_text_loss = (pos_term / num_positive + neg_term).masked_fill(~vis_with_pos, 0).sum()
            
            return vis_to_text_loss


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        
        # indices: list of tensor tuples
        # like [(tensor([ 5, 42, 51, 61]), tensor([2, 3, 0, 1])), (tensor([20]), tensor([0]))]
        
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            # 'verb_labels': self.loss_gt_verb_recon,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'entropy_bound':self.loss_entropy_bound,
            'kl_divergence':self.loss_kl_divergence,
            'verb_hm':self.loss_verb_hm,
            'semantic_similar':self.loss_semantic_similar,
            'verb_threshold':self.loss_verb_threshold,
            'masked_entity_modeling':self.loss_masked_entity_modeling,
            'verb_tagger':self.loss_verb_tagger,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        if self.triplet_filtering:
            bs, num_query = outputs['pred_verb_logits'].shape[:2]
            cost_triplet = {bs_i:{} for bs_i in range(bs)}
            indices, cost_list = self.matcher(outputs_without_aux, targets, return_cost = True)
            C = 1 * cost_list[6] + 1 * cost_list[5] + 1 * cost_list[4] + \
                2.5 * cost_list[2] + 1 * cost_list[0]
            
            query_global = 0
            target_global = 0
            for bs_i, (I, J) in enumerate(indices):
                # target_cost = C[query_global + I, target_global + J] # scale giou to the range from 0 to 1
                for I_i, J_i in zip(I, J):
                    assert J_i not in cost_triplet[bs_i].keys()
                    cost_triplet[bs_i][int(J_i)] = C[query_global + I_i, target_global + J_i]
                query_global += num_query
                target_global += J.shape[0]
            
            # C = self.cost_obj_class * cost_obj_class + self.cost_obj_class * cost_sub_class + \
            #     self.cost_verb_class * cost_verb_class + \
            #     self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            # [cost_giou, (cost_sub_giou, cost_obj_giou), 
            #                  cost_bbox, (cost_sub_bbox, cost_obj_bbox), 
            #                  cost_verb_class, cost_sub_class, cost_obj_class]

            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, cost_list_i = self.matcher(aux_outputs, targets, return_cost = True)
                C_i = 1 * cost_list_i[6] + 1 * cost_list_i[5] + 1 * cost_list_i[4] + \
                      2.5 * cost_list_i[2] + 1 * cost_list_i[0]

                query_global = 0
                target_global = 0
                for bs_i, (I, J) in enumerate(indices):
                    # target_cost = C_i[query_global + I, target_global + J] # scale giou to the range from 0 to 1
                    for I_i, J_i in zip(I, J):
                        cost_triplet[bs_i][int(J_i)] += C_i[query_global + I_i, target_global + J_i]
                    query_global += num_query
                    target_global += J.shape[0]
            
            ## Perform outlier detection
            cost_triplet_list = []
            for c in cost_triplet.values():
                cost_triplet_list += list(c.values())
            if len(cost_triplet_list) > 0:
                cost_triplet_list = torch.stack(cost_triplet_list)
                up_thre = torch.mean(cost_triplet_list) + torch.std(cost_triplet_list) * 0.5
                
                flag_dict = {} # We keep it if it's True
                # gt_sum = 0
                # keep_sum = 0
                for bs_i, c in cost_triplet.items():
                    flag_i = torch.ones((len(c),), device = outputs['pred_verb_logits'].device).bool()
                    # gt_sum += len(c)
                    for j, c_j in c.items():
                        flag_i[j] = (c_j <= up_thre)
                    flag_dict[bs_i] = flag_i
                    # keep_sum += flag_i.float().sum()
                # print('Keeping ratio: {:.2f}'.format(keep_sum/gt_sum))

                for bs_i, t in enumerate(targets):
                    t['obj_labels'] = t['obj_labels'][flag_dict[bs_i]]
                    t['sub_labels'] = t['sub_labels'][flag_dict[bs_i]]
                    t['verb_labels'] = t['verb_labels'][flag_dict[bs_i]]
                    t['sub_boxes'] = t['sub_boxes'][flag_dict[bs_i]]
                    t['obj_boxes'] = t['obj_boxes'][flag_dict[bs_i]]
                # target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                #     target['sub_labels'] = torch.zeros((0,), dtype=torch.int64)
                #     target['verb_labels'] = torch.zeros((0, len(rel_unique)), dtype=torch.float32)
                #     target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                #     target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                
            
            

        # Retrieve the matching between the outputs of the last layer and the targets
        if not self.verb_tagger:
            indices = self.matcher(outputs_without_aux, targets)
        else:
            indices = None

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.verb_tagger:
                    indices = self.matcher(aux_outputs, targets)
                else:
                    indices = None

                for loss in self.losses:
                    if loss == 'verb_hm':
                        continue
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOI(nn.Module):
    def __init__(self, subject_category_id, sigmoid = True, temperature = False, 
                        zero_shot_hoi_eval = False, verb_curing = False):
        super().__init__()
        self.subject_category_id = subject_category_id
        self.sigmoid = sigmoid
        print('Post-processing with sigmoid? ', self.sigmoid)
        self.temperature = temperature
        if self.temperature:
            self.tao = 0.07
        print('Post-processing with temperature? ', self.temperature)
        self.zero_shot_hoi_eval = zero_shot_hoi_eval
        print('Zero-shot eval on hoi dataset? ', self.zero_shot_hoi_eval)
        self.verb_curing = verb_curing
        if self.verb_curing:
            assert self.sigmoid == True
        print('Post-processing with verb_curing? ', self.verb_curing)

        obj_verb_co = np.load('datasets/priors/obj_verb_cooccurrence.npz')['cond_prob_co_matrices']
        obj_verb_co = torch.tensor(obj_verb_co).float()
        obj_verb_co = obj_verb_co + 0.1/obj_verb_co.shape[1]
        obj_verb_co = obj_verb_co / obj_verb_co.sum(dim=1).unsqueeze(dim = 1)
        self.register_buffer('obj_verb_co', obj_verb_co)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']
        # shape [bs, 100, 81]
        # shape [bs, 100, 117]
        # shape [bs, 100, 4]
        # shape [bs, 100, 4]
        if self.zero_shot_hoi_eval:
            assert 'pred_sub_logits' in outputs.keys()
            out_sub_logits = outputs['pred_sub_logits']
            if self.temperature:
                sub_prob = F.softmax(out_sub_logits/self.tao, -1)
            else:
                sub_prob = F.softmax(out_sub_logits, -1)
            sub_scores, sub_labels = sub_prob[..., :-1].max(-1)
            sub_mask = sub_labels == self.subject_category_id
            # sub_mask = sub_labels == 1 # For testing
            # out_obj_logits = out_obj_logits[sub_mask]
            # out_verb_logits = out_verb_logits[sub_mask]
            # out_sub_boxes = out_sub_boxes[sub_mask]
            # out_obj_boxes = out_obj_boxes[sub_mask]
            # print(sub_mask.shape, sub_mask.sum())
            # print(sub_labels[0])

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2  # h, w

        if self.temperature:
            obj_prob = F.softmax(out_obj_logits/self.tao, -1)
        else:
            obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)  
        # obj_prob[..., :-1] ([bs,100,80]) deletes the final class for no objects 
        # [bs, 100] [bs, 100] =  torch.max() returns values and indices

        if self.sigmoid:
            if self.verb_curing:
                verb_scores = out_verb_logits.sigmoid() * outputs['curing_score']
            else:
                verb_scores = out_verb_logits.sigmoid()
        else:
            verb_scores = out_verb_logits
        

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]  # scale 0~1. to 0~w and 0~h

        results = []
        for b_idx, (os, ol, vs, sb, ob, op) in enumerate(zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes, obj_prob[..., :-1])):
            if self.zero_shot_hoi_eval:
                s_flag = sub_mask[b_idx]
                os, ol, vs, sb, ob, op = \
                    os[s_flag], ol[s_flag], vs[s_flag], sb[s_flag], ob[s_flag], op[s_flag]
            
            sl = torch.full_like(ol, self.subject_category_id)
            # sl(Subject label) denotes the person label, set subject_category_id by default
            # sl shape [100]

            l = torch.cat((sl, ol))
            # l shape [200]
            b = torch.cat((sb, ob))
            # b shape [200, 4]
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1) # multiply the object score, a general score for an classified object
            # [100, 117] * [100, 1] = [100, 117]
            # Alternation
            # vs = vs * torch.matmul(op, self.obj_verb_co.to(op.device))

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})
        return results


class PostProcessSGG(nn.Module):
    def __init__(self, sigmoid = True, 
                       zero_shot_sgg_eval = False):
        super().__init__()
        # self.subject_category_id = subject_category_id
        self.sigmoid = sigmoid
        print('Post-processing with sigmoid? ', self.sigmoid)
        self.zero_shot_sgg_eval = zero_shot_sgg_eval
        print('Zero-shot eval on hoi dataset? ', self.zero_shot_sgg_eval)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_sub_logits, out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = \
                                                                        outputs['pred_sub_logits'], \
                                                                        outputs['pred_obj_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']
        # shape [bs, 100, 81]
        # shape [bs, 100, 117]
        # shape [bs, 100, 4]
        # shape [bs, 100, 4]

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2  # h, w

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)
        sub_prob = F.softmax(out_sub_logits, -1)
        sub_scores, sub_labels = sub_prob[..., :-1].max(-1)

        # obj_prob[..., :-1] ([bs,100,80]) deletes the final class for no objects 
        # [bs, 100] [bs, 100] =  torch.max() returns values and indices

        if self.sigmoid:
            verb_scores = out_verb_logits.sigmoid()
        else:
            verb_scores = out_verb_logits
        

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]  # scale 0~1. to 0~w and 0~h

        results = []
        for b_idx, (ss, sl, os, ol, vs, sb, ob, op) in enumerate(zip(sub_scores, sub_labels, obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes, obj_prob[..., :-1])):
            l = torch.cat((sl, ol))
            # l shape [200]
            b = torch.cat((sb, ob))
            # b shape [200, 4]
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1) * ss.unsqueeze(1) # multiply the object score, a general score for an classified object
            # [100, 117] * [100, 1] * [100, 1] = [100, 117]

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)