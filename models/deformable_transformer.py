# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn

def count_fusion(x, y):
    return F.relu(x + y) - (x - y)*(x - y)

class SepDeformableTransformerHOIv3(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        ho_encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.ho_encoder = DeformableTransformerEncoder(ho_encoder_layer, num_encoder_layers)

        ho_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.ho_decoder = TransformerDecoderHOI(ho_decoder_layer, num_decoder_layers, 
                                                return_intermediate_dec, ParSe = True)

        verb_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.verb_decoder = TransformerDecoderHOI(verb_decoder_layer, num_decoder_layers, 
                                                  return_intermediate_dec, ParSe = False)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            # self.reference_points_subobj = nn.Linear(d_model, 4)
            self.reference_points_sub = nn.Linear(d_model, 2)
            self.reference_points_obj = nn.Linear(d_model, 2)

        # self.ho_fusion_1 = nn.Linear(d_model, d_model)
        # self.ho_fusion_2 = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points_sub.weight.data, gain=1.0)
            constant_(self.reference_points_sub.bias.data, 0.)
            xavier_uniform_(self.reference_points_obj.weight.data, gain=1.0)
            constant_(self.reference_points_obj.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.ho_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt, verb_tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            query_num = query_embed.shape[1]
            reference_points_sub = self.reference_points_sub(query_embed[:,:query_num//2]).sigmoid()
            reference_points_obj = self.reference_points_obj(query_embed[:,query_num//2:]).sigmoid()
            # reference_points = self.reference_points_subobj(query_embed).sigmoid().view((bs, -1, 2, 2)).permute((2,0,1,3))  # [bs, 100, 4]
            # reference_points = self.reference_points_subobj(query_embed).sigmoid() # for ref_points == 2
            init_reference_out = reference_points = (reference_points_sub, reference_points_obj)

        # decoder
        hs_ho, inter_references = self.ho_decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        inter_references_out = inter_references
        
        verb_reference_points = inter_references_out[-1]
        verb_query_embed = hs_ho[-1][:, :query_num//2] + hs_ho[-1][:, query_num//2:]
        # verb_query_embed = count_fusion(self.ho_fusion_1(hs_ho[-1][:, :query_num//2]), self.ho_fusion_2(hs_ho[-1][:, query_num//2:]))
        # verb_query_embed = self.ho_fusion_1(hs_ho[-1][:, :query_num//2]) + self.ho_fusion_2(hs_ho[-1][:, query_num//2:])
        # verb_query_embed = self.ho_fusion_1(hs_ho[-1][:, :query_num//2]) * self.ho_fusion_2(hs_ho[-1][:, query_num//2:])
        merge_verb_tgt = verb_tgt[:query_num//2] + verb_tgt[query_num//2:]
        merge_verb_tgt = merge_verb_tgt.unsqueeze(dim = 0).expand(bs, -1, -1)
        hs_verb, verb_inter_references = self.verb_decoder(merge_verb_tgt, verb_reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, verb_query_embed, mask_flatten)

        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        
        return hs_ho, hs_verb, init_reference_out, inter_references_out, None, None


class DeformableTransformerHOI(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoderHOI(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points_subobj = nn.Linear(d_model, 4)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points_subobj.weight.data, gain=1.0)
            constant_(self.reference_points_subobj.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points_subobj(query_embed).sigmoid().view((bs, -1, 2, 2)).permute((2,0,1,3))  # [bs, 100, 4]
            # reference_points = self.reference_points_subobj(query_embed).sigmoid() # for ref_points == 2
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class TransformerDecoderHOI(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, ParSe = False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.ParSe = ParSe
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.sub_bbox_embed = None
        self.obj_bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, attn_mask=None, key_padding_mask=None):
        output = tgt
        sub_ref_points, obj_ref_points = reference_points
        assert sub_ref_points.shape[1] == obj_ref_points.shape[1]
        ref_pair_num = obj_ref_points.shape[1]
        # sub_ref_points = obj_ref_points = reference_points

        intermediate = []
        # intermediate_reference_points = []
        intermediate_sub_ref_points = []
        intermediate_obj_ref_points = []
        for lid, layer in enumerate(self.layers):
            if not self.ParSe:
                if sub_ref_points.shape[-1] == 4 and obj_ref_points.shape[-1] == 4:
                    reference_points_input = 0.5*(sub_ref_points + obj_ref_points)[:, :, None] \
                                            * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                else:
                    assert sub_ref_points.shape[-1] == 2 and obj_ref_points.shape[-1] == 2
                    reference_points_input = 0.5*(sub_ref_points + obj_ref_points)[:, :, None] * src_valid_ratios[:, None]
            else:
                # Disentangled decoding
                if sub_ref_points.shape[-1] == 4 and obj_ref_points.shape[-1] == 4:
                    reference_points_sub = sub_ref_points[:,:,None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:,None]
                    reference_points_obj = obj_ref_points[:,:,None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:,None]
                    reference_points_input = torch.cat((reference_points_sub, reference_points_obj), dim = 1)
                else:
                    assert sub_ref_points.shape[-1] == 2 and obj_ref_points.shape[-1] == 2
                    # reference_points_sub shape: bs,100,4,2  4 is the feature level
                    reference_points_sub = sub_ref_points[:, :, None] * src_valid_ratios[:, None] 
                    reference_points_obj = obj_ref_points[:, :, None] * src_valid_ratios[:, None]
                    reference_points_input = torch.cat((reference_points_sub, reference_points_obj), dim = 1)
                    # print(reference_points_input.shape)
            
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, 
                           key_padding_mask=key_padding_mask, attn_mask=attn_mask)

            # hack implementation for iterative bounding box refinement
            if self.sub_bbox_embed is not None:
                if not self.ParSe:
                    sub_tmp = self.sub_bbox_embed[lid](output)
                else:
                    sub_tmp = self.sub_bbox_embed[lid](output[:,:ref_pair_num])

                if sub_ref_points.shape[-1] == 4:
                    new_sub_ref_points = sub_tmp + inverse_sigmoid(sub_ref_points)
                    new_sub_ref_points = new_sub_ref_points.sigmoid()
                else:
                    assert sub_ref_points.shape[-1] == 2
                    new_sub_ref_points = sub_tmp
                    new_sub_ref_points[..., :2] = sub_tmp[..., :2] + inverse_sigmoid(sub_ref_points)
                    new_sub_ref_points = new_sub_ref_points.sigmoid()
                sub_ref_points = new_sub_ref_points.detach()
            
            if self.obj_bbox_embed is not None:
                if not self.ParSe:
                    obj_tmp = self.obj_bbox_embed[lid](output)
                else:
                    obj_tmp = self.obj_bbox_embed[lid](output[:,ref_pair_num:])

                if obj_ref_points.shape[-1] == 4:
                    new_obj_ref_points = obj_tmp + inverse_sigmoid(obj_ref_points)
                    new_obj_ref_points = new_obj_ref_points.sigmoid()
                else:
                    assert obj_ref_points.shape[-1] == 2
                    new_obj_ref_points = obj_tmp
                    new_obj_ref_points[..., :2] = obj_tmp[..., :2] + inverse_sigmoid(obj_ref_points)
                    new_obj_ref_points = new_obj_ref_points.sigmoid()
                obj_ref_points = new_obj_ref_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_sub_ref_points.append(sub_ref_points)
                intermediate_obj_ref_points.append(obj_ref_points)

        intermediate_ref_points = torch.stack((torch.stack(intermediate_sub_ref_points), torch.stack(intermediate_obj_ref_points)), dim = 0).transpose(0,1)
        if self.return_intermediate:
            return torch.stack(intermediate), intermediate_ref_points
        
        return output, reference_points


class TransformerDecoderVerbHOI(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.sub_bbox_embed = None
        self.obj_bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        sub_ref_points, obj_ref_points = reference_points
        # sub_ref_points = obj_ref_points = reference_points

        intermediate = []
        # intermediate_reference_points = []
        intermediate_sub_ref_points = []
        intermediate_obj_ref_points = []
        for lid, layer in enumerate(self.layers):
            if sub_ref_points.shape[-1] == 4 and obj_ref_points.shape[-1] == 4:
                reference_points_input = 0.5*(sub_ref_points + obj_ref_points)[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert sub_ref_points.shape[-1] == 2 and obj_ref_points.shape[-1] == 2
                reference_points_input = 0.5*(sub_ref_points + obj_ref_points)[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.sub_bbox_embed is not None:
                sub_tmp = self.sub_bbox_embed[lid](output)
                if sub_ref_points.shape[-1] == 4:
                    new_sub_ref_points = sub_tmp + inverse_sigmoid(sub_ref_points)
                    new_sub_ref_points = new_sub_ref_points.sigmoid()
                else:
                    assert sub_ref_points.shape[-1] == 2
                    new_sub_ref_points = sub_tmp
                    new_sub_ref_points[..., :2] = sub_tmp[..., :2] + inverse_sigmoid(sub_ref_points)
                    new_sub_ref_points = new_sub_ref_points.sigmoid()
                sub_ref_points = new_sub_ref_points.detach()
            
            if self.obj_bbox_embed is not None:
                obj_tmp = self.obj_bbox_embed[lid](output)
                if obj_ref_points.shape[-1] == 4:
                    new_obj_ref_points = obj_tmp + inverse_sigmoid(obj_ref_points)
                    new_obj_ref_points = new_obj_ref_points.sigmoid()
                else:
                    assert obj_ref_points.shape[-1] == 2
                    new_obj_ref_points = obj_tmp
                    new_obj_ref_points[..., :2] = obj_tmp[..., :2] + inverse_sigmoid(obj_ref_points)
                    new_obj_ref_points = new_obj_ref_points.sigmoid()
                obj_ref_points = new_obj_ref_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_sub_ref_points.append(sub_ref_points)
                intermediate_obj_ref_points.append(obj_ref_points)

        intermediate_ref_points = torch.stack((torch.stack(intermediate_sub_ref_points), torch.stack(intermediate_obj_ref_points)), dim = 0).transpose(0,1)
        if self.return_intermediate:
            return torch.stack(intermediate), intermediate_ref_points

        return output, reference_points


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None




class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class RLIPv2_DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, roberta_layer, VLFuse_layer, num_layers, fusion_interval = 2, fusion_last_vis = False, lang_aux_loss = False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.fusion_interval = fusion_interval
        self.roberta_layers = _get_clones(roberta_layer, num_layers//self.fusion_interval)
        self.VLFuse_layers = _get_clones(VLFuse_layer, num_layers//self.fusion_interval)
        self.fusion_last_vis = fusion_last_vis
        self.lang_aux_loss = lang_aux_loss

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, lang_hidden=None, lang_masks=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        # print(src.shape, spatial_shapes.shape, level_start_index.shape, valid_ratios.shape, pos.shape, padding_mask.shape)
        # torch.Size([16, 19816, 256]) torch.Size([4, 2]) torch.Size([4]) torch.Size([16, 4, 2]) torch.Size([16, 19816, 256]) torch.Size([16, 19816])
        
        # TODO
        # Invert the attention mask as the mask in the DETR's transformer is different from the the one in the RoBERTa.
        inv_padding_mask = ~padding_mask
        inv_lang_masks = ~lang_masks

        if self.fusion_last_vis:
            last_inv_padding_mask = inv_padding_mask[:,level_start_index[-1]:]
            last_pos = pos[:,level_start_index[-1]:]
            fused_visual_dict_features = {'src':output, 'padding_mask':last_inv_padding_mask, 'pos':last_pos}
        else:
            fused_visual_dict_features = {'src':output, 'padding_mask':inv_padding_mask, 'pos':pos}
        fused_language_dict_features = {'hidden':lang_hidden, 'masks':inv_lang_masks}

        multi_lay_lang = [] # A list to store language fusion features from multiple RoBERTa layers
        for idx, layer in enumerate(self.layers):
            if (idx)%self.fusion_interval == 0: # and idx<=2:
                fusion_idx = (idx)//self.fusion_interval
                # print('Round {} Fusion'.format(fusion_idx))

                # Select vision features from the last layer
                if self.fusion_last_vis:
                    full_src = fused_visual_dict_features['src']
                    part_src = full_src[:,level_start_index[-1]:].clone()
                    fused_visual_dict_features['src'] = part_src
                
                # VLFuse (We should use inverted 'padding_masks' and 'masks' for the inference of VLFuse_layers.)
                fuse_input_dict = {"visual": fused_visual_dict_features, 
                                   "lang": fused_language_dict_features}
                fuse_output_dict = self.VLFuse_layers[fusion_idx](fuse_input_dict)
                fused_visual_dict_features = fuse_output_dict["visual"]
                fused_language_dict_features = fuse_output_dict["lang"]

                # Replace part_src with full_src with fused vision featues from the last layer
                if self.fusion_last_vis:
                    assert len(full_src[:,level_start_index[-1]:]) == len(fused_visual_dict_features['src'])
                    full_src[:,level_start_index[-1]:] = fused_visual_dict_features['src']
                    fused_visual_dict_features['src'] = full_src

                # Language path (We should use the inverted 'masks' for th inference of roberta_layers.)
                fused_language_dict_features['hidden'] = self.roberta_layers[fusion_idx](
                                            hidden_states = fused_language_dict_features['hidden'],
                                            attention_mask = fused_language_dict_features['masks'])
                multi_lay_lang.append(fused_language_dict_features['hidden'])
        
            # Vision path (We should use original 'padding_masks' for the inference of Vision path.)
            fused_visual_dict_features['src'] = layer(fused_visual_dict_features['src'], 
                                                    pos, 
                                                    reference_points, 
                                                    spatial_shapes, 
                                                    level_start_index, 
                                                    padding_mask)
        if self.lang_aux_loss:
            # Select 3-layer language features
            if self.fusion_interval == 2:
                multi_lay_lang = torch.stack(multi_lay_lang, dim = 0)
            elif self.fusion_interval == 1:
                multi_lay_lang = torch.stack(multi_lay_lang[::2], dim = 0)
        else:
            multi_lay_lang = multi_lay_lang[-1]

        # return visual output and language hidden
        return fused_visual_dict_features['src'], multi_lay_lang # fused_language_dict_features['hidden']


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None,
                key_padding_mask=None, attn_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        # print(f'tgt2 has nan before attn? {torch.isnan(tgt).sum()}', f'tgt2 shape: {tgt.shape}')
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0].transpose(0, 1)
        # print(f'tgt2 has nan after attn? {torch.isnan(tgt2).sum()}')
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points) 
                # if self.bbox_embed is not None: Reference points are changed, 
                # Otherwise, the reference points are actually identical.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    if args.hoi:
        return DeformableTransformerHOI(
            d_model=args.hidden_dim,
            nhead=args.nheads,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=args.num_feature_levels,
            dec_n_points=args.dec_n_points,
            enc_n_points=args.enc_n_points,
            two_stage=args.two_stage,
            two_stage_num_proposals=args.num_queries)
    else:
        return DeformableTransformer(
            d_model=args.hidden_dim,
            nhead=args.nheads,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=args.num_feature_levels,
            dec_n_points=args.dec_n_points,
            enc_n_points=args.enc_n_points,
            two_stage=args.two_stage,
            two_stage_num_proposals=args.num_queries)