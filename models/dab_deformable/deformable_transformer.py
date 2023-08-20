# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DModified from eformable DETR
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
from .ops.modules import MSDeformAttn
from torch.nn.utils.rnn import pad_sequence
from models.modeling_roberta import RobertaLayer
from transformers.models.roberta.modeling_roberta import RobertaConfig
from transformers import RobertaModel, RobertaTokenizerFast, BertTokenizerFast, BertModel
from models.fuse_helper import RLIPv2_VLFuse
from models.deformable_transformer import RLIPv2_DeformableTransformerEncoder
from models.ParSetransformer import CrossModelTransformerEncoder, FeatureResizer, TransformerEncoderLayer

# class SepDeformableTransformerHOIv3(nn.Module):
#     def __init__(self, d_model=256, nhead=8,
#                  num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
#                  activation="relu", return_intermediate_dec=False,
#                  num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
#                  two_stage=False, two_stage_num_proposals=300):
#         super().__init__()

#         self.d_model = d_model
#         self.nhead = nhead
#         self.two_stage = two_stage
#         self.two_stage_num_proposals = two_stage_num_proposals

#         ho_encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
#                                                           dropout, activation,
#                                                           num_feature_levels, nhead, enc_n_points)
#         self.ho_encoder = DeformableTransformerEncoder(ho_encoder_layer, num_encoder_layers)

#         ho_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
#                                                           dropout, activation,
#                                                           num_feature_levels, nhead, dec_n_points)
#         self.ho_decoder = TransformerDecoderHOI(ho_decoder_layer, num_decoder_layers, 
#                                                 return_intermediate_dec, ParSe = True)

#         verb_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
#                                                           dropout, activation,
#                                                           num_feature_levels, nhead, dec_n_points)
#         self.verb_decoder = TransformerDecoderHOI(verb_decoder_layer, num_decoder_layers, 
#                                                   return_intermediate_dec, ParSe = False)

#         self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

#         if two_stage:
#             self.enc_output = nn.Linear(d_model, d_model)
#             self.enc_output_norm = nn.LayerNorm(d_model)
#             self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
#             self.pos_trans_norm = nn.LayerNorm(d_model * 2)
#         else:
#             # self.reference_points_subobj = nn.Linear(d_model, 4)
#             self.reference_points_sub = nn.Linear(d_model, 2)
#             self.reference_points_obj = nn.Linear(d_model, 2)

#         # self.ho_fusion_1 = nn.Linear(d_model, d_model)
#         # self.ho_fusion_2 = nn.Linear(d_model, d_model)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, MSDeformAttn):
#                 m._reset_parameters()
#         if not self.two_stage:
#             xavier_uniform_(self.reference_points_sub.weight.data, gain=1.0)
#             constant_(self.reference_points_sub.bias.data, 0.)
#             xavier_uniform_(self.reference_points_obj.weight.data, gain=1.0)
#             constant_(self.reference_points_obj.bias.data, 0.)
#         normal_(self.level_embed)

#     def get_proposal_pos_embed(self, proposals):
#         num_pos_feats = 128
#         temperature = 10000
#         scale = 2 * math.pi

#         dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
#         dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
#         # N, L, 4
#         proposals = proposals.sigmoid() * scale
#         # N, L, 4, 128
#         pos = proposals[:, :, :, None] / dim_t
#         # N, L, 4, 64, 2
#         pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
#         return pos

#     def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
#         N_, S_, C_ = memory.shape
#         base_scale = 4.0
#         proposals = []
#         _cur = 0
#         for lvl, (H_, W_) in enumerate(spatial_shapes):
#             mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
#             valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
#             valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

#             grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
#                                             torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
#             grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

#             scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
#             grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
#             wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
#             proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
#             proposals.append(proposal)
#             _cur += (H_ * W_)
#         output_proposals = torch.cat(proposals, 1)
#         output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
#         output_proposals = torch.log(output_proposals / (1 - output_proposals))
#         output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
#         output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

#         output_memory = memory
#         output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
#         output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
#         output_memory = self.enc_output_norm(self.enc_output(output_memory))
#         return output_memory, output_proposals

#     def get_valid_ratio(self, mask):
#         _, H, W = mask.shape
#         valid_H = torch.sum(~mask[:, :, 0], 1)
#         valid_W = torch.sum(~mask[:, 0, :], 1)
#         valid_ratio_h = valid_H.float() / H
#         valid_ratio_w = valid_W.float() / W
#         valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
#         return valid_ratio

#     def forward(self, srcs, masks, pos_embeds, query_embed=None):
#         assert self.two_stage or query_embed is not None

#         # prepare input for encoder
#         src_flatten = []
#         mask_flatten = []
#         lvl_pos_embed_flatten = []
#         spatial_shapes = []
#         for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
#             bs, c, h, w = src.shape
#             spatial_shape = (h, w)
#             spatial_shapes.append(spatial_shape)
#             src = src.flatten(2).transpose(1, 2)
#             mask = mask.flatten(1)
#             pos_embed = pos_embed.flatten(2).transpose(1, 2)
#             lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
#             lvl_pos_embed_flatten.append(lvl_pos_embed)
#             src_flatten.append(src)
#             mask_flatten.append(mask)
#         src_flatten = torch.cat(src_flatten, 1)
#         mask_flatten = torch.cat(mask_flatten, 1)
#         lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
#         spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
#         level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
#         valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

#         # encoder
#         memory = self.ho_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

#         # prepare input for decoder
#         bs, _, c = memory.shape
#         if self.two_stage:
#             output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

#             # hack implementation for two-stage Deformable DETR
#             enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
#             enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

#             topk = self.two_stage_num_proposals
#             topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
#             topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
#             topk_coords_unact = topk_coords_unact.detach()
#             reference_points = topk_coords_unact.sigmoid()
#             init_reference_out = reference_points
#             pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
#             query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
#         else:
#             query_embed, tgt, verb_tgt = torch.split(query_embed, c, dim=1)
#             query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
#             tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
#             query_num = query_embed.shape[1]
#             reference_points_sub = self.reference_points_sub(query_embed[:,:query_num//2]).sigmoid()
#             reference_points_obj = self.reference_points_obj(query_embed[:,query_num//2:]).sigmoid()
#             # reference_points = self.reference_points_subobj(query_embed).sigmoid().view((bs, -1, 2, 2)).permute((2,0,1,3))  # [bs, 100, 4]
#             # reference_points = self.reference_points_subobj(query_embed).sigmoid() # for ref_points == 2
#             init_reference_out = reference_points = (reference_points_sub, reference_points_obj)

#         # decoder
#         hs_ho, inter_references = self.ho_decoder(tgt, reference_points, memory,
#                                             spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
#         inter_references_out = inter_references
        
#         verb_reference_points = inter_references_out[-1]
#         verb_query_embed = hs_ho[-1][:, :query_num//2] + hs_ho[-1][:, query_num//2:]
#         # verb_query_embed = count_fusion(self.ho_fusion_1(hs_ho[-1][:, :query_num//2]), self.ho_fusion_2(hs_ho[-1][:, query_num//2:]))
#         # verb_query_embed = self.ho_fusion_1(hs_ho[-1][:, :query_num//2]) + self.ho_fusion_2(hs_ho[-1][:, query_num//2:])
#         # verb_query_embed = self.ho_fusion_1(hs_ho[-1][:, :query_num//2]) * self.ho_fusion_2(hs_ho[-1][:, query_num//2:])
#         merge_verb_tgt = verb_tgt[:query_num//2] + verb_tgt[query_num//2:]
#         merge_verb_tgt = merge_verb_tgt.unsqueeze(dim = 0).expand(bs, -1, -1)
#         hs_verb, verb_inter_references = self.verb_decoder(merge_verb_tgt, verb_reference_points, memory,
#                                             spatial_shapes, level_start_index, valid_ratios, verb_query_embed, mask_flatten)

#         if self.two_stage:
#             return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        
#         return hs_ho, hs_verb, init_reference_out, inter_references_out, None, None




class RLIP_ParSeDABDeformableTransformer_v2(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=False,
                 pass_pos_and_query=True, text_encoder_type="roberta-base", freeze_text_encoder=False,
                 args = None):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab
        self.fusion_type = args.fusion_type

        if self.fusion_type != "GLIP_attn":
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
            self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        ho_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.ho_decoder = DABDeformableTransformerDecoderHOI(ho_decoder_layer, num_decoder_layers, return_intermediate_dec, 
                                                             use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed,
                                                             ParSe = True)

        verb_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, do_self_attn = True)
        self.verb_decoder = DABDeformableTransformerDecoderHOI(verb_decoder_layer, num_decoder_layers, return_intermediate_dec,
                                                               use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed,
                                                               ParSe = False)
        
        self.verb_tgt_generator = MultiBranchFusion(256, 256, 256, 16)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))


        # Cross-modal fusion
        if self.fusion_type == "MDETR_attn":
            # TODO: To be implemented
            # None
            obj_fusion_layer = TransformerEncoderLayer(d_model, 8, dim_feedforward, dropout, 
                                                    activation, normalize_before = True)
            obj_encoder_norm = nn.LayerNorm(d_model)
            self.obj_fusion = CrossModelTransformerEncoder(obj_fusion_layer, num_decoder_layers, 
                                                        obj_encoder_norm, return_intermediate = True)

            verb_fusion_layer = TransformerEncoderLayer(d_model, 8, dim_feedforward, dropout, 
                                                    activation, normalize_before = True)
            verb_encoder_norm = nn.LayerNorm(d_model)
            self.verb_fusion = CrossModelTransformerEncoder(verb_fusion_layer, num_decoder_layers, 
                                                            verb_encoder_norm, return_intermediate = True)
        elif self.fusion_type == "GLIP_attn":
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
            RobertaConf = RobertaConfig.from_pretrained(text_encoder_type)
            # Memory-efficient RoBERTa config 
            # RobertaConf.hidden_size = 256
            # RobertaConf.intermediate_size = 1024
            # RobertaConf.num_attention_heads = 8
            roberta_layer = RobertaLayer(config = RobertaConf)
            VLFuse_layer = RLIPv2_VLFuse(args)
            # RLIPv2_DABDeformableTransformerEncoder
            self.encoder = RLIPv2_DeformableTransformerEncoder(encoder_layer, 
                                                               roberta_layer,
                                                               VLFuse_layer,
                                                               num_encoder_layers,
                                                               fusion_interval=args.fusion_interval,
                                                               fusion_last_vis=args.fusion_last_vis,
                                                               lang_aux_loss=args.lang_aux_loss)
        elif self.fusion_type == "no_fusion":
            None


        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if not self.use_dab:
                # self.reference_points = nn.Linear(d_model, 2)
                self.reference_points_sub = nn.Linear(d_model, 2)
                self.reference_points_obj = nn.Linear(d_model, 2)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self._reset_parameters()

        self.pass_pos_and_query = pass_pos_and_query
        if "roberta" in text_encoder_type:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type, cache_dir = '/mnt/data-nas/peizhi/jacob/.cache/huggingface/transformers', local_files_only = True)
            self.text_encoder = RobertaModel.from_pretrained(text_encoder_type, cache_dir = '/mnt/data-nas/peizhi/jacob/.cache/huggingface/transformers', local_files_only = True)
        elif "bert" in text_encoder_type:
            self.tokenizer = BertTokenizerFast.from_pretrained(text_encoder_type)
            self.text_encoder = BertModel.from_pretrained(text_encoder_type)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        # self.verb_tagger = args.verb_tagger
        # self.label_noise_scale = args.label_noise_scale
        # self.box_noise_scale = args.box_noise_scale
        # if self.verb_tagger:
        #     self.coord_proj = nn.Linear(4, d_model)
        #     self.verb_tgt = nn.Embedding(args.num_queries, d_model)
        self.verb_query_tgt_type = args.verb_query_tgt_type
        if "MBF" in self.verb_query_tgt_type:
            self.verb_tgt_generator = MultiBranchFusion(256, 256, 256, 16)
        print(f"We are using self.verb_query_tgt_type: {self.verb_query_tgt_type}.")


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
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
    
    def forward(self, srcs=None, 
                      masks=None, 
                      pos_embeds=None, 
                      query_embed=None,
                      text=None,
                      encode_and_save=True,
                      text_memory=None,
                      img_memory=None,
                      text_attention_mask=None,
                      obj_pred_names_sums=None,
                      spatial_shapes=None,
                      level_start_index=None,
                      valid_ratios=None,
                    #   targets=None,
                    #   key_padding_mask=None,
                    #   attn_mask=None
                      ):
        """
        Input:
            - srcs: List([bs, c, h, w])
            - masks: List([bs, h, w])
        """
        assert self.two_stage or query_embed is not None

        if encode_and_save:
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

            # if self.pass_pos_and_query:
            #     ho_tgt = torch.zeros_like(ho_query_embed)
            # else:
            #     src, ho_tgt, ho_query_embed, pos_embed = src + 0.1 * pos_embed, ho_query_embed, None, None

            # This is used to tell whether this is training stage or eval stage.
            if isinstance(text, list) and isinstance(text[0], tuple):  
                # This is used when training
                # Encode the text
                obj_pred_names_sums = []
                flat_text = []  # shape: [text_num, ]
                for cur_text in text:
                    obj_pred_names_sums.append((len(cur_text[0]), len(cur_text[1])))
                    flat_text += cur_text[0]
                    flat_text += cur_text[1]
                obj_pred_names_sums = torch.tensor(obj_pred_names_sums)

                flat_tokenized = self.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(src_flatten.device)
                # tokenizer: dict_keys(['input_ids', 'attention_mask'])
                #            'input_ids' shape: [text_num, max_token_num]
                #            'attention_mask' shape: [text_num, max_token_num]
                # print(tokenized.input_ids.shape, tokenized.attention_mask.shape)
                encoded_flat_text = self.text_encoder(**flat_tokenized)
                # encoded_text: odict_keys(['last_hidden_state', 'pooler_output'])
                #               'last_hidden_state' shape: [text_num, max_token_num, feature_dim]
                #               'pooler_output' shape: [text_num, feature_dim]
                # print(encoded_text.last_hidden_state.shape, encoded_text.pooler_output.shape)

                # Create equal-length text for all batches if the text lens do not match
                flat_idx = 0
                batch_obj_text = []
                batch_pred_text = []
                for obj_num, pred_num in obj_pred_names_sums:
                    batch_obj_text.append(encoded_flat_text.pooler_output[flat_idx:(flat_idx + obj_num)])
                    batch_pred_text.append(encoded_flat_text.pooler_output[(flat_idx + obj_num):(flat_idx + obj_num + pred_num)])
                    flat_idx += (obj_num + pred_num)
                obj_text_memory = pad_sequence(batch_obj_text) 
                pred_text_memory = pad_sequence(batch_pred_text)
                text_memory = torch.cat([obj_text_memory, pred_text_memory], dim = 0)
                # text_memory: [max_token_num, batch_num, feature_dim] like [35, 4, 768]
                #              which are padded to equal lengths across all batches
                # batch_text list len: [batch_num, ] like [4,]
                text_attention_mask = ~(text_memory.sum(dim = -1) > 0)
                # text_attention_mask: [max_token_num, batch_num] like [35, 4]
                
                # Encoder,注意encoder的修改不改变原来的逻辑
                if self.fusion_type == "GLIP_attn":
                    # Resizing first if we want to perform VLFuse with 256-dim vectors.
                    # text_memory_resized = self.resizer(text_memory)
                    text_memory_resized = text_memory

                    if text_memory_resized.shape[1] != bs:
                        text_memory_resized = text_memory_resized.repeat(1, bs, 1)
                        text_attention_mask = text_attention_mask.repeat(1, bs)
                    img_memory, text_memory_resized = self.encoder(src_flatten,
                                                spatial_shapes,
                                                level_start_index,
                                                valid_ratios,
                                                lvl_pos_embed_flatten,
                                                mask_flatten,
                                                lang_hidden=text_memory_resized.transpose(0,1),
                                                lang_masks=text_attention_mask.transpose(0,1))                    
                              
                    # Transpose only if we do not perform resizing first.
                    # text_memory_resized = text_memory_resized.transpose(0,1)
                    # text_memory_resized = self.resizer(text_memory_resized.transpose(0,1))
                    if len(text_memory_resized.shape) == 3: # which means that "text_memory" does not have the "hs_layer" dimension, i.e. lang_aux_loss = False.
                        text_memory_resized = self.resizer(text_memory_resized.transpose(0,1))
                    elif len(text_memory_resized.shape) == 4: # which means that "text_memory" has the "hs_layer" dimension, i.e. lang_aux_loss = True.
                        text_memory_resized = self.resizer(text_memory_resized.transpose(1,2))
                else:
                    img_memory = self.encoder(src_flatten,
                                                spatial_shapes,
                                                level_start_index,
                                                valid_ratios,
                                                lvl_pos_embed_flatten,
                                                mask_flatten)
                    text_memory_resized = self.resizer(text_memory)
                    if text_memory_resized.shape[1] != bs:
                        text_memory_resized = text_memory_resized.repeat(1, bs, 1)
                        text_attention_mask = text_attention_mask.repeat(1, bs)

                # # Transpose memory because pytorch's attention expects sequence first
                # text_memory = encoded_text.last_hidden_state.transpose(0, 1)
                # # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                # text_attention_mask = tokenized.attention_mask.ne(1).bool()

                # # Resize the encoder hidden states to be of the same d_model as the decoder
                # text_memory_resized = self.resizer(text_memory)
            else:
                # This is used when evaluation
                # The text is already encoded, use as it is.
                # text_attention_mask, text_memory_resized, tokenized = text
                # Encoder
                if self.fusion_type == "GLIP_attn":
                    text_attention_mask, text_memory, obj_pred_names_sums = text
                    img_memory, text_memory = self.encoder(src_flatten,
                                                spatial_shapes,
                                                level_start_index,
                                                valid_ratios,
                                                lvl_pos_embed_flatten,
                                                mask_flatten,
                                                lang_hidden=text_memory.transpose(0,1),
                                                lang_masks=text_attention_mask.transpose(0,1))
                    if len(text_memory.shape) == 3: # which means that "text_memory" does not have the "hs_layer" dimension, i.e. lang_aux_loss = False.
                        text_memory_resized = self.resizer(text_memory.transpose(0,1))
                    elif len(text_memory.shape) == 4: # which means that "text_memory" has the "hs_layer" dimension, i.e. lang_aux_loss = True.
                        text_memory_resized = self.resizer(text_memory.transpose(1,2))
                else:
                    img_memory = self.encoder(src_flatten,
                                                spatial_shapes,
                                                level_start_index,
                                                valid_ratios,
                                                lvl_pos_embed_flatten,
                                                mask_flatten)
                    text_attention_mask, text_memory_resized, obj_pred_names_sums = text

            # assert img_memory.shape[2] == text_memory_resized.shape[2] # == ho_tgt.shape[1]  # == batch_num
            memory_cache = {
                "text_memory_bf_resize": text_memory,
                "text_memory_resized": text_memory_resized,
                "text_memory": text_memory_resized,
                "img_memory": img_memory,
                "masks": mask_flatten,
                "text_attention_mask": text_attention_mask,
                "pos_embed": lvl_pos_embed_flatten,
                "ho_query_embed": query_embed, # ho_query_embed,
                "obj_pred_names_sums":obj_pred_names_sums,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
                # "key_padding_mask": key_padding_mask,
                # "attn_mask": attn_mask,
            }
            return memory_cache

        else:

            # prepare input for decoder
            bs, _, c = img_memory.shape
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
            elif self.use_dab:
                reference_points = query_embed[..., self.d_model*2:].sigmoid()
                query_num = query_embed.shape[0]
                reference_points_sub, reference_points_obj = reference_points[:query_num//2], reference_points[query_num//2:]
                tgt = query_embed[..., :self.d_model]
                tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
                verb_tgt = query_embed[..., self.d_model:self.d_model*2]
                verb_tgt = verb_tgt.unsqueeze(0).expand(bs, -1, -1)
                init_reference_out = reference_points = (reference_points_sub, reference_points_obj)
                # TODO Ensure this is right.
            else:
                # query_embed, tgt = torch.split(query_embed, c, dim=1)
                # query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
                # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
                # reference_points = self.reference_points(query_embed).sigmoid() 
                #     # bs, num_quires, 2
                # init_reference_out = reference_points
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
            # import ipdb; ipdb.set_trace()
            mask_flatten = masks
            hs_ho, inter_references = self.ho_decoder(tgt, reference_points, img_memory,
                                                spatial_shapes, level_start_index, valid_ratios, 
                                                query_pos=query_embed if not self.use_dab else None, 
                                                src_padding_mask=mask_flatten)
            inter_references_out = inter_references

            verb_reference_points = inter_references_out[-1]
            # verb_query_embed = hs_ho[-1][:, :query_num//2] + hs_ho[-1][:, query_num//2:]
            verb_query_embed = None

            if self.verb_query_tgt_type == "vanilla":
                ### verb tgt v1
                merge_verb_tgt = verb_tgt[:,:query_num//2] + verb_tgt[:,query_num//2:]
            ### verb tgt v2
            # merge_verb_tgt = self.verb_tgt_generator(hs_ho[-3][:, :query_num//2], hs_ho[-3][:, query_num//2:]) + \
            #     self.verb_tgt_generator(hs_ho[-2][:, :query_num//2], hs_ho[-2][:, query_num//2:]) + \
            #         self.verb_tgt_generator(hs_ho[-1][:, :query_num//2], hs_ho[-1][:, query_num//2:])
            elif self.verb_query_tgt_type == "MBF":
                ### verb tgt v3
                merge_verb_tgt = self.verb_tgt_generator(hs_ho[-1][:, :query_num//2], hs_ho[-1][:, query_num//2:])
            ### verb tgt v4
            # merge_verb_tgt = hs_ho[-1][:, :query_num//2] + hs_ho[-1][:, query_num//2:] + \
            #             verb_tgt[:,:query_num//2] + verb_tgt[:,query_num//2:]
            elif self.verb_query_tgt_type == "vanilla_MBF":
                ### verb tgt v5
                merge_verb_tgt = self.verb_tgt_generator(hs_ho[-1][:, :query_num//2], hs_ho[-1][:, query_num//2:]) + \
                                    verb_tgt[:,:query_num//2] + verb_tgt[:,query_num//2:]
            ### verb tgt v6
            # merge_verb_tgt = hs_ho[-1][:, :query_num//2] + hs_ho[-1][:, query_num//2:]
            else:
                assert False
            
            hs_verb, verb_inter_references = self.verb_decoder(merge_verb_tgt, verb_reference_points, img_memory,
                                                spatial_shapes, level_start_index, valid_ratios, 
                                                query_pos=verb_query_embed if not self.use_dab else None, 
                                                src_padding_mask=mask_flatten)

            # Cross-modal fusion
            if self.fusion_type == "MDETR_attn":
                max_obj_text_len = torch.max(obj_pred_names_sums[:,0])
                max_pred_text_len = torch.max(obj_pred_names_sums[:,1])
                text_len = max_obj_text_len + max_pred_text_len

                obj_text = text_memory[:max_obj_text_len]
                pred_text = text_memory[max_obj_text_len:(max_obj_text_len + max_pred_text_len)]
                obj_mask = text_attention_mask[:max_obj_text_len]
                pred_mask = text_attention_mask[max_obj_text_len:(max_obj_text_len + max_pred_text_len)]
                assert (max_obj_text_len + max_pred_text_len) == text_memory.shape[0] == text_attention_mask.shape[0]
                
                hs_ho_cross_seq = torch.cat((hs_ho[-1].permute(1,0,2), obj_text), dim = 0)
                hs_ho_mask = torch.zeros(hs_ho[-1].shape[:2], dtype=torch.bool, device=hs_ho.device)
                hs_ho_cross_mask = torch.cat((hs_ho_mask, obj_mask.permute(1, 0)), dim = 1)
                hs_ho_cross_seq = self.obj_fusion(hs_ho_cross_seq, src_key_padding_mask = hs_ho_cross_mask)
                obj_text_dec = torch.stack([h[(h.shape[0]-max_obj_text_len):] for h in hs_ho_cross_seq])
                # if obj_text_dec.shape[1] !=max_obj_text_len:
                #     print('obj_text_dec.shape[1] != max_obj_text_len', obj_text_dec.shape, max_obj_text_len)
                hs_ho_dec = torch.stack([h[:(h.shape[0]-max_obj_text_len)].permute(1, 0, 2) for h in hs_ho_cross_seq])

                hs_verb_cross_seq = torch.cat((hs_verb[-1].permute(1,0,2), pred_text), dim = 0)
                hs_verb_mask = torch.zeros(hs_verb[-1].shape[:2], dtype=torch.bool, device=hs_verb.device)
                hs_verb_cross_mask = torch.cat((hs_verb_mask, pred_mask.permute(1, 0)), dim = 1)
                hs_verb_cross_seq = self.verb_fusion(hs_verb_cross_seq, src_key_padding_mask = hs_verb_cross_mask)
                pred_text_dec = torch.stack([h[(h.shape[0]-max_pred_text_len):] for h in hs_verb_cross_seq])
                # if pred_text_dec.shape[1] !=max_pred_text_len:
                #     print('pred_text_dec.shape[1] !=max_pred_text_len', pred_text_dec.shape, max_pred_text_len)
                hs_verb_dec = torch.stack([h[:(h.shape[0]-max_pred_text_len)].permute(1, 0, 2) for h in hs_verb_cross_seq])

                text_dec = torch.cat((obj_text_dec, pred_text_dec), dim =1)
                # text_dec = torch.cat((obj_text_dec, pred_text_dec), dim = 0)

                return hs_ho_dec, hs_verb_dec, text_dec, init_reference_out, inter_references_out, hs_ho, hs_verb, None, None
            else:
                hs_layer = hs_ho.shape[0]
                # print(text_memory.shape)
                if len(text_memory.shape) == 4 and text_memory.shape[0] == hs_layer:
                    text_dec = text_memory
                else:
                    text_dec = text_memory.unsqueeze(0).repeat(hs_layer, 1, 1, 1)
                    
                return hs_ho, hs_verb, text_dec, init_reference_out, inter_references_out, hs_ho, hs_verb, None, None


class ParSeDABDeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        ho_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.ho_decoder = DABDeformableTransformerDecoderHOI(ho_decoder_layer, num_decoder_layers, return_intermediate_dec, 
                                                             use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed,
                                                             ParSe = True)

        verb_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, do_self_attn = True)
        self.verb_decoder = DABDeformableTransformerDecoderHOI(verb_decoder_layer, num_decoder_layers, return_intermediate_dec,
                                                               use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed,
                                                               ParSe = False)
        # self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, 
        #                                                     use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed)
        # self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, 
        #                                                     use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed)
        # self.decoder = DABDeformableTransformerDecoderHOI(decoder_layer, num_decoder_layers, return_intermediate_dec, 
        #                                                     use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed)
        self.verb_query_tgt_type = args.verb_query_tgt_type
        if "MBF" in self.verb_query_tgt_type:
            self.verb_tgt_generator = MultiBranchFusion(256, 256, 256, 16)
        print(f"We are using self.verb_query_tgt_type: {self.verb_query_tgt_type}.")

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if not self.use_dab:
                # self.reference_points = nn.Linear(d_model, 2)
                self.reference_points_sub = nn.Linear(d_model, 2)
                self.reference_points_obj = nn.Linear(d_model, 2)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
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
        """
        Input:
            - srcs: List([bs, c, h, w])
            - masks: List([bs, h, w])
        """
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

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)     # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # import ipdb; ipdb.set_trace()

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
        elif self.use_dab:
            reference_points = query_embed[..., self.d_model*2:].sigmoid()
            query_num = query_embed.shape[0]
            reference_points_sub, reference_points_obj = reference_points[:query_num//2], reference_points[query_num//2:]
            tgt = query_embed[..., :self.d_model]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            verb_tgt = query_embed[..., self.d_model:self.d_model*2]
            verb_tgt = verb_tgt.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points = (reference_points_sub, reference_points_obj)
            # TODO Ensure this is right.
        else:
            # query_embed, tgt = torch.split(query_embed, c, dim=1)
            # query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            # reference_points = self.reference_points(query_embed).sigmoid() 
            #     # bs, num_quires, 2
            # init_reference_out = reference_points
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
        # import ipdb; ipdb.set_trace()
        hs_ho, inter_references = self.ho_decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, 
                                            query_pos=query_embed if not self.use_dab else None, 
                                            src_padding_mask=mask_flatten)
        inter_references_out = inter_references

        verb_reference_points = inter_references_out[-1]
        # verb_query_embed = hs_ho[-1][:, :query_num//2] + hs_ho[-1][:, query_num//2:]
        verb_query_embed = None

        if self.verb_query_tgt_type == "vanilla":
            ### verb tgt v1
            merge_verb_tgt = verb_tgt[:,:query_num//2] + verb_tgt[:,query_num//2:]
        ### verb tgt v2
        # merge_verb_tgt = self.verb_tgt_generator(hs_ho[-3][:, :query_num//2], hs_ho[-3][:, query_num//2:]) + \
        #     self.verb_tgt_generator(hs_ho[-2][:, :query_num//2], hs_ho[-2][:, query_num//2:]) + \
        #         self.verb_tgt_generator(hs_ho[-1][:, :query_num//2], hs_ho[-1][:, query_num//2:])
        elif self.verb_query_tgt_type == "MBF":
            ### verb tgt v3
            merge_verb_tgt = self.verb_tgt_generator(hs_ho[-1][:, :query_num//2], hs_ho[-1][:, query_num//2:])
        ### verb tgt v4
        # merge_verb_tgt = hs_ho[-1][:, :query_num//2] + hs_ho[-1][:, query_num//2:] + \
        #             verb_tgt[:,:query_num//2] + verb_tgt[:,query_num//2:]
        elif self.verb_query_tgt_type == "vanilla_MBF":
            ### verb tgt v5
            merge_verb_tgt = self.verb_tgt_generator(hs_ho[-1][:, :query_num//2], hs_ho[-1][:, query_num//2:]) + \
                                verb_tgt[:,:query_num//2] + verb_tgt[:,query_num//2:]
        ### verb tgt v6
        # merge_verb_tgt = hs_ho[-1][:, :query_num//2] + hs_ho[-1][:, query_num//2:]
        else:
            assert False
        

        hs_verb, verb_inter_references = self.verb_decoder(merge_verb_tgt, verb_reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, 
                                            query_pos=verb_query_embed if not self.use_dab else None, 
                                            src_padding_mask=mask_flatten)

        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs_ho, hs_verb, init_reference_out, inter_references_out, None, None


        # decoder DDETR
        # hs_ho, inter_references = self.ho_decoder(tgt, reference_points, memory,
        #                                     spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        # inter_references_out = inter_references
        
        # verb_reference_points = inter_references_out[-1]
        # verb_query_embed = hs_ho[-1][:, :query_num//2] + hs_ho[-1][:, query_num//2:]
        # # verb_query_embed = count_fusion(self.ho_fusion_1(hs_ho[-1][:, :query_num//2]), self.ho_fusion_2(hs_ho[-1][:, query_num//2:]))
        # # verb_query_embed = self.ho_fusion_1(hs_ho[-1][:, :query_num//2]) + self.ho_fusion_2(hs_ho[-1][:, query_num//2:])
        # # verb_query_embed = self.ho_fusion_1(hs_ho[-1][:, :query_num//2]) * self.ho_fusion_2(hs_ho[-1][:, query_num//2:])
        # merge_verb_tgt = verb_tgt[:query_num//2] + verb_tgt[query_num//2:]
        # merge_verb_tgt = merge_verb_tgt.unsqueeze(dim = 0).expand(bs, -1, -1)
        # hs_verb, verb_inter_references = self.verb_decoder(merge_verb_tgt, verb_reference_points, memory,
        #                                     spatial_shapes, level_start_index, valid_ratios, verb_query_embed, mask_flatten)

        # if self.two_stage:
        #     return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        
        # return hs_ho, hs_verb, init_reference_out, inter_references_out, None, None

class MultiBranchFusion(nn.Module):
    """
    Multi-branch fusion module.
    Copy-paste from https://github.com/fredzzhang/spatially-conditioned-graphs/blob/main/interaction_head.py.
    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int, spatial_size: int,
        representation_size: int, cardinality: int
    ) -> None:
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(representation_size / cardinality)
        assert sub_repr_size * cardinality == representation_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, representation_size)
            for _ in range(cardinality)
        ])
    def forward(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, 
                                                            use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(d_model, 2)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
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
        """
        Input:
            - srcs: List([bs, c, h, w])
            - masks: List([bs, h, w])
        """
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

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)     # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # import ipdb; ipdb.set_trace()

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
        elif self.use_dab:
            reference_points = query_embed[..., self.d_model:].sigmoid() 
            tgt = query_embed[..., :self.d_model]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid() 
                # bs, num_quires, 2
            init_reference_out = reference_points

        # decoder
        # import ipdb; ipdb.set_trace()
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, 
                                            query_pos=query_embed if not self.use_dab else None, 
                                            src_padding_mask=mask_flatten)

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
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - padding_mask: [bs, sum(hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_lebel, 2]
        """
        output = src
        # bs, sum(hi*wi), 256
        # import ipdb; ipdb.set_trace()
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 do_self_attn = True):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.do_self_attn = do_self_attn
        if self.do_self_attn:
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

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        if self.do_self_attn:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
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


class DABDeformableTransformerDecoderHOI(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_dab=False, d_model=256,
                       high_dim_query_update=False, no_sine_embed=False, ParSe = False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        # self.bbox_embed = None
        self.sub_bbox_embed = None
        self.obj_bbox_embed = None
        self.class_embed = None
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            if self.no_sine_embed:
                self.ref_point_head = MLP(4, d_model, d_model, 3)
            else:
                self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 2)
        
        self.ParSe = ParSe

    
    def forward(self, tgt, reference_points, src, src_spatial_shapes,       
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        if self.use_dab:
            assert query_pos is None
        bs = src.shape[0]
        # reference_points = reference_points[None].repeat(bs, 1, 1) # bs, nq, 4 (x y w h)
        sub_ref_points, obj_ref_points = reference_points
        if self.ParSe:
            sub_ref_points = sub_ref_points[None].repeat(bs, 1, 1) # bs, nq, 4 (x y w h)
            obj_ref_points = obj_ref_points[None].repeat(bs, 1, 1) # bs, nq, 4 (x y w h)
        ref_pair_num = obj_ref_points.shape[1]

        intermediate = []
        # intermediate_reference_points = []
        intermediate_sub_ref_points = []
        intermediate_obj_ref_points = []
        for lid, layer in enumerate(self.layers):
            # import ipdb; ipdb.set_trace()
            if not self.ParSe:  # It means that we are performing sequential decoding (decoding for verbs).
                if sub_ref_points.shape[-1] == 4 and obj_ref_points.shape[-1] == 4:
                    ### Default verb box generation
                    reference_points_input = 0.5*(sub_ref_points + obj_ref_points)[:, :, None] \
                                            * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                    ### Adaptive Shifted MBR (ASMBR) verb box generation
                    # reference_points_input_norm = 0.5*(sub_ref_points + obj_ref_points)[:, :, None]
                    # reference_points_input_norm[...,2:] = reference_points_input_norm[...,2:] + torch.abs(sub_ref_points[...,:2] - obj_ref_points[...,:2])[:,:,None]
                    # reference_points_input = reference_points_input_norm * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                else:
                    assert False
                    # TODO to be implemented 
                    # assert sub_ref_points.shape[-1] == 2 and obj_ref_points.shape[-1] == 2
                    # reference_points_input = 0.5*(sub_ref_points + obj_ref_points)[:, :, None] * src_valid_ratios[:, None]
            else:
                # Disentangled decoding
                if sub_ref_points.shape[-1] == 4 and obj_ref_points.shape[-1] == 4:
                    reference_points_sub = sub_ref_points[:,:,None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:,None]
                    reference_points_obj = obj_ref_points[:,:,None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:,None]
                    reference_points_input = torch.cat((reference_points_sub, reference_points_obj), dim = 1)
                else:
                    assert False
                    # TODO to be implemented 
                    # assert sub_ref_points.shape[-1] == 2 and obj_ref_points.shape[-1] == 2
                    # # reference_points_sub shape: bs,100,4,2  4 is the feature level
                    # reference_points_sub = sub_ref_points[:, :, None] * src_valid_ratios[:, None] 
                    # reference_points_obj = obj_ref_points[:, :, None] * src_valid_ratios[:, None]
                    # reference_points_input = torch.cat((reference_points_sub, reference_points_obj), dim = 1)

            

            if self.use_dab:
                # import ipdb; ipdb.set_trace()
                if self.no_sine_embed:
                    raw_query_pos = self.ref_point_head(reference_points_input)
                else:
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*2 
                    raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output)                 

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # # hack implementation for iterative bounding box refinement
            # if self.bbox_embed is not None:
            #     tmp = self.bbox_embed[lid](output)
            #     if reference_points.shape[-1] == 4:
            #         new_reference_points = tmp + inverse_sigmoid(reference_points)
            #         new_reference_points = new_reference_points.sigmoid()
            #     else:
            #         assert reference_points.shape[-1] == 2
            #         new_reference_points = tmp
            #         new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
            #         new_reference_points = new_reference_points.sigmoid()
            #     reference_points = new_reference_points.detach()

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

# ### DDETR
# class TransformerDecoderHOI(nn.Module):
#     def __init__(self, decoder_layer, num_layers, return_intermediate=False, ParSe = False):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.return_intermediate = return_intermediate
#         self.ParSe = ParSe
#         # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
#         self.sub_bbox_embed = None
#         self.obj_bbox_embed = None
#         self.class_embed = None

#     def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
#                 query_pos=None, src_padding_mask=None, attn_mask=None, key_padding_mask=None):
#         output = tgt
#         sub_ref_points, obj_ref_points = reference_points
#         assert sub_ref_points.shape[1] == obj_ref_points.shape[1]
#         ref_pair_num = obj_ref_points.shape[1]
#         # sub_ref_points = obj_ref_points = reference_points

#         intermediate = []
#         # intermediate_reference_points = []
#         intermediate_sub_ref_points = []
#         intermediate_obj_ref_points = []
#         for lid, layer in enumerate(self.layers):
#             if not self.ParSe:
#                 if sub_ref_points.shape[-1] == 4 and obj_ref_points.shape[-1] == 4:
#                     reference_points_input = 0.5*(sub_ref_points + obj_ref_points)[:, :, None] \
#                                             * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
#                 else:
#                     assert sub_ref_points.shape[-1] == 2 and obj_ref_points.shape[-1] == 2
#                     reference_points_input = 0.5*(sub_ref_points + obj_ref_points)[:, :, None] * src_valid_ratios[:, None]
#             else:
#                 # Disentangled decoding
#                 if sub_ref_points.shape[-1] == 4 and obj_ref_points.shape[-1] == 4:
#                     reference_points_sub = sub_ref_points[:,:,None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:,None]
#                     reference_points_obj = obj_ref_points[:,:,None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:,None]
#                     reference_points_input = torch.cat((reference_points_sub, reference_points_obj), dim = 1)
#                 else:
#                     assert sub_ref_points.shape[-1] == 2 and obj_ref_points.shape[-1] == 2
#                     # reference_points_sub shape: bs,100,4,2  4 is the feature level
#                     reference_points_sub = sub_ref_points[:, :, None] * src_valid_ratios[:, None] 
#                     reference_points_obj = obj_ref_points[:, :, None] * src_valid_ratios[:, None]
#                     reference_points_input = torch.cat((reference_points_sub, reference_points_obj), dim = 1)
#                     # print(reference_points_input.shape)
            
#             output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, 
#                            key_padding_mask=key_padding_mask, attn_mask=attn_mask)

#             # hack implementation for iterative bounding box refinement
#             if self.sub_bbox_embed is not None:
#                 if not self.ParSe:
#                     sub_tmp = self.sub_bbox_embed[lid](output)
#                 else:
#                     sub_tmp = self.sub_bbox_embed[lid](output[:,:ref_pair_num])

#                 if sub_ref_points.shape[-1] == 4:
#                     new_sub_ref_points = sub_tmp + inverse_sigmoid(sub_ref_points)
#                     new_sub_ref_points = new_sub_ref_points.sigmoid()
#                 else:
#                     assert sub_ref_points.shape[-1] == 2
#                     new_sub_ref_points = sub_tmp
#                     new_sub_ref_points[..., :2] = sub_tmp[..., :2] + inverse_sigmoid(sub_ref_points)
#                     new_sub_ref_points = new_sub_ref_points.sigmoid()
#                 sub_ref_points = new_sub_ref_points.detach()
            
#             if self.obj_bbox_embed is not None:
#                 if not self.ParSe:
#                     obj_tmp = self.obj_bbox_embed[lid](output)
#                 else:
#                     obj_tmp = self.obj_bbox_embed[lid](output[:,ref_pair_num:])

#                 if obj_ref_points.shape[-1] == 4:
#                     new_obj_ref_points = obj_tmp + inverse_sigmoid(obj_ref_points)
#                     new_obj_ref_points = new_obj_ref_points.sigmoid()
#                 else:
#                     assert obj_ref_points.shape[-1] == 2
#                     new_obj_ref_points = obj_tmp
#                     new_obj_ref_points[..., :2] = obj_tmp[..., :2] + inverse_sigmoid(obj_ref_points)
#                     new_obj_ref_points = new_obj_ref_points.sigmoid()
#                 obj_ref_points = new_obj_ref_points.detach()

#             if self.return_intermediate:
#                 intermediate.append(output)
#                 intermediate_sub_ref_points.append(sub_ref_points)
#                 intermediate_obj_ref_points.append(obj_ref_points)

#         intermediate_ref_points = torch.stack((torch.stack(intermediate_sub_ref_points), torch.stack(intermediate_obj_ref_points)), dim = 0).transpose(0,1)
#         if self.return_intermediate:
#             return torch.stack(intermediate), intermediate_ref_points
        
#         return output, reference_points


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_dab=False, d_model=256, high_dim_query_update=False, no_sine_embed=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            if self.no_sine_embed:
                self.ref_point_head = MLP(4, d_model, d_model, 3)
            else:
                self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 2)


    def forward(self, tgt, reference_points, src, src_spatial_shapes,       
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        if self.use_dab:
            assert query_pos is None
        bs = src.shape[0]
        reference_points = reference_points[None].repeat(bs, 1, 1) # bs, nq, 4(xywh)


        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # import ipdb; ipdb.set_trace()
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None] # bs, nq, 4, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if self.use_dab:
                # import ipdb; ipdb.set_trace()
                if self.no_sine_embed:
                    raw_query_pos = self.ref_point_head(reference_points_input)
                else:
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*2 
                    raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output)                 


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
        two_stage_num_proposals=args.num_queries,
        use_dab=True)


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

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos