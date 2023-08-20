# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np
from transformers import RobertaModel, RobertaTokenizerFast, BertTokenizerFast, BertModel
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence
from .deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder, \
                                    DeformableTransformerDecoderLayer, TransformerDecoderHOI, RLIPv2_DeformableTransformerEncoder
from models.ops.modules import MSDeformAttn
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from .modeling_roberta import RobertaLayer
from transformers.models.roberta.modeling_roberta import RobertaConfig
from .fuse_helper import RLIPv2_VLFuse
from .verb_tagger_helper import prepare_query

class ParSeDeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 pass_pos_and_query=True, text_encoder_type="roberta-base", freeze_text_encoder=False):
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

        # Cross-modal fusion
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

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            # self.reference_points_subobj = nn.Linear(d_model, 4)
            self.reference_points_sub = nn.Linear(d_model, 2)
            self.reference_points_obj = nn.Linear(d_model, 2)

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
                      ):
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

            # encoder
            img_memory = self.ho_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

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
                
                # print((text_memory*text_memory).sum(dim=-1))
                # text_memory = F.normalize(text_memory, p=2, dim=-1)
                text_memory_resized = self.resizer(text_memory)
                # text_memory_resized = F.normalize(text_memory_resized, p=2, dim=-1)
                # print((text_memory_resized*text_memory_resized).sum(dim=-1))
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
                text_attention_mask, text_memory_resized, obj_pred_names_sums = text

            assert img_memory.shape[2] == text_memory_resized.shape[2] # == ho_tgt.shape[1]  # == batch_num
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
            }
            return memory_cache

        else:
            # prepare input for decoder
            bs, _, c = img_memory.shape
            ho_query_embed = query_embed
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
                ho_query_embed, ho_tgt, verb_tgt = torch.split(ho_query_embed, c, dim=1)
                ho_query_embed = ho_query_embed.unsqueeze(0).expand(bs, -1, -1)
                ho_tgt = ho_tgt.unsqueeze(0).expand(bs, -1, -1)
                query_num = ho_query_embed.shape[1]
                reference_points_sub = self.reference_points_sub(ho_query_embed[:,:query_num//2]).sigmoid()
                reference_points_obj = self.reference_points_obj(ho_query_embed[:,query_num//2:]).sigmoid()
                # reference_points = self.reference_points_subobj(query_embed).sigmoid().view((bs, -1, 2, 2)).permute((2,0,1,3))  # [bs, 100, 4]
                # reference_points = self.reference_points_subobj(query_embed).sigmoid() # for ref_points == 2
                init_reference_out = reference_points = (reference_points_sub, reference_points_obj)

            # decoder
            mask_flatten = masks
            hs_ho, inter_references = self.ho_decoder(ho_tgt, 
                                                      reference_points, 
                                                      img_memory,
                                                      spatial_shapes, 
                                                      level_start_index, 
                                                      valid_ratios, 
                                                      ho_query_embed, 
                                                      mask_flatten)
            # hs_ho shape: [3, bs, 200, 256]
            inter_references_out = inter_references
            
            verb_reference_points = inter_references_out[-1]
            verb_query_embed = hs_ho[-1][:, :query_num//2] + hs_ho[-1][:, query_num//2:]
            merge_verb_tgt = verb_tgt[:query_num//2] + verb_tgt[query_num//2:]
            merge_verb_tgt = merge_verb_tgt.unsqueeze(dim = 0).expand(bs, -1, -1)
            hs_verb, verb_inter_references = self.verb_decoder(merge_verb_tgt, 
                                                               verb_reference_points, 
                                                               img_memory,
                                                               spatial_shapes, 
                                                               level_start_index, 
                                                               valid_ratios, 
                                                               verb_query_embed, 
                                                               mask_flatten)

            # Cross-modal fusion
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

            if self.two_stage:
                return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
            
            return hs_ho_dec, hs_verb_dec, text_dec, init_reference_out, inter_references_out, hs_ho, hs_verb, None, None


# RLIP_ParSeDTransformer_v2: With GLIP cross-modal fusion
class RLIP_ParSeDTransformer_v2(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 pass_pos_and_query=True, text_encoder_type="roberta-base", freeze_text_encoder=False,
                 args = None):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.fusion_type = args.fusion_type

        if self.fusion_type != "GLIP_attn":
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

        # Cross-modal fusion
        if self.fusion_type == "MDETR_attn":
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
            ho_encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
            RobertaConf = RobertaConfig.from_pretrained(text_encoder_type)
            # Memory-efficient RoBERTa config 
            # RobertaConf.hidden_size = 256
            # RobertaConf.intermediate_size = 1024
            # RobertaConf.num_attention_heads = 8
            roberta_layer = RobertaLayer(config = RobertaConf)
            VLFuse_layer = RLIPv2_VLFuse(args)
            self.ho_encoder = RLIPv2_DeformableTransformerEncoder(ho_encoder_layer, 
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
            # self.reference_points_subobj = nn.Linear(d_model, 4)
            self.reference_points_sub = nn.Linear(d_model, 2)
            self.reference_points_obj = nn.Linear(d_model, 2)

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

        self.verb_tagger = args.verb_tagger
        self.label_noise_scale = args.label_noise_scale
        self.box_noise_scale = args.box_noise_scale
        if self.verb_tagger:
            self.coord_proj = nn.Linear(4, d_model)
            self.verb_tgt = nn.Embedding(args.num_queries, d_model)
        
        self.verb_query_tgt_type = args.verb_query_tgt_type
        if "MBF" in self.verb_query_tgt_type:
            self.verb_tgt_generator = MultiBranchFusion(256, 256, 256, 16)
            self.verb_query_embed = nn.Embedding(args.num_queries//2, d_model)
        print(f"We are using self.verb_query_tgt_type: {self.verb_query_tgt_type}.")

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
                      targets=None,
                      key_padding_mask=None,
                      attn_mask=None
                      ):
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

                    # print('text_memory (before fusion) has nan? ', torch.isnan(text_memory_resized).sum(), text_memory_resized.shape)

                    if text_memory_resized.shape[1] != bs:
                        text_memory_resized = text_memory_resized.repeat(1, bs, 1)
                        text_attention_mask = text_attention_mask.repeat(1, bs)
                    img_memory, text_memory_resized = self.ho_encoder(src_flatten,
                                                spatial_shapes,
                                                level_start_index,
                                                valid_ratios,
                                                lvl_pos_embed_flatten,
                                                mask_flatten,
                                                lang_hidden=text_memory_resized.transpose(0,1),
                                                lang_masks=text_attention_mask.transpose(0,1))
                    
                    # print('text_memory (after fusion) has nan? ', torch.isnan(text_memory_resized).sum(), text_memory_resized.shape)
                    # print('self.text_encoder.pooler.dense.weight has nan?', torch.isnan(self.text_encoder.pooler.dense.weight).sum())
                    
                    # Transpose only if we do not perform resizing first.
                    # text_memory_resized = text_memory_resized.transpose(0,1)
                    # text_memory_resized = self.resizer(text_memory_resized.transpose(0,1))
                    if len(text_memory_resized.shape) == 3: # which means that "text_memory" does not have the "hs_layer" dimension, i.e. lang_aux_loss = False.
                        text_memory_resized = self.resizer(text_memory_resized.transpose(0,1))
                    elif len(text_memory_resized.shape) == 4: # which means that "text_memory" has the "hs_layer" dimension, i.e. lang_aux_loss = True.
                        text_memory_resized = self.resizer(text_memory_resized.transpose(1,2))
                else:
                    img_memory = self.ho_encoder(src_flatten,
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
                    img_memory, text_memory = self.ho_encoder(src_flatten,
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
                    img_memory = self.ho_encoder(src_flatten,
                                                spatial_shapes,
                                                level_start_index,
                                                valid_ratios,
                                                lvl_pos_embed_flatten,
                                                mask_flatten)
                    text_attention_mask, text_memory_resized, obj_pred_names_sums = text

            if self.verb_tagger:
                # print(kwargs.keys())
                obj_num = obj_pred_names_sums[0][0]
                ### Remark by Hangjie: 2022-10-27
                ### I add detach() here to block the gradient, and we use text features from the last layer.
                obj_label_text = text_memory_resized[-1,:obj_num].detach().transpose(0, 1) # like [8, 334, 256]
                query_embed, key_padding_mask, attn_mask = prepare_query(label_embeds = obj_label_text,
                              num_queries = query_embed.shape[0],
                              targets = targets,
                              training = self.training,
                              box_embed_func = self.coord_proj,
                              label_noise_scale = self.label_noise_scale,
                              box_noise_scale = self.box_noise_scale)
                # prepare_query(label_embeds, num_queries, targets, training, box_embed_func,
                #   verb_tagger = False, label_noise_scale = 0.2, box_noise_scale = 0.4)

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
                "key_padding_mask": key_padding_mask,
                "attn_mask": attn_mask,
            }
            return memory_cache

        else:
            # prepare input for decoder
            bs, _, c = img_memory.shape
            ho_query_embed = query_embed
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
                if not self.verb_tagger:
                    ho_query_embed, ho_tgt, verb_tgt = torch.split(ho_query_embed, c, dim=1)
                    ho_query_embed = ho_query_embed.unsqueeze(0).expand(bs, -1, -1)
                    ho_tgt = ho_tgt.unsqueeze(0).expand(bs, -1, -1)
                else:
                    # TODO: 注意这里的split，然后改一下下面的ho_query_embed，不需要这里的expand
                    # Remark by Hangjie: 2022-10-27
                    ho_query_embed, ho_tgt = torch.split(ho_query_embed, c, dim=2)
                    verb_tgt = self.verb_tgt.weight #.unsqueeze(0).expand(bs, -1, -1)

                query_num = ho_query_embed.shape[1]
                reference_points_sub = self.reference_points_sub(ho_query_embed[:,:query_num//2]).sigmoid()
                reference_points_obj = self.reference_points_obj(ho_query_embed[:,query_num//2:]).sigmoid()
                # reference_points = self.reference_points_subobj(query_embed).sigmoid().view((bs, -1, 2, 2)).permute((2,0,1,3))  # [bs, 100, 4]
                # reference_points = self.reference_points_subobj(query_embed).sigmoid() # for ref_points == 2
                init_reference_out = reference_points = (reference_points_sub, reference_points_obj)

            # decoder
            mask_flatten = masks
            hs_ho, inter_references = self.ho_decoder(ho_tgt, 
                                                      reference_points, 
                                                      img_memory,
                                                      spatial_shapes, 
                                                      level_start_index, 
                                                      valid_ratios, 
                                                      ho_query_embed, 
                                                      mask_flatten,
                                                      key_padding_mask=key_padding_mask,
                                                      attn_mask=attn_mask)
            # hs_ho shape: [3, bs, 200, 256]
            inter_references_out = inter_references
            
            verb_reference_points = inter_references_out[-1]
            if self.verb_query_tgt_type == "vanilla":
                verb_query_embed = hs_ho[-1][:, :query_num//2] + hs_ho[-1][:, query_num//2:]
                merge_verb_tgt = verb_tgt[:query_num//2] + verb_tgt[query_num//2:]
                merge_verb_tgt = merge_verb_tgt.unsqueeze(dim = 0).expand(bs, -1, -1)
            elif self.verb_query_tgt_type == "MBF":
                verb_query_embed = self.verb_query_embed.weight.unsqueeze(dim = 0).expand(bs, -1, -1)
                merge_verb_tgt = self.verb_tgt_generator(hs_ho[-1][:, :query_num//2], hs_ho[-1][:, query_num//2:])
            elif self.verb_query_tgt_type == "vanilla_MBF":
                verb_query_embed = self.verb_query_embed.weight.unsqueeze(dim = 0).expand(bs, -1, -1)
                merge_verb_tgt = self.verb_tgt_generator(hs_ho[-1][:, :query_num//2], hs_ho[-1][:, query_num//2:])
                merge_verb_tgt = merge_verb_tgt + \
                    (verb_tgt[:query_num//2] + verb_tgt[query_num//2:]).unsqueeze(dim = 0).expand(bs, -1, -1)
            else:
                assert False
            
            hs_verb, verb_inter_references = self.verb_decoder(merge_verb_tgt, 
                                                               verb_reference_points, 
                                                               img_memory,
                                                               spatial_shapes, 
                                                               level_start_index, 
                                                               valid_ratios, 
                                                               verb_query_embed, 
                                                               mask_flatten,
                                                               key_padding_mask=key_padding_mask[:,:query_num//2] if key_padding_mask is not None else None)

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

            # if self.two_stage:
            #     return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact


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


class ParSeTransformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
    ):
        super().__init__()

        self.pass_pos_and_query = pass_pos_and_query
 
        ## An encoder with inetermediate output 
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = CrossModelTransformerEncoder(encoder_layer, num_encoder_layers, 
                                                        encoder_norm, return_intermediate = True)

        # Human-Object decoder layer
        ho_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        ho_decoder_norm = nn.LayerNorm(d_model)
        self.ho_decoder = TransformerDecoder(ho_decoder_layer, num_decoder_layers, ho_decoder_norm,
                                             return_intermediate=return_intermediate_dec)
        # Interaction decoder layer
        verb_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        verb_decoder_norm = nn.LayerNorm(d_model)
        self.verb_decoder = TransformerDecoder(verb_decoder_layer, num_decoder_layers, verb_decoder_norm,
                                               return_intermediate=return_intermediate_dec)

        # self.CLS = nn.Embedding(1, d_model) if contrastive_loss else None

        self._reset_parameters()

        # self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        # self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
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

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        text=None,
        encode_and_save=True,
        text_memory=None,
        img_memory=None,
        text_attention_mask=None,
    ):
        '''
        text: a tuple of lists like (obj_classes, predicate_classes)
        '''
        if encode_and_save:
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c] 
            device = src.device
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                    # print(query_embed.shape)  # shape [100, 256]
            ho_query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
                    # print(query_embed.shape)  # shape [100, 2, 256]
            mask = mask.flatten(1)

            # if self.CLS is not None:
            #     # We add a CLS token to the image, to be used for contrastive loss

            #     CLS = self.CLS.weight.view(1, 1, -1).repeat(1, bs, 1)
            #     # Add the CLS token to the incoming features
            #     src = torch.cat((CLS, src))

            #     # Adding zeros as the first token in the sequence to be compatible with the CLS token
            #     pos_embed = torch.cat((torch.zeros(1, bs, self.d_model, device=device), pos_embed))

            #     # Adding one mask item to the beginning of the mask to be compatible with CLS token
            #     cls_pad = torch.zeros(bs, 1).bool().to(device)
            #     mask = torch.cat((cls_pad, mask), dim=1)

            if self.pass_pos_and_query:
                ho_tgt = torch.zeros_like(ho_query_embed)
            else:
                src, ho_tgt, ho_query_embed, pos_embed = src + 0.1 * pos_embed, ho_query_embed, None, None

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

                flat_tokenized = self.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
                # flat_tokenized = self.tokenizer.batch_encode_plus(flat_text, max_length = 128, padding = True, return_tensors="pt").to(device)
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
                    # batch_obj_text.append(encoded_flat_text.last_hidden_state[flat_idx:(flat_idx + obj_num)].mean(dim = 1))
                    # batch_pred_text.append(encoded_flat_text.last_hidden_state[(flat_idx + obj_num):(flat_idx + obj_num + pred_num)].mean(dim = 1))
                    flat_idx += (obj_num + pred_num)
                obj_text_memory = pad_sequence(batch_obj_text) 
                pred_text_memory = pad_sequence(batch_pred_text)
                text_memory = torch.cat([obj_text_memory, pred_text_memory], dim = 0)
                # text_memory: [max_token_num, batch_num, feature_dim] like [35, 4, 768]
                #              which are padded to equal lengths across all batches
                # batch_text list len: [batch_num, ] like [4,]
                # text_attention_mask = ~(text_memory.sum(dim = -1) > 0)
                text_attention_mask = ~(text_memory.sum(dim = -1) > 0)
                # text_attention_mask: [max_token_num, batch_num] like [35, 4]
                
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
                text_attention_mask, text_memory_resized, obj_pred_names_sums = text

            # Concat on the sequence dimension
            src = torch.cat([src, text_memory_resized], dim = 0)
            # For mask, sequence dimension is second
            text_attention_mask = text_attention_mask.transpose(0, 1)
            mask = torch.cat([mask, text_attention_mask], dim = 1)
            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)

            img_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

            text_memory = img_memory[:,-len(text_memory_resized):]
            # print(text_memory == text_memory_resized)
            
            assert img_memory.shape[2] == text_memory.shape[2] == ho_tgt.shape[1]  # == batch_num
            memory_cache = {
                "text_memory_resized": text_memory_resized,
                "text_memory": text_memory,
                "img_memory": img_memory[-1],
                "mask": mask,
                "text_attention_mask": text_attention_mask,
                "pos_embed": pos_embed,
                "ho_query_embed": ho_query_embed,
                "obj_pred_names_sums":obj_pred_names_sums,
            }
            # "tokenized": tokenized,
            # "text_pooled_op": encoded_text.pooler_output if self.CLS is not None else None,
            # "img_pooled_op": img_memory[0] if self.CLS is not None else None,  # Return the CLS token
                
            return memory_cache

        else:
            ho_query_embed = query_embed
            ho_tgt = torch.zeros_like(ho_query_embed)
            ho_decoder_out = self.ho_decoder(ho_tgt, 
                                        img_memory,
                                        text_memory,
                                        memory_key_padding_mask = mask,
                                        text_memory_key_padding_mask=text_attention_mask,
                                        pos = pos_embed, 
                                        query_pos = ho_query_embed)

            ho_decoder_out = ho_decoder_out.transpose(1, 2) # shape: [3, bs, 200, 256]

            ho_pair_num = ho_decoder_out.shape[2]//2
            h_decoder_out = ho_decoder_out[:,:,:ho_pair_num]
            obj_decoder_out = ho_decoder_out[:,:,ho_pair_num:]
            
            verb_query_embed = h_decoder_out[-1] + obj_decoder_out[-1]
            verb_query_embed = verb_query_embed.permute(1, 0, 2)
            verb_tgt = torch.zeros_like(verb_query_embed)
            verb_decoder_out = self.verb_decoder(verb_tgt, 
                                            img_memory,
                                            text_memory,
                                            memory_key_padding_mask=mask,
                                            text_memory_key_padding_mask=text_attention_mask,
                                            pos = pos_embed, 
                                            query_pos = verb_query_embed)
            verb_decoder_out = verb_decoder_out.transpose(1, 2)

            # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
            return h_decoder_out, obj_decoder_out, verb_decoder_out #, memory.permute(1, 2, 0).view(bs, c, h, w)



class RLIP_ParSeTransformer_v2(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        args = None,
    ):
        super().__init__()

        self.pass_pos_and_query = pass_pos_and_query
        self.fusion_type = args.fusion_type
        # encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        # encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(
        #     decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec
        # )
        
        ## An encoder without intermediate output 
        # encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ## An encoder with inetermediate output
        if self.fusion_type in ["MDETR_attn", "no_fusion"]:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = CrossModelTransformerEncoder(encoder_layer, num_encoder_layers, 
                                                        encoder_norm, return_intermediate = True)
        elif self.fusion_type == "GLIP_attn":
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

            RobertaConf = RobertaConfig.from_pretrained(text_encoder_type)
            # Memory-efficient RoBERTa config 
            # RobertaConf.hidden_size = 256
            # RobertaConf.intermediate_size = 1024
            # RobertaConf.num_attention_heads = 8
            roberta_layer = RobertaLayer(config = RobertaConf)
            VLFuse_layer = RLIPv2_VLFuse(args)
            self.encoder = RLIPv2_CrossModelTransformerEncoder(encoder_layer, 
                                                                  roberta_layer,
                                                                  VLFuse_layer,
                                                                  num_encoder_layers,
                                                                  fusion_interval=args.fusion_interval,
                                                                  fusion_last_vis=args.fusion_last_vis,
                                                                  lang_aux_loss=args.lang_aux_loss,
                                                                  norm = encoder_norm,
                                                                  return_intermediate = True)
        else:
            assert False



        # Human-Object decoder layer
        ho_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        ho_decoder_norm = nn.LayerNorm(d_model)
        self.ho_decoder = TransformerDecoder(ho_decoder_layer, num_decoder_layers, ho_decoder_norm,
                                             return_intermediate=return_intermediate_dec)
        # Interaction decoder layer
        verb_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        verb_decoder_norm = nn.LayerNorm(d_model)
        self.verb_decoder = TransformerDecoder(verb_decoder_layer, num_decoder_layers, verb_decoder_norm,
                                               return_intermediate=return_intermediate_dec)

        # self.CLS = nn.Embedding(1, d_model) if contrastive_loss else None

        self._reset_parameters()

        # self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        # self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
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

        self.d_model = d_model
        self.nhead = nhead

        self.verb_tagger = args.verb_tagger
        self.label_noise_scale = args.label_noise_scale
        self.box_noise_scale = args.box_noise_scale
        if self.verb_tagger:
            self.coord_proj = nn.Linear(4, d_model)
            self.verb_tgt = nn.Embedding(args.num_queries, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        text=None,
        encode_and_save=True,
        text_memory=None,
        img_memory=None,
        text_attention_mask=None,
    ):
        '''
        text: a tuple of lists like (obj_classes, predicate_classes)
        '''
        if encode_and_save:
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c] 
            device = src.device
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                    # print(query_embed.shape)  # shape [100, 256]
            ho_query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
                    # print(query_embed.shape)  # shape [100, 2, 256]
            mask = mask.flatten(1)


            if self.pass_pos_and_query:
                ho_tgt = torch.zeros_like(ho_query_embed)
            else:
                src, ho_tgt, ho_query_embed, pos_embed = src + 0.1 * pos_embed, ho_query_embed, None, None

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

                flat_tokenized = self.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
                # flat_tokenized = self.tokenizer.batch_encode_plus(flat_text, max_length = 128, padding = True, return_tensors="pt").to(device)
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
                    # batch_obj_text.append(encoded_flat_text.last_hidden_state[flat_idx:(flat_idx + obj_num)].mean(dim = 1))
                    # batch_pred_text.append(encoded_flat_text.last_hidden_state[(flat_idx + obj_num):(flat_idx + obj_num + pred_num)].mean(dim = 1))
                    flat_idx += (obj_num + pred_num)
                obj_text_memory = pad_sequence(batch_obj_text) 
                pred_text_memory = pad_sequence(batch_pred_text)
                text_memory = torch.cat([obj_text_memory, pred_text_memory], dim = 0)
                # text_memory: [max_token_num, batch_num, feature_dim] like [35, 4, 768]
                #              which are padded to equal lengths across all batches
                # batch_text list len: [batch_num, ] like [4,]
                # text_attention_mask = ~(text_memory.sum(dim = -1) > 0)
                text_attention_mask = ~(text_memory.sum(dim = -1) > 0)
                # text_attention_mask: [max_token_num, batch_num] like [35, 4]

            else:
                # This is used when evaluation
                # The text is already encoded, use as it is.
                # text_attention_mask, text_memory_resized, tokenized = text
                text_attention_mask, text_memory, obj_pred_names_sums = text

            
            if self.fusion_type == "GLIP_attn":
                # Resizing first if we want to perform VLFuse with 256-dim vectors.
                # text_memory_resized = self.resizer(text_memory)
                text_memory_resized = text_memory

                if text_memory_resized.shape[1] != bs:
                    text_memory_resized = text_memory_resized.repeat(1, bs, 1)
                    text_attention_mask = text_attention_mask.repeat(1, bs)
                img_memory, text_memory_resized = self.encoder(src,
                                          src_key_padding_mask=mask,
                                          pos=pos_embed,
                                          lang_hidden=text_memory_resized.transpose(0,1),
                                          lang_masks=text_attention_mask.transpose(0,1))
                
                # Transpose only if we do not perform resizing first.
                # text_memory_resized = text_memory_resized.transpose(0,1)
                # text_memory_resized = self.resizer(text_memory_resized.transpose(0,1))
                if len(text_memory_resized.shape) == 3: # which means that "text_memory" does not have the "hs_layer" dimension, i.e. lang_aux_loss = False.
                    text_memory_resized = self.resizer(text_memory_resized.transpose(0,1))
                elif len(text_memory_resized.shape) == 4: # which means that "text_memory" has the "hs_layer" dimension, i.e. lang_aux_loss = True.
                    text_memory_resized = self.resizer(text_memory_resized.transpose(1,2))
            
            elif self.fusion_type == "MDETR_attn":
                text_memory_resized = self.resizer(text_memory)
                if text_memory_resized.shape[1] != bs:
                    text_memory_resized = text_memory_resized.repeat(1, bs, 1)
                    text_attention_mask = text_attention_mask.repeat(1, bs)
                
                # Concat on the sequence dimension
                src = torch.cat([src, text_memory_resized], dim = 0)
                # For mask, sequence dimension is second
                text_attention_mask = text_attention_mask.transpose(0, 1)
                mask = torch.cat([mask, text_attention_mask], dim = 1)
                # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
                pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)
                img_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
                text_memory_resized = img_memory[:,-len(text_memory_resized):]
            

            # assert img_memory.shape[2] == text_memory_resized.shape[3] == ho_tgt.shape[1]  # == batch_num
            memory_cache = {
                "text_memory_bf_resized": text_memory,
                "text_memory_resized": text_memory_resized,
                "text_memory": text_memory_resized,
                "img_memory": img_memory[-1],
                "mask": mask,
                "text_attention_mask": text_attention_mask,
                "pos_embed": pos_embed,
                "ho_query_embed": ho_query_embed,
                "obj_pred_names_sums":obj_pred_names_sums,
            }
            # "tokenized": tokenized,
            # "text_pooled_op": encoded_text.pooler_output if self.CLS is not None else None,
            # "img_pooled_op": img_memory[0] if self.CLS is not None else None,  # Return the CLS token
                
            return memory_cache

        else:
            ho_query_embed = query_embed
            ho_tgt = torch.zeros_like(ho_query_embed)
            ho_decoder_out = self.ho_decoder(ho_tgt, 
                                        img_memory,
                                        text_memory,
                                        memory_key_padding_mask = mask,
                                        text_memory_key_padding_mask=text_attention_mask,
                                        pos = pos_embed, 
                                        query_pos = ho_query_embed)

            ho_decoder_out = ho_decoder_out.transpose(1, 2) # shape: [3, bs, 200, 256]

            ho_pair_num = ho_decoder_out.shape[2]//2
            h_decoder_out = ho_decoder_out[:,:,:ho_pair_num]
            obj_decoder_out = ho_decoder_out[:,:,ho_pair_num:]
            
            verb_query_embed = h_decoder_out[-1] + obj_decoder_out[-1]
            verb_query_embed = verb_query_embed.permute(1, 0, 2)
            verb_tgt = torch.zeros_like(verb_query_embed)
            verb_decoder_out = self.verb_decoder(verb_tgt, 
                                            img_memory,
                                            text_memory,
                                            memory_key_padding_mask=mask,
                                            text_memory_key_padding_mask=text_attention_mask,
                                            pos = pos_embed, 
                                            query_pos = verb_query_embed)
            verb_decoder_out = verb_decoder_out.transpose(1, 2)

            # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
            return h_decoder_out, obj_decoder_out, verb_decoder_out #, memory.permute(1, 2, 0).view(bs, c, h, w)


class CrossModelTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate = False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.norm is not None:
                intermediate.append(self.norm(output))
            else:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        else:
            return intermediate[-1]

class RLIPv2_CrossModelTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, roberta_layer, VLFuse_layer, num_layers, fusion_interval = 2, 
                       fusion_last_vis = False, lang_aux_loss = False, norm=None, return_intermediate = False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.fusion_interval = fusion_interval
        self.roberta_layers = _get_clones(roberta_layer, num_layers//self.fusion_interval)
        self.VLFuse_layers = _get_clones(VLFuse_layer, num_layers//self.fusion_interval)
        self.fusion_last_vis = fusion_last_vis
        assert self.fusion_last_vis
        # We have to assert self.fusion_last_vis because DETR uses the last level of vision features by default 
        # rather than multiple levels of features like DDETR.
        self.lang_aux_loss = lang_aux_loss
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        lang_hidden = None,
        lang_masks = None,
    ):

        output = src.transpose(0, 1)
        intermediate = []

        inv_padding_mask = ~src_key_padding_mask
        inv_lang_masks = ~lang_masks
        # We assert the variable stored in fused_visual_dict_features['src'] is in the shape of [bs, src_len, dim].
        fused_visual_dict_features = {'src':output, 'padding_mask':inv_padding_mask, 'pos':pos.transpose(0, 1)}
        fused_language_dict_features = {'hidden':lang_hidden, 'masks':inv_lang_masks}

        multi_lay_lang = [] # A list to store language fusion features from multiple RoBERTa layers
        for idx, layer in enumerate(self.layers):
            if (idx)%self.fusion_interval == 0:
                fusion_idx = (idx)//self.fusion_interval

                # VLFuse (We should use inverted 'padding_masks' and 'masks' for the inference of VLFuse_layers.)
                fuse_input_dict = {"visual": fused_visual_dict_features, 
                                   "lang": fused_language_dict_features}
                fuse_output_dict = self.VLFuse_layers[fusion_idx](fuse_input_dict)
                fused_visual_dict_features = fuse_output_dict["visual"]
                fused_language_dict_features = fuse_output_dict["lang"]

                # Language path (We should use the inverted 'masks' for th inference of roberta_layers.)
                fused_language_dict_features['hidden'] = self.roberta_layers[fusion_idx](
                                            hidden_states = fused_language_dict_features['hidden'],
                                            attention_mask = fused_language_dict_features['masks'])
                multi_lay_lang.append(fused_language_dict_features['hidden'])

            output = layer(fused_visual_dict_features['src'].transpose(0,1), src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.norm is not None:
                intermediate.append(self.norm(output))
                fused_visual_dict_features['src'] = intermediate[-1].transpose(0,1)
            else:
                intermediate.append(output)
                fused_visual_dict_features['src'] = intermediate[-1].transpose(0,1)
        
        if self.lang_aux_loss:
            # Select 3-layer language features
            if self.fusion_interval == 2:
                multi_lay_lang = torch.stack(multi_lay_lang, dim = 0)
            elif self.fusion_interval == 1:
                multi_lay_lang = torch.stack(multi_lay_lang[::2], dim = 0)
        else:
            multi_lay_lang = multi_lay_lang[-1]

        if self.return_intermediate:
            return torch.stack(intermediate), multi_lay_lang
        else:
            return intermediate[-1], multi_lay_lang

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                text_memory=text_memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                text_memory_key_padding_mask=text_memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # For now, trying one version where its self attn -> cross attn text -> cross attn image -> FFN
    def forward_post(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self attention
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to text
        # tgt2 = self.cross_attn_text(
        #     query=self.with_pos_embed(tgt, query_pos),
        #     key=text_memory,
        #     value=text_memory,
        #     attn_mask=None,
        #     key_padding_mask=text_memory_key_padding_mask,
        # )[0]
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # assert False, "not implemented yet"
        
        # Self-att
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # Cross-att
        tgt2 = self.norm3(tgt)
        tgt2 = self.cross_attn_image(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout3(tgt2)

        # FFN
        tgt2 = self.norm4(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, 
                memory, 
                text_memory,
                tgt_mask, 
                memory_mask, 
                text_memory_key_padding_mask,
                tgt_key_padding_mask, 
                memory_key_padding_mask, 
                pos, 
                query_pos
            )
        return self.forward_post(
            tgt,
            memory,
            text_memory,
            tgt_mask,
            memory_mask,
            text_memory_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
        contrastive_loss=args.contrastive_loss,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
