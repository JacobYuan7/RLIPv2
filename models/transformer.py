# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .decode import _topk
from torchvision.ops import deform_conv2d, DeformConv2d
import numpy as np
from .ParSetransformer import ParSeTransformer, ParSeDeformableTransformer, RLIP_ParSeDTransformer_v2, RLIP_ParSeTransformer_v2
from .deformable_transformer import DeformableTransformerHOI, SepDeformableTransformerHOIv3
from .DAB.transformer import ParSeDABTransformer
from .dab_deformable.deformable_transformer import ParSeDABDeformableTransformer, RLIP_ParSeDABDeformableTransformer_v2


class SepTransformerv3(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        ho_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        ho_encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.ho_encoder = TransformerEncoder(ho_encoder_layer, num_encoder_layers, ho_encoder_norm)

        verb_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        verb_encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.verb_encoder = TransformerEncoder(verb_encoder_layer, num_encoder_layers, verb_encoder_norm)

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

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        # learned tgt
        # self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC  
        # Attention!!! This H and W is not the one on the image.(image is w and h)
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c] 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                # print(query_embed.shape)  # shape [100, 256]
        ho_query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
                # print(query_embed.shape)  # shape [100, 2, 256]
        mask = mask.flatten(1)

        # Encoding
        # tgt = self.tgt_embed.weight.unsqueeze(1).repeat(1, bs, 1) 
        tgt = torch.zeros_like(query_embed)
        ho_memory = self.ho_encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # verb_memory = self.verb_encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        verb_memory = self.verb_encoder(ho_memory, src_key_padding_mask=mask, pos=pos_embed)
        memory = torch.stack((ho_memory, verb_memory), dim = 0)
        memory = memory.permute(0, 2, 3, 1).view(2, bs, c, h, w) # 2 stands for memory from ho and verb encoder
        
        # Decoding without share decoding layer
        ho_tgt = torch.zeros_like(ho_query_embed)
        ho_decoder_out = self.ho_decoder(ho_tgt, ho_memory, memory_key_padding_mask = mask,
                          pos = pos_embed, query_pos = ho_query_embed)
        ho_decoder_out = ho_decoder_out.transpose(1, 2) # shape: [3, bs, 200, 256]

        ho_pair_num = ho_decoder_out.shape[2]//2
        h_decoder_out = ho_decoder_out[:,:,:ho_pair_num]
        obj_decoder_out = ho_decoder_out[:,:,ho_pair_num:]
        
        verb_query_embed = h_decoder_out[-1] + obj_decoder_out[-1]
        verb_query_embed = verb_query_embed.permute(1, 0, 2)
        verb_tgt = torch.zeros_like(verb_query_embed)
        verb_decoder_out = self.verb_decoder(verb_tgt, verb_memory, memory_key_padding_mask=mask,
                                  pos = pos_embed, query_pos = verb_query_embed)
        verb_decoder_out = verb_decoder_out.transpose(1, 2)

        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        return h_decoder_out, obj_decoder_out, verb_decoder_out, memory



class SepTransformerv2(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

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

        # 只用Par完成HOI detection
        # 我们在inference的时候只使用ho_decoder，且参数我会暂时改成 num_decoder_layers*2

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        # learned tgt
        # self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC  
        # Attention!!! This H and W is not the one on the image.(image is w and h)
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c] 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                # print(query_embed.shape)  # shape [100, 256]
        ho_query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
                # print(query_embed.shape)  # shape [100, 2, 256]
        mask = mask.flatten(1)

        # Encoding
        # tgt = self.tgt_embed.weight.unsqueeze(1).repeat(1, bs, 1) 
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
                # print(memory.shape)
        
        # ParSe HOI detection
        # Decoding without share decoding layer
        ho_tgt = torch.zeros_like(ho_query_embed)
        ho_decoder_out = self.ho_decoder(ho_tgt, memory, memory_key_padding_mask = mask,
                          pos = pos_embed, query_pos = ho_query_embed)
        ho_decoder_out = ho_decoder_out.transpose(1, 2) # shape: [3, bs, 200, 256]

        ho_pair_num = ho_decoder_out.shape[2]//2
        h_decoder_out = ho_decoder_out[:,:,:ho_pair_num]
        obj_decoder_out = ho_decoder_out[:,:,ho_pair_num:]
        
        verb_query_embed = h_decoder_out[-1] + obj_decoder_out[-1]
        verb_query_embed = verb_query_embed.permute(1, 0, 2)
        verb_tgt = torch.zeros_like(verb_query_embed)
        verb_decoder_out = self.verb_decoder(verb_tgt, memory, memory_key_padding_mask=mask,
                                  pos = pos_embed, query_pos = verb_query_embed)
        verb_decoder_out = verb_decoder_out.transpose(1, 2)

        # # Par HOI detection
        # ho_tgt = torch.zeros_like(ho_query_embed)
        # ho_decoder_out = self.ho_decoder(ho_tgt, memory, memory_key_padding_mask = mask,
        #                   pos = pos_embed, query_pos = ho_query_embed)
        # ho_decoder_out = ho_decoder_out.transpose(1, 2) # shape: [3, bs, 200, 256]

        # ho_pair_num = ho_decoder_out.shape[2]//2
        # h_decoder_out = ho_decoder_out[:,:,:ho_pair_num]
        # obj_decoder_out = ho_decoder_out[:,:,ho_pair_num:]
        # verb_decoder_out = h_decoder_out + obj_decoder_out

        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        return h_decoder_out, obj_decoder_out, verb_decoder_out, memory.permute(1, 2, 0).view(bs, c, h, w)



class SepTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Share decoder layer
        # share_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.share_decoder = TransformerDecoder(share_decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec)

        # Human decoder layer
        h_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.h_decoder = TransformerDecoder(h_decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        # Object decoder layer
        obj_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        obj_decoder_norm = nn.LayerNorm(d_model)
        self.obj_decoder = TransformerDecoder(obj_decoder_layer, num_decoder_layers, obj_decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        # Interaction decoder layer
        verb_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        verb_decoder_norm = nn.LayerNorm(d_model)
        self.verb_decoder = TransformerDecoder(verb_decoder_layer, num_decoder_layers, verb_decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        # learned tgt
        # self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC  
        # Attention!!! This H and W is not the one on the image.(image is w and h)
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c] 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                # print(query_embed.shape)  # shape [100, 256]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # query_embed, hsp_query_embed, osp_query_embed, verbsp_query_embed = query_embed[:100], query_embed[100:200], query_embed[200:300], query_embed[300:400]
        h_query_embed, o_query_embed = query_embed[:64], query_embed[64:128]
                # print(query_embed.shape)  # shape [100, 2, 256]
        mask = mask.flatten(1)

        # Encoding
        # tgt = self.tgt_embed.weight.unsqueeze(1).repeat(1, bs, 1) 
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
                # print(memory.shape)
        
        # # Decoding with share decoding layer
        # share_decoder_out = self.share_decoder(tgt, memory, memory_key_padding_mask=mask,
        #                   pos=pos_embed, query_pos=query_embed)
        # share_decoder_out = share_decoder_out.transpose(1, 2)
        # ho_query_embed = share_decoder_out[-1]
        # ho_query_embed = ho_query_embed.permute(1, 0, 2)

        # h_tgt = torch.zeros_like(ho_query_embed)
        # h_decoder_out = self.h_decoder(h_tgt, memory, memory_key_padding_mask=mask,
        #                   pos=pos_embed, query_pos=ho_query_embed + hsp_query_embed)
        # h_decoder_out = h_decoder_out.transpose(1, 2)

        # obj_tgt = torch.zeros_like(ho_query_embed)
        # obj_decoder_out = self.obj_decoder(obj_tgt, memory, memory_key_padding_mask=mask,
        #                           pos=pos_embed, query_pos=ho_query_embed + osp_query_embed)
        # obj_decoder_out = obj_decoder_out.transpose(1, 2)
        
        # verb_query_embed = obj_decoder_out[-1] + h_decoder_out[-1] # + share_decoder_out[-1]
        # verb_query_embed = verb_query_embed.permute(1, 0, 2)
        # verb_tgt = torch.zeros_like(verb_query_embed)
        # verb_decoder_out = self.verb_decoder(verb_tgt, memory, memory_key_padding_mask=mask,
        #                           pos=pos_embed, query_pos=verb_query_embed + verbsp_query_embed)
        # verb_decoder_out = verb_decoder_out.transpose(1, 2)

        # # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        # return share_decoder_out, h_decoder_out, obj_decoder_out, verb_decoder_out, memory.permute(1, 2, 0).view(bs, c, h, w)

        # Decoding without share decoding layer
        h_tgt = torch.zeros_like(h_query_embed)
        h_decoder_out = self.h_decoder(h_tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos = h_query_embed)
        h_decoder_out = h_decoder_out.transpose(1, 2)

        obj_tgt = torch.zeros_like(o_query_embed)
        obj_decoder_out = self.obj_decoder(obj_tgt, memory, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos = o_query_embed)
        obj_decoder_out = obj_decoder_out.transpose(1, 2)
        
        verb_query_embed = obj_decoder_out[-1] + h_decoder_out[-1] # + share_decoder_out[-1]
        verb_query_embed = verb_query_embed.permute(1, 0, 2)
        verb_tgt = torch.zeros_like(verb_query_embed)
        verb_decoder_out = self.verb_decoder(verb_tgt, memory, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos = verb_query_embed)
        verb_decoder_out = verb_decoder_out.transpose(1, 2)

        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        return h_decoder_out, obj_decoder_out, verb_decoder_out, memory.permute(1, 2, 0).view(bs, c, h, w)


class SeqTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Human decoder layer
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        # Object decoder layer
        obj_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        obj_decoder_norm = nn.LayerNorm(d_model)
        self.obj_decoder = TransformerDecoder(obj_decoder_layer, num_decoder_layers, obj_decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        # Interaction decoder layer
        verb_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        verb_decoder_norm = nn.LayerNorm(d_model)
        self.verb_decoder = TransformerDecoder(verb_decoder_layer, num_decoder_layers, verb_decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        # learned tgt
        # self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC  
        # Attention!!! This H and W is not the one on the image.(image is w and h)
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c] 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                # print(query_embed.shape)  # shape [100, 256]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) 
                # print(query_embed.shape)  # shape [100, 2, 256]
        mask = mask.flatten(1)

    
        # tgt = self.tgt_embed.weight.unsqueeze(1).repeat(1, bs, 1) 
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
                # print(memory.shape)
        h_decoder_out = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        h_decoder_out = h_decoder_out.transpose(1, 2)
                # print(hs.shape)
        
        obj_query_embed = h_decoder_out[-1]
        obj_query_embed = obj_query_embed.permute(1, 0, 2)
        obj_tgt = torch.zeros_like(obj_query_embed)
        obj_decoder_out = self.obj_decoder(obj_tgt, memory, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos=obj_query_embed)
        obj_decoder_out = obj_decoder_out.transpose(1, 2)

        verb_query_embed = obj_decoder_out[-1]
        verb_query_embed = verb_query_embed.permute(1, 0, 2)
        verb_tgt = torch.zeros_like(verb_query_embed)
        verb_decoder_out = self.verb_decoder(verb_tgt, memory, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos=verb_query_embed)
        verb_decoder_out = verb_decoder_out.transpose(1, 2)

        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        return h_decoder_out, obj_decoder_out, verb_decoder_out, memory.permute(1, 2, 0).view(bs, c, h, w)


class CDN(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers_hopd=3, num_dec_layers_interaction=3, 
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_dec_layers_hopd, decoder_norm,
                                          return_intermediate=return_intermediate_dec)


        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        interaction_decoder_norm = nn.LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(interaction_decoder_layer, num_dec_layers_interaction, interaction_decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hopd_out = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hopd_out = hopd_out.transpose(1, 2)


        interaction_query_embed = hopd_out[-1]
        interaction_query_embed = interaction_query_embed.permute(1, 0, 2)

        interaction_tgt = torch.zeros_like(interaction_query_embed)
        interaction_decoder_out = self.interaction_decoder(interaction_tgt, memory, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos=interaction_query_embed)
        interaction_decoder_out = interaction_decoder_out.transpose(1, 2)

        return hopd_out, interaction_decoder_out, memory.permute(1, 2, 0).view(bs, c, h, w)



class StochasticContextTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self.latent_dim = d_model
        self.latent_mu = nn.Linear(d_model, self.latent_dim)
        self.latent_log_var = nn.Linear(d_model, self.latent_dim)
        
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) 
        # [H*W, N, C]
        memory_mu = self.latent_mu(memory)
        # [H*W, N, C]
        memory_log_var = self.latent_log_var(memory)
        
        if self.training:
            sampling_num = 1
        else:
            sampling_num = 5 
        stochastic_prior = torch.randn([sampling_num, ] + list(memory_mu.shape), 
                                        dtype = memory_mu.dtype, 
                                        device = memory_mu.device)
        memory_std = torch.exp(0.5*memory_log_var)
        stochastic_memory = memory_std * stochastic_prior + memory_mu # [sampling_num, H*W, N, C]

        hs = []
        for memory in stochastic_memory:
            # memory shape: [H*W, N, C]
            hs.append(self.decoder(tgt, memory, memory_key_padding_mask=mask,
                            pos=pos_embed, query_pos=query_embed).transpose(1, 2))
            # hs element shape: [layers, num_query, N, C]
        hs_stack = torch.stack(hs, dim = 0)
        # Should we use mean?
        # hs = torch.mean(hs_stack, dim = 0)

        return hs_stack, memory_log_var
        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)



class IterativeTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # self.reference_points_sub = nn.Linear(d_model//2, 4)
        # self.reference_points_obj = nn.Linear(d_model//2, 4)
        self.reference_points = nn.Linear(d_model, 4)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        torch.nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        torch.nn.init.constant_(self.reference_points.bias.data, 0.)
        # torch.nn.init.xavier_uniform_(self.reference_points_sub.weight.data, gain=1.0)
        # torch.nn.init.constant_(self.reference_points_sub.bias.data, 0.)
        # torch.nn.init.xavier_uniform_(self.reference_points_obj.weight.data, gain=1.0)
        # torch.nn.init.constant_(self.reference_points_obj.bias.data, 0.)

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC  
        # Attention!!! This H and W is not the one on the image.(image is w and h)
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c] 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # print(query_embed.shape)  # shape [100, 256]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) 
        # print(query_embed.shape)  # shape [100, 2, 256]
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # print(memory.shape)

        # ref_points_sub = self.reference_points_sub(query_embed[...,:self.d_model//2]).permute(1, 0, 2)
        # ref_points_obj = self.reference_points_obj(query_embed[...,self.d_model//2:]).permute(1, 0, 2)
        # ref_points = torch.cat((ref_points_sub, ref_points_obj), dim = -1)
        ref_points = self.reference_points(query_embed).permute(1, 0, 2)
        

        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # print(hs.shape)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w), ref_points



class Transformerhm(nn.Module):
    def __init__(self, num_hm_classes, num_queries, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.num_hm_classes = num_hm_classes
        self.num_queries = num_queries
        self.hm_embed = nn.Linear(d_model, self.num_hm_classes)
        self.hm_conv = nn.Conv2d(d_model, d_model, kernel_size = 3, padding=1)
        self.hm_deform = DeformConv2d(d_model, d_model, kernel_size=3, padding=1)
        self.hm_off = nn.Conv2d(d_model, 2*1*3*3, kernel_size=3, padding=1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC  
        # Attention!!! This H and W is not the one on the image.(image is w and h)
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c] 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                # print(query_embed.shape)  # shape [100, 256]
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) 
                # print(query_embed.shape)  # shape [100, 2, 256]
        mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # predicting heatmaps
        memory_hm = memory.view(h,w,bs,c).permute(2,3,0,1)
        memory_hm = F.relu(self.hm_conv(memory_hm), inplace=True)
        deform_off = self.hm_off(memory_hm)
        memory_deform_hm = self.hm_deform(memory_hm, deform_off).flatten(2).transpose(1,2)
        memory_hm = memory_hm.flatten(2).permute(2, 0, 1)  # [hw, bs, c] 
        # print(memory_deform_hm.shape)

        verb_hm = torch.sigmoid(self.hm_embed(memory_hm)).view(h, w, bs, self.num_hm_classes).permute(2,3,0,1)
        scores_verb, inds_verb, clses_verb, ys_verb, xs_verb = _topk(verb_hm, K=self.num_queries)
        scores_verb = scores_verb.view(bs, self.num_queries, 1)
        clses_verb = clses_verb.view(bs, self.num_queries, 1).float()
        query_embed = torch.gather(memory_deform_hm, dim = 1, index = inds_verb.unsqueeze(-1).expand(-1,-1,c))
        # query_embed = torch.gather(memory_hm.transpose(0,1), dim = 1, index = inds_verb.unsqueeze(-1).expand(-1,-1,c))
        query_embed = query_embed.transpose(0,1)
        tgt = torch.zeros_like(query_embed)
        
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
                # print(hs.shape)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w), verb_hm



class TransformerCoupled(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, num_verb_classes=117, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerCoupledDecoder(decoder_layer, num_decoder_layers, d_model, num_verb_classes, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        # learned tgt
        # self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC  
        # Attention!!! This H and W is not the one on the image.(image is w and h)
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c] 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                # print(query_embed.shape)  # shape [100, 256]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) 
                # print(query_embed.shape)  # shape [100, 2, 256]
        mask = mask.flatten(1)

    
        # tgt = self.tgt_embed.weight.unsqueeze(1).repeat(1, bs, 1) 
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
                # print(memory.shape)
        hs, outputs_verb_class, outputs_obj_class = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
                # print(hs.shape)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w), outputs_verb_class, outputs_obj_class



class TransformerCoupledDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, d_model, num_verb_classes, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.num_verb_classes = num_verb_classes

        self.semantic_q = nn.Linear(300, d_model)
        self.semantic_k = nn.Linear(300, d_model)
        self.semantic_v = nn.Linear(300, d_model)
        self.semantic_proj_res = nn.Linear(300, d_model)
        self.semantic_gate = nn.Linear(d_model, d_model)
        self.verb_norm = nn.LayerNorm(d_model)

        obj_verb_co = np.load('datasets/priors/obj_verb_cooccurrence.npz')['cond_prob_co_matrices']
        obj_verb_co = torch.cat((torch.tensor(obj_verb_co).float(), torch.zeros((1, num_verb_classes))), dim = 0)
        # obj_verb_co = F.log_softmax(obj_verb_co, dim = -1)
 
        # smoothing
        obj_verb_co = obj_verb_co + 0.1/(obj_verb_co.shape[0]*obj_verb_co.shape[1])
        obj_verb_co = obj_verb_co / obj_verb_co.sum()
        self.register_buffer('obj_verb_co', obj_verb_co)

        verb_word_embedding = torch.tensor(np.load('datasets/word_embedding/hico_verb_word2vec-google-news-300.npz')['embedding_list'])# [:,None]
        # verb_word_embedding = norm_tensor(verb_word_embedding) # 感觉加上会比较好
        self.register_buffer('verb_word_embedding', verb_word_embedding)

        self.obj_class_embed = None
        self.verb_class_embed = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        semantic_q = self.semantic_q(self.verb_word_embedding)
        semantic_k = self.semantic_k(self.verb_word_embedding)
        semantic_v = self.semantic_v(self.verb_word_embedding)
        semantic_att = torch.einsum('ac,bc->ab', semantic_q, semantic_k)
        semantic = F.relu(torch.matmul(semantic_att, semantic_v)) + self.semantic_proj_res(self.verb_word_embedding) # self.verb_calibration_embedding
        # semantic = norm_tensor(semantic)


        intermediate = []
        outputs_verb_class = []
        outputs_obj_class = []

        for layer in self.layers:
            # layer = self.layers[0]
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            if self.obj_class_embed is not None and self.verb_class_embed is not None:
                # print('XXX')
                output_obj_class = self.obj_class_embed(intermediate[-1]) # [2, 100, 117]
                outputs_obj_class.append(output_obj_class.transpose(0,1))
                outputs_obj_81 = output_obj_class.argmax(dim =-1).unsqueeze(-1).expand(-1,-1,self.num_verb_classes) # [6,2,100]
                obj_verb_co = self.obj_verb_co.expand(outputs_obj_81.shape[:-2]+(-1,-1))
                outputs_obj_co = torch.gather(obj_verb_co, dim =1, index = outputs_obj_81) # [6, 2, 100, 117]
                semantic_aug = torch.einsum('abc,cd->abd', outputs_obj_co, semantic)
                semantic_gate = torch.sigmoid(self.semantic_gate(intermediate[-1]))
                # print(semantic_aug.shape)
                # print(semantic_gate.shape)
                # print(intermediate[-1].shape)
                output_aug = semantic_gate * semantic_aug + intermediate[-1]
                output_verb_class = self.verb_class_embed(self.verb_norm(output_aug))
                outputs_verb_class.append(output_verb_class.transpose(0,1))
                output = output_aug
            


        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), outputs_verb_class, outputs_obj_class

        return output




class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        # learned tgt
        # self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC  
        # Attention!!! This H and W is not the one on the image.(image is w and h)
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c] 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                # print(query_embed.shape)  # shape [100, 256]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) 
                # print(query_embed.shape)  # shape [100, 2, 256]
        mask = mask.flatten(1)

    
        # tgt = self.tgt_embed.weight.unsqueeze(1).repeat(1, bs, 1) 
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
                # print(memory.shape)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
                # print(hs.shape)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)





class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

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

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            # layer = self.layers[0]
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
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
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
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

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    if args.stochastic_context_transformer:
        print('Building stochastic context transformer...')
        return StochasticContextTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    elif args.IterativeDETRHOI:
        print('Building Iterative transformer...')
        return IterativeTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    elif args.DETRHOIhm:
        print('Building DETRHOIhm...')
        return Transformerhm(
            num_hm_classes=1, 
            num_queries=args.num_queries,
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    elif args.SeqDETRHOI:
        print('Building Sequence transformer...')
        return SeqTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    elif args.SepDETRHOI:
        print('Building Fully Disentangled transformer...')
        return SepTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    elif args.ParSe:
        print('Building Fully Disentangled transformer version2...')
        return SepTransformerv2(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    elif args.SepDETRHOIv3:
        print('Building Fully Disentangled transformer version3...')
        return SepTransformerv3(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    elif args.RLIP_ParSe:
        print('Building ParSeDETR Transformer...')
        return ParSeTransformer(
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
        )
    elif args.RLIP_ParSe_v2:
        print('Building RLIP_ParSe_v2 Transformer...')
        return RLIP_ParSeTransformer_v2(
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
            args = args,
        )
    elif args.DDETRHOI:
        print('Building DeformableTransformerHOI...')
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
    elif args.ParSeD:
        print('Building SepDeformableTransformerHOIv3...')
        return SepDeformableTransformerHOIv3(
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
    elif args.RLIP_ParSeD:
        print('Building ParseDeformableTransformer...')
        return ParSeDeformableTransformer(
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
    elif args.RLIP_ParSeD_v2:
        print('Building RLIP_ParSeDTransformer_v2...')
        return RLIP_ParSeDTransformer_v2(
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
            args = args)
    elif args.CDNHOI:
        print('Building CDN Transformer...')
        return CDN(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_dec_layers_hopd=args.dec_layers_hopd,
            num_dec_layers_interaction=args.dec_layers_interaction,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    # elif args.DETRHOIcoupled:
    #     print('Building Coupled transformer...')
    #     return TransformerCoupled(
    #         d_model=args.hidden_dim,
    #         dropout=args.dropout,
    #         nhead=args.nheads,
    #         dim_feedforward=args.dim_feedforward,
    #         num_encoder_layers=args.enc_layers,
    #         num_decoder_layers=args.dec_layers,
    #         num_verb_classes=args.num_verb_classes,
    #         normalize_before=args.pre_norm,
    #         return_intermediate_dec=True,
    #     )
    elif args.ParSeDABDETR:
        print('Building DAB-DETR Transformer with ', args.transformer_activation, '(Should be prelu)')
        return ParSeDABTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=4,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
    )
    elif args.ParSeDABDDETR:
        print('Building ParSeDABDeformableTransformer...')
        return ParSeDABDeformableTransformer(
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
            use_dab=True,
        )
    elif args.RLIP_ParSeDA_v2:
        print('RLIP_ParSeDABDeformableTransformer_v2...')
        return RLIP_ParSeDABDeformableTransformer_v2(
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
            use_dab=True,
            args = args
            )
    else:
        print('Building original transformer...')
        return Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
