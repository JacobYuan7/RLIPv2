# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import pdb
import math
from util.misc import cat, permute_and_flatten
from timm.models.layers import DropPath

from transformers.activations import ACT2FN
import torch.utils.checkpoint as checkpoint

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

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


def _make_conv(input_dim, output_dim, k, stride=1):
    pad = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, (k, k), padding=(pad, pad), stride=(stride, stride)),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True)
    )


def _make_mlp(input_dim, output_dim, drop):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True),
                         nn.Dropout(drop),
                         nn.Linear(output_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True))


def _make_coord(batch, height, width):
    # relative position encoding
    xv, yv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    xv_min = (xv.float() * 2 - width) / width
    yv_min = (yv.float() * 2 - height) / height
    xv_max = ((xv + 1).float() * 2 - width) / width
    yv_max = ((yv + 1).float() * 2 - height) / height
    xv_ctr = (xv_min + xv_max) / 2
    yv_ctr = (yv_min + yv_max) / 2
    hmap = torch.ones(height, width) * (1. / height)
    wmap = torch.ones(height, width) * (1. / width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0), \
                                               xv_max.unsqueeze(0), yv_max.unsqueeze(0), \
                                               xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0), \
                                               hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0))
    coord = coord.unsqueeze(0).repeat(batch, 1, 1, 1)
    return coord


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, args=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = args.stable_softmax_2d # cfg.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D
        self.clamp_min_for_underflow = args.clamp_min_for_underflow # cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW
        self.clamp_max_for_overflow = args.clamp_max_for_overflow # cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[
            0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


class RLIPv2_BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, args=None):
        super(RLIPv2_BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = args.stable_softmax_2d # cfg.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D
        self.clamp_min_for_underflow = args.clamp_min_for_underflow # cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW
        self.clamp_max_for_overflow = args.clamp_max_for_overflow # cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, v, l, v_pos=None, attention_mask_l=None, attention_mask_v=None):
        # TODO:
        # Add with_pos_embed(v, v_pos) for attention calculation
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(self.with_pos_embed(v, v_pos)) * self.scale  # bsz, tgt_len, embed_dim
        key_states = self._shape(self.l_proj(l), -1, bsz)  # bsz, self.num_heads, -1, self.head_dim
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)  # bsz, self.num_heads, -1, self.head_dim
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)  # bsz, self.num_heads, -1, self.head_dim

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # bsz * self.num_heads, -1, self.head_dim
        key_states = key_states.view(*proj_shape) # bsz * self.num_heads, -1, self.head_dim
        value_v_states = value_v_states.view(*proj_shape) # bsz * self.num_heads, -1, self.head_dim
        value_l_states = value_l_states.view(*proj_shape) # bsz * self.num_heads, -1, self.head_dim

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) # bsz * self.num_heads, tgt_len, src_len

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l, max=50000) # Do not increase 50000, data type half has quite limited range
        
        # Following lines (if) are added to mask out visual features for the calculation of attn_weights_l
        if attention_mask_v is not None:
            # shape of attn_weights_l: [bsz, src_len, tgt_len]  src_len = lang_len; tgt_len = vis_len
            assert (attention_mask_v.dim() == 2)
            attention_mask = attention_mask_v.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, src_len, tgt_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, src_len, tgt_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, src_len, tgt_len)}"
                )
            attn_weights_l = attn_weights_l.view(bsz, self.num_heads, src_len, tgt_len) + attention_mask
            attn_weights_l = attn_weights_l.view(bsz * self.num_heads, src_len, tgt_len)

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l

# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_l=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l

class BiAttentionBlockForCheckpoint(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlockForCheckpoint, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

        self.cfg = cfg
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL:
            if not self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                self.shrink_lang = FeatureResizer(l_dim * 5, l_dim, 0.1)

    def forward(self, q0, q1, q2, q3, q4, l, attention_mask_l=None, dummy_tensor=None):

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL:
            visu_feat = []
            lang_feat = []
            for ii, feat in enumerate([q0, q1, q2, q3, q4]):
                bs, _, h, w = feat.shape
                q = feat.flatten(2).transpose(1, 2)
                
                new_v, new_l = self.single_attention_call(q, l, attention_mask_l=attention_mask_l)
                new_v = new_v.transpose(1, 2).contiguous().view(bs, -1, h, w)
                lang_feat.append(new_l)
                visu_feat.append(new_v)
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                pass
            else:
                lang_feat = self.shrink_lang(torch.cat(lang_feat, dim = -1)) # From multiple dimensions
                lang_feat = [lang_feat, None, None, None, None]
        else:
            visu_feat = []
            size_per_level, visual_features_flatten = [], []
            for ii, feat_per_level in enumerate([q0, q1, q2, q3, q4]):
                bs, c, h, w = feat_per_level.shape
                size_per_level.append([h, w])
                feat = permute_and_flatten(feat_per_level, bs, 1, c, h, w)
                visual_features_flatten.append(feat)
            visual_features_flatten = cat(visual_features_flatten, dim=1)
            new_v, new_l = self.single_attention_call(visual_features_flatten, l, attention_mask_l=attention_mask_l)
            # [bs, N, C] -> [bs, C, N]
            new_v = new_v.transpose(1, 2).contiguous()

            start = 0
            for (h, w) in size_per_level:
                new_v_per_level = new_v[:, :, start:start + h * w].view(bs, -1, h, w).contiguous()
                visu_feat.append(new_v_per_level)
                start += h * w
            
            lang_feat = [new_l, None, None, None, None]

        return visu_feat[0], visu_feat[1], visu_feat[2], visu_feat[3], visu_feat[4], lang_feat[0], lang_feat[1], lang_feat[2], lang_feat[3], lang_feat[4]

    
    def single_attention_call(self, v, l, attention_mask_l=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l


class RLIPv2_BiAttentionBlockForCheckpoint(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, args=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(RLIPv2_BiAttentionBlockForCheckpoint, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = RLIPv2_BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         args=args)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

        ### Extra parameters for gating
        self.gating_mechanism = args.gating_mechanism
        print(f"We are using {self.gating_mechanism} as a gating mechanism.")
        if self.gating_mechanism in ["Stanh", "SDFtanh", "SFtanh", "SDFXAc", "SXAc", "SXAcLN", "SDFXAcLN"]:
            self.gamma_v_down = nn.Linear(v_dim, v_dim//4)
            self.gamma_v_up = nn.Linear(v_dim//4, v_dim)
            self.gamma_l_down = nn.Linear(l_dim, l_dim//4)
            self.gamma_l_up = nn.Linear(l_dim//4, l_dim)
        if self.gating_mechanism in ["SXAcLN", "SDFXAcLN"]:
            self.layer_norm_gating_v = nn.LayerNorm(v_dim//4)
            self.layer_norm_gating_l = nn.LayerNorm(l_dim//4)
        if self.gating_mechanism in ["SOtanh", "SDFOXAcLN"]:
            self.gamma_v_down = nn.Linear(v_dim, v_dim//2)
            self.gamma_v_one = nn.Linear(v_dim//2, 1)
            self.gamma_l_down = nn.Linear(l_dim, l_dim//2)
            self.gamma_l_one = nn.Linear(l_dim//2, 1)
        if self.gating_mechanism in ["SDFOXAcLN"]:
            self.layer_norm_gating_v = nn.LayerNorm(v_dim//2)
            self.layer_norm_gating_l = nn.LayerNorm(l_dim//2)
        if self.gating_mechanism == "MBF":
            self.MBF_v = MultiBranchFusion(v_dim, v_dim, v_dim, 16)
            self.MBF_l = MultiBranchFusion(l_dim, l_dim, l_dim, 16)


        self.args = args
        if self.args.separate_bidirectional:
            if not self.args.do_lang_proj_outside_checkpoint:
                self.shrink_lang = FeatureResizer(l_dim * 5, l_dim, 0.1)

    def forward(self, q, l, q_pos=None, attention_mask_l=None, attention_mask_v=None, dummy_tensor=None):

        if self.args.separate_bidirectional:
            # TODO: modify codes for separate_bidirectional
            None
            # visu_feat = []
            # lang_feat = []
            # for ii, feat in enumerate([q0, q1, q2, q3, q4]):
            #     bs, _, h, w = feat.shape
            #     q = feat.flatten(2).transpose(1, 2)
                
            #     new_v, new_l = self.single_attention_call(q, l, attention_mask_l=attention_mask_l)
            #     new_v = new_v.transpose(1, 2).contiguous().view(bs, -1, h, w)
            #     lang_feat.append(new_l)
            #     visu_feat.append(new_v)
            # if self.args.do_lang_proj_outside_checkpoint:
            #     pass
            # else:
            #     lang_feat = self.shrink_lang(torch.cat(lang_feat, dim = -1)) # From multiple dimensions
            #     lang_feat = [lang_feat, None, None, None, None]
        else:
            # TODO: test if pos embedding is necessary for VLFuse
            new_v, new_l = self.single_attention_call(
                                v = q, 
                                l = l,
                                v_pos = q_pos,
                                attention_mask_l=attention_mask_l,
                                attention_mask_v=attention_mask_v)
            # [bs, N, C] -> [bs, C, N]
            # new_v = new_v.transpose(1, 2).contiguous()
            lang_feat = [new_l, None, None, None]


        return new_v, lang_feat[0], lang_feat[1], lang_feat[2], lang_feat[3]

    
    def single_attention_call(self, v, l, v_pos, attention_mask_l=None, attention_mask_v=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, v_pos, attention_mask_l=attention_mask_l, attention_mask_v=attention_mask_v)
        # v, l = v + delta_v, l + delta_l

        if self.gating_mechanism == "GLIP":
            ### Original GLIP cross-modal fusion, default
            v = v + self.drop_path(self.gamma_v * delta_v)
            l = l + self.drop_path(self.gamma_l * delta_l)
        elif self.gating_mechanism == "Vtanh":
            ### Vanilla tanh (Vtanh)
            v = v + self.drop_path(torch.tanh(self.gamma_v[0]) * delta_v)
            l = l + self.drop_path(torch.tanh(self.gamma_l[0]) * delta_l)
        elif self.gating_mechanism == "Etanh":
            ### Element-wise tanh (Etanh)
            v = v + self.drop_path(torch.tanh(self.gamma_v) * delta_v)
            l = l + self.drop_path(torch.tanh(self.gamma_l) * delta_l)
        elif self.gating_mechanism == "Stanh":
            ### Self-gating tanh (Stanh)
            v = v + self.drop_path(torch.tanh(self.gamma_v_up(torch.relu(self.gamma_v_down(self.gamma_v)))) * delta_v)
            l = l + self.drop_path(torch.tanh(self.gamma_l_up(torch.relu(self.gamma_l_down(self.gamma_l)))) * delta_l)
        elif self.gating_mechanism == "SDFtanh":
            ### Self-DeltaFeature-gating tanh (SDFtanh)
            v = v + self.drop_path(torch.tanh(self.gamma_v_up(torch.relu(self.gamma_v_down(delta_v)))) * delta_v)
            l = l + self.drop_path(torch.tanh(self.gamma_l_up(torch.relu(self.gamma_l_down(delta_l)))) * delta_l)
        elif self.gating_mechanism == "SFtanh":
            ### Self-Feature-gating tanh (SFtanh)
            v = v + self.drop_path(torch.tanh(self.gamma_v_up(torch.relu(self.gamma_v_down(v)))) * delta_v)
            l = l + self.drop_path(torch.tanh(self.gamma_l_up(torch.relu(self.gamma_l_down(l)))) * delta_l)
        elif self.gating_mechanism == "SOtanh":
            ### Self-One-gating tanh (SOtanh)
            v = v + self.drop_path(torch.tanh(self.gamma_v_one(torch.relu(self.gamma_v_down(self.gamma_v)))) * delta_v)
            l = l + self.drop_path(torch.tanh(self.gamma_l_one(torch.relu(self.gamma_l_down(self.gamma_l)))) * delta_l)
        elif self.gating_mechanism == "VXAc":
            ### Vanilla tanh without any activation (VXac)
            v = v + self.drop_path(self.gamma_v[0] * delta_v)
            l = l + self.drop_path(self.gamma_l[0] * delta_l)
        elif self.gating_mechanism == "SXAc":
            ### Self-gating tanh without any activation (SXAc)
            v = v + self.drop_path(self.gamma_v_up(torch.relu(self.gamma_v_down(self.gamma_v))) * delta_v)
            l = l + self.drop_path(self.gamma_l_up(torch.relu(self.gamma_l_down(self.gamma_l))) * delta_l)
        elif self.gating_mechanism == "SDFXAc":
            ### Self-DeltaFeature-gating without any activation (SDFXAc)
            v = v + self.drop_path(self.gamma_v_up(torch.relu(self.gamma_v_down(delta_v))) * delta_v)
            l = l + self.drop_path(self.gamma_l_up(torch.relu(self.gamma_l_down(delta_l))) * delta_l)
        elif self.gating_mechanism == "SXAcLN":
            ### Self-gating tanh without any activation (SXAc)
            v = v + self.drop_path(self.gamma_v_up(torch.relu(self.layer_norm_gating_v(self.gamma_v_down(self.gamma_v)))) * delta_v)
            l = l + self.drop_path(self.gamma_l_up(torch.relu(self.layer_norm_gating_l(self.gamma_l_down(self.gamma_l)))) * delta_l)
        elif self.gating_mechanism == "SDFXAcLN":
            ### Self-DeltaFeature-gating without any activation (SDFXAc)
            v = v + self.drop_path(self.gamma_v_up(torch.relu(self.layer_norm_gating_v(self.gamma_v_down(delta_v)))) * delta_v)
            l = l + self.drop_path(self.gamma_l_up(torch.relu(self.layer_norm_gating_l(self.gamma_l_down(delta_l)))) * delta_l)
        elif self.gating_mechanism == "SDFOXAcLN":
            ### Self-DeltaFeature One-gating without any activation (SDFOXAc)
            v = v + self.drop_path(self.gamma_v_one(torch.relu(self.layer_norm_gating_v(self.gamma_v_down(delta_v)))) * delta_v)
            l = l + self.drop_path(self.gamma_l_one(torch.relu(self.layer_norm_gating_l(self.gamma_l_down(delta_l)))) * delta_l)
        elif self.gating_mechanism == "MBF":
            v = self.MBF_v(v, delta_v)
            l = self.MBF_l(l, delta_l)
        elif self.gating_mechanism == "XGating":
            # Do not use any gating functions ("XGating")
            v = v + self.drop_path(delta_v)
            l = l + self.drop_path(delta_l)
        else:
            assert False

        return v, l


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


# Single Direction MHA
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1, 
        clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q, k, v, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim = q.size()

        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(k), -1, bsz)
        value_states = self._shape(self.v_proj(v), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)


        return attn_output, attn_weights


class AttentionMLP(nn.Module):
    def __init__(self, q_dim, hidden_dim, dropout=0.1):
        super(AttentionMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(q_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, q_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class AttentionT2I(nn.Module):
    def __init__(self, q_dim, k_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, mode="i2t", use_layer_scale = False,
                 clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(AttentionT2I, self).__init__()

        # pre_layer norm
        self.layer_norm_q_1 = nn.LayerNorm(q_dim)
        self.layer_norm_k_1 = nn.LayerNorm(k_dim)
        self.attn = MultiHeadAttention(q_dim=q_dim,
                                       k_dim=k_dim,
                                       embed_dim=embed_dim,
                                       num_heads=num_heads,
                                       clamp_min_for_underflow=clamp_min_for_underflow,
                                       clamp_max_for_overflow=clamp_max_for_overflow)
        self.mode = mode

        # add layer scale for training stability
        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.gamma = nn.Parameter(init_values * torch.ones((q_dim)), requires_grad=True)


    def forward(self, q0, q1, q2, q3, q4, k, v, attention_mask, dummy_arg=None):
        qs = []
        for q_index, q in enumerate([q0, q1, q2, q3, q4]):
            bs, _, h, w = q.shape
            # (batch, seq_len, embed_size)
            q = q.flatten(2).transpose(1, 2)
            q = self.layer_norm_q_1(q)
            k, v = self.layer_norm_k_1(k), self.layer_norm_k_1(v)
            delta_q = self.attn(q, k, v, attention_mask=attention_mask)[0]
            if self.use_layer_scale:
                q = q + self.drop_path(self.gamma * delta_q)
            else:
                q = q + delta_q
            q = q.transpose(1, 2).contiguous().view(bs, -1, h, w)
            qs.append(q)


        return qs[0], qs[1], qs[2], qs[3], qs[4]

##### This is modified from https://github.com/microsoft/GLIP/blob/fd52c6361f013e70ae7682d90b3ab3ca2bd5e6bc/maskrcnn_benchmark/modeling/rpn/vldyhead.py#L350 #####
class RLIPv2_VLFuse(torch.nn.Module):
    """
    Early Fusion Module
    """
    def __init__(self, args):
        super(RLIPv2_VLFuse, self).__init__()
        self.init_configs(args)
        self.args = args

        self.use_checkpoint_fusion = False
        if hasattr(args, 'use_checkpoint_fusion'):
            print("Use checkpoint to save memory during RLIPv2_VLFuse.")
            self.use_checkpoint_fusion = args.use_checkpoint_fusion
            self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        # if hasattr(cfg.MODEL.DYHEAD, 'USE_CHECKPOINT'):
        #     self.use_checkpoint = cfg.MODEL.DYHEAD.USE_CHECKPOINT
        #     self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        # early fusion module
        print("RLIPv2 EARLY FUSION ON, USING {}".format(args.fusion_type))
        if args.fusion_type == "GLIP_attn":
            # bi-direction (text->image, image->text)
            self.b_attn = RLIPv2_BiAttentionBlockForCheckpoint(
                        v_dim=self.joint_embedding_size,
                        l_dim=self.lang_dim,
                        embed_dim=self.embed_dim,
                        num_heads=self.n_head,
                        hidden_dim=self.i2t_hidden_dim,
                        dropout=0.1,
                        drop_path=.0,
                        init_values=1.0 / args.num_feature_levels,
                        args=args,
                        )
            if self.args.separate_bidirectional and self.args.do_lang_proj_outside_checkpoint:
                self.shrink_lang = FeatureResizer(self.lang_dim * args.num_feature_levels,
                                self.lang_dim, 0.1)
        else:
            print("NO FUSION INVOLVED.")


    def init_configs(self, args):
        # common params
        self.lang_model = args.text_encoder_type # cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.joint_embedding_size = 256 # cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE
        self.joint_embedding_dropout = 0.1 # cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT
        self.joint_mlp_layers = 2 # cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_MLP_LAYERS

        # self.max_query_len = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        # self.n_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS
        # self.coord_dim = 8
        # self.joint_inp_dim = self.coord_dim + self.joint_embedding_size
        # self.joint_out_dim = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_OUT_SIZE

        # mha params
        self.n_head = 8
        self.embed_dim = 2048 # Save memory: 256 # Original: 2048
        self.t2i_hidden_dim = 1024  # 256 * 4
        self.i2t_hidden_dim = 3072  # 768 * 4

        if self.lang_model in ["bert-base-uncased", "roberta-base", "clip"]:
            self.lang_dim = 768 # after resizing: 256 # Original: 768 # cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        else:
            self.lang_dim = 1024

    def forward(self, x):
        visual_dict_features = x["visual"]
        language_dict_features = x["lang"]

        visual_features = visual_dict_features['src']
        batch_size = visual_features[0].shape[0]
        device = visual_features[0].device

        fused_visual_features = None
        fused_language_dict_features = None

        if self.args.fusion_type == "GLIP_attn":
            if self.args.use_checkpoint_fusion:
                q, l0, l1, l2, l3 = checkpoint.checkpoint(
                    self.b_attn,
                    visual_dict_features['src'],
                    language_dict_features['hidden'],
                    visual_dict_features['pos'], # This is "lvl_pos_embed_flatten".
                    language_dict_features['masks'],
                    visual_dict_features['padding_mask'],
                    self.dummy_tensor
                )
            else:
                q, l0, l1, l2, l3 = self.b_attn(
                    q = visual_dict_features['src'],
                    l = language_dict_features['hidden'],
                    q_pos = visual_dict_features['pos'], # This is "lvl_pos_embed_flatten".
                    attention_mask_l = language_dict_features['masks'],
                    attention_mask_v = visual_dict_features['padding_mask'],
                    dummy_tensor = self.dummy_tensor
                )

            visual_dict_features['src'] = q
            fused_visual_dict_features = visual_dict_features # [q0, q1, q2, q3, q4]
            
            if self.args.separate_bidirectional and self.args.do_lang_proj_outside_checkpoint:
                language_features = self.shrink_lang(torch.cat([l0, l1, l2, l3], dim = -1))
            else:
                language_features = l0

            language_dict_features['hidden'] = language_features
            fused_language_dict_features = language_dict_features
        else:
            # fused_visual_features = visual_features
            fused_visual_dict_features = visual_dict_features
            fused_language_dict_features = language_dict_features

        features_dict = {"visual": fused_visual_dict_features,
                         "lang": fused_language_dict_features}

        return features_dict