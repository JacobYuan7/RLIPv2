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
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:  # Default: return_interm_layers = False
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        # xs: ordered_dict{0: [batch, channel, w, h]} like: [2, 2048, 29, 25] [2, 2048, 18, 25]
        # for xkey, xvalue in xs.items():
        #     print(xkey)
        #     print(xvalue.shape)
        
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 load_backbone: str):
        print('name ', name)
        
        # By default, we do not use pretrained backbone.
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
        print('ResNet pretrain: False')

        # pretrained = load_backbone == 'supervised'
        if name == 'resnet50':
            if load_backbone == 'supervised':
                backbone = getattr(torchvision.models, name)(
                    replace_stride_with_dilation=[False, False, dilation],
                    pretrained=False, norm_layer=FrozenBatchNorm2d)
                default_sup_pretrain = torch.load('/mnt/data-nas/peizhi/params/resnet50-19c8e357.pth', map_location='cpu')
                backbone.load_state_dict(default_sup_pretrain, strict = True)
                print('Loading ResNet supervisedly pretrained on ImageNet from localdir...')
            # load the SwAV pre-training model from the url instead of supervised pre-training model
            elif load_backbone == 'swav':
                # checkpoint = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',map_location="cpu")
                checkpoint = torch.load('/mnt/data-nas/peizhi/params/swav_800ep_pretrain.pth.tar', map_location="cpu")
                state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
                backbone.load_state_dict(state_dict, strict=False)
                print('Loading ResNet unsupervised pretraining (SwAV) on ImageNet...')
        elif name == 'resnet101':
            if load_backbone == 'supervised':
                backbone = getattr(torchvision.models, name)(
                    replace_stride_with_dilation=[False, False, dilation],
                    pretrained=False, norm_layer=FrozenBatchNorm2d)
                default_sup_pretrain = torch.load('/mnt/data-nas/peizhi/params/resnet101-5d3b4d8f.pth', map_location='cpu')
                backbone.load_state_dict(default_sup_pretrain, strict = True)
                print('Loading ResNet supervisedly pretrained on ImageNet from localdir...')
        else:
            backbone = getattr(torchvision.models, name)(
                    replace_stride_with_dilation=[False, False, dilation],
                    pretrained=True, norm_layer=FrozenBatchNorm2d)
            print('Loading ResNet supervisedly pretrained on ImageNet from online resources...')

        
        # super().__init__(backbone, train_backbone, return_interm_layers)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


# def build_backbone(args):
#     position_embedding = build_position_encoding(args)
#     train_backbone = args.lr_backbone > 0
#     return_interm_layers = args.masks
#     backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
#     return model

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks # or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, load_backbone=args.load_backbone)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model