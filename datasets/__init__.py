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
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .hico import build as build_hico
from .vcoco import build as build_vcoco
from .vg import build as build_vg
from .mixed_dataset import build as build_mixed
from .mixed_dataset import BatchIterativeDistributedSampler
from .oi_sgg import build as build_oi_sgg
from .o365 import build as build_o365

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'o365_det':
        return build_o365(image_set, args)
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    if args.dataset_file == 'vcoco':
        return build_vcoco(image_set, args)
    if args.dataset_file in ['vg', 'vg_oi']:
        return build_vg(image_set, args)
    if args.dataset_file in ['vg_coco2017_o365', 'vg_coco2017', 'coco2017', 'vg_hico', 'vg_coco2017_hico', 'vg_coco2017_o365_hico']:
        return build_mixed(image_set, args)
    if args.dataset_file == 'oi_sgg':
        return build_oi_sgg(image_set, args)
        
    raise ValueError(f'dataset {args.dataset_file} not supported')
