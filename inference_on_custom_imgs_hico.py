# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# RLIP: Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
'''
This is modified from generate_vcoco_official.py by Hangjie Yuan.
'''
import argparse
from pathlib import Path
import numpy as np
import copy
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List
import json
import datasets.transforms as T
from PIL import Image
import os

from datasets.vcoco import build as build_dataset
from models.backbone import build_backbone
from models.DDETR_backbone import build_backbone as build_DDETR_backbone
from models.transformer import build_transformer
import util.misc as utils
from models.hoi import PostProcessHOI
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from models.hoi import OCN, ParSeD, ParSe, RLIP_ParSe, RLIP_ParSeD
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

class DETRHOI(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_obj_class = self.obj_class_embed(hs)
        outputs_verb_class = self.verb_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        return out


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


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * HOI
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--missing_category_id', default=80, type=int)

    parser.add_argument('--hoi_path', type=str)
    parser.add_argument('--param_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")

    # Align with main.py
    parser.add_argument('--load_backbone', default='supervised', type=str, choices=['swav', 'supervised'])
    parser.add_argument('--DDETRHOI', action = 'store_true',
                        help='Deformable DETR for HOI detection.')
    parser.add_argument('--SeqDETRHOI', action = 'store_true',
                        help='Sequential decoding by DETRHOI')
    parser.add_argument('--SepDETRHOI', action = 'store_true',
                        help='SepDETRHOI: Fully disentangled decoding by DETRHOI')
    parser.add_argument('--SepDETRHOIv3', action = 'store_true',
                        help='SepDETRHOIv3: Fully disentangled decoding by DETRHOI')
    parser.add_argument('--CDNHOI', action = 'store_true',
                        help='CDNHOI')
    parser.add_argument('--ParSeDABDETR', action = 'store_true',
                        help='Parallel Detection and Sequential Relation Inferring using DAB-DETR.')
    parser.add_argument('--RLIPParSeDABDETR', action = 'store_true',
                        help='RLIP-Parallel Detection and Sequential Relation Inferring using DAB-DETR.')

    parser.add_argument('--stochastic_context_transformer', action = 'store_true',
                        help='Enable the stochastic context transformer')
    parser.add_argument('--IterativeDETRHOI', action = 'store_true',
                        help='Enable the Iterative Refining model for DETRHOI')
    parser.add_argument('--DETRHOIhm', action = 'store_true',
                        help='Enable the verb heatmap query prediction for DETRHOI')
    parser.add_argument('--OCN', action = 'store_true',
                        help='Augment DETRHOI with Cross-Modal Calibrated Semantics.')
    parser.add_argument('--ParSeD', action = 'store_true',
                        help='ParSeD')
    parser.add_argument('--ParSe', action = 'store_true',
                        help='ParSe')
    parser.add_argument('--RLIP_ParSe', action = 'store_true',
                        help='RLIP-ParSe')
    parser.add_argument('--RLIP_ParSeD', action = 'store_true',
                        help='RLIP-ParSeD')
    parser.add_argument("--use_no_obj_token", dest="use_no_obj_token", action="store_true", help="Whether to use No_obj_token",)
    parser.add_argument("--use_no_verb_token", dest="use_no_verb_token", action="store_true", help="Whether to use No_verb_token",)
    parser.add_argument("--subject_class", dest="subject_class", action="store_true", help="Whether to classify the subject in a triplet",)
    parser.add_argument(
        "--no_pass_pos_and_query",
        dest="pass_pos_and_query",
        action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )
    parser.add_argument(
        "--text_encoder_type",
        default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large", "bert-base-uncased", "bert-base-cased"),
    )
    parser.add_argument(
        "--freeze_text_encoder", action="store_true", help="Whether to freeze the weights of the text encoder"
    )

    # DDETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    return parser


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    object_classes = load_hico_object_txt()
    verb_classes = load_hico_verb_txt()
    corre_mat = np.load('datasets/priors/corre_hico.npy')
    device = torch.device(args.device)

    transform = make_hico_transforms(image_set = 'val')
    batch_img_path = split_path_list('custom_imgs/', batch_size = args.batch_size)

    args.lr_backbone = 0
    args.masks = False
    if args.DDETRHOI or args.ParSeD or args.RLIP_ParSeD:
        backbone = build_DDETR_backbone(args)
    else:
        backbone = build_backbone(args)
    transformer = build_transformer(args)
    if args.OCN:        
        model = OCN(
            backbone,
            transformer,
            num_obj_classes = len(object_classes) + 1,
            num_verb_classes = len(verb_classes),
            num_queries = args.num_queries,
            dataset = 'vcoco',
        )
        print('Building OCN...')
    elif args.ParSe:
        model = ParSe(
            backbone,
            transformer,
            num_obj_classes=args.num_obj_classes,
            num_verb_classes=args.num_verb_classes,
            num_queries=args.num_queries,
            # aux_loss=args.aux_loss,
        )
        print('Building ParSe...')
    elif args.RLIP_ParSe:
        model = RLIP_ParSe(
            backbone,
            transformer,
            num_queries=args.num_queries,
            # contrastive_align_loss= (args.verb_loss_type == 'cross_modal_matching') and (args.obj_loss_type == 'cross_modal_matching'),
            contrastive_hdim=64,
            # aux_loss=args.aux_loss,
            subject_class = args.subject_class,
            use_no_verb_token = args.use_no_verb_token,
        )
        print('Building RLIP_ParSe...')
    elif args.ParSeD:
        model = ParSeD(
            backbone,
            transformer,
            num_obj_classes=args.num_obj_classes,
            num_verb_classes=args.num_verb_classes,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            # aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            two_stage=args.two_stage,
            # verb_curing=args.verb_curing,
        )
        print('Building ParSeD...')
    elif args.RLIP_ParSeD:
        model = RLIP_ParSeD(
            backbone,
            transformer,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            # aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            two_stage=args.two_stage,
            subject_class = args.subject_class,
            # verb_curing=args.verb_curing,
        )
        print('Building RLIP_ParSeD...')
    else:
        model = DETRHOI(backbone, transformer, len(object_classes) + 1, len(verb_classes),
                        args.num_queries)
    postprocessors = {'hoi': PostProcessHOI(args.subject_category_id)}
    model.to(device)

    checkpoint = torch.load(args.param_path, map_location='cpu')
    load_info = model.load_state_dict(checkpoint['model'])
    print('Loading Info: ' + str(load_info))

    if hasattr(model.transformer, 'text_encoder'):
        detections = generate_hoi_with_text(model, postprocessors, batch_img_path, verb_classes, object_classes, args.subject_category_id, device, args, transform)
    else:
        # TODO
        # detections = generate_hoi_without_text(model, post_processor, data_loader_val, device, verb_classes, args.missing_category_id)
        None

    with open(args.save_path, 'wb') as f:
        pickle.dump(detections, f, protocol=2)


@torch.no_grad()
def generate_hoi_with_text(model, postprocessors, batch_img_path, verb_text, object_text, subject_category_id, device, args, transform):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Prepare the text embeddings
    if args.use_no_obj_token:
        obj_pred_names_sums = torch.tensor([[len(object_text) + 1, len(verb_text)]])
        flat_text = object_text + ['no objects'] + verb_text
    else:
        obj_pred_names_sums = torch.tensor([[len(object_text), len(verb_text)]])
        flat_text = object_text + verb_text
    flat_tokenized = model.transformer.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
        # tokenizer: dict_keys(['input_ids', 'attention_mask'])
        #            'input_ids' shape: [text_num, max_token_num]
        #            'attention_mask' shape: [text_num, max_token_num]
    encoded_flat_text = model.transformer.text_encoder(**flat_tokenized)
    text_memory = encoded_flat_text.pooler_output
    text_memory_resized = model.transformer.resizer(text_memory)
    text_memory_resized = text_memory_resized.unsqueeze(dim = 1).repeat(1, args.batch_size, 1)
    # text_attention_mask = torch.ones(text_memory_resized.shape[:2], device = device).bool()
    text_attention_mask = torch.zeros(text_memory_resized.shape[:2], device = device).bool()
    text = (text_attention_mask, text_memory_resized, obj_pred_names_sums)
    kwargs = {'text':text}

    preds = []
    gts = []
    indices = []
    result_dict = {}
    print_freq = 500
    for one_batch_path in metric_logger.log_every(batch_img_path, print_freq, header):
        samples, orig_target_sizes = load_image(transform, one_batch_path, device)
                 
        samples = samples.to(device)
        # Prepare kwargs:
        # This step must be done in the loop, due to the fact that last epoch may not have batch_size samples
        if args.batch_size != samples.tensors.shape[0]:
            text_memory_resized_short = text_memory_resized[: , :samples.tensors.shape[0]]
            text_attention_mask_short = text_attention_mask[: , :samples.tensors.shape[0]]
            text = (text_attention_mask_short, text_memory_resized_short, obj_pred_names_sums)
            kwargs = {'text': text}

        memory_cache = model(samples, encode_and_save=True, **kwargs)
        outputs = model(samples, encode_and_save=False, memory_cache=memory_cache, **kwargs)
        # outputs: a dict, whose keys are (['pred_obj_logits', 'pred_verb_logits', 
        #                                'pred_sub_boxes', 'pred_obj_boxes', 'aux_outputs'])
        # orig_target_sizes shape [bs, 2]
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if outputs['pred_verb_logits'].shape[2] == len(verb_text) + 1:
            outputs['pred_verb_logits'] = outputs['pred_verb_logits'][:,:,:-1]
        results = postprocessors['hoi'](outputs, orig_target_sizes)
        result_dict.update({p:r for p, r in zip(one_batch_path, results)})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return result_dict


def load_hico_verb_txt(file_path = 'datasets/hico_verb_names.txt') -> List[list]:
    '''
    Output like [['train'], ['boat'], ['traffic', 'light'], ['fire', 'hydrant']]
    '''
    verb_names = []
    for line in open(file_path,'r'):
        # verb_names.append(line.strip().split(' ')[-1])
        verb_names.append(' '.join(line.strip().split(' ')[-1].split('_')))
    return verb_names

def load_hico_object_txt(file_path = 'datasets/hico_object_names.txt') -> List[list]:
    '''
    Output like [['adjust'], ['board'], ['brush', 'with'], ['buy']]
    '''
    object_names = []
    with open(file_path, 'r') as f:
        object_names = json.load(f)
    object_list = list(object_names.keys())
    return object_list


### Define transforms for inference on custom images
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target = None):
        for t in self.transforms:
            image, target = t(image, target = target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

def make_hico_transforms(image_set):

    normalize = Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'val':
        return Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def split_path_list(img_root_path, batch_size):
    # Split the list by batch_size
    path_list = os.listdir(img_root_path)
    img_path_list = []
    for path in path_list:
        if os.path.isfile(img_root_path + path):
            img_path_list.append(path)

    batch_img_path = []
    temp_img_path = []
    for img_path in img_path_list:
        temp_img_path.append(img_root_path + img_path)
        if len(temp_img_path) == batch_size:
            batch_img_path.append(temp_img_path)
            temp_img_path = []
    if len(temp_img_path) > 0:
        batch_img_path.append(temp_img_path)

    return batch_img_path


def load_image(transform, file_path_list, device):
    raw_image_list = []
    size_list = []
    for file_path in file_path_list:
        raw_image = Image.open(file_path).convert('RGB')
        w, h = raw_image.size
        raw_image_list.append(raw_image)  
        size_list.append(torch.as_tensor([int(h), int(w)]).to(device))

    image = [transform(raw_image)[0].to(device) for raw_image in raw_image_list]
    image = nested_tensor_from_tensor_list(image)
    size = torch.stack(size_list, dim = 0)
    return image, size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)