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
import torchvision

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
from models.hoi import OCN, ParSeD, ParSe, RLIP_ParSe, RLIP_ParSeD, RLIP_ParSeDA
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from models.swin.backbone import build_backbone as build_Swin_backbone
import pdb
# pdb.set_trace()

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
    parser.add_argument(
        "--verb_tagger",
        dest="verb_tagger",
        action="store_true",
        help="Whether to perform verb tagging pre-training",
    )
    parser.add_argument('--label_noise_scale', default=0.2, type=float,
                        help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float,
                        help="box noise scale to shift and scale")


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

    # RLIPv2
    parser.add_argument('--RLIP_ParSeDA_v2', action = 'store_true',
                        help='RLIP_ParSeDA_v2.') 
    parser.add_argument('--RLIP_ParSeD_v2', action = 'store_true',
                        help='RLIP_ParSeD_v2.') 
    parser.add_argument('--RLIP_ParSe_v2', action = 'store_true',
                        help='RLIP_ParSe_v2.') 
    parser.add_argument('--ParSeDABDDETR', action = 'store_true',
                        help='Parallel Detection and Sequential Relation Inferring using DAB-Deformable DETR.')
    
    # Cross-Modal Fusion
    parser.add_argument('--use_checkpoint_fusion', default=False, action='store_true', help = 'Use checkpoint to save memory.')
    parser.add_argument('--fusion_type', default = "no_fusion", choices = ("MDETR_attn", "GLIP_attn", "no_fusion"), )
    parser.add_argument('--fusion_interval', default=1, type=int, help="Fusion interval in VLFuse.")
    parser.add_argument('--fusion_last_vis', default=False, action='store_true', help = 'Whether to fuse the last layer of the vision features.')
    parser.add_argument('--lang_aux_loss', default=False, action='store_true', help = 'Whether to use aux loss to calculate the loss functions.')
    parser.add_argument('--separate_bidirectional', default=False, action='store_true', help = 'For GLIP_attn, we perform separate attention for different levels of features.')
    parser.add_argument('--do_lang_proj_outside_checkpoint', default=False, action='store_true', help = 'Use feature resizer to project the concatenation of interactive language features to the dimension of language embeddings.')
    parser.add_argument('--stable_softmax_2d', default=False, action='store_true', help = 'Use "attn_weights = attn_weights - attn_weights.max()" during BiMultiHeadAttention in VLFuse.')
    parser.add_argument('--clamp_min_for_underflow', default=False, action='store_true', help = 'Clamp attention weights (before softmax) during BiMultiHeadAttention in VLFuse.')
    parser.add_argument('--clamp_max_for_overflow', default=False, action='store_true', help = 'Clamp attention weights (before softmax) during BiMultiHeadAttention in VLFuse.')

    parser.add_argument('--gating_mechanism', default="GLIP", type=str,
                        choices=["GLIP", "Vtanh", "Etanh", "Stanh", "SDFtanh", "SFtanh", "SOtanh", "SXAc", "SDFXAc", "VXAc", "SXAcLN", "SDFXAcLN", "SDFOXAcLN", "MBF"],
                        help = "The gating mechanism used to perform language-vision feature fusion.")
    parser.add_argument('--verb_query_tgt_type', default="vanilla", type=str,
                        choices=["vanilla", "MBF", "vanilla_MBF"],
                        help = "The method used to generate queries.")
    
    ## DABDETR
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int,
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true', 
                        help="Random init the x,y of anchor boxes and freeze them.")
    parser.add_argument('--pe_temperatureH', default=20, type=int, 
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int, 
                        help="Temperature for width positional encoding.")


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
    assert args.batch_size == 1
    object_classes = load_hico_object_txt()
    verb_classes = load_hico_verb_txt()
    corre_mat = np.load('datasets/priors/corre_hico.npy')
    device = torch.device(args.device)

    transform = make_hico_transforms(image_set = 'val')
    # batch_img_path = split_path_list('custom_imgs/', batch_size = args.batch_size)

    args.lr_backbone = 0
    args.masks = False
    # if args.DDETRHOI or args.ParSeD or args.RLIP_ParSeD:
    #     backbone = build_DDETR_backbone(args)
    # else:
    #     backbone = build_backbone(args)
    if 'swin' in args.backbone:
        backbone = build_Swin_backbone(args)
    elif args.DDETRHOI or args.ParSeD or args.RLIP_ParSeD or args.RLIP_ParSeD_v2 or args.ParSeDABDDETR or args.RLIP_ParSeDA_v2:
        backbone = build_DDETR_backbone(args)
    elif args.ParSeDABDETR or args.RLIPParSeDABDETR:
        backbone = build_DABDETR_backbone(args)
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
    elif args.RLIP_ParSeD or args.RLIP_ParSeD_v2:
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
                args=args,
            )
            print('Building RLIP_ParSeD...')
    # elif args.RLIP_ParSeD:
    #     model = RLIP_ParSeD(
    #         backbone,
    #         transformer,
    #         num_queries=args.num_queries,
    #         num_feature_levels=args.num_feature_levels,
    #         # aux_loss=args.aux_loss,
    #         with_box_refine=args.with_box_refine,
    #         two_stage=args.two_stage,
    #         subject_class = args.subject_class,
    #         # verb_curing=args.verb_curing,
    #     )
    #     print('Building RLIP_ParSeD...')
    elif args.RLIP_ParSeDA_v2:
            model = RLIP_ParSeDA(
                backbone,
                transformer,
                num_queries=args.num_queries,
                num_feature_levels=args.num_feature_levels,
                # aux_loss=args.aux_loss,
                with_box_refine=args.with_box_refine,
                two_stage=args.two_stage,
                use_dab=True, 
                num_patterns=args.num_patterns,
                random_refpoints_xy=args.random_refpoints_xy,
                subject_class = args.subject_class,
                args = args,
            )
    else:
        model = DETRHOI(backbone, transformer, len(object_classes) + 1, len(verb_classes),
                        args.num_queries)
    postprocessors = {'hoi': PostProcessHOI(args.subject_category_id)}
    model.to(device)

    checkpoint = torch.load(args.param_path, map_location='cpu')
    load_info = model.load_state_dict(checkpoint['model'])
    print('Loading Info: ' + str(load_info))


    ### Prepare dataset
    Coco_train = CocoDetection(img_folder = '/mnt/data-nas/peizhi/data/coco2017/train2017',
                         ann_file = '/mnt/data-nas/peizhi/data/coco2017/annotations/instances_train2017.json',
                         transforms=make_hico_transforms('val'),
                         return_masks=False)
    # Coco_val = CocoDetection(img_folder = '/mnt/data-nas/peizhi/data/coco2017/val2017',
    #                      ann_file = '/mnt/data-nas/peizhi/data/coco2017/annotations/instances_val2017.json',
    #                      transforms=make_coco_transforms('val'),
    #                      return_masks=False)
    # official_coco_bbox = transform_coco_official_to_VG_format(Coco_train)
    # official_coco_bbox.update(transform_coco_official_to_VG_format(Coco_val))

    sampler = torch.utils.data.RandomSampler(Coco_train)
    batch_sampler = torch.utils.data.BatchSampler(
        sampler, args.batch_size, drop_last=True)
    data_loader = DataLoader(Coco_train, batch_sampler=batch_sampler,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers, shuffle = False)


    if hasattr(model.transformer, 'text_encoder'):
        detections = generate_pseudo_triplets_with_text(model, postprocessors, data_loader, verb_classes, object_classes, args.subject_category_id, device, args, transform)
    else:
        # TODO
        # detections = generate_hoi_without_text(model, post_processor, data_loader_val, device, verb_classes, args.missing_category_id)
        None

    print(detections[-1])
    with open(args.save_path, 'w') as f:
        json.dump(detections, f)


@torch.no_grad()
def generate_pseudo_triplets_with_text(model, postprocessors, data_loader, verb_text, object_text, subject_category_id, device, args, transform):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    all_rels, rels_for_coco = aggregate_rels_for_dataset()
    if args.use_no_obj_token:
        # obj_pred_names_sums = torch.tensor([[len(object_text) + 1, len(verb_text)]])
        flat_text = object_text + ['no objects'] + all_rels
    else:
        # obj_pred_names_sums = torch.tensor([[len(object_text), len(verb_text)]])
        flat_text = object_text + all_rels
    flat_tokenized = model.transformer.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
        # tokenizer: dict_keys(['input_ids', 'attention_mask'])
        #            'input_ids' shape: [text_num, max_token_num]
        #            'attention_mask' shape: [text_num, max_token_num]
    encoded_flat_text = model.transformer.text_encoder(**flat_tokenized)
    text_memory = encoded_flat_text.pooler_output
    # text_memory_resized = model.transformer.resizer(text_memory)
    if args.RLIP_ParSe_v2:
        text_memory_resized = text_memory
    elif args.RLIP_ParSeD_v2 or args.RLIP_ParSeDA_v2:
        if args.fusion_type == "GLIP_attn":
            text_memory_resized = text_memory
        else:
            text_memory_resized = model.module.transformer.resizer(text_memory)
    else:
        text_memory_resized = model.module.transformer.resizer(text_memory)
    text_memory_resized = text_memory_resized.unsqueeze(dim = 1).repeat(1, args.batch_size, 1)
    object_text_memory_resized = text_memory_resized[:-len(all_rels)]
    rel_text_memory_resized = text_memory_resized[-len(all_rels):]



    # # Prepare the text embeddings
    # if args.use_no_obj_token:
    #     obj_pred_names_sums = torch.tensor([[len(object_text) + 1, len(verb_text)]])
    #     flat_text = object_text + ['no objects'] + verb_text
    # else:
    #     obj_pred_names_sums = torch.tensor([[len(object_text), len(verb_text)]])
    #     flat_text = object_text + verb_text
    # flat_tokenized = model.transformer.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
    #     # tokenizer: dict_keys(['input_ids', 'attention_mask'])
    #     #            'input_ids' shape: [text_num, max_token_num]
    #     #            'attention_mask' shape: [text_num, max_token_num]
    # encoded_flat_text = model.transformer.text_encoder(**flat_tokenized)
    # text_memory = encoded_flat_text.pooler_output
    # text_memory_resized = model.transformer.resizer(text_memory)
    # text_memory_resized = text_memory_resized.unsqueeze(dim = 1).repeat(1, args.batch_size, 1)
    # # text_attention_mask = torch.ones(text_memory_resized.shape[:2], device = device).bool()
    # text_attention_mask = torch.zeros(text_memory_resized.shape[:2], device = device).bool()
    # text = (text_attention_mask, text_memory_resized, obj_pred_names_sums)
    # kwargs = {'text':text}

    preds = []
    gts = []
    indices = []
    result_list = []
    print_freq = 500
    for batch_i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)): 
        # if batch_i == 100:
        #     break            
        samples = samples.to(device)

        # Prepare kwargs:
        # Note that we assert batch_size == 1
        img_id = str(targets[0]['image_id'].item())
        if img_id not in rels_for_coco.keys():
            continue
        rels_for_img = rels_for_coco[img_id]
        if len(rels_for_img) <= 0:
            continue
        rels_in_all_idx = [rel_idx for rel_idx, rel in enumerate(all_rels) if rel in rels_for_img]
        rel_text_memory_resized_img = rel_text_memory_resized[rels_in_all_idx]
        text_memory_resized = torch.cat((object_text_memory_resized, rel_text_memory_resized_img), dim = 0)
        obj_pred_names_sums = torch.tensor([[len(object_text_memory_resized), len(rel_text_memory_resized_img)]])
        text_attention_mask = torch.zeros(text_memory_resized.shape[:2], device = device).bool()
        text = (text_attention_mask, text_memory_resized, obj_pred_names_sums)
        kwargs = {'text':text}

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
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if outputs['pred_verb_logits'].shape[2] == len(verb_text) + 1:
            outputs['pred_verb_logits'] = outputs['pred_verb_logits'][:,:,:-1]
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        results = filter_by_gt_object_annotations(results, targets, flat_text, all_rels)
        result_list += results
        # result_dict.update({p:r for p, r in zip(one_batch_path, results)})
        # result_dict.update({img_id: results[0]}) # because by default, batch_size == 1.

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return result_list


def filter_by_gt_object_annotations(results, targets, flat_text, all_rels, verb_thre = 0.005):
    ### Filter out invalid triplets if they are not overlapped with the gt subjects and objects.
    obj_text = flat_text[:-len(all_rels)]
    rel_text = all_rels
    object_cat_to_name = load_hico_object_dict()

    filtered_results = []
    for result, target in zip(results, targets):
        filtered_result = {}
        result_bbox = [{'category_id': obj_text[l.item()], 'bbox': b} for l, b in zip(result['labels'], result['boxes'])]
        target_bbox = [{'category_id': object_cat_to_name[l.item()], 'bbox': b} for l, b in zip(target['labels'], result['boxes'])]
        # print(result_bbox)
        # print(target_bbox)
        match_pairs_dict, match_pair_overlaps = compute_iou_mat(result_bbox, target_bbox)

        match_pair_overlaps_results = {i:[] for i in range(len(result_bbox))}
        for tar_id, res_id_list in match_pairs_dict.items():
            for res_id in res_id_list:
                match_pair_overlaps_results[res_id].append(tar_id)
        # print(match_pair_overlaps_results)

        filtered_rels = []
        relationship_id = 0
        valid_pair_ids, valid_rel_ids = torch.where(result['verb_scores'] >= verb_thre)
        for verb_idx, (valid_pair_id, valid_rel_id) in enumerate(zip(valid_pair_ids, valid_rel_ids)):
            sub_id = result['sub_ids'][valid_pair_id.item()]
            obj_id = result['obj_ids'][valid_pair_id.item()]
            if len(match_pair_overlaps_results[sub_id.item()]) > 0 and \
                        len(match_pair_overlaps_results[obj_id.item()]) > 0:
                filtered_rels.append(
                        {
                            "relationship_id": relationship_id,
                            "predicate": rel_text[valid_rel_id.item()],
                            "subject_id": int(match_pair_overlaps_results[sub_id.item()][0]),
                            "object_id": int(match_pair_overlaps_results[obj_id.item()][0]),
                            "confidence": result['verb_scores'][valid_pair_id][valid_rel_id].item(),
                        }
                    )
                relationship_id = relationship_id + 1

        # print(len(filtered_rels))
        filtered_result['relationships'] = filtered_rels
        filtered_result['objects'] = transform_coco_bbox_to_VG_format(target_bbox)
        filtered_result['image_id'] = str(target['image_id'].item())
        filtered_result['dataset'] = "coco2017"
        filtered_result['data_split'] = "train2017"
        filtered_results.append(filtered_result)
        

    return filtered_results
        

# Our Structure of scene_graph.json annotation
# {"image_id": 2407890,
# "objects": [...
#     {"object_id": 1023838, "x": 324, "y": 320, "w": 142, "h": 255, 
#         "names": "cat","synsets": ["cat.n.01"]},
#     {"object_id":  5071, "x": 359, "y": 362, "w": 72, "h": 81,
#         "names": "table", "synsets": ["table.n.01"]},
# ...],
# "relationships": [...
#     {"relationship_id": 15947, "predicate": "wears", "synsets": ["wear.v.01"],
#     "subject_id": 1023838, "object_id":  5071,
#     }
# ...]}


def transform_coco_bbox_to_VG_format(coco_bbox):
    '''
    This function transforms the bbox annotations of COCO dataset to the VG format.
    Args:
        coco_bbox (list): a list of the COCO format.
    
    Returns:
        vg_bbox (list) : a list of the COCO format.
    '''
    vg_bbox = []
    for bbox_idx, bbox in enumerate(coco_bbox):
        # if bbox["bbox"][2] > 0 and bbox["bbox"][3] > 0:   # This is to ensure boxes are valid.
        vg_bbox.append(
            {
                "object_id": bbox_idx,
                "x": bbox["bbox"][0].item(),
                "y": bbox["bbox"][1].item(),
                "w": (bbox["bbox"][2] - bbox["bbox"][0]).item(),
                "h": (bbox["bbox"][3] - bbox["bbox"][1]).item(),
                "names": bbox["category_id"],
            }
        )

    return vg_bbox




def compute_iou_mat(bbox_list1, bbox_list2, overlap_iou = 0.5):
    # gt_bboxes, pred_bboxes
    iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
    if len(bbox_list1) == 0 or len(bbox_list2) == 0:
        return {}
    for i, bbox1 in enumerate(bbox_list1):
        for j, bbox2 in enumerate(bbox_list2):
            iou_i = compute_IOU(bbox1, bbox2)
            iou_mat[i, j] = iou_i

    iou_mat_ov=iou_mat.copy()
    iou_mat[iou_mat>=overlap_iou] = 1
    iou_mat[iou_mat<overlap_iou] = 0

    match_pairs = np.nonzero(iou_mat) # return gt index array and pred index array
    match_pairs_dict = {}
    match_pair_overlaps = {}
    if iou_mat.max() > 0: # if there is a matched pair
        for i, pred_id in enumerate(match_pairs[1]):
            if pred_id not in match_pairs_dict.keys():
                match_pairs_dict[pred_id] = []
                match_pair_overlaps[pred_id]=[]
            match_pairs_dict[pred_id].append(match_pairs[0][i])
            match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
    return match_pairs_dict, match_pair_overlaps
    # dict like:
    # match_pairs_dict {pred_id: [gt_id], pred_id: [gt_id], ...}
    # match_pair_overlaps {pred_id: [gt_id], pred_id: [gt_id], ...} 
    # we may have many gt_ids for a specific pred_id, because we don't consider the class

def compute_IOU(bbox1, bbox2):
    # if isinstance(bbox1['category_id'], str):
    #     bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
    # if isinstance(bbox2['category_id'], str):
    #     bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
    if bbox1['category_id'] == bbox2['category_id']:
        rec1 = bbox1['bbox']
        rec2 = bbox2['bbox']
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
        S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
            return intersect / (sum_area - intersect)
    else:
        return 0



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


def load_hico_object_dict(file_path = 'datasets/hico_object_names.txt') -> List[list]:
    object_names = []
    with open(file_path, 'r') as f:
        object_names = json.load(f)
    object_cat_to_name = {cat:name for name, cat in object_names.items()}
    return object_cat_to_name


def aggregate_rels_for_dataset():
    paraphrases_rel_texts_for_coco = Path('/mnt/data-nas/peizhi/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_caption_nucleus10_trainval2017_Paraphrases_rel_texts_for_coco_images.json')
    with open(paraphrases_rel_texts_for_coco, 'r') as f:
        rels_for_coco_pairs = json.load(f)
        # example: [[[[4, 0], [4, 1], [4, 2], [7, 4]], ['with']]]
    
    all_rels = []
    rels_for_coco = {}
    for img_id, pairs in rels_for_coco_pairs.items():
        rel_text = []
        for pair in pairs:
            rel_text += pair[1]
        
        rels_for_coco[img_id] = rel_text
        for rel in rel_text:
            if rel not in all_rels:
                all_rels.append(rel)

    return all_rels, rels_for_coco





class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target



class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def transform_coco_official_to_VG_format(Coco):
    '''
    Args:
        Coco (class): This is a class to produce coco annotations.
    
    Returns:
        official_bbox_dict (dict): a dict of annotations in the VG format.
    '''
    object_80_dict = load_hico_object_txt()
    official_bbox_dict = {}
    coco_start_obj_idx = 10000000
    for idx, coco_data in enumerate(Coco):
        vg_bbox = []
        coco_img, coco_target = coco_data

        num_obj = coco_target['boxes'].shape[0]   # in the format of cxcywh
        coco_bbox = coco_target['boxes']
        labels = [int(l) for l in coco_target['labels']]
        box_labels = coco_target['labels']
        ### According to coco.py, target["orig_size"] = torch.as_tensor([int(h), int(w)]).
        img_h, img_w = coco_target['orig_size']

        for i in range(num_obj):
            vg_bbox.append({
                "x": (coco_bbox[i][0] - coco_bbox[i][2]/2.)*img_w,
                "y": (coco_bbox[i][1] - coco_bbox[i][3]/2.)*img_h,
                "w": coco_bbox[i][2]*img_w,
                "h": coco_bbox[i][3]*img_h,
                "object_id": coco_start_obj_idx,
                "names": object_80_dict[box_labels[i].item()],
            })
            coco_start_obj_idx+=1

        official_bbox_dict[str(coco_target['image_id'].item())] = vg_bbox
        # print(str(coco_target['image_id'].item()))

    return official_bbox_dict


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