# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import copy
import pickle
import json
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from datasets.vcoco import build as build_dataset
from datasets.coco import build_cocorel as build_dataset_cocorel
from datasets.o365 import build_o365rel as build_dataset_o365rel
from models.backbone import build_backbone
from models.DDETR_backbone import build_backbone as build_DDETR_backbone
from models.swin.backbone import build_backbone as build_Swin_backbone
from models.transformer import build_transformer
import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from models.hoi import OCN, ParSeD, ParSe, RLIP_ParSe, RLIP_ParSeD
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import os
import pdb
# pdb.set_trace()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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


class PostProcessHOI(nn.Module):

    def __init__(self, num_queries, subject_category_id, correct_mat):
        super().__init__()
        self.max_hois = 100

        self.num_queries = num_queries
        self.subject_category_id = subject_category_id

        correct_mat = np.concatenate((correct_mat, np.ones((correct_mat.shape[0], 1))), axis=1)
        self.register_buffer('correct_mat', torch.from_numpy(correct_mat))

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(b.to('cpu').numpy(), l.to('cpu').numpy())]

            hoi_scores = vs * os.unsqueeze(1)

            verb_labels = torch.arange(hoi_scores.shape[1], device=self.correct_mat.device).view(1, -1).expand(
                hoi_scores.shape[0], -1)
            object_labels = ol.view(-1, 1).expand(-1, hoi_scores.shape[1])
            masks = self.correct_mat[verb_labels.reshape(-1), object_labels.reshape(-1)].view(hoi_scores.shape)
            hoi_scores *= masks

            ids = torch.arange(b.shape[0])

            hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                    subject_id, object_id, category_id, score in zip(ids[:ids.shape[0] // 2].to('cpu').numpy(),
                                                                     ids[ids.shape[0] // 2:].to('cpu').numpy(),
                                                                     verb_labels.to('cpu').numpy(), hoi_scores.to('cpu').numpy())]

            results.append({
                'predictions': bboxes,
                'hoi_prediction': hois
            })

        return results


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--img_inference_batch', default=16, type=int)

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
    parser.add_argument('--drop_path_rate', default=0.2, type=float,
                        help="drop_path_rate applied in the Swin transformer.")
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
    parser.add_argument('--pretrained_swin', type=str, default='',
                        help='Pretrained model path for the swin backbone only!!!')
    parser.add_argument('--use_checkpoint', action='store_true') # This is for Swin-transformer to save memory.
    parser.add_argument('--vg_rel_texts_for_coco_images', type=str, required=False)
    parser.add_argument('--vg_rel_texts_for_o365_images', type=str, required=False)
    parser.add_argument('--vg_rel_texts_for_hico_objects_path', type=str, required=False)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--save_keep_names_freq_path', type=str, required=False)
    parser.add_argument('--save_filtering_anno_path', type=str, required=False)
    parser.add_argument('--save_filtering_keep_names_freq_path', type=str, required=False)
    parser.add_argument('--save_filtering_ood_path', type=str, required=False)
    parser.add_argument('--merge_keep_names_freq_path_1', type=str, required=False)
    parser.add_argument('--merge_keep_names_freq_path_2', type=str, required=False)
    parser.add_argument('--merge_keep_names_freq_path_save', type=str, required=False)
    parser.add_argument('--merge_rel_det_annos_path_1', type=str, required=False)
    parser.add_argument('--merge_rel_det_annos_path_2', type=str, required=False)
    parser.add_argument('--merge_rel_det_annos_path_save', type=str, required=False)

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

    # RLIP v2 and Verb Tagger
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--o365_path', type=str)
    parser.add_argument('--o365_segment', default=0, type=int, help='number of feature levels')
    parser.add_argument('--RLIP_ParSeD_v2', action = 'store_true',
                        help='RLIP_ParSeD_v2.') 
    parser.add_argument('--RLIP_ParSeDA_v2', action = 'store_true',
                        help='RLIP_ParSeDA_v2.') 
    parser.add_argument('--RLIP_ParSe_v2', action = 'store_true',
                        help='RLIP_ParSe_v2.') 
    parser.add_argument('--fusion_type', default = "no_fusion", choices = ("MDETR_attn", "GLIP_attn", "no_fusion"), )
    parser.add_argument('--fusion_interval', default=1, type=int, help="Fusion interval in VLFuse.")
    parser.add_argument('--fusion_last_vis', default=False, action='store_true', help = 'Whether to fuse the last layer of the vision features.')
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")
    parser.add_argument("--verb_tagger", dest="verb_tagger", action="store_true", help="Whether to perform verb tagging pre-training")
    parser.add_argument('--stable_softmax_2d', default=False, action='store_true', help = 'Use "attn_weights = attn_weights - attn_weights.max()" during BiMultiHeadAttention in VLFuse.')
    parser.add_argument('--clamp_min_for_underflow', default=False, action='store_true', help = 'Clamp attention weights (before softmax) during BiMultiHeadAttention in VLFuse.')
    parser.add_argument('--clamp_max_for_overflow', default=False, action='store_true', help = 'Clamp attention weights (before softmax) during BiMultiHeadAttention in VLFuse.')
    parser.add_argument('--separate_bidirectional', default=False, action='store_true', help = 'For GLIP_attn, we perform separate attention for different levels of features.')
    parser.add_argument('--lang_aux_loss', default=False, action='store_true', help = 'Whether to use aux loss to calculate the loss functions.')
    parser.add_argument('--label_noise_scale', default=0.2, type=float,
                        help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float,
                        help="box noise scale to shift and scale")
    parser.add_argument('--use_checkpoint_fusion', default=False, action='store_true', help = 'Use checkpoint to save memory.')
    
    parser.add_argument('--relation_threshold', default=0.2, type=float)
    parser.add_argument('--gating_mechanism', default="GLIP", type=str,
                        choices=["GLIP", "Vtanh", "Etanh", "Stanh", "SDFtanh", "SFtanh", "SOtanh", "SXAc", "SDFXAc", "VXAc", "SXAcLN", "SDFXAcLN", "SDFOXAcLN", "MBF"],
                        help = "The gating mechanism used to perform language-vision feature fusion.")
    parser.add_argument('--verb_query_tgt_type', default="vanilla", type=str,
                        choices=["vanilla", "MBF", "vanilla_MBF"],
                        help = "The gating mechanism used to perform language-vision feature fusion.")

    return parser


def main_tagger(args):
    device = torch.device(args.device)

    if args.dataset_file == "coco":
        dataset_tag = build_dataset_cocorel(image_set='tagger', args=args)
        dataset_tag_val = build_dataset_cocorel(image_set='tagger_val', args=args)
    elif args.dataset_file == "o365_det":
        dataset_tag = build_dataset_o365rel(image_set='tagger', args=args)
        dataset_tag_val = build_dataset_o365rel(image_set='tagger_val', args=args)

    sampler_tag = torch.utils.data.SequentialSampler(dataset_tag)
    sampler_tag_val = torch.utils.data.SequentialSampler(dataset_tag_val)

    data_loader_tag = DataLoader(dataset_tag, args.batch_size, sampler = sampler_tag,
                                 drop_last = False, collate_fn = utils.collate_fn, num_workers = args.num_workers)
    data_loader_tag_val = DataLoader(dataset_tag_val, args.batch_size, sampler = sampler_tag_val,
                                 drop_last = False, collate_fn = utils.collate_fn, num_workers = args.num_workers)

    args.lr_backbone = 0
    args.masks = False

    if 'swin' in args.backbone:
        backbone = build_Swin_backbone(args)
    elif args.DDETRHOI or args.ParSeD or args.RLIP_ParSeD or args.RLIP_ParSeD_v2:
        backbone = build_DDETR_backbone(args)
    else:
        backbone = build_backbone(args)
    transformer = build_transformer(args)

    if args.RLIP_ParSeD or args.RLIP_ParSeD_v2:
        # model = RLIP_ParSeD(
        #     backbone,
        #     transformer,
        #     num_queries=args.num_queries,
        #     num_feature_levels=args.num_feature_levels,
        #     # aux_loss=args.aux_loss,
        #     with_box_refine=args.with_box_refine,
        #     two_stage=args.two_stage,
        #     subject_class = args.subject_class,
        #     # verb_curing=args.verb_curing,
        #     args=args,
        # )
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
                # masked_entity_modeling=args.masked_entity_modeling,
                # pseudo_verb=args.pseudo_verb,
                # matcher=matcher if args.masked_entity_modeling else None,
                args=args,
            )
        print('Building RLIP_ParSeD...')
    else:
        print('Verb tagger does not support the specified model')
        assert False
    
    model.to(device)

    checkpoint = torch.load(args.param_path, map_location='cpu')
    load_info = model.load_state_dict(checkpoint['model'])
    print('Loading Info: ' + str(load_info))

    if hasattr(model.transformer, 'text_encoder'):
        # detections = verb_tagger_with_text(model, post_processor, data_loader_tag, dataset_tag, device, verb_classes, args.missing_category_id, args)
        post_processor = None
        if args.dataset_file == "coco":
            data_split_train = "train2017"
            data_split_val = "val2017"
            detections_val = verb_tagger_with_text(model, post_processor, data_loader_tag_val, dataset_tag_val, device, args,
                                            data_split = data_split_val,
                                            relation_threshold = args.relation_threshold,
                                            img_inference_batch = args.img_inference_batch)
            detections = verb_tagger_with_text(model, post_processor, data_loader_tag, dataset_tag, device, args,
                                           data_split = data_split_train,
                                           relation_threshold = args.relation_threshold,
                                           img_inference_batch = args.img_inference_batch)
        elif args.dataset_file == "o365_det":
            data_split_train = "train"
            data_split_val = "val"
            detections_val = []
            detections = []
            if args.o365_segment in [1,2,3]: # should be [1,2,3]!!!!!!!!!!!!! 第一个地方！！
                detections_val = []
            elif args.o365_segment == 4:
                detections_val = verb_tagger_with_text_o365(model, post_processor, data_loader_tag_val, dataset_tag_val, device, args,
                                            data_split = data_split_val,
                                            relation_threshold = args.relation_threshold,
                                            img_inference_batch = args.img_inference_batch)
            else:
                assert False
            detections = verb_tagger_with_text_o365(model, post_processor, data_loader_tag, dataset_tag, device, args,
                                           data_split = data_split_train,
                                           relation_threshold = args.relation_threshold,
                                           img_inference_batch = args.img_inference_batch)


        # if args.dataset_file == "o365_det" and args.o365_segment in [1,2,3]:
        #     detections_val = []
        # else:
        #     detections_val = verb_tagger_with_text(model, post_processor, data_loader_tag_val, dataset_tag_val, device, args,
        #                                     data_split = data_split_val,
        #                                     relation_threshold = args.relation_threshold,
        #                                     img_inference_batch = args.img_inference_batch)
        # detections = verb_tagger_with_text(model, post_processor, data_loader_tag, dataset_tag, device, args,
        #                                    data_split = data_split_train,
        #                                    relation_threshold = args.relation_threshold,
        #                                    img_inference_batch = args.img_inference_batch)
        detections = detections + detections_val

        with open(args.save_path,'w') as f:
            json.dump(detections, f)
            print(f"Saving to {args.save_path}.")
        

    # with open(args.save_path, 'wb') as f:
    #     pickle.dump(detections, f, protocol=2)


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                     24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                     37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                     48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                     58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                     72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                     82, 84, 85, 86, 87, 88, 89, 90)

    verb_classes = ['hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk', 'look_obj', 'hit_instr', 'hit_obj',
                    'eat_obj', 'eat_instr', 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj',
                    'throw_obj', 'catch_obj', 'cut_instr', 'cut_obj', 'run', 'work_on_computer_instr',
                    'ski_instr', 'surf_instr', 'skateboard_instr', 'smile', 'drink_instr', 'kick_obj',
                    'point_instr', 'read_obj', 'snowboard_instr']

    device = torch.device(args.device)

    dataset_val = build_dataset(image_set='val', args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    assert args.batch_size == 1
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler = sampler_val,
                                 drop_last = False, collate_fn = utils.collate_fn, num_workers = args.num_workers)

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
            num_obj_classes = len(valid_obj_ids) + 1,
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
        model = DETRHOI(backbone, transformer, len(valid_obj_ids) + 1, len(verb_classes),
                        args.num_queries)
    post_processor = PostProcessHOI(args.num_queries, args.subject_category_id, dataset_val.correct_mat)
    model.to(device)
    post_processor.to(device)

    checkpoint = torch.load(args.param_path, map_location='cpu')
    load_info = model.load_state_dict(checkpoint['model'])
    print('Loading Info: ' + str(load_info))

    if not hasattr(model.transformer, 'text_encoder'):
        detections = generate(model, post_processor, data_loader_val, device, verb_classes, args.missing_category_id)
    else:
        detections = generate_with_text(model, post_processor, data_loader_val, dataset_val, device, verb_classes, args.missing_category_id, args)

    with open(args.save_path, 'wb') as f:
        pickle.dump(detections, f, protocol=2)


@torch.no_grad()
def verb_tagger_with_text(model, post_processor, data_loader_tag, dataset_tag, device, args, data_split,
                          relation_threshold = 0.2, img_inference_batch = 16):
    '''
    :param: img_inference_batch: This parameter decides the batch size when inferring on one image. 
                                 (One image might have a large number of groups to infer relations.) 
    
    '''
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate:'

    coco_obj_dict = load_hico_object_txt()
    coco_obj = [j for i,j in coco_obj_dict.items()]
    coco_obj_label_to_80 = torch.empty((90 + 1, )).fill_(torch.inf).long()
    for idx, (i,_) in enumerate(coco_obj_dict.items()):
        coco_obj_label_to_80[i] = idx
    print(coco_obj_label_to_80)

    # Read in the coco dataset for object detection
    # coco_json = '/mnt/data-nas/peizhi/data/coco2017/annotations/instances_train2017.json'
    # with open(coco_json, "r") as f:
    #     coco_annos = json.load(f)
    coco_start_obj_idx = 10000000
    coco_start_rel_idx = 10000000

    start_time = time.time()
    coco_pseudo_relation_annos = []
    for img_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader_tag, 100, header)):
        ### Skip images (just for debugging)
        # if img_idx < 1600:
        #     continue

        samples = samples.to(device)
        # targets[0]: dict_keys(['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size', 'relation_candidates'])
        image_id = targets[0]['image_id']
        relation_candidates = targets[0]['relation_candidates']
        boxes = targets[0]['boxes'] # in the format of cxcywh
        box_labels = targets[0]['labels']
        ### According to coco.py, target["orig_size"] = torch.as_tensor([int(h), int(w)]).
        # img_w, img_h = targets[0]['orig_size'] 
        img_h, img_w = targets[0]['orig_size']


        box_labels_80 = coco_obj_label_to_80[box_labels]

        ### Test reading in images.
        # img_file_name = str(targets[0]['image_id'].item()).zfill(12) + '.jpg'
        # coco2017_img_folder = Path('/mnt/data-nas/peizhi/data/coco2017/train2017')
        # img = Image.open(coco2017_img_folder / img_file_name).convert('RGB')
        # w, h = img.size
        # print(f'Size of the read-in image: {w, h}')

        ### Prepare the text inputs and targets
        kwargs_list = []
        rel_list = [] # This is stored to tag relation labels.
        rel_text_merge = []
        sub_boxes = []
        obj_boxes = []
        sub_labels = []
        obj_labels = []
        for idx, (pair_i, rel_i) in enumerate(relation_candidates):
            ### Merge the texts if we have img_inference_batch > 1.
            rel_text_merge = merge_list_b_to_list_a(rel_text_merge, rel_i)

            sub_idx = [s for s,o in pair_i]
            obj_idx = [o for s,o in pair_i]
            sub_boxes.append(boxes[sub_idx].to(device))
            obj_boxes.append(boxes[obj_idx].to(device))
            sub_labels.append(box_labels_80[sub_idx].to(device))
            obj_labels.append(box_labels_80[obj_idx].to(device))

            if (idx+1) % img_inference_batch == 0:
                ### Prepare the text inputs
                if args.use_no_obj_token:
                    obj_pred_names_sums = torch.tensor([[len(coco_obj) + 1, len(rel_text_merge)]])
                    flat_text = coco_obj + ['no objects'] + rel_text_merge
                else:
                    obj_pred_names_sums = torch.tensor([[len(coco_obj), len(rel_text_merge)]])
                    flat_text = coco_obj + rel_text_merge
                flat_tokenized = model.transformer.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
                # tokenizer: dict_keys(['input_ids', 'attention_mask'])
                #            'input_ids' shape: [text_num, max_token_num]
                #            'attention_mask' shape: [text_num, max_token_num]
                encoded_flat_text = model.transformer.text_encoder(**flat_tokenized)
                text_memory = encoded_flat_text.pooler_output
                if args.fusion_type == 'GLIP_attn':
                    text_memory_resized = text_memory
                else:
                    text_memory_resized = model.transformer.resizer(text_memory)
                current_bs = img_inference_batch
                text_memory_resized = text_memory_resized.unsqueeze(dim = 1).repeat(1, current_bs, 1)
                text_attention_mask = torch.zeros(text_memory_resized.shape[:2], device = device).bool()
                text = (text_attention_mask, text_memory_resized, obj_pred_names_sums)

                # if args.batch_size != samples.tensors.shape[0]:
                # text_memory_resized_short = text_memory_resized[: , :samples.tensors.shape[0]]
                # text_attention_mask_short = text_attention_mask[: , :samples.tensors.shape[0]]
                # text = (text_attention_mask_short, text_memory_resized_short, obj_pred_names_sums)
                # kwargs = {'text': text}
                
                targets = []
                for group_sub_boxes, group_obj_boxes, group_sub_labels, group_obj_labels in zip(sub_boxes, obj_boxes, sub_labels, obj_labels):
                    targets.append({'sub_boxes':group_sub_boxes,
                                    'obj_boxes':group_obj_boxes,
                                    'sub_labels':group_sub_labels,
                                    'obj_labels':group_obj_labels})
                kwargs_list.append({'text':text, 'targets':targets})
                rel_list.append(rel_text_merge)
                
                rel_text_merge = []
                sub_boxes = []
                obj_boxes = []
                sub_labels = []
                obj_labels = []
        
        ### Prepare the text inputs
        left_bs = len(sub_boxes)
        if left_bs > 0:
            if args.use_no_obj_token:
                obj_pred_names_sums = torch.tensor([[len(coco_obj) + 1, len(rel_text_merge)]])
                flat_text = coco_obj + ['no objects'] + rel_text_merge
            else:
                obj_pred_names_sums = torch.tensor([[len(coco_obj), len(rel_text_merge)]])
                flat_text = coco_obj + rel_text_merge
            flat_tokenized = model.transformer.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
            # tokenizer: dict_keys(['input_ids', 'attention_mask'])
            #            'input_ids' shape: [text_num, max_token_num]
            #            'attention_mask' shape: [text_num, max_token_num]
            encoded_flat_text = model.transformer.text_encoder(**flat_tokenized)
            text_memory = encoded_flat_text.pooler_output
            if args.fusion_type == 'GLIP_attn':
                text_memory_resized = text_memory
            else:
                text_memory_resized = model.transformer.resizer(text_memory)
            
            text_memory_resized = text_memory_resized.unsqueeze(dim = 1).repeat(1, left_bs, 1)
            text_attention_mask = torch.zeros(text_memory_resized.shape[:2], device = device).bool()
            text = (text_attention_mask, text_memory_resized, obj_pred_names_sums)

            # if args.batch_size != samples.tensors.shape[0]:
            # text_memory_resized_short = text_memory_resized[: , :samples.tensors.shape[0]]
            # text_attention_mask_short = text_attention_mask[: , :samples.tensors.shape[0]]
            # text = (text_attention_mask_short, text_memory_resized_short, obj_pred_names_sums)
            # kwargs = {'text': text}
            
            targets = []
            for group_sub_boxes, group_obj_boxes, group_sub_labels, group_obj_labels in zip(sub_boxes, obj_boxes, sub_labels, obj_labels):
                targets.append({'sub_boxes':group_sub_boxes,
                        'obj_boxes':group_obj_boxes,
                        'sub_labels':group_sub_labels,
                        'obj_labels':group_obj_labels})
            kwargs_list.append({'text':text, 'targets':targets})
            rel_list.append(rel_text_merge)


        ### Inference on each group of triplets
        outputs_img = []
        for group_i_kwargs in kwargs_list:
            current_bs = len(group_i_kwargs['targets'])
            group_i_samples = NestedTensor(
                        tensors = samples.tensors.repeat(current_bs, 1, 1, 1),
                        mask = samples.mask.repeat(current_bs, 1, 1))
            memory_cache = model(group_i_samples, encode_and_save=True, **group_i_kwargs)
            outputs = model(group_i_samples, encode_and_save=False, memory_cache=memory_cache, **group_i_kwargs)
            # outputs: dict_keys(['pred_obj_logits', 'pred_verb_logits', 'pred_sub_boxes', 'pred_obj_boxes', 'aux_outputs'])
            outputs_img.append(outputs)


        ### check the query generation of verb_tagger_helper.py 
        ### 检查一下是否box的格式和VGRel的是不是相同的: YEs.
        ### 检查一下one_box 的格式是不是x1 y1 x2 y2: No. cxcywh

        ### Post-process outputs
        img_pseudo_relation_annos = {}
        img_pseudo_relation_annos["image_id"] = image_id.item()
        img_pseudo_relation_annos["dataset"] = 'coco2017'
        img_pseudo_relation_annos["data_split"] = data_split
        img_pseudo_relation_annos["objects"] = []
        for one_box, one_label in zip(boxes, box_labels):   # in the format of cxcywh
            one_obj = {"object_id": coco_start_obj_idx, 
                       "x": ((one_box[0] - one_box[2]/2.) * img_w).item(),
                       "y": ((one_box[1] - one_box[3]/2.) * img_h).item(),
                       "w": ((one_box[2]) * img_w).item(),
                       "h": ((one_box[3]) * img_h).item(),
                       "names": coco_obj_dict[int(one_label)]
                    }
            # print(one_obj)
            img_pseudo_relation_annos["objects"].append(one_obj)
            coco_start_obj_idx += 1

        img_pseudo_relation_annos["relationships"] = []
        bs_global = 0
        if len(rel_list)>1:
            print([len(l) for l in rel_list])
        assert len(outputs_img) == len(rel_list)
        for outputs, rel_text_merge in zip(outputs_img, rel_list):
            verbs = outputs['pred_verb_logits'].sigmoid()
            bs_idx_list, pair_idx_list, verb_idx_list = torch.where(verbs > relation_threshold)
            for bs_idx, pair_idx, verb_idx in zip(bs_idx_list, pair_idx_list, verb_idx_list):
                pair_i, rel_i = relation_candidates[bs_global + bs_idx.item()]
                
                num_possible_pairs = len(pair_i)
                if pair_idx.item() < num_possible_pairs:
                    sub_idx = pair_i[pair_idx.item()][0]
                    obj_idx = pair_i[pair_idx.item()][1]
                    # if verb_idx.item()>=len(rel_text_merge):
                    #     print(kwargs_list[0]['text'][1].shape, kwargs_list[1]['text'][1].shape)
                    #     print(len(rel_list[0]), len(rel_list[1]))
                    #     print(verb_idx.item(), len(rel_text_merge))
                    #     print(kwargs_list[0]['text'][2], kwargs_list[1]['text'][2])
                    #     print(len(outputs_img))
                    one_rel = {
                        "relationship_id": coco_start_rel_idx,
                        "predicate": rel_text_merge[verb_idx.item()],
                        "subject_id": img_pseudo_relation_annos["objects"][sub_idx]["object_id"],
                        "object_id": img_pseudo_relation_annos["objects"][obj_idx]["object_id"],
                        "confidence": verbs[bs_idx, pair_idx, verb_idx].item(),
                    }
                    img_pseudo_relation_annos["relationships"].append(one_rel)
                    coco_start_rel_idx += 1
            bs_global += verbs.shape[0]
        
        coco_pseudo_relation_annos.append(img_pseudo_relation_annos)

        # if img_idx == 0:
        #     break

        # TODO: TODO: TODO:
        # 增加代码支持一部分数据推理完之后就保存一部分，不要全部推理完才可以保存。
        # 一个比较简单的实现就是把下面保存的代码放到这里来，加上img_idx的条件
    
    print(f'Processing time: {int(time.time() - start_time)} seconds.')
    
    ### Save to disk.
    # with open(args.save_path,'w') as f:
    #     json.dump(coco_pseudo_relation_annos, f)
    return coco_pseudo_relation_annos

    ### Inference statistics of the COCO dataset
    # RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_greater0_v3.json
    # Generate:  [118286/118287]  eta: 0:00:00    time: 0.5514  data: 0.0041  max mem: 21979
    # Generate: Total time: 7:31:35 (0.2291 s / it)
    # Processing time: 27095 seconds.
    # vg_rel_texts_for_coco_images_greater5_v3.json
    # Generate:  [118286/118287]  eta: 0:00:00    time: 0.2034  data: 0.0023  max mem: 8428
    # Generate: Total time: 6:24:59 (0.1953 s / it)
    # Processing time: 23099 seconds.
    # vg_rel_texts_for_coco_images_greater5_v2.json
    # Generate:  [118286/118287]  eta: 0:00:00    time: 0.1228  data: 0.0021  max mem: 8097
    # Generate: Total time: 4:00:15 (0.1219 s / it)
    # Processing time: 14415 seconds

@torch.no_grad()
def verb_tagger_with_text_o365(model, post_processor, data_loader_tag, dataset_tag, device, args, data_split,
                               relation_threshold = 0.2, img_inference_batch = 16):
    '''
    :param: img_inference_batch: This parameter decides the batch size when inferring on one image. 
                                 (One image might have a large number of groups to infer relations.) 
    
    '''
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate:'


    o365_obj_dict = load_o365_categories()
    o365_obj = [j for i,j in o365_obj_dict.items()]
    o365_obj_label_to_365 = torch.empty((365 + 1, )).fill_(torch.inf).long()
    for idx, (i,_) in enumerate(o365_obj_dict.items()):
        o365_obj_label_to_365[i] = idx
    print(o365_obj_label_to_365)


    # Read in the coco dataset for object detection
    # coco_json = '/mnt/data-nas/peizhi/data/coco2017/annotations/instances_train2017.json'
    # with open(coco_json, "r") as f:
    #     coco_annos = json.load(f)
    o365_start_obj_idx = 20000000
    o365_start_rel_idx = 20000000

    start_time = time.time()
    coco_pseudo_relation_annos = []
    for img_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader_tag, 1000, header)):
        ### Skip images (just for debugging)
        # if img_idx < 1600:
        #     continue
        # if img_idx == 1000:
        #     break

        samples = samples.to(device)
        # targets[0]: dict_keys(['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size', 'relation_candidates'])
        image_id = targets[0]['image_id']

        relation_candidates = targets[0]['relation_candidates']
        boxes = targets[0]['boxes'] # in the format of cxcywh
        box_labels = targets[0]['labels']
        ### According to coco.py, target["orig_size"] = torch.as_tensor([int(h), int(w)]).
        # img_w, img_h = targets[0]['orig_size'] 
        img_h, img_w = targets[0]['orig_size']
        if check_list_out_of_bounds(relation_candidates, boxes):
            print(f"{image_id} is our-of-bounds.")
            continue


        box_labels_365 = o365_obj_label_to_365[box_labels]

        ### Test reading in images.
        # img_file_name = str(targets[0]['image_id'].item()).zfill(12) + '.jpg'
        # coco2017_img_folder = Path('/mnt/data-nas/peizhi/data/coco2017/train2017')
        # img = Image.open(coco2017_img_folder / img_file_name).convert('RGB')
        # w, h = img.size
        # print(f'Size of the read-in image: {w, h}')

        ### Prepare the text inputs and targets
        kwargs_list = []
        rel_list = [] # This is stored to tag relation labels.
        rel_text_merge = []
        sub_boxes = []
        obj_boxes = []
        sub_labels = []
        obj_labels = []
        for idx, (pair_i, rel_i) in enumerate(relation_candidates):
            ### Merge the texts if we have img_inference_batch > 1.
            rel_text_merge = merge_list_b_to_list_a(rel_text_merge, rel_i)

            # Guard against index is out of bounds for 'sub_boxes.append(boxes[sub_idx].to(device))' and 'obj_boxes.append(boxes[obj_idx].to(device))'
            # sub_idx = [s for s,o in pair_i]
            # obj_idx = [o for s,o in pair_i]
            sub_idx = []
            obj_idx = []
            for s,o in pair_i:
                if s < len(boxes) and o < len(boxes):
                    sub_idx.append(s)
                    obj_idx.append(o)
                else:
                    print(f"{image_id} has out-of-bounds sub_idx or obj_idx. len(boxes)={len(boxes)}, s={s}, o={o}")
                    continue

            sub_boxes.append(boxes[sub_idx].to(device))
            obj_boxes.append(boxes[obj_idx].to(device))
            sub_labels.append(box_labels_365[sub_idx].to(device))
            obj_labels.append(box_labels_365[obj_idx].to(device))

            if (idx+1) % img_inference_batch == 0:
                ### Prepare the text inputs
                if args.use_no_obj_token:
                    obj_pred_names_sums = torch.tensor([[len(o365_obj) + 1, len(rel_text_merge)]])
                    flat_text = o365_obj + ['no objects'] + rel_text_merge
                else:
                    obj_pred_names_sums = torch.tensor([[len(o365_obj), len(rel_text_merge)]])
                    flat_text = o365_obj + rel_text_merge
                flat_tokenized = model.transformer.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
                # tokenizer: dict_keys(['input_ids', 'attention_mask'])
                #            'input_ids' shape: [text_num, max_token_num]
                #            'attention_mask' shape: [text_num, max_token_num]
                encoded_flat_text = model.transformer.text_encoder(**flat_tokenized)
                text_memory = encoded_flat_text.pooler_output
                if args.fusion_type == 'GLIP_attn':
                    text_memory_resized = text_memory
                else:
                    text_memory_resized = model.transformer.resizer(text_memory)
                current_bs = img_inference_batch
                text_memory_resized = text_memory_resized.unsqueeze(dim = 1).repeat(1, current_bs, 1)
                text_attention_mask = torch.zeros(text_memory_resized.shape[:2], device = device).bool()
                text = (text_attention_mask, text_memory_resized, obj_pred_names_sums)

                # if args.batch_size != samples.tensors.shape[0]:
                # text_memory_resized_short = text_memory_resized[: , :samples.tensors.shape[0]]
                # text_attention_mask_short = text_attention_mask[: , :samples.tensors.shape[0]]
                # text = (text_attention_mask_short, text_memory_resized_short, obj_pred_names_sums)
                # kwargs = {'text': text}
                
                targets = []
                for group_sub_boxes, group_obj_boxes, group_sub_labels, group_obj_labels in zip(sub_boxes, obj_boxes, sub_labels, obj_labels):
                    targets.append({'sub_boxes':group_sub_boxes,
                                    'obj_boxes':group_obj_boxes,
                                    'sub_labels':group_sub_labels,
                                    'obj_labels':group_obj_labels})
                kwargs_list.append({'text':text, 'targets':targets})
                rel_list.append(rel_text_merge)
                
                rel_text_merge = []
                sub_boxes = []
                obj_boxes = []
                sub_labels = []
                obj_labels = []
        
        ### Prepare the text inputs
        left_bs = len(sub_boxes)
        if left_bs > 0:
            if args.use_no_obj_token:
                obj_pred_names_sums = torch.tensor([[len(o365_obj) + 1, len(rel_text_merge)]])
                flat_text = o365_obj + ['no objects'] + rel_text_merge
            else:
                obj_pred_names_sums = torch.tensor([[len(o365_obj), len(rel_text_merge)]])
                flat_text = o365_obj + rel_text_merge
            flat_tokenized = model.transformer.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
            # tokenizer: dict_keys(['input_ids', 'attention_mask'])
            #            'input_ids' shape: [text_num, max_token_num]
            #            'attention_mask' shape: [text_num, max_token_num]
            encoded_flat_text = model.transformer.text_encoder(**flat_tokenized)
            text_memory = encoded_flat_text.pooler_output
            if args.fusion_type == 'GLIP_attn':
                text_memory_resized = text_memory
            else:
                text_memory_resized = model.transformer.resizer(text_memory)
            
            text_memory_resized = text_memory_resized.unsqueeze(dim = 1).repeat(1, left_bs, 1)
            text_attention_mask = torch.zeros(text_memory_resized.shape[:2], device = device).bool()
            text = (text_attention_mask, text_memory_resized, obj_pred_names_sums)

            # if args.batch_size != samples.tensors.shape[0]:
            # text_memory_resized_short = text_memory_resized[: , :samples.tensors.shape[0]]
            # text_attention_mask_short = text_attention_mask[: , :samples.tensors.shape[0]]
            # text = (text_attention_mask_short, text_memory_resized_short, obj_pred_names_sums)
            # kwargs = {'text': text}
            
            targets = []
            for group_sub_boxes, group_obj_boxes, group_sub_labels, group_obj_labels in zip(sub_boxes, obj_boxes, sub_labels, obj_labels):
                targets.append({'sub_boxes':group_sub_boxes,
                        'obj_boxes':group_obj_boxes,
                        'sub_labels':group_sub_labels,
                        'obj_labels':group_obj_labels})
            kwargs_list.append({'text':text, 'targets':targets})
            rel_list.append(rel_text_merge)


        ### Inference on each group of triplets
        outputs_img = []
        for group_i_kwargs in kwargs_list:
            current_bs = len(group_i_kwargs['targets'])
            group_i_samples = NestedTensor(
                        tensors = samples.tensors.repeat(current_bs, 1, 1, 1),
                        mask = samples.mask.repeat(current_bs, 1, 1))
            memory_cache = model(group_i_samples, encode_and_save=True, **group_i_kwargs)
            outputs = model(group_i_samples, encode_and_save=False, memory_cache=memory_cache, **group_i_kwargs)
            # outputs: dict_keys(['pred_obj_logits', 'pred_verb_logits', 'pred_sub_boxes', 'pred_obj_boxes', 'aux_outputs'])
            outputs_img.append(outputs)


        ### check the query generation of verb_tagger_helper.py 
        ### 检查一下是否box的格式和VGRel的是不是相同的: YEs.
        ### 检查一下one_box 的格式是不是x1 y1 x2 y2: No. cxcywh

        ### Post-process outputs
        img_pseudo_relation_annos = {}
        img_pseudo_relation_annos["image_id"] = image_id.item()
        img_pseudo_relation_annos["dataset"] = 'o365'
        img_pseudo_relation_annos["data_split"] = data_split
        img_pseudo_relation_annos["objects"] = []
        for one_box, one_label in zip(boxes, box_labels):   # in the format of cxcywh
            one_obj = {"object_id": o365_start_obj_idx, 
                       "x": ((one_box[0] - one_box[2]/2.) * img_w).item(),
                       "y": ((one_box[1] - one_box[3]/2.) * img_h).item(),
                       "w": ((one_box[2]) * img_w).item(),
                       "h": ((one_box[3]) * img_h).item(),
                       "names": o365_obj_dict[int(one_label)]
                    }
            # print(one_obj)
            img_pseudo_relation_annos["objects"].append(one_obj)
            o365_start_obj_idx += 1

        img_pseudo_relation_annos["relationships"] = []
        bs_global = 0
        if len(rel_list)>1:
            print([len(l) for l in rel_list])
        assert len(outputs_img) == len(rel_list)
        for outputs, rel_text_merge in zip(outputs_img, rel_list):
            verbs = outputs['pred_verb_logits'].sigmoid()
            bs_idx_list, pair_idx_list, verb_idx_list = torch.where(verbs > relation_threshold)
            for bs_idx, pair_idx, verb_idx in zip(bs_idx_list, pair_idx_list, verb_idx_list):
                pair_i, rel_i = relation_candidates[bs_global + bs_idx.item()]
                
                num_possible_pairs = len(pair_i)
                if pair_idx.item() < num_possible_pairs:
                    sub_idx = pair_i[pair_idx.item()][0]
                    obj_idx = pair_i[pair_idx.item()][1]
                    # if verb_idx.item()>=len(rel_text_merge):
                    #     print(kwargs_list[0]['text'][1].shape, kwargs_list[1]['text'][1].shape)
                    #     print(len(rel_list[0]), len(rel_list[1]))
                    #     print(verb_idx.item(), len(rel_text_merge))
                    #     print(kwargs_list[0]['text'][2], kwargs_list[1]['text'][2])
                    #     print(len(outputs_img))
                    one_rel = {
                        "relationship_id": o365_start_rel_idx,
                        "predicate": rel_text_merge[verb_idx.item()],
                        "subject_id": img_pseudo_relation_annos["objects"][sub_idx]["object_id"],
                        "object_id": img_pseudo_relation_annos["objects"][obj_idx]["object_id"],
                        "confidence": verbs[bs_idx, pair_idx, verb_idx].item(),
                    }
                    img_pseudo_relation_annos["relationships"].append(one_rel)
                    o365_start_rel_idx += 1
            bs_global += verbs.shape[0]
        
        coco_pseudo_relation_annos.append(img_pseudo_relation_annos)

        # if img_idx == 0:
        #     break

        # TODO: TODO: TODO:
        # 增加代码支持一部分数据推理完之后就保存一部分，不要全部推理完才可以保存。
        # 一个比较简单的实现就是把下面保存的代码放到这里来，加上img_idx的条件
    
    print(f'Processing time: {int(time.time() - start_time)} seconds.')
    
    ### Save to disk.
    # with open(args.save_path,'w') as f:
    #     json.dump(coco_pseudo_relation_annos, f)
    return coco_pseudo_relation_annos


    

# Ori Structure of HICO annotation
# {"file_name": "HICO_test2015_00000006.jpg", 
# "hoi_annotation": [{"subject_id": 0, "object_id": 1, "category_id": 50}], 
# "annotations": [{"bbox": [31, 114, 638, 427], "category_id": 1}, 
#                 {"bbox": [28, 213, 636, 477], "category_id": 15}]
# }
# Ori Structure of scene_graph.json annotation
# {"image_id": 2407890,
# "objects": [...
#     {"object_id": 1023838, "x": 324, "y": 320, "w": 142, "h": 255, 
#         "names": ["cat"],"synsets": ["cat.n.01"]},
#     {"object_id":  5071, "x": 359, "y": 362, "w": 72, "h": 81,
#         "names": ["table"], "synsets": ["table.n.01"]},
# ...],
# "relationships": [...
#     {"relationship_id": 15947, "predicate": "wears", "synsets": ["wear.v.01"],
#     "subject_id": 1023838, "object_id":  5071,
#     }
# ...]}
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


def check_list_out_of_bounds(relation_candidates, boxes):
    out_of_bounds_flag = False
    for idx, (pair_i, rel_i) in enumerate(relation_candidates):
        for s,o in pair_i:
            if s >= len(boxes) or o >= len(boxes):
                out_of_bounds_flag = True
    return out_of_bounds_flag



def merge_list_b_to_list_a(list_a, list_b):
    for l in list_b:
        if l not in list_a:
            list_a.append(l)
    return list_a


def generate_keep_names_freq(
        anno_path = None,
        save_path = None,
        annos = None):
    '''
    This function generates a .json file containing the relationship names, object names and 
    their corresponding frequencies.

    :param: anno_path: file path for the generated pseudo-labels.
    :param: save_path: file path to be saved.
    '''
    if anno_path is not None:
        with open(anno_path, "r") as f:
            annos = json.load(f)
    
    obj_keep_dict = {}
    rel_keep_dict = {}
    hoi_triplets = 0
    triplets = 0
    for anno in annos:
        objects_anno = anno['objects']
        relationships_anno = anno['relationships']
        relationships_anno = add_local_object_id(relationships_anno, objects_anno)
        triplets += len(relationships_anno)

        for obj in objects_anno:
            if obj['names'] not in obj_keep_dict.keys():
                obj_keep_dict[obj['names']] = 1
            else:
                obj_keep_dict[obj['names']] += 1

        for rel in relationships_anno:
            if rel['predicate'] not in rel_keep_dict.keys():
                rel_keep_dict[rel['predicate']] = 1
            else:
                rel_keep_dict[rel['predicate']] += 1
            if objects_anno[rel['subject_id_local']]['names'] == 'person':
                hoi_triplets += 1

    rel_keep_dict = dict(sorted(rel_keep_dict.items(), key = lambda item:item[1], reverse = True))
    obj_keep_dict = dict(sorted(obj_keep_dict.items(), key = lambda item:item[1], reverse = True))
    print(f'There are {len(rel_keep_dict)} kinds of relationships and {len(obj_keep_dict)} kinds of objects.')
    print(f'There are {triplets} triplets, {hoi_triplets} of which are hoi triplets in {anno_path}.')

    # with open('vg_keep_names_v1_no_lias_freq.json', 'w') as outfile:
    if save_path is not None:
        with open(save_path, 'w') as outfile:
            json.dump({"relationship_names": list(rel_keep_dict.keys()),
                    "object_names": list(obj_keep_dict.keys()),
                    "relationship_freq": rel_keep_dict,
                    "object_freq": obj_keep_dict}, outfile)
        print(f"Saving to {save_path}.")
    
    ### /mnt/data-nas/peizhi/data/coco2017/annotations/RLIPv2_train2017_threshold25_Tagger2_Noi24_20e.json.
    # There are 1020 kinds of relationships and 80 kinds of objects.
    # There are 3001338 triplets, 1210789 of which are hoi triplets in /mnt/data-nas/peizhi/data/coco2017/annotations/RLIPv2_train2017_threshold25_Tagger2_Noi24_20e.json.


def add_local_object_id(relationships_anno, objects_anno):
    '''
    This function add local object id to the relationship annotations. 
    This can ease the usage of relationship annotations. 

    relationships_anno: relationship annotations of a single image
    objects_anno: object annotations of a single image
    '''
    objects_trans = {} # {global_id: local_id}
    for idx_obj, cur_obj_anno in enumerate(objects_anno):
        objects_trans[cur_obj_anno['object_id']] = idx_obj
    # print(f"sum of objs:{idx_obj+1}")

    for cur_rel_anno in relationships_anno:
        cur_rel_anno['subject_id_local'] = objects_trans[cur_rel_anno['subject_id']]
        cur_rel_anno['object_id_local'] = objects_trans[cur_rel_anno['object_id']]

    return relationships_anno


def fuse_multi_keep_names_freq(
            keep_names_freq_list,
            merge_keep_names_freq_path):
    '''
    This function fuses multiple keep_names_freq.json files into one
    in order to perform pre-training on mixed datasets.
    '''
    annos_list = []
    for keep_names_freq_path in keep_names_freq_list:
        with open(keep_names_freq_path, "r") as f:
            annos_list.append(json.load(f))
    # keys in annos:
    # ["relationship_names", "object_names", "relationship_freq", "object_freq"]
    
    ### We should treat "relationship_freq" and "object_freq" as anchors
    ### since we need ordered "relationship_names" and "object_names".
    base_annos = annos_list[0]
    for annos in annos_list[1:]:
        for rel in annos["relationship_freq"].keys():
            if rel not in base_annos["relationship_freq"].keys():
                base_annos["relationship_freq"][rel] = annos["relationship_freq"][rel]
            else:
                base_annos["relationship_freq"][rel] += annos["relationship_freq"][rel]
        
        for obj in annos["object_freq"].keys():
            if obj not in base_annos["object_freq"].keys():
                base_annos["object_freq"][obj] = annos["object_freq"][obj]
            else:
                base_annos["object_freq"][obj] += annos["object_freq"][obj]
    
    ### We should rank according to frequencies to enable further usage.
    rel_keep_dict = dict(sorted(base_annos["relationship_freq"].items(), key = lambda item:item[1], reverse = True))
    obj_keep_dict = dict(sorted(base_annos["object_freq"].items(), key = lambda item:item[1], reverse = True))
    print(f'There are {len(rel_keep_dict)} kinds of relationships and {len(obj_keep_dict)} kinds of objects.')
    print(f'There are {sum(rel_keep_dict.values())} relationships in the merged json file.')
    # print(f'There are {triplets} triplets, {hoi_triplets} of which are hoi triplets in {anno_path}.')

    # with open('vg_keep_names_v1_no_lias_freq.json', 'w') as outfile:
    with open(merge_keep_names_freq_path, 'w') as outfile:
        json.dump({"relationship_names": list(rel_keep_dict.keys()),
                   "object_names": list(obj_keep_dict.keys()),
                   "relationship_freq": rel_keep_dict,
                   "object_freq": obj_keep_dict}, outfile)


def fuse_rel_det_annos(
    rel_det_annos_list,
    merge_rel_det_annos_path,
    anno_shuffle = True,
    filter_overlap = False,
    save = False,
):
    '''
    This function fuses multiple json file containing relation annotations from multiple sources.
    '''
    annos_list = []
    for rel_det_annos_path in rel_det_annos_list:
        with open(rel_det_annos_path, "r") as f:
            annos_list += json.load(f)
    print(f'We have {len(annos_list)} images in total.')

    if anno_shuffle:
        np.random.shuffle(annos_list)
        print('Shuffling is performed.')
    
    if filter_overlap:
        image_id_list = []
        for anno in annos_list:
            if int(anno["image_id"]) not in image_id_list:
                image_id_list.append(int(anno["image_id"]))
        print(f'We have {len(image_id_list)} images after deduplication.')
    
    if save:
        with open(merge_rel_det_annos_path, 'w') as outfile:
            json.dump(annos_list, outfile)

    # COCO + VG
    # We have 226364 images in total.
    # We have 225116 images after deduplication.

def produce_stat_for_pseudo_labels(
    anno_path,
    relation_threshold = 0.2,
):
    with open(anno_path, "r") as f:
        annos = json.load(f)
    
    rel_dict = {}
    for anno in annos:
        obj_anno = anno["objects"]
        rel_anno = anno["relationships"]
        for rel in rel_anno:
            if rel["confidence"] >= relation_threshold:
                if rel["predicate"] in rel_dict.keys():
                    rel_dict[rel["predicate"]] += 1
                else:
                    rel_dict[rel["predicate"]] = 1
    print(f"We have {sum(rel_dict.values())} relationships in {len(rel_dict)} different kinds.")
    # relation_threshold = 0.2
    # We have 5828544 relationships in 1141 different kinds
    # relation_threshold = 0.3
    # We have 1541595 relationships in 903 different kinds.
    # relation_threshold = 0.4
    # We have 381030 relationships in 671 different kinds.
    # relation_threshold = 0.5
    # We have 82148 relationships in 379 different kinds.
    # relation_threshold = 0.6
    # We have 11853 relationships in 115 different kinds.


def one_triplet_filter_relations(
    anno_path,
    K = 2,
    relation_threshold = 0.2,
    save_filtering_anno_path = None,
    save_filtering_keep_names_freq_path = None,
):
    '''
    This function filters relations if it has more than K relations even if it has confidence higher than a threshold.
    '''
    with open(anno_path, "r") as f:
        annos = json.load(f)
    
    ori_rel_nums = sum([len(anno['relationships']) for anno in annos])
    filtered_rel_nums = []
    new_annos = []
    for anno in annos:
        new_annos.append(anno)
        rel_anno = anno["relationships"]
        rel_dict = {} # {[18927843, 18927845]: [[0, 0.263], [1, 0.385], [2, 0.457]]}
        for idx, rel in enumerate(rel_anno):
            # print(rel["subject_id"], rel["object_id"])
            # print(rel_dict)
            if (rel["subject_id"], rel["object_id"]) not in rel_dict.keys():
                rel_dict[(rel["subject_id"], rel["object_id"])] = [[idx, rel["confidence"]]]
            else:
                rel_dict[(rel["subject_id"], rel["object_id"])].append([idx, rel["confidence"]])
        
        filtered_rel_dict = {}
        filtered_rel_num = 0
        for i, j in rel_dict.items():
            if len(j) > K:
                new_j = list(sorted(j, key = lambda item:item[1], reverse = True))
                new_j = new_j[:K]
                filtered_rel_dict[i] = new_j
                filtered_rel_num += len(new_j)
            else:
                filtered_rel_dict[i] = j
                filtered_rel_num += len(j)
        filtered_rel_nums.append(filtered_rel_num)

        # keep indices
        filtered_rel_idx = []
        for _, f_rel in filtered_rel_dict.items():
            for ff_rel in f_rel:
                filtered_rel_idx.append(ff_rel[0])
        new_annos[-1]["relationships"] = [rel_anno[i] for i in filtered_rel_idx]

    print(f"Original number of relationships: {ori_rel_nums}, after filtering: {sum(filtered_rel_nums)}.")

    generate_keep_names_freq(annos = new_annos,
                             save_path = save_filtering_keep_names_freq_path)
    
    if save_filtering_anno_path is not None:
        with open(save_filtering_anno_path, 'w') as outfile:
            json.dump(new_annos, outfile)
        print(f"Saving to {save_filtering_anno_path}.")

    # K = 2, relatino_threshold = 0.2    
    # Original number of relationships: 5828544, after filtering: 1904600.
    # There are 864 kinds of relationships and 80 kinds of objects.
    # There are 1904600 triplets, 676308 of which are hoi triplets in None.

def filter_out_of_distribution(
    anno_path,
    vg_rel_texts_for_hico_objects_path,
    save_filtering_ood_path,
):
    with open(anno_path, "r") as f:
        annos = json.load(f)
    
    with open(vg_rel_texts_for_hico_objects_path, "r") as f:
        vg_rel_texts_for_hico = json.load(f)
    
    ori_rel_nums = sum([len(anno['relationships']) for anno in annos])
    for anno_idx, anno in enumerate(annos):
        rel_anno = anno["relationships"]
        obj_anno = anno["objects"]
        obj_dict = {obj["object_id"]: obj for obj in obj_anno}

        new_rel_anno = []
        for rel in rel_anno:
            sub_idx = rel["subject_id"]
            obj_idx = rel["object_id"]
            sub_name = obj_dict[sub_idx]["names"]
            obj_name = obj_dict[obj_idx]["names"]
            sub_obj_cand = vg_rel_texts_for_hico[sub_name][obj_name]

            if rel["predicate"] in sub_obj_cand:
                new_rel_anno.append(rel)
        
        annos[anno_idx]["relationships"] = new_rel_anno

    filtered_rel_nums = sum([len(anno['relationships']) for anno in annos])
    print(f"Original number of relationships: {ori_rel_nums}, after filtering: {filtered_rel_nums}.")

    # print(vg_rel_texts_for_hico)
    if save_filtering_ood_path is not None:
        with open(save_filtering_ood_path, 'w') as outfile:
            json.dump(annos, outfile)
        print(f"Saving to {save_filtering_ood_path}.")

@torch.no_grad()
def generate_with_text(model, post_processor, data_loader, dataset_val, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate:'

    # Prepare the text embeddings
    if args.use_no_obj_token:
        obj_pred_names_sums = torch.tensor([[len(dataset_val.object_text) + 1, len(dataset_val.verb_text)]])
        flat_text = dataset_val.object_text + ['no objects'] + dataset_val.verb_text
    else:
        obj_pred_names_sums = torch.tensor([[len(dataset_val.object_text), len(dataset_val.verb_text)]])
        flat_text = dataset_val.object_text + dataset_val.verb_text
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
    
    detections = []
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)

        if args.batch_size != samples.tensors.shape[0]:
            text_memory_resized_short = text_memory_resized[: , :samples.tensors.shape[0]]
            text_attention_mask_short = text_attention_mask[: , :samples.tensors.shape[0]]
            text = (text_attention_mask_short, text_memory_resized_short, obj_pred_names_sums)
            kwargs = {'text': text}
        
        memory_cache = model(samples, encode_and_save=True, **kwargs)
        outputs = model(samples, encode_and_save=False, memory_cache=memory_cache, **kwargs)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if outputs['pred_verb_logits'].shape[2] == len(dataset_val.verb_text) + 1:
            outputs['pred_verb_logits'] = outputs['pred_verb_logits'][:,:,:-1]
        results = post_processor(outputs, orig_target_sizes)

        for img_results, img_targets in zip(results, targets):
            for hoi in img_results['hoi_prediction']:
                detection = {
                    'image_id': img_targets['img_id'],
                    'person_box': img_results['predictions'][hoi['subject_id']]['bbox'].tolist()
                }
                if img_results['predictions'][hoi['object_id']]['category_id'] == missing_category_id:
                    object_box = [np.nan, np.nan, np.nan, np.nan]
                else:
                    object_box = img_results['predictions'][hoi['object_id']]['bbox'].tolist()
                cut_agent = 0
                hit_agent = 0
                eat_agent = 0
                for idx, score in zip(hoi['category_id'], hoi['score']):
                    verb_class = verb_classes[idx]
                    score = score.item()
                    if len(verb_class.split('_')) == 1:
                        detection['{}_agent'.format(verb_class)] = score
                    elif 'cut_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        cut_agent = score if score > cut_agent else cut_agent
                    elif 'hit_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        hit_agent = score if score > hit_agent else hit_agent
                    elif 'eat_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        eat_agent = score if score > eat_agent else eat_agent
                    else:
                        detection[verb_class] = object_box + [score]
                        detection['{}_agent'.format(
                            verb_class.replace('_obj', '').replace('_instr', ''))] = score
                detection['cut_agent'] = cut_agent
                detection['hit_agent'] = hit_agent
                detection['eat_agent'] = eat_agent
                detections.append(detection)
    return detections




@torch.no_grad()
def generate(model, post_processor, data_loader, device, verb_classes, missing_category_id):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate:'

    detections = []
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = post_processor(outputs, orig_target_sizes)

        for img_results, img_targets in zip(results, targets):
            for hoi in img_results['hoi_prediction']:
                detection = {
                    'image_id': img_targets['img_id'],
                    'person_box': img_results['predictions'][hoi['subject_id']]['bbox'].tolist()
                }
                if img_results['predictions'][hoi['object_id']]['category_id'] == missing_category_id:
                    object_box = [np.nan, np.nan, np.nan, np.nan]
                else:
                    object_box = img_results['predictions'][hoi['object_id']]['bbox'].tolist()
                cut_agent = 0
                hit_agent = 0
                eat_agent = 0
                for idx, score in zip(hoi['category_id'], hoi['score']):
                    verb_class = verb_classes[idx]
                    score = score.item()
                    if len(verb_class.split('_')) == 1:
                        detection['{}_agent'.format(verb_class)] = score
                    elif 'cut_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        cut_agent = score if score > cut_agent else cut_agent
                    elif 'hit_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        hit_agent = score if score > hit_agent else hit_agent
                    elif 'eat_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        eat_agent = score if score > eat_agent else eat_agent
                    else:
                        detection[verb_class] = object_box + [score]
                        detection['{}_agent'.format(
                            verb_class.replace('_obj', '').replace('_instr', ''))] = score
                detection['cut_agent'] = cut_agent
                detection['hit_agent'] = hit_agent
                detection['eat_agent'] = eat_agent
                detections.append(detection)

    return detections

def load_hico_object_txt(file_path = '/mnt/data-nas/peizhi/jacob/RLIP/datasets/hico_object_names.txt'):
    '''
    Output like 
        ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',]
    '''
    with open(file_path, 'r') as f:
        object_names = json.load(f)
    object_dict = {j:i for i,j in object_names.items()}
    return object_dict

def load_o365_categories():
    o365_paraphrases_path = '/mnt/data-nas/peizhi/jacob/RLIP/datasets/priors/o365_obj_paraphrase.json' 
    with open(o365_paraphrases_path, "r") as f:
        para_dict = json.load(f)
    id_to_categories = {}
    start_idx = 1
    for name in para_dict.keys():
        id_to_categories[start_idx] = name
        start_idx += 1
    return id_to_categories

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    ### Inferring pseudo-labels
    main_tagger(args)

    # ### Generate frequency stat file for annos
    # generate_keep_names_freq(anno_path = args.save_path,
    #                          save_path = args.save_keep_names_freq_path)

    # ### Merge keep_names_freq.json
    # fuse_multi_keep_names_freq(
    #     keep_names_freq_list = [args.merge_keep_names_freq_path_1, args.merge_keep_names_freq_path_2],
    #     merge_keep_names_freq_path = args.merge_keep_names_freq_path_save,
    # )

    # ### Merge relation detection datasets
    # fuse_rel_det_annos(
    #     rel_det_annos_list = [args.merge_rel_det_annos_path_1, args.merge_rel_det_annos_path_2],
    #     merge_rel_det_annos_path = args.merge_rel_det_annos_path_save,
    #     anno_shuffle = False,
    #     save = True,
    # )

    # produce_stat_for_pseudo_labels(
    #     anno_path = args.save_path,
    #     relation_threshold = args.relation_threshold
    # )
    
    # one_triplet_filter_relations(
    #     anno_path = args.save_path,
    #     save_filtering_anno_path = args.save_filtering_anno_path,
    #     save_filtering_keep_names_freq_path = args.save_filtering_keep_names_freq_path,
    # )

    # filter_out_of_distribution(
    #     anno_path = args.save_path,
    #     vg_rel_texts_for_hico_objects_path = args.vg_rel_texts_for_hico_objects_path,
    #     save_filtering_ood_path = args.save_filtering_ood_path,
    # )


    