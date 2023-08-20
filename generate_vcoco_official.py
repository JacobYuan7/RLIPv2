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
import numpy as np
import copy
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.vcoco import build as build_dataset
from models.backbone import build_backbone
from models.DDETR_backbone import build_backbone as build_DDETR_backbone
from models.transformer import build_transformer
from models.swin.backbone import build_backbone as build_Swin_backbone
import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from models.hoi import OCN, ParSeD, ParSe, RLIP_ParSe, RLIP_ParSeD, RLIP_ParSeDA
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

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--drop_path_rate', default=0.2, type=float,
                        help="drop_path_rate applied in the Swin transformer.")
    parser.add_argument('--pretrained_swin', type=str, default='',
                        help='Pretrained model path for the swin backbone only!!!')
    parser.add_argument('--use_checkpoint', action='store_true') # This is for Swin-transformer to save memory.


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
    parser.add_argument('--RLIP_ParSeD_v2', action = 'store_true',
                        help='RLIP_ParSeD_v2.') 
    parser.add_argument('--RLIP_ParSeDA_v2', action = 'store_true',
                        help='RLIP_ParSeDA_v2.') 
    parser.add_argument('--RLIP_ParSe_v2', action = 'store_true',
                        help='RLIP_ParSe_v2.') 
    parser.add_argument('--ParSeDABDDETR', action = 'store_true',
                        help='Parallel Detection and Sequential Relation Inferring using DAB-Deformable DETR.')
    # parser.add_argument('--ParSeDABDETR', action = 'store_true',
    #                     help='Parallel Detection and Sequential Relation Inferring using DAB-DETR.')
    # parser.add_argument('--RLIPParSeDABDETR', action = 'store_true',
    #                     help='RLIP-Parallel Detection and Sequential Relation Inferring using DAB-DETR.')
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

    # DDETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

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

    # Cross-Modal Fusion
    parser.add_argument('--use_checkpoint_fusion', default=False, action='store_true', help = 'Use checkpoint to save memory.')
    parser.add_argument('--fusion_type', default = "no_fusion", choices = ("MDETR_attn", "GLIP_attn", "no_fusion"), )
    parser.add_argument('--fusion_interval', default=1, type=int, help="Fusion interval in VLFuse.")
    parser.add_argument('--fusion_last_vis', default=False, action='store_true', help = 'Whether to fuse the last layer of the vision features.')
    parser.add_argument('--lang_aux_loss', default=False, action='store_true', help = 'Whether to use aux loss to calculate the loss functions.')
    parser.add_argument('--gating_mechanism', default="GLIP", type=str,
                        choices=["GLIP", "Vtanh", "Etanh", "Stanh", "SDFtanh", "SFtanh", "SOtanh", "SXAc", "SDFXAc", "VXAc", "SXAcLN", "SDFXAcLN", "SDFOXAcLN", "MBF"],
                        help = "The gating mechanism used to perform language-vision feature fusion.")
    parser.add_argument('--verb_query_tgt_type', default="vanilla", type=str,
                        choices=["vanilla", "MBF", "vanilla_MBF"],
                        help = "The method used to generate queries.")
    parser.add_argument('--separate_bidirectional', default=False, action='store_true', help = 'For GLIP_attn, we perform separate attention for different levels of features.')
    parser.add_argument('--do_lang_proj_outside_checkpoint', default=False, action='store_true', help = 'Use feature resizer to project the concatenation of interactive language features to the dimension of language embeddings.')
    parser.add_argument('--stable_softmax_2d', default=False, action='store_true', help = 'Use "attn_weights = attn_weights - attn_weights.max()" during BiMultiHeadAttention in VLFuse.')
    parser.add_argument('--clamp_min_for_underflow', default=False, action='store_true', help = 'Clamp attention weights (before softmax) during BiMultiHeadAttention in VLFuse.')
    parser.add_argument('--clamp_max_for_overflow', default=False, action='store_true', help = 'Clamp attention weights (before softmax) during BiMultiHeadAttention in VLFuse.')



    return parser


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

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler = sampler_val,
                                 drop_last = False, collate_fn = utils.collate_fn, num_workers = args.num_workers)

    args.lr_backbone = 0
    args.masks = False
    if 'swin' in args.backbone:
        backbone = build_Swin_backbone(args)
    elif args.DDETRHOI or args.ParSeD or args.RLIP_ParSeD or args.RLIP_ParSeDA_v2 or args.RLIP_ParSeD_v2:
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
        print('Building RLIP_ParSeD/RLIP_ParSeD_v2...')
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
def generate_with_text(model, post_processor, data_loader, dataset_val, device, verb_classes, missing_category_id, args):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

    