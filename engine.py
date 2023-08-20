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
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools
from random import choice, uniform
import time
import json

import torch
import torch.nn.functional as F
import util.misc as utils
from util.optim import adjust_learning_rate
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.hico_eval import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator
from datasets.oi_sgg_eval import OISGGEvaluator
from models.matcher import build_matcher

def exponential_inc_iterative_loss(loss_dict_reduced, weight_dict, model):
    loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                for k, v in loss_dict_reduced.items() if k in weight_dict}
    for k, v in loss_dict_reduced.items():
        None
    return loss_dict_reduced_scaled


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter("lr_backbone", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if len(optimizer.param_groups) == 3:
        metric_logger.add_meter("lr_text_encoder", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        if args.subject_class:
            metric_logger.add_meter('sub_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    num_training_steps = int(len(data_loader) * args.epochs)
    if args.gradient_strategy == "gradient_accumulation":
        assert args.iterative_paradigm is not None
        iterative_paradigm = [int(d) for d in args.iterative_paradigm.split(',')]
    
    print_freq = 600
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        curr_step = epoch * len(data_loader) + i
        samples = samples.to(device)
        if hasattr(model.module.transformer, 'text_encoder'):
            text = [(t['obj_classes'], t[ 'verb_classes']) for t in targets]
            # # Put all tensor variables into the targets
            # targets = [{k: v.to(device) for k, v in t.items() if k not in ['filename', 'image_id', 'obj_classes', 'verb_classes']} for t in targets]
            # Put all target variables into 'targets'
            new_targets = []
            for t in targets:
                t_dict = {}
                for k, v in t.items():
                    if k not in ['filename', 'image_id', 'obj_classes', 'verb_classes']:
                        t_dict[k] = v.to(device)
                    else:
                        t_dict[k] = v
                new_targets.append(t_dict)
            targets = new_targets
            
            kwargs = {'targets':new_targets, 'text':text}
        else:
            targets = [{k: v.to(device) for k, v in t.items() if k not in ['filename', 'image_id', 'obj_classes', 'verb_classes']} for t in targets]
            kwargs = {'targets':targets}
        
        if hasattr(model.module.transformer, 'text_encoder'):
            kwargs = merge_batch_data(kwargs, 
                                      use_no_obj_token = args.use_no_obj_token,
                                      use_all_text_labels = args.use_all_text_labels,
                                      negative_text_sampling = args.negative_text_sampling,
                                      sampling_stategy = args.sampling_stategy,
                                      data_loader = data_loader)
            memory_cache = model(samples, encode_and_save=True, **kwargs)
            outputs = model(samples, encode_and_save=False, memory_cache=memory_cache, **kwargs)
            if hasattr(data_loader.dataset, 'rel_feature') and 'hard_mining' in args.sampling_stategy:
                update_rel_obj_memory(data_loader, memory_cache['text_memory_bf_resize'], kwargs['text'])
        else:
            outputs = model(samples, **kwargs)
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # if hasattr(model,'Iterative'):
        #     loss_dict_reduced_scaled = exponential_inc_iterative_loss(
        #         loss_dict_reduced, weight_dict, model)
        # else:
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}                                 

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # ### Original gradient update
        # optimizer.zero_grad()
        # losses.backward()
        # if max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # optimizer.step()

        ### New gradient update
        if args.gradient_strategy == "gradient_accumulation":
            assert len(iterative_paradigm) > 1 
            ### Instantiation 1
            if (i+1) % len(iterative_paradigm) == 1:
                accumulation_losses = losses
            elif (i+1) % len(iterative_paradigm) == 0:
                # print("gradient_accumulation")
                accumulation_losses += losses * 1 # 1 # 0.2 
                optimizer.zero_grad()
                accumulation_losses = accumulation_losses # / len(iterative_paradigm)
                accumulation_losses.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            else:
                accumulation_losses += losses

            # ### Instantiation 2
            # losses = losses / len(iterative_paradigm)   
            # losses.backward()
            # # * weight_list[iterative_paradigm[(i+1) % len(iterative_paradigm)]]
            # # If you want to add loss weights, you can use a new list containing weights like [w_dataset0, w_dataset1, w_dataset2].
            # if (i+1) % len(iterative_paradigm) == 0:
            #     print("gradient_accumulation")
            #     optimizer.zero_grad()
            #     if max_norm > 0:
            #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            #     optimizer.step()
        elif args.gradient_strategy == "vanilla":
            # print('vanilla')
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()


        if args.schedule is not None:
            adjust_learning_rate(
                optimizer,
                epoch,
                curr_step,
                num_training_steps=num_training_steps,
                args=args,
            )
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        if 'obj_class_error' in loss_dict_reduced.keys():
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        if 'sub_class_error' in loss_dict_reduced.keys():
            metric_logger.update(sub_class_error=loss_dict_reduced['sub_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        if len(optimizer.param_groups) == 3:
            metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])
    
        # print(loss_dict_reduced['sub_class_error'])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    
    print_freq = 100
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    # filename_list = []
    print_freq = 500
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # targets: tuple, len(tuple) = batch_size
        #          element in tuple: a dict, whose keys are ['orig_size', 'size', 'boxes', 'labels', 'id', 'hois']
                 
        # print(targets[0]['orig_size'])
        # print(targets[0]['size'])
        # print('')
        samples = samples.to(device)

        outputs = model(samples)
        # outputs: a dict, whose keys are (['pred_obj_logits', 'pred_verb_logits', 
        #                                'pred_sub_boxes', 'pred_obj_boxes', 'aux_outputs'])
        # print(outputs[''])
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes shape [bs, 2]
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        # print(len(list(itertools.chain.from_iterable(utils.all_gather(results)))))
        # print(list(itertools.chain.from_iterable(utils.all_gather(results)))[0])
        
        # preds: merge predicted batch data
        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        # gts: merge ground truth batch data
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))
        
        # # Add for evaluation
        # filename_list += [t['filename'] for t in targets]

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        # np.savez('eval_stat_best.npz', preds = preds, gts = gts, subject_category_id = subject_category_id, 
        #                         rare_triplets = data_loader.dataset.rare_triplets,
        #                         non_rare_triplets = data_loader.dataset.non_rare_triplets, 
        #                         correct_mat = data_loader.dataset.correct_mat,
        #                         filename_list = filename_list)
        evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, subject_category_id, data_loader.dataset.correct_mat, args)

    stats = evaluator.evaluate()

    return stats



@torch.no_grad()
def evaluate_hoi_with_text(dataset_file, model, postprocessors, data_loader, dataset_val, subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Prepare the text embeddings
    if args.use_no_obj_token:
        obj_pred_names_sums = torch.tensor([[len(dataset_val.object_text) + 1, len(dataset_val.verb_text)]])
        flat_text = dataset_val.object_text + ['no objects'] + dataset_val.verb_text
    else:
        obj_pred_names_sums = torch.tensor([[len(dataset_val.object_text), len(dataset_val.verb_text)]])
        flat_text = dataset_val.object_text + dataset_val.verb_text
    flat_tokenized = model.module.transformer.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
        # tokenizer: dict_keys(['input_ids', 'attention_mask'])
        #            'input_ids' shape: [text_num, max_token_num]
        #            'attention_mask' shape: [text_num, max_token_num]
    encoded_flat_text = model.module.transformer.text_encoder(**flat_tokenized)
    text_memory = encoded_flat_text.pooler_output
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

    # # Add codes for saving word embeddings
    # save_txt_dict = {}
    # for t, m in zip(flat_text, text_memory):
    #     save_txt_dict[t] = m.cpu().numpy()
    # np.savez('/mnt/data-nas/peizhi/jacob/visualization/tSNE/RLIP-ParSe_no_pretrain.npz', 
    #          save_txt_dict = save_txt_dict)

    print("It is running evaluate_hoi_with_text......")

    preds = []
    gts = []
    indices = []
    filename_list = []
    print_freq = 500
    inf_total_time = 0
    for batch_i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # targets: tuple, len(tuple) = batch_size
        #          element in tuple: a dict, whose keys are ['orig_size', 'size', 'boxes', 'labels', 'id', 'hois']
                 
        samples = samples.to(device)
        # Prepare kwargs:
        # This step must be done in the loop, due to the fact that last epoch may not have batch_size samples
        if args.batch_size != samples.tensors.shape[0]:
            text_memory_resized_short = text_memory_resized[: , :samples.tensors.shape[0]]
            text_attention_mask_short = text_attention_mask[: , :samples.tensors.shape[0]]
            text = (text_attention_mask_short, text_memory_resized_short, obj_pred_names_sums)
            kwargs = {'text': text}

        start_time = time.time()
        memory_cache = model(samples, encode_and_save=True, **kwargs)
        outputs = model(samples, encode_and_save=False, memory_cache=memory_cache, **kwargs)
        # outputs: a dict, whose keys are (['pred_obj_logits', 'pred_verb_logits', 
        #                                'pred_sub_boxes', 'pred_obj_boxes', 'aux_outputs'])
        # orig_target_sizes shape [bs, 2]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if outputs['pred_verb_logits'].shape[2] == len(dataset_val.verb_text) + 1:
            outputs['pred_verb_logits'] = outputs['pred_verb_logits'][:,:,:-1]
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        ### Print FPS: this is only valid when batch_size == 1.
        inf_total_time += (time.time() - start_time)
        if (batch_i+1)%300 == 0:
            print(f"FPS: {(batch_i+1)/inf_total_time}. (this is only valid when batch_size == 1)")
        
        # preds: merge predicted batch data
        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        # gts: merge ground truth batch data
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        # Add for evaluation
        filename_list += [t['filename'] for t in targets]

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        # np.savez('/mnt/data-nas/peizhi/jacob/eval_stat_RLIP-ParSeD_VG.npz', preds = preds, gts = gts, subject_category_id = subject_category_id, 
        #                         rare_triplets = data_loader.dataset.rare_triplets,
        #                         non_rare_triplets = data_loader.dataset.non_rare_triplets, 
        #                         correct_mat = data_loader.dataset.correct_mat,
        #                         filename_list = filename_list)
        evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, subject_category_id, data_loader.dataset.correct_mat, args)

    stats = evaluator.evaluate()

    return stats


@torch.no_grad()
def evaluate_sgg_with_text(dataset_file, model, postprocessors, data_loader, dataset_val, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Prepare the text embeddings
    if args.use_no_obj_token:
        obj_pred_names_sums = torch.tensor([[len(dataset_val.object_text) + 1, len(dataset_val.verb_text)]])
        flat_text = dataset_val.object_text + ['no objects'] + dataset_val.verb_text
    else:
        obj_pred_names_sums = torch.tensor([[len(dataset_val.object_text), len(dataset_val.verb_text)]])
        flat_text = dataset_val.object_text + dataset_val.verb_text
    flat_tokenized = model.module.transformer.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
        # tokenizer: dict_keys(['input_ids', 'attention_mask'])
        #            'input_ids' shape: [text_num, max_token_num]
        #            'attention_mask' shape: [text_num, max_token_num]
    encoded_flat_text = model.module.transformer.text_encoder(**flat_tokenized)
    text_memory = encoded_flat_text.pooler_output
    if args.fusion_type=="GLIP_attn":
        text_memory_resized = text_memory
    else:
        text_memory_resized = model.module.transformer.resizer(text_memory)
    text_memory_resized = text_memory_resized.unsqueeze(dim = 1).repeat(1, args.batch_size, 1)
    # text_attention_mask = torch.ones(text_memory_resized.shape[:2], device = device).bool()
    text_attention_mask = torch.zeros(text_memory_resized.shape[:2], device = device).bool()
    text = (text_attention_mask, text_memory_resized, obj_pred_names_sums)
    kwargs = {'text':text}

    # # Add codes for saving word embeddings
    # save_txt_dict = {}
    # for t, m in zip(flat_text, text_memory):
    #     save_txt_dict[t] = m.cpu().numpy()
    # np.savez('/mnt/data-nas/peizhi/jacob/visualization/tSNE/RLIP-ParSe_no_pretrain.npz', 
    #          save_txt_dict = save_txt_dict)

    print("It is running evaluate_sgg_with_text......")

    preds = []
    gts = []
    indices = []
    filename_list = []
    print_freq = 500
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # if i == 100:
        #     break
        # targets: tuple, len(tuple) = batch_size
        #          element in tuple: a dict, whose keys are ['orig_size', 'size', 'boxes', 'labels', 'id', 'hois']
                 
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
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if outputs['pred_verb_logits'].shape[2] == len(dataset_val.verb_text) + 1:
            outputs['pred_verb_logits'] = outputs['pred_verb_logits'][:,:,:-1]
        results = postprocessors['sgg'](outputs, orig_target_sizes)

        # print(len(list(itertools.chain.from_iterable(utils.all_gather(results)))))
        # print(list(itertools.chain.from_iterable(utils.all_gather(results)))[0])
        
        # preds: merge predicted batch data
        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        # gts: merge ground truth batch data
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        # Add for evaluation
        filename_list += [t['filename'] for t in targets]

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'oi_sgg':
        evaluator_100 = OISGGEvaluator(preds, gts, data_loader.dataset.correct_mat, topK = 100, args = args)
        evaluator_50 = OISGGEvaluator(preds, gts, data_loader.dataset.correct_mat, topK = 50, args = args)

    stats_100 = evaluator_100.evaluate()
    stats = evaluator_50.evaluate()
    stats.update(stats_100)
    evaluator_50.print_res(stats)

    return stats



@torch.no_grad()
def evaluate_hoi_with_text_matching_uniformity(dataset_file, model, postprocessors, data_loader, dataset_val, subject_category_id, device, args):
    model.eval()
    matcher = build_matcher(args)
    verb_class_dict = {i:[] for i in range(117)}
    save_relation_ft_path = '/mnt/data-nas/peizhi/jacob/Uniformity/LSE_RQL_RPL_relation_feature_2.npz'

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Prepare the text embeddings
    if args.use_no_obj_token:
        obj_pred_names_sums = torch.tensor([[len(dataset_val.object_text) + 1, len(dataset_val.verb_text)]])
        flat_text = dataset_val.object_text + ['no objects'] + dataset_val.verb_text
    else:
        obj_pred_names_sums = torch.tensor([[len(dataset_val.object_text), len(dataset_val.verb_text)]])
        flat_text = dataset_val.object_text + dataset_val.verb_text
    flat_tokenized = model.module.transformer.tokenizer.batch_encode_plus(flat_text, padding="longest", return_tensors="pt").to(device)
        # tokenizer: dict_keys(['input_ids', 'attention_mask'])
        #            'input_ids' shape: [text_num, max_token_num]
        #            'attention_mask' shape: [text_num, max_token_num]
    encoded_flat_text = model.module.transformer.text_encoder(**flat_tokenized)
    text_memory = encoded_flat_text.pooler_output
    text_memory_resized = model.module.transformer.resizer(text_memory)
    text_memory_resized = text_memory_resized.unsqueeze(dim = 1).repeat(1, args.batch_size, 1)
    # text_attention_mask = torch.ones(text_memory_resized.shape[:2], device = device).bool()
    text_attention_mask = torch.zeros(text_memory_resized.shape[:2], device = device).bool()
    text = (text_attention_mask, text_memory_resized, obj_pred_names_sums)
    kwargs = {'text':text}
    

    preds = []
    gts = []
    indices = []
    print_freq = 500
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # targets: tuple, len(tuple) = batch_size
        #          element in tuple: a dict, whose keys are ['orig_size', 'size', 'boxes', 'labels', 'id', 'hois']
        if hasattr(model.module.transformer, 'text_encoder'):
            text = [(t['obj_classes'], t[ 'verb_classes']) for t in targets]
            # # Put all tensor variables into the targets
            # targets = [{k: v.to(device) for k, v in t.items() if k not in ['filename', 'image_id', 'obj_classes', 'verb_classes']} for t in targets]
            # Put all target variables into 'targets'
            new_targets = []
            for t in targets:
                t_dict = {}
                for k, v in t.items():
                    if k not in ['filename', 'image_id', 'obj_classes', 'verb_classes', 'id']:
                        # print(k)
                        t_dict[k] = v.to(device)
                    else:
                        t_dict[k] = v
                new_targets.append(t_dict)
            targets = new_targets
            
            # kwargs = {'targets':new_targets, 'text':text}


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
        indices = matcher(outputs, targets, return_cost = False)
        # print(outputs.keys())
        verb_feature = outputs['verb_decoder_out']
        for bs_idx, (src_idx, gt_idx) in enumerate(indices):
            for s, g in zip(src_idx, gt_idx):
                verb_tensor = targets[bs_idx]['verb_labels'][g]
                verb_idx_list = torch.nonzero(verb_tensor).squeeze(dim = -1)
                for vil in verb_idx_list:
                    verb_class_dict[vil.item()].append(verb_feature[bs_idx][s.item()].cpu().numpy())
                    # print(len(verb_class_dict))
        # with open(save_relation_ft_path, 'w') as outfile:
        #     json.dump(verb_class_dict, outfile)
        
        

        # outputs: a dict, whose keys are (['pred_obj_logits', 'pred_verb_logits', 
        #                                'pred_sub_boxes', 'pred_obj_boxes', 'aux_outputs'])
        # orig_target_sizes shape [bs, 2]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if outputs['pred_verb_logits'].shape[2] == len(dataset_val.verb_text) + 1:
            outputs['pred_verb_logits'] = outputs['pred_verb_logits'][:,:,:-1]
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        # print(len(list(itertools.chain.from_iterable(utils.all_gather(results)))))
        # print(list(itertools.chain.from_iterable(utils.all_gather(results)))[0])
        
        # preds: merge predicted batch data
        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        # gts: merge ground truth batch data
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

    np.savez_compressed(save_relation_ft_path, 
                        verb_class_dict = verb_class_dict)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        # np.savez('eval_stat_best.npz', preds = preds, gts = gts, subject_category_id = subject_category_id, 
        #                         rare_triplets = data_loader.dataset.rare_triplets,
        #                         non_rare_triplets = data_loader.dataset.non_rare_triplets, 
        #                         correct_mat = data_loader.dataset.correct_mat)
        evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, subject_category_id, data_loader.dataset.correct_mat)

    stats = evaluator.evaluate()

    return stats


def merge_batch_data(kwargs, 
                     use_no_obj_token, 
                     use_all_text_labels, 
                     negative_text_sampling = 0, 
                     sampling_stategy = None,
                     data_loader = None):
    '''
    This function merges the text data from all samples, which allows for different input text lengths.
    This creates more negative samples for each iteration.
    '''
    # Merge sub and obj labels
    targets = kwargs['targets']
    text = kwargs['text']
    obj_text_list = [obj_text for (obj_text, _) in text]
    verb_text_list = [verb_text for (_, verb_text) in text]
    obj_label_list = [t['obj_labels'] for t in targets]
    sub_label_list = [t['sub_labels'] for t in targets]
    verb_label_list = [t['verb_labels'] for t in targets]
    if use_all_text_labels:
        assert sum([len(i) - len(j) for i,j in zip(obj_text_list[:-1], obj_text_list[1:])]) == 0
        assert sum([len(i) - len(j) for i,j in zip(verb_text_list[:-1], verb_text_list[1:])]) == 0

    merged_obj_text, new_obj_label_list = merge_obj_text(obj_text_list, obj_label_list)
    merged_sub_text, new_sub_label_list = merge_obj_text(obj_text_list, sub_label_list)
    merged_verb_text, new_verb_label_list = merge_verb_text(verb_text_list, verb_label_list)
    assert sum([0 if i == j else 1 for i, j in zip(merged_obj_text, merged_sub_text)]) == 0

    # if len(merged_obj_text) != 0 and len(merged_verb_text) == 0:
    #     print(merged_obj_text, merged_verb_text, verb_label_list)

    ### Sample negative texts to reach the number of 'negative_text_sampling'.
    # This step should be completed before the addition of no_obj_token.
    sampling_stategy = (sampling_stategy, sampling_stategy) if '+' not in sampling_stategy else sampling_stategy.split('+')
    merged_obj_text, _ = sample_text(merged_list = merged_obj_text, 
                                  text_type = 'obj', 
                                  negative_text_sampling = int(negative_text_sampling * 2/3.), 
                                  data_loader = data_loader,
                                  obj_target = (new_sub_label_list, new_obj_label_list),
                                  sampling_stategy = sampling_stategy[0]) # 'freq' 'hard_mining'
    merged_verb_text, new_verb_label_list = sample_text(merged_list = merged_verb_text, 
                                  text_type = 'rel', 
                                  negative_text_sampling = negative_text_sampling - int(negative_text_sampling*2/3.), 
                                  data_loader = data_loader,
                                  verb_target = new_verb_label_list,
                                  sampling_stategy = sampling_stategy[1]) # 'freq'

    ### add no_obj_token at the end of 'merged_obj_text'
    if use_no_obj_token:
        merged_obj_text.append('no objects')
    
    kwargs['text'] = [(merged_obj_text, merged_verb_text)]
    for idx, t in enumerate(targets):
        t['obj_labels'] = new_obj_label_list[idx]
        t['sub_labels'] = new_sub_label_list[idx]
        t['verb_labels'] = new_verb_label_list[idx]
    kwargs['targets'] = targets

    return kwargs

def merge_obj_text(text_list, label_list):
    '''
    text_list: 
        extract from "targets": List of every sample's text list
    label_list: 
        extract from "targets": List of every sample's obj label tensor
    '''
    # get the text version of label_list
    text_label_lsit = []
    for cur_text, cur_label in zip(text_list, label_list):
        text_label_lsit.append([cur_text[l] for l in cur_label])
    # merge text
    merged_text = []
    for cur_text in text_list:
        for t in cur_text:
            if t not in merged_text:
                merged_text.append(t)
    # get new label_list
    new_label_list = []
    for cur_text_label in text_label_lsit:
        new_label_list.append(torch.tensor([merged_text.index(tl) for tl in cur_text_label], dtype=torch.int64, device = label_list[0].device))
    assert new_label_list[0].shape == label_list[0].shape

    return merged_text, new_label_list

def merge_verb_text(text_list, label_list):
    '''
    text_list: 
        extract from "targets": List of every sample's text list
    label_list: 
        extract from "targets": List of every sample's verb label tensor
    '''
    # get the text version of label_list
    text_label_list = []
    for cur_text, cur_label in zip(text_list, label_list):
        idx_tensor = torch.tensor(range(0, cur_label.shape[1]))
        cur_text_label = []
        for label_tensor in cur_label:
            kept_mask = label_tensor == 1
            label_idx = idx_tensor[kept_mask]
            cur_text_label.append([cur_text[idx] for idx in label_idx])
        text_label_list.append(cur_text_label)
    # merge text
    merged_text = []
    for cur_text in text_list:
        for t in cur_text:
            if t not in merged_text:
                merged_text.append(t)
    # get new label_list
    new_label_list = []
    for cur_text_label in text_label_list:
        cur_label = []
        for ctl in cur_text_label:
            zero_tensor = torch.zeros((len(merged_text),), dtype=torch.float32, device = label_list[0].device)
            for cur_ctl in ctl:
                zero_tensor[merged_text.index(cur_ctl)] = 1
            cur_label.append(zero_tensor)
        if len(cur_text_label) > 0:
            new_label_list.append(torch.stack(cur_label, dim = 0))
        else:
            new_label_list.append(torch.zeros((0, len(merged_text)), dtype=torch.float32, device = label_list[0].device))
    
    return merged_text, new_label_list

def sample_text(merged_list, 
                text_type, 
                negative_text_sampling, 
                data_loader, 
                verb_target = None,
                obj_target = None,
                sampling_stategy = 'random'):
    """
    This function is implemented to sample texts to fill a list with a predefined length.
    
    merged_list: the merged text list
    text_type: in ['obj', 'rel']
    negative_text_sampling: the len of the text list after sampling
    data_loader: the dataloader which contains the full list of texts
    verb_target: if we are sample for verbs, we have to change its target format 
                 to match the predicted tensor and the target tensor 
    sampling_stategy: in ['random', 'frequency']
    """
    assert text_type in ['obj', 'rel']
    # If this batch does not have any positive triplet samples, hard_mining will degrade to frequency based sampling.
    if sampling_stategy == 'hard_mining':
        target = obj_target[0] if text_type == 'obj' else verb_target
        all_target = torch.cat(target, dim = 0)
        if all_target.shape[0] == 0:
            sampling_stategy = 'freq'

    if len(merged_list) >= negative_text_sampling:
        return merged_list, verb_target
    else:
        if sampling_stategy in ['random', 'freq']:
            full_text = data_loader.dataset.object_names if text_type == 'obj' else data_loader.dataset.relationship_names
            full_text_freq = data_loader.dataset.object_freq if text_type == 'obj' else data_loader.dataset.relationship_freq
            prob_cumsum = list(np.cumsum(list(full_text_freq.values())) / sum(list(full_text_freq.values())))
            while len(merged_list) < negative_text_sampling:
                if sampling_stategy == 'random':
                    random_t = choice(full_text)
                elif sampling_stategy == 'freq':
                    random_p = uniform(0,1)
                    for i,j in enumerate(prob_cumsum):
                        if random_p <= j:
                            random_t = full_text[i]
                            break
                if random_t not in merged_list:
                    merged_list.append(random_t)
        elif sampling_stategy == 'hard_mining':
            ### Calculate global similarity between the merged list (query list) with the feature sequence
            text_seq, feature_seq = data_loader.dataset.obj_feature if text_type == 'obj' else data_loader.dataset.rel_feature
            device = obj_target[0][0].device if text_type == 'obj' else verb_target[0].device
            feature_seq = feature_seq.to(feature_seq)
            merged_list_feature = torch.stack([feature_seq[text_seq.index(i)] for i in merged_list], dim = 0)
            ## Distance metric 1: Cosine sim
            merged_text_1 = F.normalize(merged_list_feature, p=2, dim=-1)
            feature_seq_2 = F.normalize(feature_seq, p=2, dim=-1)
            merged_sim = torch.einsum('ab,cb->ac', merged_text_1, feature_seq_2)
            ## Distance metric 2: Euclidean sim, CPU out of memory
            # seq_len, text_dim = feature_seq.shape
            # merged_len, text_dim = merged_list_feature.shape
            # merged_text_1 = merged_list_feature.unsqueeze(dim = 1).expand(merged_len, seq_len, text_dim).reshape(-1, text_dim)
            # feature_seq_2 = feature_seq.unsqueeze(dim = 0).expand(merged_len, seq_len, text_dim).reshape(-1, text_dim)
            # merged_sim = F.pairwise_distance(merged_text_1, feature_seq_2, p = 2).view(merged_len, seq_len)
            # merged_sim = merged_sim.max(-1)[0].unsqueeze(dim = -1) - merged_sim
            ## Distance metric 2: Euclidean sim using 'for' operation
            # seq_len, text_dim = feature_seq.shape
            # merged_len, text_dim = merged_list_feature.shape
            # merged_sim = []
            # start_t = time.time()
            # for mlf in merged_list_feature:
            #     mlf_1 = mlf.unsqueeze(dim = 0).expand_as(feature_seq)
            #     merged_sim.append(F.pairwise_distance(mlf_1, feature_seq, p=2))
            # merged_sim = torch.stack(merged_sim, dim = 0)
            # merged_sim = merged_sim.max(-1)[0].unsqueeze(dim = -1) - merged_sim
            # print(time.time() - start_t)
            ## Distance metric 2: Euclidean sim using 'cdist' operation
            # seq_len, text_dim = feature_seq.shape
            # merged_len, text_dim = merged_list_feature.shape
            # # start_t = time.time()
            # merged_sim = torch.cdist(merged_list_feature, feature_seq, p=2) # 0.2 second for one batch
            # merged_sim = merged_sim.max(-1)[0].unsqueeze(dim = -1) - merged_sim
            # # print(time.time() - start_t)
            
            ### Aggregate similarity for the merged list (query list)
            if text_type == 'obj':
                sub_obj_t = torch.cat(obj_target[0] + obj_target[1], dim = 0)
                query_sim = merged_sim[sub_obj_t]
            elif text_type == 'rel':
                verb_t = torch.cat(verb_target, dim = 0)
                query_sim = torch.stack([merged_sim[vt.bool()].sum(dim = 0) for vt in torch.cat(verb_target, dim = 0)])
            query_sim = query_sim / query_sim.max(-1)[0].unsqueeze(-1)
            # Since we have made sure that we have more than 1 triplet if performing hard mining,
            # we do not need to ensure the query.shape[0] to be >=1.
            query_sim = query_sim.sum(dim = 0) # sum over all queries, making it [seq_len,]

            ### sample via ranking
            sorted_q_v, sorted_q_ind = torch.sort(query_sim, dim = 0, descending = True)
            # print(sorted_q_v[0:10000:100], sorted_q_v[0:10000:1000], sorted_q_v[0:-1:10000])
            sorted_flag = 0
            while len(merged_list) < negative_text_sampling:
                if text_seq[sorted_q_ind[sorted_flag]] not in merged_list:
                    merged_list.append(text_seq[sorted_q_ind[sorted_flag]])
                sorted_flag += 1

        assert len(merged_list) == negative_text_sampling
    
    if text_type == 'rel':
        assert verb_target is not None
        new_verb_target = []
        padded_zeros_len = negative_text_sampling - verb_target[0].shape[-1]
        for verb_t in verb_target:
            padded_zero_tensor = torch.zeros((len(verb_t), padded_zeros_len), 
                                             dtype=torch.float32, 
                                             device = verb_t.device)
            new_verb_target.append(torch.cat((verb_t, padded_zero_tensor), dim = -1))
        return merged_list, new_verb_target
    else:
        return merged_list, verb_target

def update_rel_obj_memory(data_loader, text_memory_bf_resize, text):
    ### Update obj features
    obj_text = text[0][0]
    len_obj = len(obj_text)
    new_obj_ft = text_memory_bf_resize.squeeze(dim = 1)[:len_obj]
    for o, nft in zip(obj_text, new_obj_ft):
        o_ind = data_loader.dataset.obj_feature[0].index(o)
        data_loader.dataset.obj_feature[1][o_ind] = nft
    
    ### Update rel features
    rel_text = text[0][1]
    len_rel = len(rel_text)
    new_rel_ft = text_memory_bf_resize.squeeze(dim = 1)[len_obj:(len_obj + len_rel)]
    for r, nft in zip(rel_text, new_rel_ft):
        r_ind = data_loader.dataset.rel_feature[0].index(r)
        data_loader.dataset.rel_feature[1][r_ind] = nft     

if __name__=="__main__":
    None