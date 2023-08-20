# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from itertools import combinations, permutations
import time
from mmcv.image import imread, imwrite
from mmcv.utils import is_str
from enum import Enum
import cv2

import json
import sys
sys.path.append("..")
import transforms as T
# from .. import transforms as T
import sng_parser
import numpy as np
from PIL import Image
import clip


def transform_sentences_to_triplets(save_path_sng = None):
    coco_captions_train_path = '/Path/To/data/annotations_trainval2014/captions_train2014.json'  # Oracle captions
    coco_captions_val_path = '/Path/To/data/annotations_trainval2014/captions_val2014.json'  # Oracle captions
    with open(coco_captions_train_path, 'r') as f:
        coco_captions_train = json.load(f)
    with open(coco_captions_val_path, 'r') as f:
        coco_captions_val = json.load(f)

    ### Merge train and val
    coco_caps = {}
    sng_parsed = {}
    start_t = time.time()
    for cap_idx, one_cap in enumerate((coco_captions_train['annotations'] + coco_captions_val['annotations'])):
        if (cap_idx+1)%5000 == 0:
            print(f"Processing {cap_idx+1} / {len(coco_captions_train['annotations'] + coco_captions_val['annotations'])}")

        cap_graph = sng_parser.parse(one_cap['caption'])
        if one_cap['image_id'] not in coco_caps.keys():
            coco_caps[one_cap['image_id']] = {'image_id': one_cap['image_id'], 
                                              'id': [one_cap['id']],
                                              'caption': [one_cap['caption']]}
            sng_parsed[one_cap['image_id']] = [cap_graph]
        else:
            if one_cap['id'] not in coco_caps[one_cap['image_id']]['id']:
                coco_caps[one_cap['image_id']]['id'].append(one_cap['id'])
                coco_caps[one_cap['image_id']]['caption'].append(one_cap['caption'])
                sng_parsed[one_cap['image_id']].append(cap_graph)
    print(f"{sum([len(s) for _,s in sng_parsed.items()])} captions are processed.")
    print(f"Processing time: {int(time.time() - start_t)}s.")
    
    # graph = sng_parser.parse('A blue boat themed bathroom with a life preserver on the wall')
    ### Save to disk
    if save_path_sng is not None:
        with open(save_path_sng, "w") as f:
            json.dump(sng_parsed, f)

### Format of the parsed sentence:
# {'entities': [{'head': 'boat',
#                'lemma_head': 'boat',
#                'lemma_span': 'a blue boat',
#                'modifiers': [{'dep': 'det', 'lemma_span': 'a', 'span': 'A'},
#                              {'dep': 'amod',
#                               'lemma_span': 'blue',
#                               'span': 'blue'}],
#                'span': 'A blue boat',
#                'span_bounds': (0, 3),
#                'type': 'unknown'},
#               {'head': 'bathroom',
#                'lemma_head': 'bathroom',
#                'lemma_span': 'bathroom',
#                'modifiers': [],
#                'span': 'bathroom',
#                'span_bounds': (4, 5),
#                'type': 'scene'},
#               {'head': 'life preserver',
#                'lemma_head': 'life preserver',
#                'lemma_span': 'a life preserver',
#                'modifiers': [],
#                'span': 'a life preserver',
#                'span_bounds': (6, 9),
#                'type': 'unknown'},
#               {'head': 'wall',
#                'lemma_head': 'wall',
#                'lemma_span': 'the wall',
#                'modifiers': [{'dep': 'det',
#                               'lemma_span': 'the',
#                               'span': 'the'}],
#                'span': 'the wall',
#                'span_bounds': (10, 12),
#                'type': 'unknown'}],
#  'relations': [{'lemma_relation': 'theme',
#                 'object': 1,
#                 'relation': 'themed',
#                 'subject': 0},
#                {'lemma_relation': 'with',
#                 'object': 2,
#                 'relation': 'with',
#                 'subject': 0},
#                {'lemma_relation': 'on',
#                 'object': 3,
#                 'relation': 'on',
#                 'subject': 2}]}


def coco_scene_graph_match_with_bbox(
    scene_graph_path = None,
    bbox_path = None,
    save_path_pseudo_relations = None,
    match_strategy = 'original_text',
    CLIP_mode = None,
):
    '''
    We match the parsed scene graphs (either from oracle COCO captions or from generated COCO captions) with the COCO detection bboxes.
    We will save pseudo_relations_coco (list) to disk.

    Args:
        scene_graph_path (string): a string indicating the path for scene graphs
        bbox_path (list): a list of path (support for coco datasetc containing train2017 and val2017)
        save_path_pseudo_relations (string): a path to store the annotations for pseudo relations
        match_strategy (string): A choice from ['original_text', 'paraphrases'], indicating whether we should use paraphrases to tell that 
                                    parsed relations are valid using COCO GT bbox annotations.
        CLIP_mode (string): The CLIP model we use to filter the valid pairs.

    Returns:
        None
    '''
    if scene_graph_path:
        with open(scene_graph_path, 'r') as f:
            sng_parsed = json.load(f)
    sng_parsed = grammartical_tranform(sng_parsed)
    if bbox_path:
        if len(bbox_path) == 1:
            with open(bbox_path[0], 'r') as f:
                bboxes = json.load(f)["annotations"]
        elif len(bbox_path) == 2:
            with open(bbox_path[0], 'r') as f:
                train_bboxes = json.load(f)["annotations"]
            with open(bbox_path[1], 'r') as f:
                val_bboxes = json.load(f)["annotations"]
            # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
            bboxes = train_bboxes + val_bboxes
    
    # convert bboxes to a dict with image_id as keys
    bboxes_dict = {}
    for bbox in bboxes:
        image_id = str(bbox['image_id'])
        if image_id not in bboxes_dict.keys():
            bboxes_dict[image_id] = [bbox,]
        else:
            bboxes_dict[image_id].append(bbox)
    
    ### We aggregate all object names in a given coco image.
    object_80_dict = load_hico_object_txt()
    object_80_list = list(object_80_dict.keys())
    bboxes_names_dict = {}
    for image_id, bboxes_img in bboxes_dict.items():
        assert image_id not in bboxes_names_dict.keys()
        bboxes_names_dict[image_id] = []
        for one_bbox_img in bboxes_img:
            if object_80_dict[one_bbox_img["category_id"]] not in bboxes_names_dict[image_id]:
                bboxes_names_dict[image_id].append(object_80_dict[one_bbox_img["category_id"]])
    print(f"{len(bboxes_names_dict.keys())} images are in the trainval2017.")
    print(f"However, {len(sng_parsed)} images are in the COCO caption2014.\n")
    # print(list(bboxes_names_dict.keys())[:100])
    # print(list(sng_parsed.keys())[:100])

    ### We filter out relations which do not have corresponding objects in the image.
    # we firstly filter out parsed 'entities' if they are not within 80 classes
    matcher = MatchWithGTbboxes(match_strategy = match_strategy)
    sng_new1 = {}  # This excludes the images that they do not appear in the trainval2017 (caption2014 has more).
    for image_id, graph_list in sng_parsed.items():
        if image_id in bboxes_names_dict.keys():
            bboxes_names = bboxes_names_dict[image_id]
            graph_list_new = []
            for graph in graph_list:
                # new_graph = match_one_img_bboxes_with_one_scene_graph(
                #                 bboxes_names = bboxes_names,
                #                 scene_graph_one_sent = graph)
                new_graph = matcher.match_one_img_bboxes_with_one_scene_graph(
                                bboxes_names = bboxes_names,
                                scene_graph_one_sent = graph)
                if len(new_graph['relations']) > 0:
                    graph_list_new.append(new_graph)
            sng_new1[image_id] = graph_list_new
        else:
            continue
    print(f"After filtering, we have {len(sng_new1)} images with parsed scene graphs.\n")
    
    ### We give some stat about the left relations and objects in relations.   
    sng_parsed_dist = compute_relation_distribution_from_scene_graphs(sng_parsed)
    sng_new1_dist = compute_relation_distribution_from_scene_graphs(sng_new1)
    print(f"We have {sum(sng_parsed_dist.values())} relations in {len(sng_parsed_dist)} kinds before filtering.")
    print(f"We have {sum(sng_new1_dist.values())} relations in {len(sng_new1_dist)} kinds after filtering.\n")
    # check_span_lemma_span(sng_new1)
    # Oracle COCO captions:
    # We have 1086109 relations in 5390 kinds before filtering.
    # We have 39933 relations in 847 kinds after filtering.
    sng_parsed_obj_dist = compute_object_distribution_from_scene_graphs(sng_parsed)
    print(f"We have {sum(sng_parsed_obj_dist.values())} objects in {len(sng_parsed_obj_dist)} kinds before filtering.\n")
    # for i,j in sng_parsed_obj_dist.items():
    #     if j > 1000:
    #         print(i, ' ')
    
    ### We match relations with given GT bbox annotations.
    # We traverse all the filtered relation scene graphs.
    # We add "train2017" or "val2017" indicator to the annotations to facilitate image read-in.
    if CLIP_mode is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(CLIP_mode, device=device)

    train_img_ids = np.unique([str(box['image_id']) for box in train_bboxes])
    val_img_ids = np.unique([str(box['image_id']) for box in val_bboxes])
    coco2017_folder = Path('/Path/To/data/coco2017')
    # coco_start_obj_idx = 10000000
    coco_start_rel_idx = 10000000
    pseudo_relations_coco = []

    CLIP_pseudo_rel_num = 0
    naive_pseudo_rel_num = 0
    for img_idx, (image_id, scene_graphs) in enumerate(sng_new1.items()):
        if (img_idx+1) % 100 == 0:
            print(f"CLIP filtered rels/naive rels: {CLIP_pseudo_rel_num}/{naive_pseudo_rel_num}")
            print(f'Matching scene graphs with images: {img_idx+1}/{len(sng_new1)}')
            # break
        # if (img_idx+1) <= 39500:
        #     continue

        rels_img = []

        bboxes_img = bboxes_dict[image_id]
        bboxes_img = transform_coco_bbox_to_VG_format(coco_bbox = bboxes_img,
                                                      object_80_dict = object_80_dict)

        relationships_coco_name_list = []
        relationships_span_name_list = []
        for scene_graph in scene_graphs:
            for triplet in scene_graph['relations']:
                relationships_coco_name_list.append([
                    triplet['subject_coco_name'], triplet['relation'], triplet['object_coco_name']
                ])
                # We define this list containing span names for further usage if span name is better.
                relationships_span_name_list.append([
                    triplet['subject_span'], triplet['relation'], triplet['object_span']
                ])
        
        if CLIP_mode is not None:
            img_file_name = str(image_id).zfill(12) + '.jpg'
            if image_id in val_img_ids:
                img_path = coco2017_folder / 'val2017' / img_file_name
            else:
                img_path = coco2017_folder / 'train2017' / img_file_name
            img_file = Image.open(img_path)
            CLIP_vis_features, CLIP_text_features = prepare_CLIP_features(
                                  relationships_name_list = relationships_coco_name_list,
                                  bboxes_img = bboxes_img,
                                  img_file = img_file,
                                  model = model, 
                                  preprocess = preprocess,
                                  device = device)

        for triplet_idx, triplet_name in enumerate(relationships_coco_name_list):
            if CLIP_mode is not None:
                interaction_text_ft = CLIP_text_features[f"a photo of {triplet_name[0]} {triplet_name[1]} {triplet_name[2]}"]
                # background_text_ft = CLIP_text_features[f"a photo of {triplet_name[0]} not interacting with {triplet_name[2]}"]
                background_text_ft = CLIP_text_features[f"a photo of {triplet_name[0]} having no relation with {triplet_name[2]}"]
                text_ft = torch.stack([interaction_text_ft, background_text_ft], dim = 0)
                text_ft /= text_ft.norm(dim=-1, keepdim=True)
                
            for sub_idx, sub_bbox in enumerate(bboxes_img):
                if sub_bbox["names"] == triplet_name[0]:
                    for obj_idx, obj_bbox in enumerate(bboxes_img):
                        if obj_bbox["names"] == triplet_name[2]:
                            naive_pseudo_rel_num += 1
                            if CLIP_mode is None:
                                rels_img.append(
                                    {
                                        "relationship_id": coco_start_rel_idx,
                                        "predicate": triplet_name[1],
                                        "subject_id": bboxes_img[sub_idx]["object_id"],
                                        "object_id": bboxes_img[obj_idx]["object_id"],
                                    }
                                )
                                coco_start_rel_idx += 1
                            else:
                                # Some union regions are too small, so that they have w=0 or h=0. We do not include these union regions.
                                if (sub_idx, obj_idx) not in CLIP_vis_features.keys():
                                    continue
                                vis_ft = CLIP_vis_features[(sub_idx, obj_idx)]
                                vis_ft /= vis_ft.norm(dim=-1, keepdim=True)
                                cos_score = (100. * vis_ft @ text_ft.T).softmax(-1)
                                # if cos_score[0] > 0.9:
                                rels_img.append(
                                    {
                                        "relationship_id": coco_start_rel_idx,
                                        "predicate": triplet_name[1],
                                        "subject_id": bboxes_img[sub_idx]["object_id"],
                                        "object_id": bboxes_img[obj_idx]["object_id"],
                                        "confidence": cos_score[0].item(),
                                        "overlap": check_overlap([sub_bbox["x"], sub_bbox["y"], sub_bbox["w"], sub_bbox["h"]], [obj_bbox["x"], obj_bbox["y"], obj_bbox["w"], obj_bbox["h"]])
                                    }
                                )
                                coco_start_rel_idx += 1
                                CLIP_pseudo_rel_num += 1
                                



        pseudo_relations_coco.append(
            {
                "image_id": image_id,
                "objects": bboxes_img,
                "relationships": rels_img,
                "dataset": "coco2017",
                "data_split": "val2017" if image_id in val_img_ids else "train2017",
            }
        )


    pseudo_relations_coco_dist = compute_relation_distribution_from_vg_format(pseudo_relations_coco)
    print(f"We have {sum(pseudo_relations_coco_dist.values())} relations in {len(pseudo_relations_coco_dist)} kinds after matching with GT bboxes.")


    ### Save to disk.
    if save_path_pseudo_relations is not None:
        with open(save_path_pseudo_relations, "w") as f:
            json.dump(pseudo_relations_coco, f)
        print(f'Finishing saving to {save_path_pseudo_relations}.')




def visualize_pseudo_relations(anno_path,
                               one_rel_per_img = False,
                               relation_threshold = 0.):
    '''
    This function aims to visualize images with relation annotations.

    Args:
        anno_path (string): This is a string indicating the annotations to be visualized.
        one_rel_per_img (bool): This param indicates whether we only visualize one relation in one image.

    Returns:
        None
    '''
    coco2017_folder = Path('/Path/To/data/coco2017')
    save_folder = Path('/Path/To/jacob/visualization/pseudo_relations_CLIP_2')
    with open(anno_path, 'r') as f:
        rel_annos = json.load(f)

    rel_annos_dict = {int(rel_anno['image_id']):rel_anno for rel_anno in rel_annos}
    rel_annos_dict = dict(sorted(rel_annos_dict.items(), key = lambda item:item[0], reverse = True))
    rel_annos = list(rel_annos_dict.values())

    for img_anno in rel_annos[:500]:
        h_bboxes = []
        h_label_texts = []
        o_bboxes = []
        o_label_texts = []

        rels = img_anno["relationships"]
        objs = img_anno["objects"]
        objs_dict = {o["object_id"]:o for o in objs}
        if len(rels) == 0:
            continue 
        for rel in rels:
            if relation_threshold > 0.:
                if "confidence" in rel.keys():
                    if rel["confidence"] < relation_threshold:
                        continue
            sub_box = objs_dict[rel["subject_id"]]
            obj_box = objs_dict[rel["object_id"]]
            h_bboxes.append(np.array([sub_box["x"], sub_box["y"], sub_box["x"]+sub_box["w"], sub_box["y"]+sub_box["h"]]))
            h_label_texts.append(f"{sub_box['names']} {rel['predicate']}")
            o_bboxes.append(np.array([obj_box["x"], obj_box["y"], obj_box["x"]+obj_box["w"], obj_box["y"]+obj_box["h"]]))
            o_label_texts.append(obj_box["names"])
        if len(h_bboxes)==0:
            continue
        h_bboxes = np.stack(h_bboxes, axis = 0)
        o_bboxes = np.stack(o_bboxes, axis = 0)

        if img_anno["dataset"] == "coco2017":
            img_file_name = str(img_anno['image_id']).zfill(12) + '.jpg'
            if 'data_split' not in img_anno.keys():
                img_path = coco2017_folder / 'train2017' / img_file_name
            else:
                img_path = coco2017_folder / img_anno['data_split'] / img_file_name
        
        if not one_rel_per_img:
            out_path = save_folder / img_file_name
            imshow_det_hoi_bboxes(img_path,
                            h_bboxes,
                            h_label_texts,
                            o_bboxes,
                            o_label_texts,
                            score_thr=0,
                            thickness=2,
                            font_scale=.5,
                            show=False,
                            win_name='',
                            wait_time=0,
                            out_file=out_path)
        else:
            for rel_idx in range(len(h_bboxes)):
                out_path = save_folder / f"{str(img_anno['image_id']).zfill(12)}_{rel_idx}.jpg"
                h_bboxes_i = h_bboxes[rel_idx][None,:]
                h_label_texts_i = [h_label_texts[rel_idx]]
                o_bboxes_i = o_bboxes[rel_idx][None,:]
                o_label_texts_i = [o_label_texts[rel_idx]]
                imshow_det_hoi_bboxes(img_path,
                                    h_bboxes_i,
                                    h_label_texts_i,
                                    o_bboxes_i,
                                    o_label_texts_i,
                                    score_thr=0,
                                    thickness=2,
                                    font_scale=.75,
                                    show=False,
                                    win_name='',
                                    wait_time=0,
                                    out_file=out_path)


def visualize_pseudo_relations_CLIP_RTagger(
    CLIP_anno_path,
    RTagger_anno_path,
    one_rel_per_img = False,
    relation_threshold = 0.):

    coco2017_folder = Path('/Path/To/data/coco2017')
    save_folder_CLIP = Path('/Path/To/jacob/visualization/pseudo_relations_CLIP_2')
    save_folder_RTagger = Path('/Path/To/jacob/visualization/pseudo_relations_R-Tagger')
    with open(CLIP_anno_path, 'r') as f:
        CLIP_rel_annos = json.load(f)
    with open(RTagger_anno_path, 'r') as f:
        RTagger_rel_annos = json.load(f)

    RTagger_rel_annos_dict = {int(rel_anno['image_id']):rel_anno for rel_anno in RTagger_rel_annos}
    CLIP_rel_annos_dict = {int(rel_anno['image_id']):rel_anno for rel_anno in CLIP_rel_annos}
    CLIP_rel_annos_dict = dict(sorted(CLIP_rel_annos_dict.items(), key = lambda item:item[0], reverse = True))
    CLIP_rel_annos = list(CLIP_rel_annos_dict.values())

    draw_idx = 0 # until it reaches a pre-defined number like: 500.
    for img_anno in CLIP_rel_annos:
        if int(img_anno['image_id']) not in RTagger_rel_annos_dict.keys():
            continue
        draw_idx += 1
        print('xxxxx')
        if draw_idx > 2000:
            break

        # Visualize CLIP
        h_bboxes = []
        h_label_texts = []
        o_bboxes = []
        o_label_texts = []

        rels = img_anno["relationships"]
        objs = img_anno["objects"]
        objs_dict = {o["object_id"]:o for o in objs}
        if len(rels) == 0:
            continue 
        for rel in rels:
            if relation_threshold > 0.:
                if "confidence" in rel.keys():
                    if rel["confidence"] < relation_threshold:
                        continue
            sub_box = objs_dict[rel["subject_id"]]
            obj_box = objs_dict[rel["object_id"]]
            h_bboxes.append(np.array([sub_box["x"], sub_box["y"], sub_box["x"]+sub_box["w"], sub_box["y"]+sub_box["h"]]))
            h_label_texts.append(f"{sub_box['names']} {rel['predicate']}")
            o_bboxes.append(np.array([obj_box["x"], obj_box["y"], obj_box["x"]+obj_box["w"], obj_box["y"]+obj_box["h"]]))
            o_label_texts.append(obj_box["names"])
        if len(h_bboxes)==0:
            continue
        h_bboxes = np.stack(h_bboxes, axis = 0)
        o_bboxes = np.stack(o_bboxes, axis = 0)

        if img_anno["dataset"] == "coco2017":
            img_file_name = str(img_anno['image_id']).zfill(12) + '.jpg'
            if 'data_split' not in img_anno.keys():
                img_path = coco2017_folder / 'train2017' / img_file_name
            else:
                img_path = coco2017_folder / img_anno['data_split'] / img_file_name
        
        if not one_rel_per_img:
            out_path = save_folder_CLIP / img_file_name
            imshow_det_hoi_bboxes(img_path,
                            h_bboxes,
                            h_label_texts,
                            o_bboxes,
                            o_label_texts,
                            score_thr=0,
                            thickness=2,
                            font_scale=.5,
                            show=False,
                            win_name='',
                            wait_time=0,
                            out_file=out_path)
        else:
            for rel_idx in range(len(h_bboxes)):
                out_path = save_folder_CLIP / f"{str(img_anno['image_id']).zfill(12)}_{rel_idx}.jpg"
                h_bboxes_i = h_bboxes[rel_idx][None,:]
                h_label_texts_i = [h_label_texts[rel_idx]]
                o_bboxes_i = o_bboxes[rel_idx][None,:]
                o_label_texts_i = [o_label_texts[rel_idx]]
                imshow_det_hoi_bboxes(img_path,
                                    h_bboxes_i,
                                    h_label_texts_i,
                                    o_bboxes_i,
                                    o_label_texts_i,
                                    score_thr=0,
                                    thickness=2,
                                    font_scale=.75,
                                    show=False,
                                    win_name='',
                                    wait_time=0,
                                    out_file=out_path)
        
        # Visualize R-Tagger
        img_anno = RTagger_rel_annos_dict[int(img_anno['image_id'])]
        h_bboxes = []
        h_label_texts = []
        o_bboxes = []
        o_label_texts = []

        rels = img_anno["relationships"]
        objs = img_anno["objects"]
        objs_dict = {o["object_id"]:o for o in objs}
        if len(rels) == 0:
            continue 
        for rel in rels:
            if relation_threshold > 0.:
                if "confidence" in rel.keys():
                    if rel["confidence"] < relation_threshold:
                        continue
            sub_box = objs_dict[rel["subject_id"]]
            obj_box = objs_dict[rel["object_id"]]
            h_bboxes.append(np.array([sub_box["x"], sub_box["y"], sub_box["x"]+sub_box["w"], sub_box["y"]+sub_box["h"]]))
            h_label_texts.append(f"{sub_box['names']} {rel['predicate']}")
            o_bboxes.append(np.array([obj_box["x"], obj_box["y"], obj_box["x"]+obj_box["w"], obj_box["y"]+obj_box["h"]]))
            o_label_texts.append(obj_box["names"])
        if len(h_bboxes)==0:
            continue
        h_bboxes = np.stack(h_bboxes, axis = 0)
        o_bboxes = np.stack(o_bboxes, axis = 0)

        if img_anno["dataset"] == "coco2017":
            img_file_name = str(img_anno['image_id']).zfill(12) + '.jpg'
            if 'data_split' not in img_anno.keys():
                img_path = coco2017_folder / 'train2017' / img_file_name
            else:
                img_path = coco2017_folder / img_anno['data_split'] / img_file_name
        
        if not one_rel_per_img:
            out_path = save_folder_RTagger / img_file_name
            imshow_det_hoi_bboxes(img_path,
                            h_bboxes,
                            h_label_texts,
                            o_bboxes,
                            o_label_texts,
                            score_thr=0,
                            thickness=2,
                            font_scale=.5,
                            show=False,
                            win_name='',
                            wait_time=0,
                            out_file=out_path)
        else:
            for rel_idx in range(len(h_bboxes)):
                out_path = save_folder_RTagger / f"{str(img_anno['image_id']).zfill(12)}_{rel_idx}.jpg"
                h_bboxes_i = h_bboxes[rel_idx][None,:]
                h_label_texts_i = [h_label_texts[rel_idx]]
                o_bboxes_i = o_bboxes[rel_idx][None,:]
                o_label_texts_i = [o_label_texts[rel_idx]]
                imshow_det_hoi_bboxes(img_path,
                                    h_bboxes_i,
                                    h_label_texts_i,
                                    o_bboxes_i,
                                    o_label_texts_i,
                                    score_thr=0,
                                    thickness=2,
                                    font_scale=.75,
                                    show=False,
                                    win_name='',
                                    wait_time=0,
                                    out_file=out_path)


def transform_to_verb_tagger_input_format(
    scene_graph_path = None,
    bbox_path = None,
    save_path_rel_texts_for_coco_images = None,
    match_strategy = 'original_text',
    bbox_overlap = False,
):
    '''
    This function transforms the parsed scene graphs into a form which can be processed by the Verb Tagger.
    The beginning part of code is identical to the function coco_scene_graph_match_with_bbox.

    Args:
        scene_graph_path (string): a string indicating the path for scene graphs
        bbox_path (list): a list of path (support for coco datasetc containing train2017 and val2017)
        save_path_rel_texts_for_coco_images (string): a path to store the annotations for pseudo relations
        match_strategy (string): A choice from ['original_text', 'paraphrases'], indicating whether we should use paraphrases to tell that 
                                    parsed relations are valid using COCO GT bbox annotations.
        bbox_overlap (bool): whether we use "Overlap" as a prior knowledge to filter out pairs.

    Returns:
        None
    '''
    if scene_graph_path:
        with open(scene_graph_path, 'r') as f:
            sng_parsed = json.load(f)
    sng_parsed = grammartical_tranform(sng_parsed)
    if bbox_path:
        if len(bbox_path) == 1:
            with open(bbox_path[0], 'r') as f:
                bboxes = json.load(f)["annotations"]
        elif len(bbox_path) == 2:
            with open(bbox_path[0], 'r') as f:
                train_bboxes = json.load(f)["annotations"]
            with open(bbox_path[1], 'r') as f:
                val_bboxes = json.load(f)["annotations"]
            # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
            bboxes = train_bboxes + val_bboxes
    num_queries = 200
    num_pairs = num_queries//2
    
    # convert bboxes to a dict with image_id as keys
    bboxes_dict = {}
    for bbox in bboxes:
        image_id = str(bbox['image_id'])
        if image_id not in bboxes_dict.keys():
            bboxes_dict[image_id] = [bbox,]
        else:
            bboxes_dict[image_id].append(bbox)
    
    ### We aggregate all object names in a given coco image.
    object_80_dict = load_hico_object_txt()
    object_80_list = list(object_80_dict.keys())
    bboxes_names_dict = {}
    for image_id, bboxes_img in bboxes_dict.items():
        assert image_id not in bboxes_names_dict.keys()
        bboxes_names_dict[image_id] = []
        for one_bbox_img in bboxes_img:
            if object_80_dict[one_bbox_img["category_id"]] not in bboxes_names_dict[image_id]:
                bboxes_names_dict[image_id].append(object_80_dict[one_bbox_img["category_id"]])
    print(f"{len(bboxes_names_dict.keys())} images are in the trainval2017.")
    print(f"However, {len(sng_parsed)} images are in the COCO caption2014.\n")
    # print(list(bboxes_names_dict.keys())[:100])
    # print(list(sng_parsed.keys())[:100])

    ### We filter out relations which do not have corresponding objects in the image.
    # we firstly filter out parsed 'entities' if they are not within 80 classes
    matcher = MatchWithGTbboxes(match_strategy = match_strategy)
    sng_new1 = {}  # This excludes the images that they do not appear in the trainval2017 (caption2014 has more).
    for image_id, graph_list in sng_parsed.items():
        if image_id in bboxes_names_dict.keys():
            bboxes_names = bboxes_names_dict[image_id]
            graph_list_new = []
            for graph in graph_list:
                # new_graph = match_one_img_bboxes_with_one_scene_graph(
                #                 bboxes_names = bboxes_names,
                #                 scene_graph_one_sent = graph)
                new_graph = matcher.match_one_img_bboxes_with_one_scene_graph(
                                bboxes_names = bboxes_names,
                                scene_graph_one_sent = graph)
                if len(new_graph['relations']) > 0:
                    graph_list_new.append(new_graph)
            sng_new1[image_id] = graph_list_new
        else:
            continue
    print(f"After filtering, we have {len(sng_new1)} images with parsed scene graphs.\n")

    ### We match relations with given GT bbox annotations.
    # We traverse all the filtered relation scene graphs to obatin all possible triplets.
    # We traverse all the possible pairs to obtain pairs with valid subject classes and object classes.
    # By default, I will not use "Overlap" as prior knowledge to lower the number of possible pairs.
    Coco = CocoDetection(img_folder = '/Path/To/data/coco2017/train2017',
                         ann_file = '/Path/To/data/coco2017/annotations/instances_train2017.json',
                         transforms=make_coco_transforms('val'),
                         return_masks=False)
    official_coco_bbox = transform_coco_official_to_VG_format(Coco)
    
    coco_start_rel_idx = 10000000
    pseudo_relations_coco = []
    rel_cand = {}
    for img_idx, (image_id, scene_graphs) in enumerate(sng_new1.items()):
        if (img_idx+1) % 10000 == 0:
            print(f'Matching scene graphs with images: {img_idx+1}/{len(sng_new1)}')
            # print(rel_cand[int(image_id)])

        rels_img = []
        bboxes_img = official_coco_bbox[image_id]
        # bboxes_img = bboxes_dict[image_id]
        # bboxes_img = transform_coco_bbox_to_VG_format(coco_bbox = bboxes_img,
        #                                               object_80_dict = object_80_dict)
        num_obj = len(bboxes_img)
        possible_pairs = list(permutations(range(0, num_obj), 2))
        # possible_pairs_text = [[text_labels[pair[0]], text_labels[pair[1]]] for pair in possible_pairs]
        # num_possible_pairs = len(possible_pairs)

        relationships_coco_name_list = []
        relationships_span_name_list = []
        for scene_graph in scene_graphs:
            for triplet in scene_graph['relations']:
                relationships_coco_name_list.append([
                    triplet['subject_coco_name'], triplet['relation'], triplet['object_coco_name']
                ])
                # We define this list containing span names for further usage if span name is better.
                relationships_span_name_list.append([
                    triplet['subject_span'], triplet['relation'], triplet['object_span']
                ])
        
        valid_pairs = []
        valid_rel_texts = []
        for pair in possible_pairs:
            sub_obj = bboxes_img[pair[0]]
            obj_obj = bboxes_img[pair[1]]
            sub_text = sub_obj["names"]
            obj_text = obj_obj["names"]
            if bbox_overlap:
                if not check_overlap([sub_obj["x"], sub_obj["y"], sub_obj["w"], sub_obj["h"]], 
                                     [obj_obj["x"], obj_obj["y"], obj_obj["w"], obj_obj["h"]]):
                    continue

            for coco_name_triplet in relationships_coco_name_list:
                if coco_name_triplet[0] == sub_text and coco_name_triplet[2] == obj_text:
                    valid_pairs.append(pair)
                    valid_rel_texts.append(coco_name_triplet[1])
        
        num_groups = len(valid_pairs)//num_pairs + 1
        rel_cand[int(image_id)] = []
        for i in range(0, num_groups):
            if i == num_groups-1:
                i_pairs = valid_pairs[i*num_pairs:]
                i_pair_texts = valid_rel_texts[i*num_pairs:]
            else:
                i_pairs = valid_pairs[i*num_pairs:(i+1)*num_pairs]
                i_pair_texts = valid_rel_texts[i*num_pairs:(i+1)*num_pairs]

            # Merge pair texts
            i_rel_texts = []
            for t in i_pair_texts:
                if t not in i_rel_texts:
                    i_rel_texts.append(t)
            
            rel_cand[int(image_id)].append([i_pairs, i_rel_texts])
        
    if save_path_rel_texts_for_coco_images:
        with open(save_path_rel_texts_for_coco_images, "w") as f:
            json.dump(rel_cand, f)
        print(f"Saving to {save_path_rel_texts_for_coco_images}.")


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

def grammartical_tranform(sng):
    '''
    This function aims to make some grammatical transformation to the texts.
    By now, I only change all 'span' texts to lower cases.
    
    Args:
        sng (dict): This is a dict containing all the scene graphs in one dataset.
    
    Returns:
        sng (dict): This is a dict containing transformed version of sng.
    '''
    object_dist = {}
    for image_id, scene_graph in sng.items():
        for one_graph in scene_graph:
            for one_obj in one_graph['entities']:
                one_obj['span'] = one_obj['span'].lower()
    return sng


def prepare_CLIP_features(
    relationships_name_list,
    bboxes_img,
    img_file,
    model, 
    preprocess,
    device,
    img_batch_size = 64):
    '''
    This function prepares all the visual features for the minimum bounding rectangles and 
    all the text features for relation triplets.

    Args:
        relationships_name_list (list): a list containing all possible triplets.
        bboxes_img (list): a list containing all bboxes in the image.
        img_file (PIL image): the image file read by PIL package.
        model (CLIP model): the CLIP model.
        preprocess (CLIP preprocess): preprocessing steps from the CLIP model.
        device (device): device.
        img_batch_size (int): the batch size we use when performing image inference.

    Returns:
        CLIP_vis_features (torch tensors): a tensor containing all features for all the minimum bounding rectangles.
        CLIP_text_features (torch tensors): a tensor containing all text features for relation triplets.
    '''
    all_min_bounding_rects = []
    all_min_rects_idx = []
    for sub_idx, sub_bbox in enumerate(bboxes_img):
        for obj_idx, obj_bbox in enumerate(bboxes_img):
            if sub_bbox["w"] == 0 or sub_bbox["h"] == 0 or obj_bbox["w"] == 0 or obj_bbox["h"] == 0:
                continue

            min_bounding_rect = (min(sub_bbox["x"], obj_bbox["x"]), min(sub_bbox["y"], obj_bbox["y"]),
                                 max(sub_bbox["x"]+sub_bbox["w"], obj_bbox["x"]+obj_bbox["w"]), 
                                 max(sub_bbox["y"]+sub_bbox["h"], obj_bbox["y"]+obj_bbox["h"]))
            min_bounding_region = img_file.crop(min_bounding_rect)
            # We have to ensure that the cropped region has w > 0 and h > 0.
            if min_bounding_region.size[0] == 0 or min_bounding_region.size[1] == 0:
                continue

            min_bounding_region = preprocess(min_bounding_region).unsqueeze(0).to(device)
            all_min_bounding_rects.append(min_bounding_region)
            all_min_rects_idx.append((sub_idx, obj_idx))
    all_min_bounding_rects = torch.cat(all_min_bounding_rects, dim = 0)

    text = [f"a photo of {triplet_name[0]} {triplet_name[1]} {triplet_name[2]}" for triplet_name in relationships_name_list]
    # background_text = [f"a photo of {triplet_name[0]} not interacting with {triplet_name[2]}" for triplet_name in relationships_name_list]
    background_text = [f"a photo of {triplet_name[0]} having no relation with {triplet_name[2]}" for triplet_name in relationships_name_list]
    text_tokenized = clip.tokenize(text).to(device)
    background_text_tokenized = clip.tokenize(background_text).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokenized)
        background_text_features = model.encode_text(background_text_tokenized)
        text_features_dict = {t:f for t,f in zip(text, text_features)}
        text_features_dict.update({t:f for t,f in zip(background_text, background_text_features)})
        
        bs_img = torch.split(all_min_bounding_rects, img_batch_size)
        region_features = torch.cat([model.encode_image(one_bs_img) for one_bs_img in bs_img])
        assert region_features.shape[0] == len(all_min_rects_idx)
        region_features_dict = {i:f for i,f in zip(all_min_rects_idx, region_features)}
        # for sub_idx in range(len(bboxes_img)):
        #     for obj_idx in range(len(bboxes_img)):
        #         region_features_dict[(sub_idx, obj_idx)] = region_features[sub_idx*len(bboxes_img) + obj_idx]
                # 应该在上面就把region和（sub idx，obj idx对应起来）
        
    return region_features_dict, text_features_dict


class MatchWithGTbboxes():
    def __init__(self, match_strategy):
        self.match_strategy = match_strategy
        if match_strategy == 'paraphrases':
            paraphrase_file = "/Path/To/jacob/RLIP/datasets/priors/hico_obj_paraphrase.json"
            with open(paraphrase_file, 'r') as f:
                self.obj_paraphrase = json.load(f)

    def match_with_paraphrases(self, coco_name, entity_span):
        '''
        We use human-collected paraphrases to match the name to the entity span.

        Args:
            coco_name (string): this is the string indicating the class text of this category.
            entity_span (string): this is the span name in the parsed sentence, it is usually free-form texts.
        
        Returns:
            match_flag (bool): this is the flag indicating whether they are matched.
        '''
        coco_name_paraphrases = self.obj_paraphrase[coco_name]
        match_flag = False
        for paraphrase in coco_name_paraphrases:
            if paraphrase in entity_span:
                match_flag = True
        return match_flag
    
    def match_one_img_bboxes_with_one_scene_graph(
            self,
            bboxes_names,
            scene_graph_one_sent,
            # match_strategy = 'original_text',
        ):
        '''
        Args:
            bboxes_names (list): This is a list containing all the bbox names in an image (obtained from GT bbox annotations).
            scene_graph_one_sent (dict): This is the scene graph parsed from one sentence.
            match_strategy (string): A choice from ['original_text', 'paraphrases'], indicating whether we should use paraphrases to tell that 
                                    parsed relations are valid using COCO GT bbox annotations.
        
        Returns:
            new_scene_graph (dict): This dict exlcludes all the entities not covered by the :param:bboxes_names.
                !!! Note: We add its corresponding 1. coco class name
                                                   2. lemma_span of the subject and object
                                                      into this dict.
        '''
        new_scene_graph = {'entities':scene_graph_one_sent['entities'], 
                           'relations':[]}
        # We keep all the entities to ensure that we do not need to change the subject/object (id) in relations.
        
        keep_entity_idx = []
        for entity_idx, entity in enumerate(scene_graph_one_sent['entities']):
            for name in bboxes_names:
                if self.match_strategy == 'original_text':
                    if name in entity['span']:
                        entity['coco_name'] = name
                        keep_entity_idx.append(entity_idx)
                elif self.match_strategy == 'paraphrases':
                    if self.match_with_paraphrases(coco_name = name, entity_span = entity['span']):
                        entity['coco_name'] = name
                        keep_entity_idx.append(entity_idx)
                else:
                    assert False
                
        for rel in scene_graph_one_sent['relations']:
            if rel['subject'] in keep_entity_idx and rel['object'] in keep_entity_idx:
                rel['subject_coco_name'] = new_scene_graph['entities'][rel['subject']]['coco_name']
                rel['object_coco_name'] = new_scene_graph['entities'][rel['object']]['coco_name']
                rel['subject_span'] = new_scene_graph['entities'][rel['subject']]['span']
                rel['object_span'] = new_scene_graph['entities'][rel['object']]['span']
                new_scene_graph['relations'].append(rel)
        
        return new_scene_graph


def compute_relation_distribution_from_scene_graphs(sng):
    '''
    Args:
        sng (dict): This is a dict containing all the scene graphs in one dataset.
            The format is like:
            {
                '12345':[
                    {
                        'entities': a list,
                        'relations': a list,
                    },
                    {
                        'entities': a list,
                        'relations': a list,
                    },
                    ...
                ],
                '23456':...,
                '34567':...,
            }
    
    Returns
        relation_dist (dict): a new dict, showing the relation distribution of this dataset.
    '''
    relation_dist = {}
    for image_id, scene_graph in sng.items():
        for one_graph in scene_graph:
            # print(one_graph)
            for one_rel in one_graph['relations']:
                if one_rel['relation'] not in relation_dist.keys():
                    relation_dist[one_rel['relation']] = 1
                else:
                    relation_dist[one_rel['relation']] += 1
    relation_dist = dict(sorted(relation_dist.items(), key = lambda item:item[1], reverse = True))

    return relation_dist


def compute_object_distribution_from_scene_graphs(sng):
    '''
    Args:
        sng (dict): This is a dict containing all the scene graphs in one dataset.
            The format is like:
            {
                '12345':[
                    {
                        'entities': a list,
                        'relations': a list,
                    },
                    {
                        'entities': a list,
                        'relations': a list,
                    },
                    ...
                ],
                '23456':...,
                '34567':...,
            }
    
    Returns
        object_dist (dict): a new dict, showing the object distribution of this dataset.
    '''
    object_dist = {}
    for image_id, scene_graph in sng.items():
        for one_graph in scene_graph:
            # for one_obj in one_graph['entities']:
            entities = one_graph['entities']
            for one_rel in one_graph['relations']:
                sub_span = entities[one_rel['subject']]['span']
                obj_span = entities[one_rel['object']]['span']
                if sub_span not in object_dist.keys():
                    object_dist[sub_span] = 1
                else:
                    object_dist[sub_span] += 1
                
                if obj_span not in object_dist.keys():
                    object_dist[obj_span] = 1
                else:
                    object_dist[obj_span] += 1

    object_dist = dict(sorted(object_dist.items(), key = lambda item:item[1], reverse = True))

    return object_dist


def check_span_lemma_span(sng):
    '''
    This function tries to find the differences of 'span' and 'lemma_span'.
    
    Args:
        sng (dict): This is a dict containing all the scene graphs in one dataset.
            The format is like:
            {
                '12345':[
                    {
                        'entities': a list,
                        'relations': a list,
                    },
                    {
                        'entities': a list,
                        'relations': a list,
                    },
                    ...
                ],
                '23456':...,
                '34567':...,
            }
        
    Returns:
        None
    '''
    # From my observation, they mainly differ in grammartical form, like the capital letters and plural forms.
    for image_id, scene_graph in sng.items():
        for one_graph in scene_graph:
            for one_obj in one_graph['entities']:
                if one_obj['span'].lower() != one_obj['lemma_span']:
                    print(one_obj['span'], ' ', one_obj['lemma_span'])


def compute_relation_distribution_from_vg_format(vg_format_list):
    '''
    Args:
        vg_format_list: This is a list of dicts containing annotations of the VG format.
            The format of the dict is like:
                {
                    "image_id": 2407890,
                    "objects": [...
                        {"object_id": 1023838, "x": 324, "y": 320, "w": 142, "h": 255, 
                            "names": "cat","synsets": ["cat.n.01"]},
                        {"object_id":  5071, "x": 359, "y": 362, "w": 72, "h": 81,
                            "names": "table", "synsets": ["table.n.01"]},
                    ...],
                    "relationships": [...
                        {"relationship_id": 15947, "predicate": "wears", "synsets": ["wear.v.01"],
                        "subject_id": 1023838, "object_id":  5071}
                    ...]
                }
    
    Returns:
    :return: a new dict, showing the relation distribution of this dataset.
    '''
    relation_dist = {}
    for vg_format_anno in vg_format_list:
        for one_rel in vg_format_anno["relationships"]:
            if one_rel['predicate'] not in relation_dist.keys():
                relation_dist[one_rel['predicate']] = 1
            else:
                relation_dist[one_rel['predicate']] += 1

    relation_dist = dict(sorted(relation_dist.items(), key = lambda item:item[1], reverse = True))

    return relation_dist

def transform_coco_bbox_to_VG_format(coco_bbox, object_80_dict):
    '''
    This function transforms the bbox annotations of COCO dataset to the VG format.
    Args:
        coco_bbox (list): a list of the COCO format.
        object_80_dict (dict): a dict that maps the 'category_id' to the coco class text.
    
    Returns:
        vg_bbox (list) : a list of the COCO format.
    '''
    vg_bbox = []
    for bbox in coco_bbox:
        if bbox["bbox"][2] > 0 and bbox["bbox"][3] > 0:   # This is to ensure boxes are valid.
            vg_bbox.append(
                {
                    "object_id": bbox["id"],
                    "x": bbox["bbox"][0],
                    "y": bbox["bbox"][1],
                    "w": bbox["bbox"][2],
                    "h": bbox["bbox"][3],
                    "names": object_80_dict[bbox["category_id"]],
                }
            )

    return vg_bbox
        
def check_overlap(bbox1, bbox2):
    '''
    This function aims to check whether two bboxes are overlapped.
    If so, it returns True, otherwise it returns False.
    
    Args:
        bbox1 (list): [x, y, w, h]
        bbox2 (list): [x, y, w, h]

    Returns:
        overlap_flag (bool): whether two bboxes are overlapped.
    '''
    cx1, cy1 = bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2
    cx2, cy2 = bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2
    if (abs(cx1 - cx2) < bbox1[2]/2 + bbox2[2]/2) and (abs(cy1 - cy2) < bbox1[3]/2 + bbox2[3]/2):
        return True
    else:
        return False

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


def imshow_det_hoi_bboxes(img,
                      h_bboxes,
                      h_labels,
                      o_bboxes,
                      o_labels,
                      link_ho=False,
                      h_class_names=None,
                      o_class_names=None,
                      score_thr=0,
                      h_bbox_color='green',
                      h_text_color='green',
                      o_bbox_color='red',
                      o_text_color='red',
                      thickness=1,
                      font_scale=0.5,
                      show=False,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        link_ho (bool): Indicate whether we draw a line linking subject and object box.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert h_bboxes.ndim == 2 and o_bboxes.ndim == 2
    # assert h_labels.ndim == 1 and o_labels.ndim == 1
    assert h_bboxes.shape[0] == len(h_labels) and o_bboxes.shape[0] == len(o_labels)
    assert h_bboxes.shape[1] == 4 or h_bboxes.shape[1] == 5
    assert o_bboxes.shape[1] == 4 or o_bboxes.shape[1] == 5
    img = imread(img)
    img = np.ascontiguousarray(img)

    if score_thr > 0:
        assert h_bboxes.shape[1] == 5 # scores are only for human verbs
        scores = h_bboxes[:, -1]
        inds = scores > score_thr
        h_bboxes = h_bboxes[inds, :]
        h_labels = h_labels[inds]
        o_bboxes = o_bboxes[inds, :]
        o_labels = o_labels[inds]

    h_bbox_color = color_val(h_bbox_color)
    h_text_color = color_val(h_text_color)
    o_bbox_color = color_val(o_bbox_color)
    o_text_color = color_val(o_text_color)

    for h_bbox, h_label in zip(h_bboxes, h_labels):
        h_bbox_int = h_bbox.astype(np.int32)
        left_top = (h_bbox_int[0], h_bbox_int[1])
        right_bottom = (h_bbox_int[2], h_bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, h_bbox_color, thickness=thickness)
        h_label_text = h_class_names[
            h_label] if h_class_names is not None else f'{h_label}'
        if len(h_bbox) > 4:
            h_label_text += f'|{h_bbox[-1]:.02f}'
        cv2.putText(img, h_label_text, (h_bbox_int[0], h_bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, h_text_color)
    
    for o_bbox, o_label in zip(o_bboxes, o_labels):
        o_bbox_int = o_bbox.astype(np.int32)
        left_top = (o_bbox_int[0], o_bbox_int[1])
        right_bottom = (o_bbox_int[2], o_bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, o_bbox_color, thickness=thickness)
        o_label_text = o_class_names[
            o_label] if o_class_names is not None else f'{o_label}'
        if len(o_bbox) > 4:
            o_label_text += f'|{o_bbox[-1]:.02f}'
        cv2.putText(img, o_label_text, (o_bbox_int[0], o_bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, o_text_color)
    
    for h_bbox, o_bbox in zip(h_bboxes, o_bboxes):
        h_bbox_int = h_bbox.astype(np.int32)
        h_cxcy = (((h_bbox_int[0]+h_bbox_int[2])/2.).astype(np.int32), ((h_bbox_int[1]+h_bbox_int[3])/2.).astype(np.int32))
        o_bbox_int = o_bbox.astype(np.int32)
        o_cxcy = (((o_bbox_int[0]+o_bbox_int[2])/2.).astype(np.int32), ((o_bbox_int[1]+o_bbox_int[3])/2.).astype(np.int32))
        cv2.line(img, h_cxcy, o_cxcy, color=h_bbox_color, thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
    return img

class Color(Enum):
    """An enum that defines common colors.
    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color):
    """Convert various input to color tuples.
    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs
    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if is_str(color):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')

def load_hico_object_txt(file_path = '/Path/To/jacob/RLIP/datasets/hico_object_names.txt'):
    '''
    Output like 
        ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',]
    '''
    with open(file_path, 'r') as f:
        object_names = json.load(f)
    object_dict = {j:i for i,j in object_names.items()}
    return object_dict

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


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

if __name__=="__main__":
    # transform_sentences_to_triplets()
    # transform_sentences_to_triplets(save_path_sng = '/Path/To/data/annotations_trainval2014/SceneGraph_OracleCaps_trainval2014.json')

    # coco_scene_graph_match_with_bbox(
    #     scene_graph_path = '/Path/To/data/annotations_trainval2014/SceneGraph_OracleCaps_trainval2014.json',
    #     bbox_path = ['/Path/To/data/coco2017/annotations/instances_train2017.json',
    #                  '/Path/To/data/coco2017/annotations/instances_val2017.json'],
    #     save_path_pseudo_relations = '/Path/To/data/coco2017/annotations/RLIPv2_trainval2017_OracleCaps_Paraphrases_CLIP_ViT-L14_Attr_Confidence_Overlap.json',
    #     match_strategy = 'paraphrases',
    #     CLIP_mode = 'ViT-B/16')
    # 'ViT-B/32'
    # 'ViT-B/16'
    # 'ViT-L/14'
    # RLIPv2_trainval2017_OracleCaps_Overlap_Paraphrases.json

    # visualize_pseudo_relations(anno_path = '/Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_OracleCaps_Paraphrases.json',
    #                            one_rel_per_img = True,
    #                            relation_threshold = 0.2)
    # '/Path/To/data/coco2017/annotations/RLIPv2_trainval2017_OracleCaps_Overlap_Paraphrases.json'
    # RLIPv2_trainval2017_OracleCaps_Paraphrases_CLIP_ViT-L14_Attr_Confidence_Overlap.json

    # transform_to_verb_tagger_input_format(
    #     scene_graph_path = '/Path/To/data/annotations_trainval2014/SceneGraph_OracleCaps_trainval2014.json',
    #     bbox_path = ['/Path/To/data/coco2017/annotations/instances_train2017.json',],
    #                 #  '/Path/To/data/coco2017/annotations/instances_val2017.json'],
    #     save_path_rel_texts_for_coco_images = '/Path/To/data/coco2017/annotations/SceneGraph_OracleCaps_Paraphrases_rel_texts_for_coco_images.json',
    #     match_strategy = 'paraphrases',
    #     bbox_overlap = False)


    ### Visualization of the paper
    # visualize_pseudo_relations(anno_path = '/Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05.json',
    #                            one_rel_per_img = True,
    #                            relation_threshold = 0.2)

    # visualize_pseudo_relations(anno_path = '/Path/To/data/coco2017/annotations/RLIPv2_trainval2017_OracleCaps_Paraphrases_CLIP_ViT-L14_thre08.json',
    #                            one_rel_per_img = True,
    #                            relation_threshold = 0.2)

    ### Joint Visualization of CLIP and R-Tagger
    visualize_pseudo_relations_CLIP_RTagger(
        CLIP_anno_path = '/Path/To/data/coco2017/annotations/RLIPv2_trainval2017_OracleCaps_Paraphrases_CLIP_ViT-L14_thre08.json',
        RTagger_anno_path = '/Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05.json',
        one_rel_per_img = False,
        relation_threshold = 0.2)

        
                   