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
import sng_parser
import numpy as np
from PIL import Image
import clip
from coco_caption_helper import transform_coco_official_to_VG_format, grammartical_tranform, check_overlap, load_hico_object_txt, MatchWithGTbboxes, \
        compute_relation_distribution_from_scene_graphs, compute_object_distribution_from_scene_graphs, CocoDetection, make_coco_transforms

def transform_BLIP_sentences_to_triplets(
    coco_captions_BLIP_path = None,
    save_path_sng = None,
):
    '''
    This function transforms the BLIP captions into scene graphs.

    Args:
        coco_captions_BLIP_path (string): 
        save_path_sng (string): 
    
    Returns:
        None
    '''
    # coco_captions_BLIP_path = '/Path/To/data/coco2017/annotations/BLIP_captions/model_large_caption_nucleus5.json'
    with open(coco_captions_BLIP_path, 'r') as f:
        coco_captions = json.load(f)

    coco_caps = {}
    sng_parsed = {}
    start_t = time.time()
    global_caption_id = 0
    for cap_idx, (image_id, image_caps) in enumerate(coco_captions.items()):
        if (cap_idx+1)%5000 == 0:
            print(f"Processing {cap_idx+1} / {len(coco_captions)}")
        
        cap_graph = []
        for one_cap in image_caps:
            cap_graph.append(sng_parser.parse(one_cap))
        sng_parsed[image_id] = cap_graph

    
    print(f"{sum([len(s) for _,s in sng_parsed.items()])} captions are processed.")
    print(f"Processing time: {int(time.time() - start_t)}s.")
    
    ### Save to disk
    if save_path_sng is not None:
        with open(save_path_sng, "w") as f:
            json.dump(sng_parsed, f)
        print(f"Saving to {save_path_sng}.")



def transform_BLIP_sngs_to_verb_tagger_input_format(
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

    ### We give some stat about the left relations and objects in relations.   
    sng_parsed_dist = compute_relation_distribution_from_scene_graphs(sng_parsed)
    sng_new1_dist = compute_relation_distribution_from_scene_graphs(sng_new1)
    print(f"We have {sum(sng_parsed_dist.values())} relations in {len(sng_parsed_dist)} kinds before filtering.")
    print(f"We have {sum(sng_new1_dist.values())} relations in {len(sng_new1_dist)} kinds after filtering.\n")
    sng_parsed_obj_dist = compute_object_distribution_from_scene_graphs(sng_parsed)
    print(f"We have {sum(sng_parsed_obj_dist.values())} objects in {len(sng_parsed_obj_dist)} kinds before filtering.\n")
    

    ### We match relations with given GT bbox annotations.
    # We traverse all the filtered relation scene graphs to obatin all possible triplets.
    # We traverse all the possible pairs to obtain pairs with valid subject classes and object classes.
    # By default, I will not use "Overlap" as prior knowledge to lower the number of possible pairs.
    Coco_train = CocoDetection(img_folder = '/Path/To/data/coco2017/train2017',
                         ann_file = '/Path/To/data/coco2017/annotations/instances_train2017.json',
                         transforms=make_coco_transforms('val'),
                         return_masks=False)
    Coco_val = CocoDetection(img_folder = '/Path/To/data/coco2017/val2017',
                         ann_file = '/Path/To/data/coco2017/annotations/instances_val2017.json',
                         transforms=make_coco_transforms('val'),
                         return_masks=False)
    official_coco_bbox = transform_coco_official_to_VG_format(Coco_train)
    official_coco_bbox.update(transform_coco_official_to_VG_format(Coco_val))
    
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
                if [triplet['subject_coco_name'], triplet['relation'], triplet['object_coco_name']] not in relationships_coco_name_list:
                    relationships_coco_name_list.append([
                        triplet['subject_coco_name'], triplet['relation'], triplet['object_coco_name']
                    ])
                # We define this list containing span names for further usage if span name is better.
                if [triplet['subject_span'], triplet['relation'], triplet['object_span']] not in relationships_span_name_list:
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


def merge_instance_coco_trainval2017():
    train2017_path = '/Path/To/data/coco2017/annotations/instances_train2017.json'
    val2017_path = '/Path/To/data/coco2017/annotations/instances_val2017.json'



if __name__=="__main__":
    ### BLIP
    # Use generated captions 
    # transform_BLIP_sentences_to_triplets(
    #     coco_captions_BLIP_path = '/Path/To/data/coco2017/annotations/BLIP_captions/model_large_caption_nucleus20.json',
    #     save_path_sng = '/Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_caption_nucleus20_trainval2017.json',
    # )

    # transform_BLIP_sngs_to_verb_tagger_input_format(
    #     scene_graph_path = '/Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_caption_nucleus20_trainval2017.json',
    #     bbox_path = ['/Path/To/data/coco2017/annotations/instances_train2017.json',
    #                  '/Path/To/data/coco2017/annotations/instances_val2017.json'],
    #     save_path_rel_texts_for_coco_images = '/Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_caption_nucleus20_trainval2017_Paraphrases_rel_texts_for_coco_images.json',
    #     match_strategy = 'paraphrases',
    #     bbox_overlap = False)

    # transform_BLIP_sngs_to_verb_tagger_input_format(
    #     scene_graph_path = '/Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_beam_trainval2017.json',
    #     bbox_path = ['/Path/To/data/coco2017/annotations/instances_train2017.json',
    #                  '/Path/To/data/coco2017/annotations/instances_val2017.json'],
    #     save_path_rel_texts_for_coco_images = '/Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_beam_trainval2017_Paraphrases_rel_texts_for_coco_images.json',
    #     match_strategy = 'paraphrases',
    #     bbox_overlap = False)
    

    ### BLIP-2
    # # Use generated captions 
    # transform_BLIP_sentences_to_triplets(
    #     coco_captions_BLIP_path = '/Path/To/data/coco2017/annotations/BLIP2_captions/coco_opt2.7b_beam.json',
    #     save_path_sng = '/Path/To/data/coco2017/annotations/BLIP2_captions/SceneGraph_coco_opt2.7b_beam_trainval2017.json',
    # )

    # transform_BLIP_sngs_to_verb_tagger_input_format(
    #     scene_graph_path = '/Path/To/data/coco2017/annotations/BLIP2_captions/SceneGraph_coco_opt2.7b_beam_trainval2017.json',
    #     bbox_path = ['/Path/To/data/coco2017/annotations/instances_train2017.json',
    #                  '/Path/To/data/coco2017/annotations/instances_val2017.json'],
    #     save_path_rel_texts_for_coco_images = '/Path/To/data/coco2017/annotations/BLIP2_captions/SceneGraph_coco_opt2.7b_beam_trainval2017_Paraphrases_rel_texts_for_coco_images.json',
    #     match_strategy = 'paraphrases',
    #     bbox_overlap = False)


    # transform_BLIP_sentences_to_triplets(
    #     coco_captions_BLIP_path = '/Path/To/data/coco2017/annotations/BLIP2_captions/coco_opt2.7b_nucleus10.json',
    #     save_path_sng = '/Path/To/data/coco2017/annotations/BLIP2_captions/SceneGraph_coco_opt2.7b_nucleus10_trainval2017.json',
    # )

    # transform_BLIP_sngs_to_verb_tagger_input_format(
    #     scene_graph_path = '/Path/To/data/coco2017/annotations/BLIP2_captions/SceneGraph_coco_opt2.7b_nucleus10_trainval2017.json',
    #     bbox_path = ['/Path/To/data/coco2017/annotations/instances_train2017.json',
    #                  '/Path/To/data/coco2017/annotations/instances_val2017.json'],
    #     save_path_rel_texts_for_coco_images = '/Path/To/data/coco2017/annotations/BLIP2_captions/SceneGraph_coco_opt2.7b_nucleus10_trainval2017_Paraphrases_rel_texts_for_coco_images.json',
    #     match_strategy = 'paraphrases',
    #     bbox_overlap = False)