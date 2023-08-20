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
import os
from typing import Any, Callable, List, Optional, Tuple

import json
import sys
sys.path.append("..")
import transforms as T
# from o365 import O365Detection
import sng_parser
import numpy as np
from PIL import Image
import clip
from coco_caption_helper import transform_coco_official_to_VG_format, grammartical_tranform, check_overlap, load_hico_object_txt, MatchWithGTbboxes, \
        compute_relation_distribution_from_scene_graphs, compute_object_distribution_from_scene_graphs, CocoDetection, make_coco_transforms, \
            transform_coco_bbox_to_VG_format, imshow_det_hoi_bboxes

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
    with open(coco_captions_BLIP_path, 'r') as f:
        coco_captions = json.load(f)
    print(f"Finish loading captions:{coco_captions_BLIP_path}...")

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
    save_path_rel_texts_for_o365_images = None,
    match_strategy = 'original_text',
    bbox_overlap = False,
):
    '''
    This function transforms the parsed scene graphs into a form which can be processed by the Verb Tagger.
    The beginning part of code is identical to the function coco_scene_graph_match_with_bbox.

    Args:
        scene_graph_path (string): a string indicating the path for scene graphs
        bbox_path (list): a list of path (support for o365 dataset containing train and val)
        save_path_rel_texts_for_o365_images (string): a path to store the annotations for pseudo relations
        match_strategy (string): A choice from ['original_text', 'paraphrases'], indicating whether we should use paraphrases to tell that 
                                    parsed relations are valid using Objects 365 GT bbox annotations.
        bbox_overlap (bool): whether we use "Overlap" as a prior knowledge to filter out pairs.

    Returns:
        None
    '''
    start_time = time.time()
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
    print(f"Reading annotations costs {int(time.time() - start_time)} seconds.")
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
    
    ### We aggregate all object names in a given o365 image.
    start_time = time.time()
    object_365_dict = load_o365_categories()
    object_365_list = list(object_365_dict.keys())
    bboxes_names_dict = {}
    for image_id, bboxes_img in bboxes_dict.items():
        assert image_id not in bboxes_names_dict.keys()
        bboxes_names_dict[image_id] = []
        for one_bbox_img in bboxes_img:
            if object_365_dict[one_bbox_img["category_id"]] not in bboxes_names_dict[image_id]:
                bboxes_names_dict[image_id].append(object_365_dict[one_bbox_img["category_id"]])
    print(f"{len(bboxes_names_dict.keys())} images are in the o365.")
    print(f"Aggregating object names costs {int(time.time() - start_time)} seconds.")
    # print(f"However, {len(sng_parsed)} images are in the COCO caption2014.\n")

    ### We filter out relations which do not have corresponding objects in the image.
    # we firstly filter out parsed 'entities' if they are not within 80 classes
    start_time = time.time()
    matcher = MatchWithGTbboxes_O365(match_strategy = match_strategy)
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
    print(f"Filter out invalid relations costs {int(time.time() - start_time)} seconds.")

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
    # Coco_train = CocoDetection(img_folder = '/Path/To/data/coco2017/train2017',
    #                      ann_file = '/Path/To/data/coco2017/annotations/instances_train2017.json',
    #                      transforms=make_coco_transforms('val'),
    #                      return_masks=False)
    # Coco_val = CocoDetection(img_folder = '/Path/To/data/coco2017/val2017',
    #                      ann_file = '/Path/To/data/coco2017/annotations/instances_val2017.json',
    #                      transforms=make_coco_transforms('val'),
    #                      return_masks=False)
    # official_coco_bbox = transform_coco_official_to_VG_format(Coco_train)
    # official_coco_bbox.update(transform_coco_official_to_VG_format(Coco_val))
    

    O365_train = O365Detection(img_folder = '/Path/To/data/Objects365/',
                            ann_file = '/Path/To/data/Objects365/train/zhiyuan_objv2_train.json',
                            transforms=make_coco_transforms('val'),
                            return_masks=False,
                            image_id_to_filepath= '/Path/To/data/Objects365/image_id_to_filepath.json')
    O365_val = O365Detection(img_folder = '/Path/To/data/Objects365/',
                            ann_file = '/Path/To/data/Objects365/val/zhiyuan_objv2_val.json',
                            transforms=make_coco_transforms('val'),
                            return_masks=False,
                            image_id_to_filepath= '/Path/To/data/Objects365/image_id_to_filepath.json')
    official_o365_bbox = transform_o365_official_to_VG_format(O365_train) # Processing time: 1405 seconds
    official_o365_bbox.update(transform_o365_official_to_VG_format(O365_val))
    # official_o365_bbox = transform_o365_official_to_VG_format(O365_val)
    
    
    start_time = time.time()
    o365_start_rel_idx = 20000000
    pseudo_relations_o365 = []
    rel_cand = {}
    for img_idx, (image_id, scene_graphs) in enumerate(sng_new1.items()):
        if (img_idx+1) % 10000 == 0:
            print(f'Matching scene graphs with images: {img_idx+1}/{len(sng_new1)}')
            # print(rel_cand[int(image_id)])

        rels_img = []

        # Using official dataloader
        bboxes_img = official_o365_bbox[image_id]
        # Using bbox from the json file
        # bboxes_img = bboxes_dict[image_id]
        # bboxes_img = transform_coco_bbox_to_VG_format(coco_bbox = bboxes_img,
        #                                               object_80_dict = object_365_dict)
        
        num_obj = len(bboxes_img)
        possible_pairs = list(permutations(range(0, num_obj), 2))
        # possible_pairs_text = [[text_labels[pair[0]], text_labels[pair[1]]] for pair in possible_pairs]
        # num_possible_pairs = len(possible_pairs)

        relationships_o365_name_list = []
        relationships_span_name_list = []
        for scene_graph in scene_graphs:
            for triplet in scene_graph['relations']:
                if [triplet['subject_o365_name'], triplet['relation'], triplet['object_o365_name']] not in relationships_o365_name_list:
                    relationships_o365_name_list.append([
                        triplet['subject_o365_name'], triplet['relation'], triplet['object_o365_name']
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

            for o365_name_triplet in relationships_o365_name_list:
                if o365_name_triplet[0] == sub_text and o365_name_triplet[2] == obj_text:
                    valid_pairs.append(pair)
                    valid_rel_texts.append(o365_name_triplet[1])
        
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

    print(f"Match relations with given GT bbox annotations costs {int(time.time() - start_time)} seconds.")
        
    if save_path_rel_texts_for_o365_images:
        with open(save_path_rel_texts_for_o365_images, "w") as f:
            json.dump(rel_cand, f)
        print(f"Saving to {save_path_rel_texts_for_o365_images}.")



class O365Detection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, image_id_to_filepath):
        super(O365Detection, self).__init__(img_folder, ann_file, transforms, return_masks)
        with open(image_id_to_filepath, "r") as f:
            self.image_id_to_filepath = json.load(f)
        
        with open(ann_file, "r") as f:
            self.image_info = json.load(f)['images']
        self.image_info = {info["id"]:info for info in self.image_info}

        self.prepare = ConvertCocoPolysToMask(return_masks)
    
    def _load_image(self, id: int) -> Image.Image:
        # print(self.coco.loadImgs(id)[0].keys())
        # dict_keys(['height', 'id', 'license', 'width', 'file_name', 'url'])
        # print(self.coco.loadImgs(id))
        path = self.image_id_to_filepath[str(self.coco.loadImgs(id)[0]["id"])]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    
    # def _load_target(self, id: int) -> List[Any]:
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        target = self._load_target(id)# self.coco.loadAnns(self.coco.getAnnIds(id))
        image_id = self.coco.loadImgs(id)[0]["id"]
        target = {'image_id': image_id, 'annotations': target}
        image_info = self.image_info[image_id]
        target = self.prepare(target = target, w = image_info['width'], h = image_info['height'])
        # return self.coco.loadAnns(self.coco.getAnnIds(id))
        return target


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, target, w, h):
        # w, h = image.size

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
        if (~keep).sum()>0:
            print((~keep).sum())
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

        return target

def transform_o365_official_to_VG_format(O365):
    '''
    Args:
        O365 (class): This is a class to produce O365 annotations.
    
    Returns:
        official_bbox_dict (dict): a dict of annotations in the VG format.
    '''
    start_time = time.time()
    print(f"Running transform_o365_official_to_VG_format...")

    object_80_dict = load_o365_categories()
    official_bbox_dict = {}
    o365_start_obj_idx = 20000000
    for idx, o365_data in enumerate(O365):
        if (idx+1)%20000 == 0:
            print(f"Finish processing {idx+1}/{len(O365)}")

        vg_bbox = []
        # o365_target = O365.__getitem__(image_id) # This can be faster than __getitem__() because we do not load images from local files.
        o365_target = o365_data

        num_obj = o365_target['boxes'].shape[0]   # in the format of cxcywh
        o365_bbox = o365_target['boxes']
        labels = [int(l) for l in o365_target['labels']]
        box_labels = o365_target['labels']
        ### According to coco.py, target["orig_size"] = torch.as_tensor([int(h), int(w)]).
        img_h, img_w = o365_target['orig_size']

        for i in range(num_obj):
            vg_bbox.append({
                "x": (o365_bbox[i][0] - o365_bbox[i][2]/2.)*img_w,
                "y": (o365_bbox[i][1] - o365_bbox[i][3]/2.)*img_h,
                "w": o365_bbox[i][2]*img_w,
                "h": o365_bbox[i][3]*img_h,
                "object_id": o365_start_obj_idx,
                "names": object_80_dict[box_labels[i].item()],
            })
            o365_start_obj_idx+=1

        official_bbox_dict[str(o365_target['image_id'].item())] = vg_bbox
    print(f"Processing time: {int(time.time() - start_time)}")

    return official_bbox_dict



class MatchWithGTbboxes_O365():
    def __init__(self, match_strategy):
        self.match_strategy = match_strategy
        if match_strategy == 'paraphrases':
            # paraphrase_file = "/Path/To/jacob/RLIP/datasets/priors/hico_obj_paraphrase.json"
            # with open(paraphrase_file, 'r') as f:
            #     self.obj_paraphrase = json.load(f)
            self.obj_paraphrase = load_o365_paraphrases()

    def match_with_paraphrases(self, o365_name, entity_span):
        '''
        We use human-collected paraphrases to match the name to the entity span.

        Args:
            o365_name (string): this is the string indicating the class text of this category.
            entity_span (string): this is the span name in the parsed sentence, it is usually free-form texts.
        
        Returns:
            match_flag (bool): this is the flag indicating whether they are matched.
        '''
        o365_name_paraphrases = self.obj_paraphrase[o365_name]
        match_flag = False
        for paraphrase in o365_name_paraphrases:
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
                                    parsed relations are valid using o365 GT bbox annotations.
        
        Returns:
            new_scene_graph (dict): This dict exlcludes all the entities not covered by the :param:bboxes_names.
                !!! Note: We add its corresponding 1. o365 class name
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
                        entity['o365_name'] = name
                        keep_entity_idx.append(entity_idx)
                elif self.match_strategy == 'paraphrases':
                    if self.match_with_paraphrases(o365_name = name, entity_span = entity['span']):
                        entity['o365_name'] = name
                        keep_entity_idx.append(entity_idx)
                else:
                    assert False
                
        for rel in scene_graph_one_sent['relations']:
            if rel['subject'] in keep_entity_idx and rel['object'] in keep_entity_idx:
                rel['subject_o365_name'] = new_scene_graph['entities'][rel['subject']]['o365_name']
                rel['object_o365_name'] = new_scene_graph['entities'][rel['object']]['o365_name']
                rel['subject_span'] = new_scene_graph['entities'][rel['subject']]['span']
                rel['object_span'] = new_scene_graph['entities'][rel['object']]['span']
                new_scene_graph['relations'].append(rel)
        
        return new_scene_graph



def load_o365_paraphrases():
    o365_paraphrases_path = '/Path/To/jacob/RLIP/datasets/priors/o365_obj_paraphrase.json' 
    with open(o365_paraphrases_path, "r") as f:
        para_dict = json.load(f)
    return para_dict

def load_o365_categories():
    o365_paraphrases_path = '/Path/To/jacob/RLIP/datasets/priors/o365_obj_paraphrase.json' 
    with open(o365_paraphrases_path, "r") as f:
        para_dict = json.load(f)
    id_to_categories = {}
    start_idx = 1
    for name in para_dict.keys():
        id_to_categories[start_idx] = name
        start_idx += 1
    return id_to_categories


def visualize_pseudo_relations_o365(anno_path,
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
    o365_folder = Path('/Path/To/data/Objects365')
    save_folder = Path('/Path/To/jacob/visualization/pseudo_relations_o365')
    with open(anno_path, 'r') as f:
        rel_annos = json.load(f)
    with open(o365_folder / 'image_id_to_filepath.json', "r") as f:
        image_id_to_filepath_o365 = json.load(f)

    for img_anno in rel_annos[:100]:
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

        if img_anno["dataset"] == "o365":
            # img_file_name = str(img_anno['image_id']).zfill(12) + '.jpg'
            # if 'data_split' not in img_anno.keys():
            #     img_path = coco2017_folder / 'train2017' / img_file_name
            # else:
            #     img_path = coco2017_folder / img_anno['data_split'] / img_file_name
            
            img_path = o365_folder / image_id_to_filepath_o365[str(img_anno['image_id'])]
            # img = Image.open(o365_folder / img_file_name).convert('RGB')
        
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


### Execute once
def obtain_id_to_filename(
    save_id_to_filepath = None,
):
    '''
    This function aims to obtain a json file containing a dictionary 
    that maps image_id from Objects365 trainval set to its file paths.
    like: '900461': 'train/patch16/objects365_v2_00900461.jpg'
    
    Args:
        save_id_to_filepath: the saved file path.
    '''
    start_time = time.time()
    anno_path = ['/Path/To/data/Objects365/train/zhiyuan_objv2_train.json',
                 '/Path/To/data/Objects365/val/zhiyuan_objv2_val.json']

    with open(anno_path[0], 'r') as f:
        train_annos = json.load(f)
        print('Finish reading training annotations.')
    with open(anno_path[1], 'r') as f:
        val_annos = json.load(f)
        print('Finish reading validation annotations.')
    print(f"Reading annotations costs {int(time.time() - start_time)} seconds.")

    # dict_keys(['images', 'annotations', 'categories', 'licenses'])
    # example of 'annotations': {'id': 51, 'iscrowd': 0, 'isfake': 0, 'area': 1584.6324365356109, 'isreflected': 0, 'bbox': [491.3955078011, 88.1856689664, 35.65588379410002, 44.442382796800004], 'image_id': 420917, 'category_id': 84}
    # example of 'images': {'height': 512, 'id': 420917, 'license': 5, 'width': 769, 'file_name': 'images/v1/patch8/objects365_v1_00420917.jpg', 'url': ''}
    bboxes = train_annos["annotations"] + val_annos["annotations"]
    # bboxes = bboxes[:10000] # for testing

    # Convert bboxes to a dict with image_id as keys
    start_time = time.time()
    bboxes_dict = {}
    for bbox in bboxes:
        image_id = str(bbox['image_id'])
        if image_id not in bboxes_dict.keys():
            bboxes_dict[image_id] = [bbox,]
        else:
            bboxes_dict[image_id].append(bbox)
    print(f"Converting bbox annotations costs {int(time.time() - start_time)} seconds.")
    print(len(bboxes_dict))

    # Obtain paths for all images
    # Note that we need to exclude invalid images (very few images are not contained in the dataset.).
    start_time = time.time()
    o365_folder = Path('/Path/To/data/Objects365')
    val_img_ids = np.unique([str(img['id']) for img in val_annos['images']])
    id_to_filename = {str(img['id']):img['file_name'].split('/')[-2:] for img in (train_annos["images"] + val_annos["images"])}
    valid_img_path_list = []
    valid_image_id_list = []
    for idx, image_id in enumerate(bboxes_dict.keys()):
        if (idx+1)%10000 == 0:
            print(f"Finish processing {idx+1} / {len(bboxes_dict)}.")

        img_file_name = id_to_filename[image_id]
        patch_name, img_name = img_file_name[0], img_file_name[1] 
        if image_id in val_img_ids:
            img_path = o365_folder / 'val' / patch_name / img_name
        else:
            img_path = o365_folder / 'train' / patch_name / img_name

        if os.path.exists(img_path):
            valid_img_path_list.append(img_path)
            valid_image_id_list.append(image_id)
        else:
            print(f'{img_path} does not exist.')
    assert len(valid_img_path_list) == len(valid_image_id_list)
    print(f"Obtaining paths costs {int(time.time() - start_time)} seconds.")

    id_to_filepath = {}
    for img_pth, img_id in zip(valid_img_path_list, valid_image_id_list):
        partial_path = str(img_pth).split('/')[-3:]
        id_to_filepath[img_id] = f"{partial_path[0]}/{partial_path[1]}/{partial_path[2]}"
    
    if save_id_to_filepath:
        with open(save_id_to_filepath, "w") as f:
            json.dump(id_to_filepath, f)
        print(f"Saving to {save_id_to_filepath}.")

### Execute once
def delete_invalid_images_from_anno_file():
    invalid_id = [908726, 320532, 320534] # image_id
    with open('/Path/To/data/Objects365/train/zhiyuan_objv2_train_official.json', "r") as f:
        train_anno = json.load(f)
        # dict_keys(['images', 'annotations', 'categories', 'licenses'])
        # example of 'annotations': {'id': 51, 'iscrowd': 0, 'isfake': 0, 'area': 1584.6324365356109, 'isreflected': 0, 'bbox': [491.3955078011, 88.1856689664, 35.65588379410002, 44.442382796800004], 'image_id': 420917, 'category_id': 84}
        # example of 'images': {'height': 512, 'id': 420917, 'license': 5, 'width': 769, 'file_name': 'images/v1/patch8/objects365_v1_00420917.jpg', 'url': ''}
        
    new_annotations = []
    for anno in train_anno['annotations']:
        if anno["image_id"] not in invalid_id:
            new_annotations.append(anno)
    print(f"Original annos: {len(train_anno['annotations'])}, after deletion: {len(new_annotations)}")
    
    new_images = []
    for image in train_anno['images']:
        if anno["id"] not in invalid_id:
            new_images.append(image)
    print(f"Original images: {len(train_anno['images'])}, after deletion: {len(new_images)}")

    # train_anno['annotations'] = new_annotations
    # train_anno['images'] = new_images   
    # save_path = '/Path/To/data/Objects365/train/zhiyuan_objv2_train.json'
    # with open(save_path, "w") as f:
    #     json.dump(train_anno, f)





if __name__=="__main__":

    # transform_BLIP_sngs_to_verb_tagger_input_format(
    #     scene_graph_path = '/Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_beam_trainval2017.json',
    #     bbox_path = ['/Path/To/data/coco2017/annotations/instances_train2017.json',
    #                  '/Path/To/data/coco2017/annotations/instances_val2017.json'],
    #     save_path_rel_texts_for_coco_images = '/Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_beam_trainval2017_Paraphrases_rel_texts_for_coco_images.json',
    #     match_strategy = 'paraphrases',
    #     bbox_overlap = False)



    ### Part I
    ### Obtain file paths for Objects365 dataset.
    # obtain_id_to_filename(
    #     save_id_to_filepath = '/Path/To/data/Objects365/image_id_to_filepath.json',
    # )
    # obtain_id_to_filename()
    # /Path/To/data/Objects365/train/patch16/objects365_v2_00908726.jpg does not exist
    # /Path/To/data/Objects365/train/patch6/objects365_v1_00320532.jpg does not exist.
    # /Path/To/data/Objects365/train/patch6/objects365_v1_00320534.jpg does not exist

    ### Part II (we do not do this)
    ### delete invalid images from the training annotation file
    # delete_invalid_images_from_anno_file()

    ### Part III
    ### Objects365: scene graph parsing
    # transform_BLIP_sentences_to_triplets(
    #     coco_captions_BLIP_path = '/Path/To/data/Objects365/BLIP_captions/model_large_caption_nucleus10_o365trainval_1_4.json',
    #     save_path_sng = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_1_4.json',
    # )
    # 4555648 captions are processed.   Processing time: 26813s.
    # transform_BLIP_sentences_to_triplets(
    #     coco_captions_BLIP_path = '/Path/To/data/Objects365/BLIP_captions/model_large_caption_nucleus10_o365trainval_2_4.json',
    #     save_path_sng = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_2_4.json',
    # )
    # transform_BLIP_sentences_to_triplets(
    #     coco_captions_BLIP_path = '/Path/To/data/Objects365/BLIP_captions/model_large_caption_nucleus10_o365trainval_3_4.json',
    #     save_path_sng = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_3_4.json',
    # )
    # 4555648 captions are processed.   Processing time: 26429s.
    # transform_BLIP_sentences_to_triplets(
    #     coco_captions_BLIP_path = '/Path/To/data/Objects365/BLIP_captions/model_large_caption_nucleus10_o365trainval_4_4.json',
    #     save_path_sng = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_4_4.json',
    # )
    
    ### Part IV
    ### Objects365: BLIP scene graphs to verb tagger input format
    # transform_BLIP_sngs_to_verb_tagger_input_format(
    #     scene_graph_path = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_1_4.json',
    #     bbox_path = ['/Path/To/data/Objects365/train/zhiyuan_objv2_train.json',
    #                  '/Path/To/data/Objects365/val/zhiyuan_objv2_val.json'],
    #     save_path_rel_texts_for_o365_images = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_1_4_Paraphrases_rel_texts_for_o365_images.json',
    #     match_strategy = 'paraphrases',
    #     bbox_overlap = False)

    # ### Objects365: BLIP scene graphs to verb tagger input format
    # transform_BLIP_sngs_to_verb_tagger_input_format(
    #     scene_graph_path = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_2_4.json',
    #     bbox_path = ['/Path/To/data/Objects365/train/zhiyuan_objv2_train.json',
    #                  '/Path/To/data/Objects365/val/zhiyuan_objv2_val.json'],
    #     save_path_rel_texts_for_o365_images = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_2_4_Paraphrases_rel_texts_for_o365_images.json',
    #     match_strategy = 'paraphrases',
    #     bbox_overlap = False)
    
    ### Objects365: BLIP scene graphs to verb tagger input format
    # transform_BLIP_sngs_to_verb_tagger_input_format(
    #     scene_graph_path = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_3_4.json',
    #     bbox_path = ['/Path/To/data/Objects365/train/zhiyuan_objv2_train.json',
    #                  '/Path/To/data/Objects365/val/zhiyuan_objv2_val.json'],
    #     save_path_rel_texts_for_o365_images = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_3_4_Paraphrases_rel_texts_for_o365_images.json',
    #     match_strategy = 'paraphrases',
    #     bbox_overlap = False)
    # We have 7990151 relations in 4150 kinds before filtering.
    # We have 1037966 relations in 1587 kinds after filtering.
    # We have 15980302 objects in 452157 kinds before filtering.
    
    ### Objects365: BLIP scene graphs to verb tagger input format
    # transform_BLIP_sngs_to_verb_tagger_input_format(
    #     scene_graph_path = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_4_4.json',
    #     bbox_path = ['/Path/To/data/Objects365/train/zhiyuan_objv2_train.json',
    #                  '/Path/To/data/Objects365/val/zhiyuan_objv2_val.json'],
    #     save_path_rel_texts_for_o365_images = '/Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_4_4_Paraphrases_rel_texts_for_o365_images.json',
    #     match_strategy = 'paraphrases',
    #     bbox_overlap = False)

    visualize_pseudo_relations_o365(anno_path = '/Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_4_val_4.json',
                               one_rel_per_img = True,
                               relation_threshold = 0.2)