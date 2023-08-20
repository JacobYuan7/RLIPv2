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

def deduplicate_verb_tagger_output(tagger_path,
                                   save = False):
    '''
    This function aims to filter out those duplicate relationships after we run the verb tagger.
    We keep the one with the highest confidence score.

    Args:
        tagger_path (string): the path to the output of the verb_tagger.
        save (bool): whether to save the annotations after deduplication. 
    '''
    with open(tagger_path, 'r') as f:
        tagger_output = json.load(f)
    
    new_tagger = []
    print(f"We have {sum([len(anno['relationships']) for anno in tagger_output])} relationships before deduplition.")
    for anno in tagger_output:
        rels = anno["relationships"]
        
        max_conf_rels = {} # {(1,2):{'on':0.34, 'has':0.27}}
        unique_rels = []
        # Step1: Some relationships have duplicate, so we record the maximum confidence score.
        for rel in rels:
            if (rel["subject_id"], rel["object_id"]) not in max_conf_rels.keys():
                max_conf_rels[(rel["subject_id"], rel["object_id"])] = {rel["predicate"]: rel["confidence"]}
            else:
                if rel["predicate"] not in max_conf_rels[(rel["subject_id"], rel["object_id"])].keys():
                    max_conf_rels[(rel["subject_id"], rel["object_id"])][rel["predicate"]] = rel["confidence"]
                else:
                    # This means that there are duplicate rels, so we compute the maximum 
                    max_conf_rels[(rel["subject_id"], rel["object_id"])][rel["predicate"]] = \
                        max(rel["confidence"], max_conf_rels[(rel["subject_id"], rel["object_id"])][rel["predicate"]])

        # Step2:  We keep the one with the highest confidence score.
        for rel in rels:
            if rel["confidence"] == max_conf_rels[(rel["subject_id"], rel["object_id"])][rel["predicate"]]:
                unique_rels.append(rel)
        anno["relationships"] = unique_rels
    print(f"We have {sum([len(anno['relationships']) for anno in tagger_output])} relationships after deduplition.")

    if save:
        save_path = tagger_path.replace('.json', '_deduplicate.json')
        with open(save_path, "w") as f:
            json.dump(tagger_output, f)
        print(f"Saving to {save_path}.")



def merge_segments_from_verb_tagger(
    json_list = [],
    dataset_change_to = None,
    save_merged_file = None,
):
    '''
    This function merges annotations generated from "generate_relations_using_verb_tagger.py".
    
    Args:
        json_list (list): a list containing jsons for all annotations.
        dataset_change_to (string): the dataset indicator for this dataset.
                If not specified, then we will not change the "dataset" value. 
    '''
    json_annos = []
    for json_file in json_list:
        with open(json_file, "r") as f:
            json_annos += json.load(f)
    print("Finish loading the annotaions.")
    
    image_id_list = []
    new_json_annos = []
    for anno in json_annos:
        if dataset_change_to is not None:
            anno["dataset"] = dataset_change_to
        new_json_annos.append(anno)
        image_id_list.append(anno["image_id"])
        # if anno["image_id"] not in image_id_list:
        #     new_json_annos.append(anno)
        # image_id_list.append(anno["image_id"])
    print(f"Original images: {len(new_json_annos)}, after deduplication: {len(np.unique(image_id_list))}.")

    if save_merged_file:
        with open(save_merged_file, "w") as f:
            json.dump(new_json_annos, f)
        print(f"Saving to {save_merged_file}.")
    

if __name__=="__main__":
    ### Deduplicate generated relation triplets
    # deduplicate_verb_tagger_output(tagger_path = '/Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus5_thre05.json',
    #                                save = False)
    # deduplicate_verb_tagger_output(tagger_path = '/Path/To/data/coco2017/annotations/swin/RLIPv2_SwinL_trainval2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05.json',
    #                                save = False)
    # deduplicate_verb_tagger_output(tagger_path = '/Path/To/data/coco2017/annotations/swin/RLIPv2_SwinT_10ep_trainval2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05.json',
    #                                save = False)
    # deduplicate_verb_tagger_output(tagger_path = '/Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05.json',
    #                                save = False)

    # RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_OracleCaps_Paraphrases_thre05.json
    # RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_nucleus5_thre05.json


    ### Merge 4 segments from verb tagger for Objects365
    merge_segments_from_verb_tagger(
        json_list = ['/Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_1_4.json',
                     '/Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_2_4.json',
                     '/Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_3_4.json',
                     '/Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_4_val_4.json',
                     '/Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_4_train_4.json',],
                    #  '/Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_4_4.json'],
        dataset_change_to = 'o365',
        save_merged_file = '/Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_1234_4.json',
        )