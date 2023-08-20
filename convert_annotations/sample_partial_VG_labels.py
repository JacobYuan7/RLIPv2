# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pathlib import Path
from PIL import Image
import json
from collections import defaultdict, OrderedDict
import numpy as np
import sys

import torch
import torch.utils.data
import torchvision

# import datasets.transforms as T
import cv2
import os
import math
import random

def sample(ratio = 0.5, save_json = False):
    anno_file = '/Path/To/data/VG/annotations/scene_graphs_preprocessv1.json'
    with open(anno_file, 'r') as f:
        all_annotations = json.load(f)
    
    label_file = '/Path/To/jacob/RLIP/datasets/vg_keep_names_v1_no_lias_freq.json'
    with open(label_file, 'r') as f:
        all_labels = json.load(f)
    
    relationship_names = all_labels["relationship_names"]
    object_names = all_labels["object_names"]
    print(len(relationship_names), len(object_names))

    rel_sample_indices = random.sample(range(0, len(relationship_names)), int(len(relationship_names)*ratio))
    obj_sample_indices = random.sample(range(0, len(object_names)), int(len(object_names)*ratio))
    sample_rel = [relationship_names[r] for r in rel_sample_indices]
    sample_obj = [object_names[o] for o in obj_sample_indices]

    for idx, anno in enumerate(all_annotations):
        obj_dict = {}
        for obj in anno['objects']:
            if obj['names'] in sample_obj:
                obj_dict[obj['object_id']] = obj
        
        rel_list = []
        for rel in anno['relationships']:
            if rel['predicate'] in sample_rel and \
               rel['subject_id'] in obj_dict.keys() and \
               rel['object_id'] in obj_dict.keys():
               rel_list.append(rel)
        
        anno['objects'] = list(obj_dict.values())
        anno['relationships'] = rel_list
    
    if save_json:
        save_anno_file = '/Path/To/data/VG/annotations/scene_graphs_preprocessv1_random_sample50.json'
        with open(save_anno_file, 'w') as outfile:
            json.dump(all_annotations, outfile)
        print('Successfully save to {}.'.format(save_anno_file))
    


def check_stat():
    anno_file = '/Path/To/data/VG/annotations/scene_graphs_preprocessv1.json'
    with open(anno_file, 'r') as f:
        all_annotations = json.load(f)

    unique_obj_names = []
    obj_num = 0
    unique_rel_names = []
    rel_num = 0
    for idx, anno in enumerate(all_annotations):
        obj_num += len(anno['objects'])
        rel_num += len(anno['relationships'])
        for obj in anno['objects']:
            if obj['names'] not in unique_obj_names:
                unique_obj_names.append(obj['names'])
        for rel in anno['relationships']:
            if rel['predicate'] not in unique_rel_names:
                unique_rel_names.append(rel['predicate'])
    print('unique_obj_names:{}, obj_num:{}, unique_rel_names:{}, rel_num:{}'.format(len(unique_obj_names), obj_num, len(unique_rel_names),rel_num))



def generate_freq_file(save_json = False):
    # anno_file = '/Path/To/data/VG/annotations/scene_graphs_preprocessv1_random_sample50.json'
    # anno_file = '/Path/To/data/VG/annotations/scene_graphs_preprocessv2.json'
    anno_file = "/Path/To/data/open-imagev6/annotations/OI_trainval_pretraining.json"
    with open(anno_file, 'r') as f:
        all_annotations = json.load(f)

    unique_obj_names = {}
    obj_num = 0
    unique_rel_names = {}
    rel_num = 0
    for idx, anno in enumerate(all_annotations):
        obj_num += len(anno['objects'])
        rel_num += len(anno['relationships'])
        for obj in anno['objects']:
            if obj['names'] not in unique_obj_names.keys():
                unique_obj_names[obj['names']] = 1
            else:
                unique_obj_names[obj['names']] += 1
        for rel in anno['relationships']:
            if rel['predicate'] not in unique_rel_names.keys():
                unique_rel_names[rel['predicate']] = 1
            else:
                unique_rel_names[rel['predicate']] += 1
    unique_obj_names = dict(sorted(unique_obj_names.items(), key = lambda item:item[1], reverse = True))
    unique_rel_names = dict(sorted(unique_rel_names.items(), key = lambda item:item[1], reverse = True))
    print('unique_obj_names:{}, obj_num:{}, unique_rel_names:{}, rel_num:{}'.format(len(unique_obj_names), obj_num, len(unique_rel_names),rel_num))
    
    
    save_dict = {'relationship_names':list(unique_rel_names.keys()), 
                 'object_names':list(unique_obj_names.keys()),
                 'relationship_freq': unique_rel_names,
                 'object_freq': unique_obj_names}
    if save_json:
        # save_anno_file = '/Path/To/jacob/RLIP/datasets/vg_keep_names_v1_random_sample50_freq.json'
        # save_anno_file = '/Path/To/jacob/RLIP/datasets/vg_keep_names_v2_freq.json'
        save_anno_file = "/Path/To/jacob/RLIP/datasets/OI_keep_names_trainval_pretraining.json"
        with open(save_anno_file, 'w') as outfile:
            json.dump(save_dict, outfile)
        print('Successfully save to {}.'.format(save_anno_file))


if __name__=="__main__":
    # check_stat()
    # v2 (Excluding long-tail classes): 
    # unique_obj_names:497, obj_num:1728054, unique_rel_names:151, rel_num:1271210
    # v1 (Original)
    # unique_obj_names:100298, obj_num:3802374, unique_rel_names:36515, rel_num:1987331
    # sample 50%
    # unique_obj_names:50149, obj_num:1867150, unique_rel_names:7283, rel_num:322245

    # sample(ratio = 0.5, save_json = True)

    generate_freq_file(save_json = True)

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