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

def merge_OI_with_VG(save_OI_VG_anno = False,
                     save_keep_names_freq = False):
    OI_train_anno = "/Path/To/data/open-imagev6/annotations/vrd-train-anno.json"
    OI_val_anno = "/Path/To/data/open-imagev6/annotations/vrd-val-anno.json"
    OI_cat_anno = "/Path/To/data/open-imagev6/annotations/categories_dict.json"
    with open(OI_train_anno, 'r') as f:
        all_annotations = json.load(f)
    with open(OI_val_anno, 'r') as f:
        all_annotations = all_annotations + json.load(f)
    with open(OI_cat_anno, 'r') as f:
        all_cat= json.load(f)
    obj_cat_name = all_cat["obj"] 
    rel_cat_name = all_cat["rel"]
    print('Rel_cat_name in OI: {}'.format(rel_cat_name))
    print("Obj cat num: {:d}, rel cat num: {:d}".format(len(obj_cat_name), len(rel_cat_name)))
    
    object_global_id = 0
    rel_global_id = 0
    OI_list = []
    unique_obj_in_OI = []
    unique_rel_in_OI = []
    for anno in all_annotations:
        # box: x1 y1, x2 y2
        box_list = []
        for b, l in zip(anno["bbox"], anno["det_labels"]):
            cur_box = {
                "object_id": object_global_id,
                "x":b[0],
                "y":b[1],
                "w":b[2] - b[0],
                "h":b[3] - b[1],
                "names":obj_cat_name[l]
            }
            if obj_cat_name[l] not in unique_obj_in_OI:
                unique_obj_in_OI.append(obj_cat_name[l])

            box_list.append(cur_box)
            object_global_id += 1
        
        rel_list = []
        for rel in anno["rel"]:
            cur_rel = {
                "relationship_id": rel_global_id,
                "predicate": rel_cat_name[rel[2]],
                "subject_id": box_list[rel[0]]["object_id"],
                "object_id":  box_list[rel[1]]["object_id"],
            }
            if rel_cat_name[rel[2]] not in unique_rel_in_OI:
                unique_rel_in_OI.append(rel_cat_name[rel[2]])

            rel_list.append(cur_rel)
            rel_global_id += 1
        
        cur_OI = {
            "image_id": anno["img_fn"],
            "objects": box_list,
            "relationships": rel_list,
        }
        OI_list.append(cur_OI)
    print(OI_list[-3], OI_list[-2], OI_list[-1])
    print(len(OI_list), object_global_id, rel_global_id, len(unique_obj_in_OI), len(unique_rel_in_OI))
    
    ### Merge VG with OI
    VG_anno_path = Path('/Path/To/data/VG') / 'annotations' / 'scene_graphs_preprocessv1.json'
    with open(VG_anno_path, 'r') as f:
        VG_annos = json.load(f)
    OI_VG_anno = OI_list + VG_annos
    if save_OI_VG_anno:
        OI_VG_anno_path = Path("/Path/To/data/open-imagev6/annotations/OI_trainval_VG_all_pretraining.json")
        with open(OI_VG_anno_path,'w') as f:
            json.dump(OI_VG_anno, f)
        print('Saving to {}'.format(OI_VG_anno))

    ### Check hico and classes
    # hico_obj_path = Path('/Path/To/jacob/RLIP/datasets/hico_object_names.txt')
    # with open(hico_obj_path, 'r') as f:
    #     hico_obj = json.load(f)
    # hico_obj = list(hico_obj.keys())
    # overlap_obj = []
    # exclude_obj = []
    # for o in hico_obj:
    #     if o in unique_obj_in_OI:
    #         overlap_obj.append(o)
    #     else:
    #         exclude_obj.append(o)
    # print(len(overlap_obj), overlap_obj, exclude_obj)

    ### Calculate and save stat for OI_VG
    rel_sum = OrderedDict()
    obj_sum = OrderedDict()
    for idx_anno, cur_anno in enumerate(OI_VG_anno):
        for idx_rel, cur_rel in enumerate(cur_anno['relationships']):
            if cur_rel['predicate'] in rel_sum.keys():
                rel_sum[cur_rel['predicate']] += 1
            else:
                rel_sum[cur_rel['predicate']] = 1
        for idx_obj, cur_obj in enumerate(cur_anno['objects']):
            if cur_obj['names'] in obj_sum.keys():
                obj_sum[cur_obj['names']] += 1
            else:
                obj_sum[cur_obj['names']] = 1

    rel_sum = sorted(rel_sum.items(), key = lambda item:item[1], reverse = True)
    obj_sum = sorted(obj_sum.items(), key = lambda item:item[1], reverse = True)
    rel_keep = []
    rel_freq = {}
    obj_keep = []
    obj_freq = {}
    for idx, (i, j) in enumerate(rel_sum):
        # if j >= 500:
        # if j >= 5:
        if j >= 0:
            rel_keep.append(i)
            rel_freq[i] = j
    print('rel_keep:', len(rel_keep))
    for idx, (i, j) in enumerate(obj_sum):
        # if j >= 1000:
        # if j >= 5:
        if j >= 0:
            obj_keep.append(i)
            obj_freq[i] = j
    print('obj_keep:', len(obj_keep))
    if save_keep_names_freq:
        with open('/Path/To/jacob/RLIP/datasets/OI_trainval_VG_all_keep_names_freq.json', 'w') as outfile:
            json.dump({"relationship_names": rel_keep,
                    "object_names": obj_keep,
                    "relationship_freq": rel_freq,
                    "object_freq": obj_freq}, outfile)
        print('save vg_keep_names.json !!!')
    
    # OI_VG的统计数据要加上去
    # adding_OI = False 顺便改一下加载的prior knowledge的文件位置


def split_OI_from_OI_VG(save_OI_anno = False):
    OI_VG_json = "/Path/To/data/open-imagev6/annotations/OI_trainval_VG_all_pretraining.json"
    with open(OI_VG_json, 'r') as f:
        all_annotations = json.load(f)
    print(len(all_annotations))
    print(all_annotations[0])

    OI_annos = []
    for anno in all_annotations:
        anno_img_name = str(anno['image_id'])
        has_alpha = 0
        for a in anno_img_name:
            if a.isalpha():
                has_alpha = 1
                break
        if has_alpha:
            OI_annos.append(anno)
    print("VG images: {}, OI images:{}".format(len(all_annotations) - len(OI_annos), len(OI_annos)))
    
    if save_OI_anno:
        OI_json = "/Path/To/data/open-imagev6/annotations/OI_trainval_pretraining.json"
        with open(OI_json,'w') as f:
            json.dump(OI_annos, f)
        print('Saving to {}'.format(OI_json))
    
    # unique_obj_names:286, obj_num:518312, unique_rel_names:30, rel_num:353281
    

if __name__=="__main__":
    split_OI_from_OI_VG(save_OI_anno = True)

    # # # merge_OI_with_VG()

    # # # Check OI_trainval_VG_all_keep_names_freq.json
    # # # keep_names_file = Path("/Path/To/jacob/RLIP/datasets/OI_trainval_VG_all_keep_names_freq.json")
    # # # with open(keep_names_file, "r") as f:
    # # #     vg_keep_names = json.load(f)
    # # # # print(vg_keep_names["relationship_freq"])
    # # # print(vg_keep_names["relationship_names"][:100])

    # print(os.path.exists('/Path/To/data/open-imagev6/OI_VG_images/2320297.jpg'))
    # print(os.path.exists('/Path/To/data/VG/images/2320297.jpg'))
    # ori_VG = os.listdir('/Path/To/data/VG/images/')
    # ori_OI = os.listdir('/Path/To/data/open-imagev6/images/')
    # VG_OI = os.listdir('/Path/To/data/open-imagev6/OI_VG_images/')
    # print(len(VG_OI))
    # for vo in VG_OI:
    #     if '.jpg' not in vo:
    #         print(vo)
    # # VG and OI overlap? No overlap
    # # for v in ori_VG:
    # #     if v in ori_OI:
    # #         print(v)
    
    # # # Check what's missing in OI_VG.
    # # # missing_VG_OI = []
    # # # ori_VG_OI = ori_VG + ori_OI
    # # # for ovi in ori_VG_OI:
    # # #     if ovi not in VG_OI:
    # # #         missing_VG_OI.append(ovi)
    # # # with open('/Path/To/data/open-imagev6/missing_images.json', 'w') as outfile:
    # # #     json.dump(missing_VG_OI, outfile)
    # # # print(missing_VG_OI)
    # # with open('/Path/To/data/open-imagev6/missing_images.json', 'r') as f:
    # #     all_missing= json.load(f)
    # # print(len(all_missing))

    
    # # VG_OI_dest = os.listdir('/Path/To/data/open-imagev6/OI_VG_images/')
    # # VG_OI_src = os.listdir('/Path/To/data/open-imagev6/OI_VG_images/images/')
    # # src_list = os.listdir(VG_OI_src)
    # # for s in range(src_list):



# {
#     "bbox": [
#       [
#         235,
#         104,
#         786,
#         679
#       ],
#       [
#         406,
#         275,
#         767,
#         542
#       ]
#     ],
#     "det_labels": [
#       307,
#       586
#     ],
#     "rel": [
#       [
#         0,
#         1,
#         28
#       ],
#       [
#         0,
#         1,
#         1
#       ]
#     ],
#     "img_size": [
#       1024,
#       683
#     ],
#     "img_fn": "de635c5d3ecd3c46"
#   }

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