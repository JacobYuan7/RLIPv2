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

def convert_OI_SGG_to_HICO(
    anno_list = [],
    save_path = None,
):
    '''
    This function aims to convert the OI annotations (SGG) to the HICO format,
    which could possibly unify the fine-tuning of HOI and SGG.
    Note that we change the key "hoi_annotation" to "rel_annotations" to fit in with SGG task.

    Args:
        anno_list (list): a list containing the annotation file path to the OI train set, test set or trainval set.
        save_path (string): the path to save the transformed 
    
    Returns:
        None
    '''
    OI_cat_anno = "/Path/To/data/open-imagev6/annotations/categories_dict.json"
    if len(anno_list) == 1:
        with open(anno_list[0], 'r') as f:
            all_annotations = json.load(f)
    elif len(anno_list) == 2:
        with open(anno_list[0], 'r') as f:
            all_annotations = json.load(f)
        with open(anno_list[1], 'r') as f:
            all_annotations = all_annotations + json.load(f)
    else:
        assert False
    with open(OI_cat_anno, 'r') as f:
        all_cat= json.load(f)
    
    obj_cat_name = all_cat["obj"] 
    rel_cat_name = all_cat["rel"]
    print('Rel_cat_name in OI: {}'.format(rel_cat_name))
    print('Obj_cat_name in OI: {}'.format(obj_cat_name))
    print("Obj cat num: {:d}, rel cat num: {:d}".format(len(obj_cat_name), len(rel_cat_name)))
    # print(all_annotations[0])

    object_global_id = 0
    rel_global_id = 0
    OI_list = []
    for anno in all_annotations:
        # box: x1 y1, x2 y2
        box_list = []
        for b, l in zip(anno["bbox"], anno["det_labels"]):
            cur_box = {
                "bbox": b,
                "category_id": l,
            }

            box_list.append(cur_box)
            object_global_id += 1
        
        rel_list = []
        for rel in anno["rel"]:
            cur_rel = {
                "subject_id": rel[0], 
                "object_id": rel[1], 
                "category_id": rel[2],
            }

            rel_list.append(cur_rel)
            rel_global_id += 1

        cur_OI = {
            "file_name": f"{anno['img_fn']}.jpg",
            "rel_annotations": rel_list,
            "annotations": box_list,
        }
        OI_list.append(cur_OI)
    # unique_obj_in_OI = get_unique_obj_names(all_annotations, obj_cat_name)
    # unique_rel_in_OI = rel_cat_name
    # print(f"There are {len(unique_rel_in_OI)} unique relations.")
    # print(f"There are {len(unique_obj_in_OI)} unique objects.")
    print(len(OI_list))

    if save_path:
        with open(save_path,'w') as f:
            json.dump(OI_list, f)
        print('Saving to {}'.format(save_path))


def generate_OI_SGG_category_corre(
    trainval_anno_list = [],
    test_anno = None,
    save_categories_oi_sgg_dict_path = None,
    save_corre_oi_sgg_trainval_test_path = None,
    save_corre_oi_sgg_test_path = None,
):
    '''
    This function aims to generate 1) the file storing the object names and rel names in the OI SGG dataset
    and 2) the file storing the binary mask to filter out predictions.

    Args:

    Returns:
        None
    '''
    ### Process and save the category names dict.
    OI_cat_anno = "/Path/To/data/open-imagev6/annotations/categories_dict.json"
    with open(OI_cat_anno, 'r') as f:
        all_cat= json.load(f)
    if len(trainval_anno_list) == 1:
        with open(trainval_anno_list[0], 'r') as f:
            trainval_annotations = json.load(f)
    elif len(trainval_anno_list) == 2:
        with open(trainval_anno_list[0], 'r') as f:
            trainval_annotations = json.load(f)
        with open(trainval_anno_list[1], 'r') as f:
            trainval_annotations = trainval_annotations + json.load(f)
    else:
        assert False
    with open(test_anno, 'r') as f:
        test_annotations = json.load(f)

    obj_cat_name = all_cat["obj"] 
    rel_cat_name = all_cat["rel"]
    unique_obj_in_OI = get_unique_obj_names(trainval_annotations + test_annotations, obj_cat_name)
    unique_rel_in_OI = rel_cat_name
    categories_oi_sgg = {"obj": unique_obj_in_OI, "rel":unique_rel_in_OI}

    if save_categories_oi_sgg_dict_path:
        with open(save_categories_oi_sgg_dict_path,'w') as f:
            json.dump(categories_oi_sgg, f)
        print('Saving to {}.'.format(save_categories_oi_sgg_dict_path))
    
    # ########## Check the number of object kinds in the trainval and test set ##########
    # # There are 286 kinds in the trainval set.
    # # There are 214 kinds in the test set.
    # # There are 288 kinds in the trainval + test set.

    # trainval_obj_list = []
    # for anno in trainval_annotations:
    #     obj_labels = anno["det_labels"]
    #     for o in obj_labels:
    #         if o not in trainval_obj_list:
    #             trainval_obj_list.append(o)
    # print(sorted(trainval_obj_list))
    # print(len(trainval_obj_list))

    # test_obj_list = []
    # for anno in test_annotations:
    #     obj_labels = anno["det_labels"]
    #     for o in obj_labels:
    #         if o not in test_obj_list:
    #             test_obj_list.append(o)
    # print(sorted(test_obj_list))
    # print(len(test_obj_list))

    # merge_obj_list = trainval_obj_list
    # for o in test_obj_list:
    #     if o not in trainval_obj_list:
    #         merge_obj_list.append(o)
    # print(sorted(merge_obj_list))
    # print(len(merge_obj_list))
    # ###########################################################################

    ### Process and save the binary mask.
    ### We need to check whether the distribution of the trainval/train set and test set are identical.
    unique_obj_cat = [obj_cat_name.index(obj) for obj in unique_obj_in_OI]
    print(unique_obj_cat)
    corre_oi_sgg_trainval = np.zeros((len(unique_obj_in_OI), len(unique_rel_in_OI), len(unique_obj_in_OI))) # [288, 30, 288]
    corre_oi_sgg_test = np.zeros((len(unique_obj_in_OI), len(unique_rel_in_OI), len(unique_obj_in_OI)))
    for anno in trainval_annotations:
        obj_labels = anno["det_labels"]
        for rel in anno["rel"]:
            sub_cat = unique_obj_cat.index(obj_labels[rel[0]])
            obj_cat = unique_obj_cat.index(obj_labels[rel[1]])
            category_triplet = [sub_cat, rel[2], obj_cat]
            corre_oi_sgg_trainval[category_triplet[0], category_triplet[1], category_triplet[2]] = 1.
    for anno in test_annotations:
        obj_labels = anno["det_labels"]
        for rel in anno["rel"]:
            sub_cat = unique_obj_cat.index(obj_labels[rel[0]])
            obj_cat = unique_obj_cat.index(obj_labels[rel[1]])
            category_triplet = [sub_cat, rel[2], obj_cat]
            corre_oi_sgg_test[category_triplet[0], category_triplet[1], category_triplet[2]] = 1.
    print(f"There are {np.sum(corre_oi_sgg_trainval)} triplets in the trainval set, while there are {np.sum(corre_oi_sgg_test)} in the test set.")
    
    # Check whether all the test triplets are in the trainval set.
    # print(np.sum((corre_oi_sgg_trainval+corre_oi_sgg_test)>0))
    new_trip_num = 0
    for i in range(len(unique_obj_in_OI)):
        for j in range(len(unique_rel_in_OI)):
            for k in range(len(unique_obj_in_OI)):
                if corre_oi_sgg_test[i][j][k] == 1:
                    if corre_oi_sgg_trainval[i][j][k] == 0:
                        new_trip_num += 1
    print(f"{new_trip_num} kinds of triplets are not in the trainval set.")

    ### Save the corre
    corre_oi_sgg_trainval_test = ((corre_oi_sgg_trainval + corre_oi_sgg_test)>0).astype(float)
    if save_corre_oi_sgg_trainval_test_path is not None:
        np.save(save_corre_oi_sgg_trainval_test_path, corre_oi_sgg_trainval_test)
        print(f"Saving to {save_corre_oi_sgg_trainval_test_path}.")
    if save_corre_oi_sgg_test_path is not None:
        np.save(save_corre_oi_sgg_test_path, corre_oi_sgg_test)
        print(f"Saving to {save_corre_oi_sgg_test_path}.")


def get_unique_obj_names(all_annotations, obj_cat_name):
    '''
    This function aims to obtain a list of unique OpenImage object names.
    The order of the object names is in the original order.

    Args:
        all_annotations (list): a list containing OI relation annotations (read-in from the json file). 
            Note!!! This must contain the annotations from the trainval and test set.
        obj_cat_name (list): a list containing all the object names in OpenImages.

    Returns:
        unique_obj_in_OI (list): a list containing all unique object names appearing in all OI relation annotations
    '''
    unique_obj_in_OI = []
    for anno in all_annotations:
        for l in anno["det_labels"]:
            if obj_cat_name[l] not in unique_obj_in_OI:
                unique_obj_in_OI.append(obj_cat_name[l])
    
    reorder_unique_obj_in_OI = []
    for name in obj_cat_name:
        if name in unique_obj_in_OI:
            reorder_unique_obj_in_OI.append(name)

    return reorder_unique_obj_in_OI



# OpenImage relation detection annotations
# {'bbox': [[56, 133, 880, 357], [53, 129, 878, 357]], 
#  'det_labels': [285, 18], 
#  'rel': [[0, 1, 16]], 
#  'img_size': [1024, 768], 
#  'img_fn': '003e363ce1f302cd'}

# Ori Structure of HICO annotation
# {
# "file_name": "HICO_test2015_00000006.jpg", 
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

if __name__ == "__main__":
    # convert_OI_SGG_to_HICO(
    #     anno_list = ["/Path/To/data/open-imagev6/annotations/vrd-train-anno.json",
    #                  "/Path/To/data/open-imagev6/annotations/vrd-val-anno.json"],
    #     save_path = "/Path/To/data/open-imagev6/annotations/OI_SGG_trainval_finetuning.json")
    # There are 30 unique relations.
    # There are 286 unique objects.

    # convert_OI_SGG_to_HICO(
    #     anno_list = ["/Path/To/data/open-imagev6/annotations/vrd-train-anno.json"],
    #     save_path = "/Path/To/data/open-imagev6/annotations/OI_SGG_train_finetuning.json")
    # There are 30 unique relations.
    # There are 286 unique objects.

    # convert_OI_SGG_to_HICO(
    #     anno_list = ["/Path/To/data/open-imagev6/annotations/vrd-test-anno.json"],
    #     save_path = "/Path/To/data/open-imagev6/annotations/OI_SGG_test.json")
    # There are 30 unique relations.
    # There are 214 unique objects.

    # generate_OI_SGG_category_corre(
    #     trainval_anno_list = ["/Path/To/data/open-imagev6/annotations/vrd-train-anno.json",
    #                           "/Path/To/data/open-imagev6/annotations/vrd-val-anno.json"],
    #     test_anno = "/Path/To/data/open-imagev6/annotations/vrd-test-anno.json",
    #     save_categories_oi_sgg_dict_path = "/Path/To/data/open-imagev6/annotations/OI_SGG_trainval_test_categories_dict.json",
    #     save_corre_oi_sgg_trainval_test_path = "/Path/To/data/open-imagev6/annotations/OI_SGG_trainval_test_corre_mat.npy",
    #     save_corre_oi_sgg_test_path = "/Path/To/data/open-imagev6/annotations/OI_SGG_test_corre_mat.npy",
    # )
    None