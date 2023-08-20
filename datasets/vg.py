# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
VG detection dataset.
"""
import argparse
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict, OrderedDict
import numpy as np
import sys
sys.path.append("..") 
from util.image import draw_umich_gaussian, gaussian_radius

import torch
import torch.utils.data
import torchvision

# import datasets.transforms as T
from datasets import transforms as T
import cv2
import os
import math
from itertools import combinations, permutations
import copy
import random
# import h5py


class VGRelDetection(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, scene_graphs_anno_file, transforms, num_queries, use_all_text_labels = None, adding_OI = False):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(scene_graphs_anno_file, 'r') as f:
            self.annotations = json.load(f)

        # print(len(self.annotations))
        # print(type(self.annotations))
        # print(self.annotations[0])
        self._transforms = transforms
        self.num_queries = num_queries
        self.use_all_text_labels = use_all_text_labels
        if self.use_all_text_labels:
            # Using v2 means that we will only use the one file that excludes long-tail classes 
            # with open("/Path/To/jacob/RLIP/datasets/vg_keep_names_v2.json", "r") as f:
            #     vg_keep_names = json.load(f)
            with open("/Path/To/jacob/RLIP/datasets/vg_keep_names_v1_no_lias_freq.json", "r") as f:
                vg_keep_names = json.load(f)
            # with open("/Path/To/jacob/RLIP/datasets/OI_keep_names_trainval_pretraining.json", "r") as f:
            #     vg_keep_names = json.load(f)
            self.relationship_names = vg_keep_names["relationship_names"]
            self.object_names = vg_keep_names["object_names"]
        else:
            if not adding_OI:
                keep_names_file = "/Path/To/jacob/RLIP/datasets/vg_keep_names_v1_no_lias_freq.json"
                # keep_names_file = "/Path/To/jacob/RLIP/datasets/vg_keep_names_v2_freq.json"
                # keep_names_file = "/Path/To/jacob/RLIP/datasets/vg_keep_names_v1_random_sample50_freq.json"
            else:
                keep_names_file = "/Path/To/jacob/RLIP/datasets/OI_trainval_VG_all_keep_names_freq.json"

            with open(keep_names_file, "r") as f:
                vg_keep_names = json.load(f)
            self.relationship_names = vg_keep_names["relationship_names"]
            self.object_names = vg_keep_names["object_names"]
            if "relationship_freq" in vg_keep_names.keys():
                self.relationship_freq = vg_keep_names["relationship_freq"]
            if "object_freq" in vg_keep_names.keys():
                self.object_freq = vg_keep_names["object_freq"]
            
            self.rel_feature = np.load('/Path/To/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_sup-simcse-roberta-large.npz', allow_pickle = True)['rel_feature'].item()
            # self.rel_feature = np.load('/Path/To/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_text_feature.npz', allow_pickle = True)['rel_feature'].item()
            # verb_text_bf_fusion = torch.stack([torch.from_numpy(self.rel_feature[rt]).to(memory_cache["text_memory_resized"].device) for rt in rel_text], dim = 0)
            list_rel_ft = [torch.from_numpy(i) for i in list(self.rel_feature.values())]
            self.rel_feature = (list(self.rel_feature.keys()), torch.stack(list_rel_ft, dim = 0))
            self.obj_feature = np.load('/Path/To/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_sup-simcse-roberta-large.npz', allow_pickle = True)['obj_feature'].item()
            # self.obj_feature = np.load('/Path/To/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_text_feature.npz', allow_pickle = True)['obj_feature'].item()
            list_obj_ft = [torch.from_numpy(i) for i in list(self.obj_feature.values())]
            self.obj_feature = (list(self.obj_feature.keys()), torch.stack(list_obj_ft, dim = 0))

        if img_set == 'pretrain':
            self.ids = []
            # use to filter unavailable images by annotations
            self.ids = list(range(len(self.annotations)))
        else:
            self.ids = list(range(len(self.annotations)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
                        (After using the function 'self._transforms(img, target)')
        """
        img_anno = self.annotations[self.ids[idx]]
        objects_anno = img_anno['objects'] # data type: list
        relationships_anno = img_anno['relationships'] # data type: list
        
        img_file_name = str(img_anno['image_id']) + '.jpg'
        img = Image.open(self.img_folder / img_file_name).convert('RGB')
        # print(img_anno['file_name'])
        # print(img.size)
        w, h = img.size
        
        # make sure that #queries are more than #bboxes
        if self.img_set == 'pretrain' and len(relationships_anno) > self.num_queries:
            relationships_anno = relationships_anno[:self.num_queries]
        # collect coordinates and names for all bboxes
        boxes = []
        for cur_box in objects_anno:
            cur_box_cor = [cur_box['x'], cur_box['y'], cur_box['x'] + cur_box['w'], cur_box['y'] + cur_box['h']]
            boxes.append(cur_box_cor)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # print(f"boxes shape:{boxes.shape}")

        if self.use_all_text_labels:
            obj_unique = unique_name_dict_from_list(self.object_names)
            rel_unique = unique_name_dict_from_list(self.relationship_names)
        else:
            obj_unique = unique_name_dict_from_anno(objects_anno, 'objects')
            rel_unique = unique_name_dict_from_anno(relationships_anno, 'relationships')
        obj_classes = [(idx_cur_box, obj_unique[cur_box['names']]) for idx_cur_box, cur_box in enumerate(objects_anno)]
        obj_classes = torch.tensor(obj_classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'pretrain':
            # HICO: clamp the box and drop those unreasonable ones
            # VG: I have checked that all boxes are 
            boxes[:, 0::2].clamp_(min=0, max=w)  # xyxy    clamp x to 0~w
            boxes[:, 1::2].clamp_(min=0, max=h)  # xyxy    clamp y to 0~h

            # This 'keep' can be removed because all w and h are assured to be greater than 0.
            # keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            # boxes = boxes[keep]  # may have a problem
            # obj_classes = obj_classes[keep]

            # construct target dict
            target['boxes'] = boxes
            target['labels'] = obj_classes  # like [[0, 0][1, 56][2, 0][3, 0]...]
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if self._transforms is not None:
                img, target = self._transforms(img, target) # target['boxes'].shape and target['labels'].shape may change

            # ******This 'keep' should be maintained because self._transform may eliminate some boxes******
            # which means that target['boxes'].shape and target['labels'].shape may change
            
            kept_box_indices = [label[0] for label in target['labels']] # enumerated indices for kept (0, 1, 2, 3, 4, ...)
            # Guard against situations with target['labels'].shape[0] = 0, which can not perform target['labels'][:, 1].
            if target['labels'].shape[0] > 0:
                target['labels'] = target['labels'][:, 1] # object classes in 80 classes

            sub_labels, obj_labels, predicate_labels, sub_boxes, obj_boxes = [], [], [], [], []
            sub_obj_pairs = []
            relationships_anno_local = add_local_object_id(relationships_anno, objects_anno)
            for cur_rel_local in relationships_anno_local:
                # Make sure that sub and obj are not eliminated by self._transform.
                if cur_rel_local['subject_id_local'] not in kept_box_indices or \
                        cur_rel_local['object_id_local'] not in kept_box_indices:
                    continue

                sub_obj_pair = (cur_rel_local['subject_id_local'], cur_rel_local['object_id_local'])
                if sub_obj_pair in sub_obj_pairs:
                    predicate_labels[sub_obj_pairs.index(sub_obj_pair)][rel_unique[cur_rel_local['predicate']]] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    # genrate labels for sub, obj and predicate
                    kept_sub_idx = kept_box_indices.index(cur_rel_local['subject_id_local'])
                    kept_obj_idx = kept_box_indices.index(cur_rel_local['object_id_local'])
                    cur_sub_label = target['labels'][kept_sub_idx]
                    cur_obj_label = target['labels'][kept_obj_idx]
                    sub_labels.append(cur_sub_label)
                    obj_labels.append(cur_obj_label)
                    predicate_label = [0 for _ in range(len(rel_unique))]
                    predicate_label[rel_unique[cur_rel_local['predicate']]] = 1
                    predicate_labels.append(predicate_label)
                    
                    # generate box coordinates for sub and obj
                    # print(f"target['boxes'].shape: {target['boxes'].shape}")
                    cur_sub_box = target['boxes'][kept_sub_idx]
                    cur_obj_box = target['boxes'][kept_obj_idx]
                    sub_boxes.append(cur_sub_box)
                    obj_boxes.append(cur_obj_box)
            
            target['image_id'] = img_anno['image_id']
            target['obj_classes'] = list(dict(obj_unique).keys())
            target['verb_classes'] = list(dict(rel_unique).keys())
            # target['obj_classes'] = generate_class_names_list(obj_unique) 
            # target['predicate_classes'] = generate_class_names_list(rel_unique) 
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['sub_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(rel_unique)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                # target['obj_classes'] = list(dict(obj_unique).keys())
                # target['verb_classes'] = list(dict(rel_unique).keys())
                target['obj_labels'] = torch.stack(obj_labels)
                target['sub_labels'] = torch.stack(sub_labels)
                target['verb_labels'] = torch.as_tensor(predicate_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
        
        return img, target


    def set_rare_hois(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        counts = defaultdict(lambda: 0)
        for img_anno in annotations:
            hois = img_anno['hoi_annotation']
            bboxes = img_anno['annotations']
            for hoi in hois:
                triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                           self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                           self._valid_verb_ids.index(hoi['category_id']))
                counts[triplet] += 1
        self.rare_triplets = []
        self.non_rare_triplets = []
        for triplet, count in counts.items():
            if count < 10:
                self.rare_triplets.append(triplet)
            else:
                self.non_rare_triplets.append(triplet)

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)
        # self.correct_mat shape: [117, 80]
        # print(self.correct_mat.shape)



class VGRelTagger(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, scene_graphs_anno_file, transforms, num_queries, use_all_text_labels = None, adding_OI = False,\
                     pos_neg_ratio = 0., verb_tagger = True):
        '''
        pos_neg_ratio: #positive sub-obj query pairs/#negative sub-obj query pairs
                       'positive' means that the subject and the object has a valid relation label.
        '''
        self.img_set = img_set
        self.img_folder = img_folder
        with open(scene_graphs_anno_file, 'r') as f:
            self.annotations = json.load(f)

        self._transforms = transforms
        self.num_queries = num_queries
        self.use_all_text_labels = use_all_text_labels
        if self.use_all_text_labels:
            # Using v2 means that we will only use the one file that excludes long-tail classes 
            # with open("/Path/To/jacob/RLIP/datasets/vg_keep_names_v2.json", "r") as f:
            #     vg_keep_names = json.load(f)
            with open("/Path/To/jacob/RLIP/datasets/vg_keep_names_v1_no_lias_freq.json", "r") as f:
                vg_keep_names = json.load(f)
            # with open("/Path/To/jacob/RLIP/datasets/OI_keep_names_trainval_pretraining.json", "r") as f:
            #     vg_keep_names = json.load(f)
            self.relationship_names = vg_keep_names["relationship_names"]
            self.object_names = vg_keep_names["object_names"]
        else:
            if not adding_OI:
                keep_names_file = "/Path/To/jacob/RLIP/datasets/vg_keep_names_v1_no_lias_freq.json"
                # keep_names_file = "/Path/To/jacob/RLIP/datasets/vg_keep_names_v2_freq.json"
                # keep_names_file = "/Path/To/jacob/RLIP/datasets/vg_keep_names_v1_random_sample50_freq.json"
            else:
                keep_names_file = "/Path/To/jacob/RLIP/datasets/OI_trainval_VG_all_keep_names_freq.json"

            with open(keep_names_file, "r") as f:
                vg_keep_names = json.load(f)
            self.relationship_names = vg_keep_names["relationship_names"]
            self.object_names = vg_keep_names["object_names"]
            if "relationship_freq" in vg_keep_names.keys():
                self.relationship_freq = vg_keep_names["relationship_freq"]
            if "object_freq" in vg_keep_names.keys():
                self.object_freq = vg_keep_names["object_freq"]
            
            self.rel_feature = np.load('/Path/To/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_sup-simcse-roberta-large.npz', allow_pickle = True)['rel_feature'].item()
            # self.rel_feature = np.load('/Path/To/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_text_feature.npz', allow_pickle = True)['rel_feature'].item()
            # verb_text_bf_fusion = torch.stack([torch.from_numpy(self.rel_feature[rt]).to(memory_cache["text_memory_resized"].device) for rt in rel_text], dim = 0)
            list_rel_ft = [torch.from_numpy(i) for i in list(self.rel_feature.values())]
            self.rel_feature = (list(self.rel_feature.keys()), torch.stack(list_rel_ft, dim = 0))
            self.obj_feature = np.load('/Path/To/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_sup-simcse-roberta-large.npz', allow_pickle = True)['obj_feature'].item()
            # self.obj_feature = np.load('/Path/To/data/VG/word_embedding/vg_keep_names_v1_no_lias_freq_text_feature.npz', allow_pickle = True)['obj_feature'].item()
            list_obj_ft = [torch.from_numpy(i) for i in list(self.obj_feature.values())]
            self.obj_feature = (list(self.obj_feature.keys()), torch.stack(list_obj_ft, dim = 0))

        ### Filter out images with zero object annotations
        # new_annotations = []
        # for anno in self.annotations:
        #     if len(anno['objects']) > 3:
        #         new_annotations.append(anno)
        # print(f'Deleting {len(self.annotations) - len(new_annotations)} images with zero object.')
        # self.annotations = new_annotations

        if img_set == 'pretrain':
            self.ids = []
            # use to filter unavailable images by annotations
            self.ids = list(range(len(self.annotations)))

        else:
            self.ids = list(range(len(self.annotations)))

        
        self.num_pairs = num_queries//2
        self.required_num_pos_pairs = int(self.num_pairs/(pos_neg_ratio + 1)*pos_neg_ratio + 0.5)
        self.required_num_neg_pairs = self.num_pairs - self.required_num_pos_pairs
        self.verb_tagger = verb_tagger

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
                        (After using the function 'self._transforms(img, target)')
        """
        img_anno = self.annotations[self.ids[idx]]
        objects_anno = img_anno['objects'] # data type: list
        relationships_anno = img_anno['relationships'] # data type: list
        
        img_file_name = str(img_anno['image_id']) + '.jpg'
        img = Image.open(self.img_folder / img_file_name).convert('RGB')
        # print(img_anno['file_name'])
        # print(img.size)
        w, h = img.size
        
        # make sure that #queries are more than #bboxes
        if self.img_set == 'pretrain' and len(relationships_anno) > self.num_queries:
            relationships_anno = relationships_anno[:self.num_queries]
        # collect coordinates and names for all bboxes
        boxes = []
        for cur_box in objects_anno:
            cur_box_cor = [cur_box['x'], cur_box['y'], cur_box['x'] + cur_box['w'], cur_box['y'] + cur_box['h']]
            boxes.append(cur_box_cor)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # print(f"boxes shape:{boxes.shape}")

        if self.use_all_text_labels:
            obj_unique = unique_name_dict_from_list(self.object_names)
            rel_unique = unique_name_dict_from_list(self.relationship_names)
        else:
            obj_unique = unique_name_dict_from_anno(objects_anno, 'objects')
            rel_unique = unique_name_dict_from_anno(relationships_anno, 'relationships')
        obj_classes = [(idx_cur_box, obj_unique[cur_box['names']]) for idx_cur_box, cur_box in enumerate(objects_anno)]
        obj_classes = torch.tensor(obj_classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])

        ### The following code is responsible for reading positive samples.
        if self.img_set == 'pretrain':
            # HICO: clamp the box and drop those unreasonable ones
            # VG: I have checked that all boxes are 
            boxes[:, 0::2].clamp_(min=0, max=w)  # xyxy    clamp x to 0~w
            boxes[:, 1::2].clamp_(min=0, max=h)  # xyxy    clamp y to 0~h

            # This 'keep' can be removed because all w and h are assured to be greater than 0.
            # keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            # boxes = boxes[keep]  # may have a problem
            # obj_classes = obj_classes[keep]

            # construct target dict
            target['boxes'] = boxes
            target['labels'] = obj_classes  # like [[0, 0][1, 56][2, 0][3, 0]...]
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if self._transforms is not None:
                img, target = self._transforms(img, target) # target['boxes'].shape and target['labels'].shape may change
            

            # ******This 'keep' should be maintained because self._transform may eliminate some boxes******
            # which means that target['boxes'].shape and target['labels'].shape may change
            
            kept_box_indices = [label[0] for label in target['labels']] # enumerated indices for kept (0, 1, 2, 3, 4, ...)
            # Guard against situations with target['labels'].shape[0] = 0, which can not perform target['labels'][:, 1].
            if target['labels'].shape[0] > 0:
                target['labels'] = target['labels'][:, 1] # object classes in 80 classes

            sub_labels, obj_labels, predicate_labels, sub_boxes, obj_boxes = [], [], [], [], []
            sub_obj_pairs = []
            relationships_anno_local = add_local_object_id(relationships_anno, objects_anno)
            for cur_rel_local in relationships_anno_local:
                # Make sure that sub and obj are not eliminated by self._transform.
                if cur_rel_local['subject_id_local'] not in kept_box_indices or \
                        cur_rel_local['object_id_local'] not in kept_box_indices:
                    continue

                sub_obj_pair = (cur_rel_local['subject_id_local'], cur_rel_local['object_id_local'])
                if sub_obj_pair in sub_obj_pairs:
                    predicate_labels[sub_obj_pairs.index(sub_obj_pair)][rel_unique[cur_rel_local['predicate']]] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    # genrate labels for sub, obj and predicate
                    kept_sub_idx = kept_box_indices.index(cur_rel_local['subject_id_local'])
                    kept_obj_idx = kept_box_indices.index(cur_rel_local['object_id_local'])
                    cur_sub_label = target['labels'][kept_sub_idx]
                    cur_obj_label = target['labels'][kept_obj_idx]
                    sub_labels.append(cur_sub_label)
                    obj_labels.append(cur_obj_label)
                    predicate_label = [0 for _ in range(len(rel_unique))]
                    predicate_label[rel_unique[cur_rel_local['predicate']]] = 1
                    predicate_labels.append(predicate_label)
                    
                    # generate box coordinates for sub and obj
                    # print(f"target['boxes'].shape: {target['boxes'].shape}")
                    cur_sub_box = target['boxes'][kept_sub_idx]
                    cur_obj_box = target['boxes'][kept_obj_idx]
                    sub_boxes.append(cur_sub_box)
                    obj_boxes.append(cur_obj_box)
            
            target['image_id'] = img_anno['image_id']
            target['obj_classes'] = list(dict(obj_unique).keys())
            target['verb_classes'] = list(dict(rel_unique).keys())
            # target['obj_classes'] = generate_class_names_list(obj_unique) 
            # target['predicate_classes'] = generate_class_names_list(rel_unique) 
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['sub_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(rel_unique)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                # target['obj_classes'] = list(dict(obj_unique).keys())
                # target['verb_classes'] = list(dict(rel_unique).keys())
                target['obj_labels'] = torch.stack(obj_labels)
                target['sub_labels'] = torch.stack(sub_labels)
                target['verb_labels'] = torch.as_tensor(predicate_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)


            ### Implementation 2
            ### Remark by Hangjie
            ### The following code is responsible for constructing pairs for training the verb tagger.
            ### By default, I'll perform non-overlap sampling and generate masks.
            if self.verb_tagger:
                # num_gt_pair = target['obj_labels'].shape[0]
                num_pos_pairs = len(sub_obj_pairs)
                assert num_pos_pairs == target['obj_labels'].shape[0]
                
                possible_pairs = list(permutations(kept_box_indices, 2))
                num_possible_pairs = len(possible_pairs)
                neg_possible_pairs = copy.deepcopy(possible_pairs)
                for one_pos_pair in sub_obj_pairs:
                    ### Remark by Hangjie:
                    ### Due to the noisy annotations of the vg dataset, 
                    ### we may have triplets with identical subjects and objects.
                    ### Thus, we can write code to ensure that if one_pos_pair not in neg_possible_pairs, they should have identical subject and object indices.
                    if one_pos_pair not in neg_possible_pairs:
                        assert one_pos_pair[0] == one_pos_pair[1]
                    else:
                        neg_possible_pairs.remove(one_pos_pair)
                    # assert one_pos_pair in neg_possible_pairs
                    # neg_possible_pairs.remove(one_pos_pair)
                num_neg_pairs = len(neg_possible_pairs)
                

                if num_possible_pairs < self.num_pairs:
                    # We need to generate key_padding_mask and src_padding_mask to pass into DeformableTransformerDecoderLayer.
                    # Is src_padding_mask needed?

                    full_pos_pairs = list(range(num_pos_pairs))
                    full_neg_pairs = neg_possible_pairs
                else:
                    if num_pos_pairs < self.num_pairs:
                        full_pos_pairs = list(range(num_pos_pairs))
                        full_neg_pairs = random.sample(neg_possible_pairs, self.num_pairs - num_pos_pairs)
                    else:
                        full_pos_pairs = random.sample(range(num_pos_pairs), self.num_pairs)
                        full_neg_pairs = []

                ### Compose the query pairs for training.
                target['obj_labels'] = target['obj_labels'][full_pos_pairs]
                target['sub_labels'] = target['sub_labels'][full_pos_pairs]
                target['verb_labels'] = target['verb_labels'][full_pos_pairs]
                target['sub_boxes'] = target['sub_boxes'][full_pos_pairs]
                target['obj_boxes'] = target['obj_boxes'][full_pos_pairs]

                neg_sub_labels, neg_obj_labels, neg_predicate_labels, neg_sub_boxes, neg_obj_boxes = [], [], [], [], []
                for one_neg_pair in full_neg_pairs:
                    kept_sub_idx = kept_box_indices.index(one_neg_pair[0])
                    kept_obj_idx = kept_box_indices.index(one_neg_pair[1])
                    cur_sub_label = target['labels'][kept_sub_idx]
                    cur_obj_label = target['labels'][kept_obj_idx]
                    neg_sub_labels.append(cur_sub_label)
                    neg_obj_labels.append(cur_obj_label)
                    predicate_label = [0 for _ in range(len(rel_unique))]
                    neg_predicate_labels.append(predicate_label)
                    
                    # generate box coordinates for sub and obj
                    # print(f"target['boxes'].shape: {target['boxes'].shape}")
                    cur_sub_box = target['boxes'][kept_sub_idx]
                    cur_obj_box = target['boxes'][kept_obj_idx]
                    neg_sub_boxes.append(cur_sub_box)
                    neg_obj_boxes.append(cur_obj_box)

                if len(full_neg_pairs) > 0:
                    # target['obj_labels'] = torch.cat((torch.stack(neg_obj_labels), target['obj_labels']), dim = 0)
                    # target['sub_labels'] = torch.cat((torch.stack(neg_sub_labels), target['sub_labels']), dim = 0)
                    # target['verb_labels'] = torch.cat((torch.as_tensor(neg_predicate_labels, dtype=torch.float32), target['verb_labels']), dim = 0)
                    # target['sub_boxes'] = torch.cat((torch.stack(neg_sub_boxes), target['sub_boxes']), dim = 0)
                    # target['obj_boxes'] = torch.cat((torch.stack(neg_obj_boxes), target['obj_boxes']), dim = 0)

                    target['obj_labels'] = torch.cat((target['obj_labels'], torch.stack(neg_obj_labels)), dim = 0)
                    target['sub_labels'] = torch.cat((target['sub_labels'], torch.stack(neg_sub_labels)), dim = 0)
                    target['verb_labels'] = torch.cat((target['verb_labels'], torch.as_tensor(neg_predicate_labels, dtype=torch.float32)), dim = 0)
                    target['sub_boxes'] = torch.cat((target['sub_boxes'], torch.stack(neg_sub_boxes)), dim = 0)
                    target['obj_boxes'] = torch.cat((target['obj_boxes'], torch.stack(neg_obj_boxes)), dim = 0)

        return img, target


def generate_class_names_list(unique_dict):
    '''
    unique_dict: an OrderedDict() outputted from function unique_name_dict.
    return: a list for the class names
    '''
    names_list = []
    for name in list(unique_dict.keys()):
        names_list.append(name)
    print(names_list)
    return names_list

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


def unique_name_dict_from_anno(anno, anno_type):
    '''
    This function transform the original class names to a dict without repeative keys()
        from original annotations.
    This helps to determine the local class label id.

    anno: annotation list from a single image
    anno_type: 'relationships' or 'objects'
    '''
    assert anno_type == 'relationships' or anno_type == 'objects'
    key_name = 'names' if anno_type == 'objects' else 'predicate'

    unique_dict = OrderedDict()
    label_tensor = torch.tensor([i for i in range(len(anno))])
    dict_idx = 0
    for cur_anno in anno:
        if cur_anno[key_name] not in unique_dict.keys():
            # print(cur_anno[key_name])
            unique_dict[cur_anno[key_name]] = label_tensor[dict_idx]
            dict_idx += 1
    return unique_dict


def unique_name_dict_from_list(name_list):
    '''
    This function transform the original class names to a dict without repeative keys()
        from a name list.
    This helps to determine the local class label id.

    name_list: a list for the name string
    '''
    unique_dict = OrderedDict()
    label_tensor = torch.tensor([i for i in range(len(name_list))])
    for dict_idx, n in enumerate(name_list):
        unique_dict[n] = label_tensor[dict_idx]
    return unique_dict


# Add color jitter to coco transforms
def make_vg_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set in ['train', 'pretrain']:
        # print('')
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
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



def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    assert len(input_tensor.shape) == 3
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    # input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    # print('input_tensor.shape ' + str(input_tensor.shape))
    cv2.imwrite(filename, input_tensor)


def build(image_set, args):
    root = Path(args.vg_path)
    assert root.exists(), f'provided VG path {root} does not exist'

    # /Path/To/data/VG
    if args.dataset_file == 'vg':
        if args.use_aliases:
            PATHS = {'pretrain': (root / 'images' , root / 'annotations' / 'scene_graphs_preprocess_alias.json')}
        elif args.use_all_text_labels:
            PATHS = {'pretrain': (root / 'images' , root / 'annotations' / 'scene_graphs_preprocessv1.json')}
        else:
            PATHS = {'pretrain': (root / 'images' , root / 'annotations' / 'scene_graphs_preprocessv1.json')}
            
    elif args.dataset_file == 'vg_oi':
        PATHS = {'pretrain': (root / 'OI_VG_images' , root / 'annotations' / 'OI_trainval_VG_all_pretraining.json')}


    if args.cross_modal_pretrain:
        assert image_set == 'pretrain'
        vg_img_folder, vg_anno_file = PATHS[image_set]
        print("Annotation file we use: ", vg_anno_file)
        if not args.verb_tagger:
            dataset = VGRelDetection(image_set, vg_img_folder, vg_anno_file, 
                                    transforms=make_vg_transforms(image_set), 
                                    num_queries = args.num_queries,
                                    use_all_text_labels = args.use_all_text_labels)
        else:
            dataset = VGRelTagger(image_set, vg_img_folder, vg_anno_file, 
                                    transforms=make_vg_transforms(image_set), 
                                    num_queries = args.num_queries,
                                    use_all_text_labels = args.use_all_text_labels,
                                    pos_neg_ratio = args.pos_neg_ratio, verb_tagger = True)
    

    return dataset



def triplet_nms_filter(preds):
    preds_filtered = []
    for img_preds in preds:
        pred_bboxes = img_preds['predictions']
        pred_hois = img_preds['hoi_prediction']
        all_triplets = {}
        for index, pred_hoi in enumerate(pred_hois):
            triplet = str(pred_bboxes[pred_hoi['subject_id']]['category_id']) + '_' + \
                        str(pred_bboxes[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

            if triplet not in all_triplets:
                all_triplets[triplet] = {'subs':[], 'objs':[], 'scores':[], 'indexes':[]}
            all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
            all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
            all_triplets[triplet]['scores'].append(pred_hoi['score'])
            all_triplets[triplet]['indexes'].append(index)

        all_keep_inds = []
        for triplet, values in all_triplets.items():
            subs, objs, scores = values['subs'], values['objs'], values['scores']
            keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

            keep_inds = list(np.array(values['indexes'])[keep_inds])
            all_keep_inds.extend(keep_inds)

        preds_filtered.append({
            'filename': img_preds['filename'],
            'predictions': pred_bboxes,
            'hoi_prediction': list(np.array(img_preds['hoi_prediction'])[all_keep_inds])
            })

    return preds_filtered


def check_vg_stat(dataset_path):
    '''
    Step1. How many images do this dataset have annotated with >=100 triplets?
    Step2. Is the length of 'names' for the objects can be greater than 1? 
            Yes, like ['woman', 'person', 'she', 'lady']
                      ['court', 'badminton', 'white lines', 'net', 'tennis', 'lines']
                      ['white tennis shoe', 'shoes', 'left shoe', 'shoe']
    Step3. Are there different relationships with the same sub id and obj id but differnt predicate?
            i.e. check multi-predicate relationships
            Note that this is merely a rough estimation as we only consider 2-predicate relationships.
            If there are N-predicate relationships, the nums will add up (combinations(a set with N elements, 2)). 
    Step4. Are there any boxes' w/h <=0?
    Step5. Observe the synset string.
    '''
    dataset_path = Path(dataset_path)
    print("loading scene graphs....")
    with open(dataset_path / "scene_graphs_preprocess_alias.json", "r") as f:
    # with open(dataset_path / "scene_graphs_preprocess.json", "r") as f:
    # with open(dataset_path / "scene_graphs.json", "r") as f:
    # with open(dataset_path / "scene_graphs_small_for_debug_preprocess.json", "r") as f:
    # with open(dataset_path / "scene_graphs_small_for_debug.json", "r") as f:
        VG_scene_graph = json.load(f)
    print("loading success!")

    # # Step1
    # rels_greater_100 = 0
    # objects_greater_100 = 0
    # for i in range(len(VG_scene_graph)):
    #     if len(VG_scene_graph[i]["relationships"])>100:
    #         print('Number fo relationship triplets more than 100...')
    #         print(len(VG_scene_graph[i]["relationships"]), VG_scene_graph[i]["image_id"])
    #         rels_greater_100 += 1
    #     if len(VG_scene_graph[i]["objects"])>100:
    #         print('Number of objects more than 100...')
    #         print(len(VG_scene_graph[i]["objects"]), VG_scene_graph[i]["image_id"])
    #         objects_greater_100 += 1
    # print(f"{rels_greater_100} images have more than 100 triplets, {objects_greater_100}  images have more than 100 objects.")
    # # 106 images have more than 100 triplets, 560 images have more than 100 objects.

    # # Step2
    # obj_names_greater_1 = 0
    # for i in range(len(VG_scene_graph)):
    #     for j in VG_scene_graph[i]["objects"]:
    #         if isinstance(j['names'], list):
    #             if len(j['names']) > 1:
    #                 obj_names_greater_1 += 1
    #                 print(j['names'])
    # print(f"{obj_names_greater_1} objects have greater than 2 names.")

    # # Step3
    # ori_rel_sum = 0
    # with2_rel_sum = 0
    # for idx_anno, cur_anno in enumerate(VG_scene_graph):
    #     # keep flag: if flag==1, then keep, else drop it due to repetitive annotations
    #     keep_list = np.ones(len(cur_anno["relationships"]))
    #     for idx_rel, cur_rel in enumerate(cur_anno["relationships"]):
    #         if keep_list[idx_rel]:
    #             for idx_next, next_rel in enumerate(cur_anno["relationships"][idx_rel+1:]):
    #                 if keep_list[idx_rel + 1 + idx_next]:    
    #                     if cur_rel["subject_id"] == next_rel["subject_id"] and \
    #                         cur_rel["object_id"] == next_rel["object_id"] and \
    #                         cur_rel["predicate"] != next_rel["predicate"]:
    #                         with2_rel_sum += 1
    #                         keep_list[idx_rel + 1 + idx_next] = 0
    #     ori_rel_sum += len(cur_anno["relationships"])
    # print(f"Original rel sums:{ori_rel_sum}, how many rels are there with multi-predicate:{with2_rel_sum}")
    # # after preprocess: 
    # # Original rel sums:16700, how many rels are there with multi-predicate:760
    # # Original rel sums:1992780, how many rels are there with multi-predicate:86277

    # # Step4
    # ori_bbox_sum = 0
    # nonpos_bbox_sum = 0
    # for idx_anno, cur_anno in enumerate(VG_scene_graph):
    #     ori_bbox_sum += len(cur_anno["objects"])
    #     for idx_obj, cur_obj in enumerate(cur_anno["objects"]):
    #         if cur_obj['w'] <=0 or cur_obj['h'] <=0:
    #             nonpos_bbox_sum += 1
    # print(f"Original bbox sums:{ori_bbox_sum}, how many bboxs' w or h <=0:{nonpos_bbox_sum}")
    # # Original bbox sums:3802374, how many bboxs' w or h <=0:0

    # # Step5
    # for idx_anno, cur_anno in enumerate(VG_scene_graph):
    #     for idx_obj, cur_obj in enumerate(cur_anno["objects"]):
    #         for syn in cur_obj["synsets"]:
    #             if "04" in syn:
    #                 print(syn)

    # Step6
    rel_sum = OrderedDict()
    obj_sum = OrderedDict()
    for idx_anno, cur_anno in enumerate(VG_scene_graph):
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
    rel_greater_300 = 0
    obj_greater_1000 = 0
    for idx, (i, j) in enumerate(rel_sum):
        if idx < 50:
            print(i, j)
        if j >= 300:
            rel_greater_300 += 1
    print('rel_greater_300:', rel_greater_300)
    for idx, (i, j) in enumerate(obj_sum):
        if idx < 500:
            print(i, j)
        if j >= 1000:
            obj_greater_1000 += 1
    print('obj_greater_1000:', obj_greater_1000)


def alias_dict(alias_set):
    '''
    alias_set: 'relationship_alias' or 'object_alias'
    return: a dict that the values are always be the first item in relationship_alias.txt or object_alias.txt
        like:
           {'tree': 'tree', 'trees': 'tree', 'tree is':'tree' , 'of trees':'tree' ...}
    '''
    # relationship_alias: 107607 unique names, 9003 synsets
    # object_alias: 54257 unique names, 7301 synsets
    assert alias_set == 'relationship_alias' or alias_set == 'object_alias'
    if alias_set == 'relationship_alias':
        path = '/Path/To/data/VG/annotations/relationship_alias.txt'
    elif alias_set == 'object_alias':
        path = '/Path/To/data/VG/annotations/object_alias.txt'

    alias_dict = {}
    with open(path, 'r') as f:
        for line in f:
            same_meaning_list = line.strip().split(',')
            for i in same_meaning_list:
                alias_dict[i] = same_meaning_list[0]
    return alias_dict


def convert_obj_list_to_obj_dict(objects):
    '''
    convert the object annotations to a dict whose
        keys: object_id
        values: original object annotation
    '''
    objects_dict = {}
    for i in objects:
        objects_dict[i['object_id']] = i
    return objects_dict


def compute_IOU_vg(bbox1, bbox2, object_alias_dict):
    '''
    This function computes IoU for VG annotation, the difference to the vanilla compute_IOU is that:
        here we use 'names' to denote object class rather than 'category_id' in VG
    '''
    b1_alias_name = object_alias_dict[bbox1['names']] if bbox1['names'] in object_alias_dict.keys() else bbox1['names']
    b2_alias_name = object_alias_dict[bbox2['names']] if bbox2['names'] in object_alias_dict.keys() else bbox2['names']

    # if object_alias_dict[bbox1['names']] == object_alias_dict[bbox2['names']]:
    if b1_alias_name == b2_alias_name:
        rec1 = bbox1['bbox']
        rec2 = bbox2['bbox']
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
        S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
            return intersect / (sum_area - intersect)
    else:
        return 0


def redundant_relation(relation1, relation2, objects, IoU_thre, 
                       relationship_alias_dict, object_alias_dict):
    '''
    Relation annotation from scene_graphs.json:
    {"relationship_id": 15947, "predicate": "wears", "synsets": ["wear.v.01"],
     "subject_id": 1023838, "object_id":  5071,}

    Bbox annotation from scene_graphs.json:
    {"object_id": 1023838, "x": 324, "y": 320, "w": 142, "h": 255, 
     "name": "cat","synsets": ["cat.n.01"]}

    object: objects list from scene_graphs.json

    return: True if relation1 and relation2 are repetitive:
        1. with the same subject and object class
        2. subject and object IoU both greater than IoU_thre
        3. 'predicate' is the same
    '''
    r1_sub = objects[relation1['subject_id']]
    r1_obj = objects[relation1['object_id']]
    r2_sub = objects[relation2['subject_id']]
    r2_obj = objects[relation2['object_id']]
    r1_sub_bbox = r1_sub['x'], r1_sub['y'], r1_sub['x'] + r1_sub['w'], r1_sub['y'] + r1_sub['h']
    r1_obj_bbox = r1_obj['x'], r1_obj['y'], r1_obj['x'] + r1_obj['w'], r1_obj['y'] + r1_obj['h']
    r2_sub_bbox = r2_sub['x'], r2_sub['y'], r2_sub['x'] + r2_sub['w'], r2_sub['y'] + r2_sub['h']
    r2_obj_bbox = r2_obj['x'], r2_obj['y'], r2_obj['x'] + r2_obj['w'], r2_obj['y'] + r2_obj['h']

    r1_sub_bbox = {'names': r1_sub['names'], 'bbox': r1_sub_bbox}
    r1_obj_bbox = {'names': r1_obj['names'], 'bbox': r1_obj_bbox}
    r2_sub_bbox = {'names': r2_sub['names'], 'bbox': r2_sub_bbox}
    r2_obj_bbox = {'names': r2_obj['names'], 'bbox': r2_obj_bbox}

    sub_IoU = compute_IOU_vg(r1_sub_bbox, r2_sub_bbox, object_alias_dict = object_alias_dict)
    obj_IoU = compute_IOU_vg(r1_obj_bbox, r2_obj_bbox, object_alias_dict = object_alias_dict)

    r1_alias_name = relationship_alias_dict[relation1['predicate']] if relation1['predicate'] in relationship_alias_dict.keys() else relation1['predicate']
    r2_alias_name = relationship_alias_dict[relation2['predicate']] if relation2['predicate'] in relationship_alias_dict.keys() else relation2['predicate']
    if sub_IoU >= IoU_thre and obj_IoU >= IoU_thre and r1_alias_name == r2_alias_name:
        return True
    else:
        return False


def vg_preprocess(dataset_path, IoU_thre, num_queries = 100, save_preprocess = False):
    '''
    This is added as a step of pre-processing v2.
    Step0. Make the 'names' of the objects and 'predicate' of the relationships lower case.

    Pre-processing v1
    Step1. Filter out annotations with repetitive triplets 
           (with the same subject, object and relationship labels)
    Setp2. Filter out redundant object names whose numbers are greater than 1 and
            make the value of the 'names' attribute to be a string rather than a list
            Filtering strategy: Choose the first one if #names >= 2.
    Step3. Filter out annotations with redundant triplets 
           (same classes for Subject, object and relationship labels, 
           and IoU > IoU_thre for the subject and the object)
    Step4. Discard relationships if the image has more than 100 relationships. 
           (ensure the number of querries >= the number of relationships )
    Step5. Filter out bboxes which are i) out of the range of the image and ii) w or h <=0.
           Also filter out the corresponding relationships.
    
    Following steps are pre-processing v2.
    Step7. Aim to use alias.txt to reduce the class noise in the annotations for objects and relationships.
           This function is identical to merge_label_with_alias()
           !!!!!! we do not use this step by default.
    Step8. Merge box annotations
    Step9. Filter out relations if the #object <= 1000 and #relation <= 500.
           Filter out objects if it is not in those whose #object <= 1000 and #relation <= 500.
    '''
    dataset_path = Path(dataset_path)
    print("loading scene graphs....")
    anno_file_name = "scene_graphs.json"
    # anno_file_name = "scene_graphs_small_for_debug.json"
    with open(dataset_path / anno_file_name, "r") as f:
        VG_scene_graph = json.load(f)
    print("loading success!")

    # Step0
    for idx_anno, cur_anno in enumerate(VG_scene_graph):
        for idx_obj, cur_obj in enumerate(cur_anno['objects']):
            for idx_n, n in enumerate(cur_obj['names']):
                cur_obj['names'][idx_n] = n.lower()
        for idx_rel, cur_rel in enumerate(cur_anno['relationships']):
            cur_rel['predicate'] = cur_rel['predicate'].lower()

    # Step1
    ori_rel_sum = 0
    filter_rel_sum = 0
    for idx_anno, cur_anno in enumerate(VG_scene_graph):
        # keep flag: if flag==1, then keep, else drop it due to repetitive annotations
        keep_list = np.ones(len(cur_anno["relationships"]))
        for idx_rel, cur_rel in enumerate(cur_anno["relationships"]):
            if keep_list[idx_rel]:
                for idx_next, next_rel in enumerate(cur_anno["relationships"][idx_rel+1:]):
                    if keep_list[idx_rel + 1 + idx_next]:    
                        if cur_rel["subject_id"] == next_rel["subject_id"] and \
                            cur_rel["object_id"] == next_rel["object_id"] and \
                            cur_rel["predicate"] == next_rel["predicate"]:
                            keep_list[idx_rel + 1 + idx_next] = 0
        # Filter out relationship anno with flag == 0
        ori_rel_sum += len(cur_anno["relationships"])
        cur_anno_rel = [j for i,j in enumerate(cur_anno["relationships"]) if keep_list[i]]
        filter_rel_sum += len(cur_anno_rel)
        cur_anno["relationships"] = cur_anno_rel
    print(f"Original rel sums:{ori_rel_sum}, after filtering:{filter_rel_sum}")
    # Original rel sums:21166, after filtering:17020
    # Original rel sums:2316104, after filtering:2025304

    # Step2
    ori_obj_sum = 0
    greater2_obj_sum = 0
    for idx_anno, cur_anno in enumerate(VG_scene_graph):
        for idx_obj, cur_obj in enumerate(cur_anno['objects']):
            ori_obj_sum += 1
            # There no need for this command because they are all list like ['man']
            # if isinstance(cur_obj['names'], list): 
            if len(cur_obj['names']) >= 2:
                greater2_obj_sum += 1
            cur_obj['names'] = cur_obj['names'][0]
    print(f"All objects counts:{ori_obj_sum}, objects with 2 or more labels:{greater2_obj_sum}")
    # All objects counts:31310, objects with 2 or more labels:676
    # All objects counts:3802374, objects with 2 or more labels:59478

    # Step3
    ori_rel_sum = 0
    filter_rel_sum = 0
    relationship_alias_dict = alias_dict('relationship_alias')
    object_alias_dict = alias_dict('object_alias')
    for idx_anno, cur_anno in enumerate(VG_scene_graph):
        # keep flag: if flag==1, then keep, else drop it due to repetitive annotations
        keep_list = np.ones(len(cur_anno["relationships"]))
        for idx_rel, cur_rel in enumerate(cur_anno["relationships"]):
            if keep_list[idx_rel]:
                for idx_next, next_rel in enumerate(cur_anno["relationships"][idx_rel+1:]):
                    if keep_list[idx_rel + 1 + idx_next]:
                        if redundant_relation(relation1 = cur_rel, 
                                              relation2 = next_rel, 
                                              objects = convert_obj_list_to_obj_dict(cur_anno['objects']),
                                              IoU_thre = IoU_thre,
                                              relationship_alias_dict = relationship_alias_dict,
                                              object_alias_dict = object_alias_dict):
                            keep_list[idx_rel + 1 + idx_next] = 0
        ori_rel_sum += len(cur_anno["relationships"])
        cur_anno_rel = [j for i,j in enumerate(cur_anno["relationships"]) if keep_list[i]]
        filter_rel_sum += len(cur_anno_rel)
        cur_anno["relationships"] = cur_anno_rel
    print(f"Rel sums after step1:{ori_rel_sum}, rel sums after step2:{filter_rel_sum}")
    # Rel sums after step1:17020, rel sums after step2:16512, IoU = 0.5
    # Rel sums after step1:17020, rel sums after step2:16700, IoU = 0.7
    # Rel sums after step1:17020, rel sums after step2:16982, IoU = 0.9
    # Rel sums after step1:2025304, rel sums after step2:1992836, IoU = 0.7

    # Step4
    ori_img_sum = 0
    greater100_img_sum = 0
    for idx_anno, cur_anno in enumerate(VG_scene_graph):
        ori_img_sum += 1
        if len(cur_anno["relationships"]) > num_queries:
            cur_anno["relationships"] = cur_anno["relationships"][:num_queries]
            greater100_img_sum += 1
    print(f"Images counts:{ori_img_sum}, images with rels greater than 100:{greater100_img_sum}")
    # Images counts:108077, images with rels greater than 100: 5
    
    # Step7
    # object_alias_dict = alias_dict('object_alias')
    # relationship_alias_dict = alias_dict('relationship_alias')
    # for idx_anno, cur_anno in enumerate(VG_scene_graph):
    #     for idx_obj, cur_obj in enumerate(cur_anno['objects']):
    #         cur_obj['names'] = object_alias_dict[cur_obj['names']] if cur_obj['names'] in object_alias_dict.keys() else cur_obj['names']
    #     for idx_rel, cur_rel in enumerate(cur_anno['relationships']):
    #         cur_rel['predicate'] = relationship_alias_dict[cur_rel['predicate']] if cur_rel['predicate'] in relationship_alias_dict.keys() else cur_rel['predicate']

    # Step9
    rel_sum = OrderedDict()
    obj_sum = OrderedDict()
    for idx_anno, cur_anno in enumerate(VG_scene_graph):
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
        if j >= 20:
        # if j >= 0:
            rel_keep.append(i)
            rel_freq[i] = j
    print('rel_keep:', len(rel_keep))
    for idx, (i, j) in enumerate(obj_sum):
        # if j >= 1000:
        # if j >= 5:
        if j >= 20:
        # if j >= 0:
            obj_keep.append(i)
            obj_freq[i] = j
    print('obj_keep:', len(obj_keep))
    # with open('vg_keep_names_v1_no_lias_freq.json', 'w') as outfile:
    #     json.dump({"relationship_names": rel_keep, \
    #                "object_names": obj_keep, \
    #                "relationship_freq": rel_freq,
    #                "object_freq": obj_freq}, outfile)
    # print('save vg_keep_names.json !!!')
    # if j >= 100:
    # rel_keep: 516
    # obj_keep: 2255
    # How many relationships are left after filtering? 1615853

    # if j >= 20:
    # rel_keep: 1668
    # obj_keep: 6298
    # How many relationships are left after filtering? 1778660

    for idx_anno, cur_anno in enumerate(VG_scene_graph):
        object_dict = convert_obj_list_to_obj_dict(cur_anno['objects'])
        new_anno_rel = []
        for idx_rel, cur_rel in enumerate(cur_anno['relationships']):
            r1_sub = object_dict[cur_rel['subject_id']]
            r1_obj = object_dict[cur_rel['object_id']]
            if r1_sub['names'] in obj_keep and r1_obj['names'] in obj_keep and cur_rel['predicate'] in rel_keep:
                new_anno_rel.append(cur_rel)
        cur_anno['relationships'] = new_anno_rel
    new_rel_sum = sum([len(cur_anno['relationships']) for cur_anno in VG_scene_graph])
    print("How many relationships are left after filtering?", new_rel_sum)

    # This is a sub-step of step9 which filters out objects which are not in the relations.
    # This step will exclude a large quantity of bboxes.
    # for idx_anno, cur_anno in enumerate(VG_scene_graph):
    #     rel_obj_id = []
    #     for idx_rel, cur_rel in enumerate(cur_anno['relationships']):
    #         if cur_rel['subject_id'] not in rel_obj_id:
    #             rel_obj_id.append(cur_rel['subject_id'])
    #         if cur_rel['object_id'] not in rel_obj_id:
    #             rel_obj_id.append(cur_rel['object_id'])

    #     new_anno_obj = []
    #     for idx_obj, cur_obj in enumerate(cur_anno['objects']):
    #         if cur_obj['names'] in obj_keep and cur_obj['object_id'] in rel_obj_id:
    #             new_anno_obj.append(cur_obj)
    #     cur_anno['objects'] = new_anno_obj
    # new_obj_sum = sum([len(cur_anno['objects']) for cur_anno in VG_scene_graph])
    # print("How many objects are left after filtering?", new_obj_sum)

    # Save the annotations after pre-processing
    if save_preprocess:
        with open(dataset_path / anno_file_name.replace('.', '_preprocess_greater20.'), 'w') as outfile:
            # '_preprocessv2.' '_preprocess_greater5.'  '_no_alias_test_nofiltering_relations.'  '_preprocess_v1test.'
            json.dump(VG_scene_graph, outfile)


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


def check_vg_synset_alias(synset_type,
                    dataset_path = '/Path/To/data/VG/annotations'):
    '''
    synset_type: "attribute_synsets", "object_synsets" or "relationship_synsets"
    dataset_path: path for the annotations
    '''
    dataset_path = Path(dataset_path)
    print("loading the synset file....")
    with open(dataset_path / (synset_type + ".json"), "r") as f:
        synsets = json.load(f)
    print("loading success!")

    ### Step1
    ### This step aims to check whether every object has its own "synsets" label.
    ### The answer is no. 843545 objects do not have synset labels.
    ###                   581933 relationships do not have synset labels.
    ###                   12825 objects have more than 1 synset label.
    ###                   All realtionships do not have more than 1 synset label.
    anno_file_name = "scene_graphs_preprocess.json"
    # anno_file_name = "scene_graphs_small_for_debug_preprocess.json"
    print("loading scene graphs....")
    with open(dataset_path / anno_file_name, "r") as f:
        VG_scene_graph = json.load(f)
    print("loading success!")
    objects_wo_synsets = 0
    relationships_wo_synsets = 0
    objects_synsets_greater_1 = 0
    relationships_synsets_greater_1 = 0
    for idx_anno, cur_anno in enumerate(VG_scene_graph):
        for idx_obj, cur_obj in enumerate(cur_anno['objects']):
            if len(cur_obj["synsets"]) == 0:
                objects_wo_synsets += 1
            elif len(cur_obj["synsets"]) > 1:
                objects_synsets_greater_1 += 1
        for idx_rel, cur_rel in enumerate(cur_anno['relationships']):
            if len(cur_rel["synsets"]) == 0:
                relationships_wo_synsets += 1
            elif len(cur_rel["synsets"]) > 1:
                relationships_synsets_greater_1 += 1
    print('How many objects are without synsets?', objects_wo_synsets)
    print('How many relationships are without synsets?', relationships_wo_synsets)
    print('How many objects are with more than 1 synset?', objects_synsets_greater_1)
    print('How many relationships are with more than 1 synset?', relationships_synsets_greater_1)

    ### Step2
    ### This step aims to check whether every object and relationship has its own alias.
    ### The answer is no. 528386 objects do not have alias labels.
    ###                   1018905 relationships do not have alias labels.
    anno_file_name = "scene_graphs_preprocess.json"
    # anno_file_name = "scene_graphs_small_for_debug_preprocess.json"
    print("loading scene graphs....")
    with open(dataset_path / anno_file_name, "r") as f:
        VG_scene_graph = json.load(f)
    print("loading success!")
    object_alias_dict = alias_dict('object_alias')
    relationship_alias_dict = alias_dict('relationship_alias')
    objects_wo_alias = 0
    relationships_wo_alias = 0
    for idx_anno, cur_anno in enumerate(VG_scene_graph):
        for idx_obj, cur_obj in enumerate(cur_anno['objects']):
            if cur_obj['names'] not in object_alias_dict.keys():
                objects_wo_alias += 1
        for idx_rel, cur_rel in enumerate(cur_anno['relationships']):
            if cur_rel['predicate'] not in relationship_alias_dict.keys():
                relationships_wo_alias += 1
    print('How many objects are without alias?', objects_wo_alias)
    print('How many relationships are without alias?', relationships_wo_alias)


def merge_label_with_alias(anno_file_name,
                           dataset_path = '/Path/To/data/VG/annotations'):
    '''
    This function aims to use alias.txt to reduce the class noise in the annotations 
    for objects and relationships.
    '''
    dataset_path = Path(dataset_path)
    print("loading scene graphs....")
    with open(dataset_path / anno_file_name, "r") as f:
        VG_scene_graph = json.load(f)
    print("loading success!")

    object_alias_dict = alias_dict('object_alias')
    relationship_alias_dict = alias_dict('relationship_alias')
    print(relationship_alias_dict)
    for idx_anno, cur_anno in enumerate(VG_scene_graph):
        for idx_obj, cur_obj in enumerate(cur_anno['objects']):
            cur_obj['names'] = object_alias_dict[cur_obj['names']] if cur_obj['names'] in object_alias_dict.keys() else cur_obj['names']
        for idx_rel, cur_rel in enumerate(cur_anno['relationships']):
            cur_rel['predicate'] = relationship_alias_dict[cur_rel['predicate']] if cur_rel['predicate'] in relationship_alias_dict.keys() else cur_rel['predicate']
            
    save_anno_file_name = anno_file_name.replace('.json', '_alias.json')

    # with open(dataset_path / save_anno_file_name, 'w') as outfile:
    #     json.dump(VG_scene_graph, outfile)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--vg_path', type=str, default=None,
                        help="Path to the Visual Genome dataset.")
    parser.add_argument('--cross_modal_pretrain', action = 'store_true',
                        help='Whether to perform cross-modal pretraining on VG dataset')
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    return parser

if __name__ == '__main__':
    # Pre-process data and save the data
    vg_preprocess(dataset_path = '/Path/To/data/VG/annotations',
                  IoU_thre = 0.7, 
                  save_preprocess = True)

    # Check some basic stats about VG dataset
    # check_vg_stat(dataset_path = '/Path/To/data/VG/annotations')

    # Check some basic stats about VG dataset
    # check_vg_synset_alias(synset_type = "attribute_synsets")
    # check_vg_synset_alias(synset_type = "relationship_synsets")
    # check_vg_synset_alias(synset_type = "object_synsets")

    # Use alias to reduce label redundancy
    # merge_label_with_alias('scene_graphs_preprocess.json')