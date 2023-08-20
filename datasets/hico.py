# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
HICO detection dataset.
"""
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np
import sys
sys.path.append("..") 
from util.image import draw_umich_gaussian, gaussian_radius

import torch
import torch.utils.data
import torchvision
from typing import List

import datasets.transforms as T
import cv2
import os
import math


class HICODetection(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries, args = None):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        # print(len(self.annotations))
        # print(type(self.annotations))
        # print(self.annotations[0])
        self._transforms = transforms

        self.num_queries = num_queries

        # 80 object classes
        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        # 117 verb classes
        self._valid_verb_ids = list(range(1, 118))

        if img_set == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(img_anno['annotations']):
                        break
                else:
                    self.ids.append(idx)
        else:
            self.ids = list(range(len(self.annotations)))
        # self.ids = self.ids[:1000]
        
        self.object_text = load_hico_object_txt()
        self.verb_text = load_hico_verb_txt()
        self.use_correct_subject_category_hico = args.use_correct_subject_category_hico
        if self.use_correct_subject_category_hico:
            print("Use correct subject category: 0 (rather than 1).")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        # print(img_anno['file_name'])
        # print(img.size)
        w, h = img.size
        
        # make sure that #queries are more than #bboxes
        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        # collect coordinates for all bboxes
        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Get the object index_id in the range of 80 classes. 
        # This is quite confusing because COCO has 80 classes but has ids 1~90. 
        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            # clamp the box and drop those unreasonable ones
            boxes[:, 0::2].clamp_(min=0, max=w)  # xyxy    clamp x to 0~w
            boxes[:, 1::2].clamp_(min=0, max=h)  # xyxy    clamp y to 0~h
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            # construct target dict
            target['boxes'] = boxes
            target['labels'] = classes  # like [[0, 0][1, 56][2, 0][3, 0]...]
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(img, target)
                # print(img.shape)


            # enumerated indices for kept (0, 1, 2, 3, 4, ...)
            kept_box_indices = [label[0] for label in target['labels']]
            # object classes in 80 classes
            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    # verb category_id in the range from 1 to 117
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    # Set all verb labels to 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)

            target['filename'] = img_anno['file_name']
            target['obj_classes'] = self.object_text
            target['verb_classes'] = self.verb_text
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['sub_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                # I assume the sub_labels are not used in the following experiments.
                # Thus, it's okay to set it to all ones. Otherwise, we should set it to all zeros.
                if self.use_correct_subject_category_hico:
                    target['sub_labels'] = torch.zeros((len(obj_labels),), dtype=torch.int64)
                    # print(target['sub_labels'][0])
                else:
                    target['sub_labels'] = torch.ones((len(obj_labels),), dtype=torch.int64)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
        else:
            target['filename'] = img_anno['file_name']
            target['boxes'] = boxes # 
            target['labels'] = classes # 
            target['id'] = idx # img_idx
            

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
                # if len(self._valid_verb_ids.index(hoi['category_id'])) >=2:
                #     print(self._valid_verb_ids.index(hoi['category_id']))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

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

    def set_seen_hois(self, unseen_list, hoi_list_new_path, zero_shot_setting):
        """
        This functions aims to set unseen verb class indices:
        To make the implementation easy, we still use the defined attribute "self.rare_triplets", "self.non_rare_triplets"
        It means that when we are using this function, 
            self.rare_triplets stores a list for unseen triplets (comparably hard).
            self.non_rare_triplets stores a list for seen triplets (comparably easy).
        """
        with open(hoi_list_new_path, 'r') as f:
            hoi_list_new = json.load(f)
        
        self.rare_triplets = []
        for u in unseen_list:
            assert u == int(hoi_list_new[u]["id"]) - 1
            triplet = (0, 
                       self._valid_obj_ids.index(hoi_list_new[u]["object_cat"]),
                       self._valid_verb_ids.index(hoi_list_new[u]["verb_id"]))
            self.rare_triplets.append(triplet)
        
        self.non_rare_triplets = []
        assert max(unseen_list) < 600
        seen_list = [i for i in range(600) if i not in unseen_list]
        for s in seen_list:
            assert s == int(hoi_list_new[s]["id"]) - 1
            triplet = (0, 
                       self._valid_obj_ids.index(hoi_list_new[s]["object_cat"]),
                       self._valid_verb_ids.index(hoi_list_new[s]["verb_id"]))
            self.non_rare_triplets.append(triplet)
        
        print('Set {:d} unseen (rare) triplets and {:d} seen (non-rare) triplets.'.format(len(self.rare_triplets), len(self.non_rare_triplets)))

    def remove_text_unseen(self, zero_shot_setting):
        """
        If we use zero-shot setting like 'UO' or 'UV', 
        we need to remove these texts from the input label sequence.  
        But do we need this?
        """
        assert zero_shot_setting in ['UO']
        if zero_shot_setting == 'UO' and self.img_set == 'train':
            ### This is to ensure that we exclude objs that are not in the training set.
            unseen_obj_name = ['dog', 'baseball bat', 'clock', 'elephant', 'frisbee',\
                               'pizza', 'skateboard', 'tennis racket', 'toothbrush', 'zebra']
            for u in unseen_obj_name:
                self.object_text.remove(u)
            print('Zero-shot setting: UO, {} texts remain, removing {}'.format(len(self.object_text), ', '.join(unseen_obj_name)))

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)
        # self.correct_mat shape: [117, 80]
        # print(self.correct_mat.shape)

def load_hico_verb_txt(file_path = 'datasets/hico_verb_names.txt') -> List[list]:
    '''
    Output like [['train'], ['boat'], ['traffic', 'light'], ['fire', 'hydrant']]
    '''
    verb_names = []
    for line in open(file_path,'r'):
        # verb_names.append(line.strip().split(' ')[-1])
        verb_names.append(' '.join(line.strip().split(' ')[-1].split('_')))
    return verb_names

def load_hico_object_txt(file_path = 'datasets/hico_object_names.txt') -> List[list]:
    '''
    Output like [['adjust'], ['board'], ['brush', 'with'], ['buy']]
    '''
    object_names = []
    with open(file_path, 'r') as f:
        object_names = json.load(f)
    object_list = list(object_names.keys())
    return object_list

class HICODetectionhm(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries
        self.draw_gaussian = draw_umich_gaussian

        # 80 object classes
        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        # 117 verb classes
        self._valid_verb_ids = list(range(1, 118))

        if img_set == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(img_anno['annotations']):
                        break
                else:
                    self.ids.append(idx)
        else:
            self.ids = list(range(len(self.annotations)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        # print(img_anno['file_name'])
        # print(img.size)
        w, h = img.size
        
        # make sure that #queries are more than #bboxes
        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        # collect coordinates for all bboxes
        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Get the object index_id in the range of 80 classes. 
        # This is quite confusing because COCO has 80 classes but has ids 1~90. 
        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            # clamp the box and drop those unreasonable ones
            boxes[:, 0::2].clamp_(min=0, max=w)  # xyxy    clamp x to 0~w
            boxes[:, 1::2].clamp_(min=0, max=h)  # xyxy    clamp y to 0~h
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]
            
            # construct target dict
            target['boxes'] = boxes
            target['labels'] = classes  # like [[0, 0][1, 56][2, 0][3, 0]...]
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(img, target)


            # enumerated indices for kept (0, 1, 2, 3, 4, ...)
            kept_box_indices = [label[0] for label in target['labels']]
            # object classes in 80 classes
            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            boxes_ct = np.stack(((target['boxes'][:,0] + target['boxes'][:,2])/2., (target['boxes'][:,1] + target['boxes'][:,3])/2.), axis = -1)
            # Make sure we use img.shape after the self._transforms to init the verb_hm
            verb_hm = np.zeros((1,) + img.shape[-2:], dtype=np.float32)
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    # Merge sub_obj pair with multiple verb labels into one verb labels
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    # verb category_id in the range from 1 to 117
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    # Set all verb labels to 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)
                
                # hoi_cate = self._valid_verb_ids.index(hoi['category_id'])
                hoi_cate = 0
                sub_ct = boxes_ct[kept_box_indices.index(hoi['subject_id'])]
                obj_ct = boxes_ct[kept_box_indices.index(hoi['object_id'])]
                rel_ct = np.array([(sub_ct[0] + obj_ct[0]) / 2,
                                (sub_ct[1] + obj_ct[1]) / 2], dtype=np.float32)
                radius = gaussian_radius((math.ceil(abs(sub_ct[0] - obj_ct[0])), math.ceil(abs(sub_ct[1] - obj_ct[1]))))
                radius = max(0, int(radius))
                rel_ct_int = rel_ct.astype(np.int32)
                self.draw_gaussian(verb_hm[hoi_cate], rel_ct_int, radius)
                
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['verb_hm'] = torch.from_numpy(verb_hm)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
                target['verb_hm'] = torch.from_numpy(verb_hm)
        else:
            target['boxes'] = boxes # 
            target['labels'] = classes # 
            target['id'] = idx # img_idx
            # if idx == 0:
            #     print(target['boxes'])
            #     print(target['labels'])
            #     print(target['id'])
            

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
                # if len(self._valid_verb_ids.index(hoi['category_id'])) >=2:
                #     print(self._valid_verb_ids.index(hoi['category_id']))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

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


# Add color jitter to coco transforms
def make_hico_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
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
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'

    if args.few_shot_transfer == 100:
        if args.zero_shot_setting is None:
            if args.relation_label_noise == 0:
                train_anno_file = root / 'annotations' / 'trainval_hico.json'
            else:
                train_anno_file = root / 'annotations' / 'trainval_hico_{}relation_noise.json'.format(args.relation_label_noise)
        elif args.zero_shot_setting in ['UC-RF', 'UC-NF', 'UO']:
            assert args.relation_label_noise == 0
            train_anno_file = root / 'annotations' / 'trainval_hico_{}.json'.format(args.zero_shot_setting)
        else:
            assert False
    elif args.few_shot_transfer == 10:
        assert args.zero_shot_setting is None
        train_anno_file = root / 'annotations' / 'trainval_hico_10percent.json'
    elif args.few_shot_transfer == 1:
        assert args.zero_shot_setting is None
        train_anno_file = root / 'annotations' / 'trainval_hico_1percent.json'
    print('Training anno file: {}'.format(train_anno_file))
    
    
    PATHS = {
        'train': (root / 'images' / 'train2015', train_anno_file),
        'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file = PATHS[image_set]
    if args.DETRHOIhm:
        dataset = HICODetectionhm(image_set, img_folder, anno_file, transforms=make_hico_transforms(image_set),
                                num_queries=args.num_queries)
    else:
        dataset = HICODetection(image_set, img_folder, anno_file, transforms=make_hico_transforms(image_set),
                                num_queries=args.num_queries, args = args)
    
    ### Set zero-shot indices 
    # Need to ensure that we can not use both few_shot_transfer and zero_shot_setting
    if args.zero_shot_setting is not None:
        assert args.few_shot_transfer == 100
    unseen_idx = None
    if args.zero_shot_setting == 'UC-RF':
        # UC-RF, short for Unseen combinations (rare first)
        unseen_idx = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418, 70, 416,
                    389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596, 345, 189,
                    205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229, 158, 195,
                    238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188, 216, 597,
                    77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104, 55, 50,
                    198, 168, 391, 192, 595, 136, 581]
    elif args.zero_shot_setting == 'UC-NF':
        # UC-NF, short for  Unseen combinations (non-rare first)
        unseen_idx = [38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75, 212, 472, 61,
                    457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479, 230, 385, 73,
                    159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338, 29, 594, 346,
                    456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191, 266, 304, 6, 572,
                    529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329, 246, 173, 506,
                    383, 93, 516, 64]
    elif args.zero_shot_setting == 'UO':
        # UO, short for Unseen objects
        unseen_idx = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                    126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                    294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                    338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                    429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                    463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                    537, 558, 559, 560, 561, 595, 596, 597, 598, 599]
    if unseen_idx is not None:
        # unseen_idx = [u+1 for u in unseen_idx] # convert the indices to the range of 1 to 600
        hoi_list_new_path = root / 'annotations' / 'hoi_list_new.json'

    # Remove texts from the label sequence
    # if image_set == 'train':
    #     if args.zero_shot_setting in ['UO']:
    #         dataset.remove_text_unseen(args.zero_shot_setting) 


    if image_set == 'val':
        if args.zero_shot_setting == None:
            anno_file_100percent = root / 'annotations' / 'trainval_hico.json'
            dataset.set_rare_hois(anno_file_100percent)
            print('Setting rare hois for None zero-shot setting.')
        elif args.zero_shot_setting in ['UC-RF', 'UC-NF', 'UO']:
            dataset.set_seen_hois(unseen_idx, hoi_list_new_path, args.zero_shot_setting)
            print('Setting seen hois for {}.'.format(args.zero_shot_setting))

        # dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset



if __name__ == '__main__':
    print(load_hico_verb_txt())
    print(load_hico_object_txt())