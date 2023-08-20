# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
Mixed_dataset.
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
from datasets.coco import CocoRLIPDetection
from torch.utils.data import ConcatDataset as _ConcatDataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
# from datasets.rlipv2_helper.caption_helper import check_overlap

from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
# __all__ = ["DistributedSampler", ]
T_co = TypeVar('T_co', covariant=True)


class BatchIterativeDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, batch_size: int, iterative_paradigm, 
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        # By default, the first dataset is the VisualGenome.
        # We set it as the anchor dataset.
        self.anchor_dataset_size = len(dataset.datasets[0])
        self.number_of_datasets = len(self.dataset.datasets)
        self.batch_size = batch_size
        self.iterative_paradigm = [int(d) for d in iterative_paradigm.split(',')]
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.anchor_dataset_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self.anchor_dataset_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.anchor_dataset_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.anchor_dataset_size, generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(self.anchor_dataset_size))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        ### Add indices for the extra datasets
        extra_indices = []
        global_start_idx = self.anchor_dataset_size
        for data_idx, extra_data in enumerate(self.dataset.datasets):
            if data_idx == 0:
                continue
            else:
                repeative_num = sum([data_idx == d for d in self.iterative_paradigm])
                if self.shuffle:
                    data_indices = torch.randperm(len(extra_data), generator=g).tolist()  # type: ignore[arg-type]
                else:
                    data_indices = list(range(len(extra_data)))
                data_indices = [i+global_start_idx for i in data_indices]
                global_start_idx = global_start_idx + len(extra_data)
                extra_indices.append(data_indices[:self.anchor_dataset_size * repeative_num]) # This is to match the size of the anchor dataset
        indices = [indices, ] + extra_indices

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        # assert len(indices) == self.num_samples
        iterative_indices = []
        iterative_batch_num = math.ceil(self.num_samples / self.batch_size)
        ### The true batch_size = iterative_batch_num * len(self.iterative_paradigm)
        start_flag = [0 for _ in range(len(indices))]
        for batch_idx in range(iterative_batch_num):
            if (len(indices[0]) - start_flag[0]) >= self.num_replicas * self.batch_size:
                batch_sample = self.num_replicas * self.batch_size
            else:
                batch_sample = len(indices[0]) - start_flag[0]

            for data_idx in self.iterative_paradigm:
                one_indices = indices[data_idx][start_flag[data_idx]: (start_flag[data_idx]+batch_sample)]
                iterative_indices.append(one_indices[self.rank::self.num_replicas])
                start_flag[data_idx] += batch_sample
        # [[dataset0], [dataset1], [dataset2], [dataset2], [dataset2], ....]

        assert self.num_samples == sum([len(i) for i in iterative_indices[0::len(self.iterative_paradigm)]])
        
        # assert 求和的index == self.num_samples
        # return iter(indices)

        return iter(iterative_indices)

    def __len__(self) -> int:
        # return self.num_samples
        if self.drop_last:
            return self.num_samples // self.batch_size * len(self.iterative_paradigm)  # type: ignore[arg-type]
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size * len(self.iterative_paradigm) # type: 

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """
    def __init__(self, dataset_list):
        super().__init__(dataset_list)

        # Use Pre-extracted keep_name_freq file
        # self.relationship_freq = dataset_list[-1].relationship_freq
        # self.object_freq = dataset_list[-1].object_freq
        # self.relationship_names = dataset_list[-1].relationship_names
        # self.object_names = dataset_list[-1].object_names

        # Extract the keep_name_freq online
        keep_name_freq_list = []
        for dataset in dataset_list:
            annos = dataset.annotations
            keep_name_freq_list.append(generate_keep_names_freq(annos))
        fused_keep_names_freq = fuse_multi_keep_names_freq(keep_name_freq_list)
        self.relationship_freq = fused_keep_names_freq["relationship_freq"]
        self.object_freq = fused_keep_names_freq["object_freq"]
        self.relationship_names = fused_keep_names_freq["relationship_names"]
        self.object_names = fused_keep_names_freq["object_names"]


    # def get_idxs(self, idx):
    #     dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
    #     if dataset_idx == 0:
    #         sample_idx = idx
    #     else:
    #         sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
    #     return dataset_idx, sample_idx

    # def get_img_info(self, idx):
    #     dataset_idx, sample_idx = self.get_idxs(idx)


def generate_keep_names_freq(
        annos = None):
    '''
    This function generates a .json file containing the relationship names, object names and 
    their corresponding frequencies.

    :param: anno_path: file path for the generated pseudo-labels.
    :param: save_path: file path to be saved.
    ''' 
    obj_keep_dict = {}
    rel_keep_dict = {}
    hoi_triplets = 0
    triplets = 0
    for anno in annos:
        objects_anno = anno['objects']
        relationships_anno = anno['relationships']
        relationships_anno = add_local_object_id(relationships_anno, objects_anno)
        triplets += len(relationships_anno)

        for obj in objects_anno:
            if obj['names'] not in obj_keep_dict.keys():
                obj_keep_dict[obj['names']] = 1
            else:
                obj_keep_dict[obj['names']] += 1

        for rel in relationships_anno:
            if rel['predicate'] not in rel_keep_dict.keys():
                rel_keep_dict[rel['predicate']] = 1
            else:
                rel_keep_dict[rel['predicate']] += 1
            if objects_anno[rel['subject_id_local']]['names'] == 'person':
                hoi_triplets += 1

    rel_keep_dict = dict(sorted(rel_keep_dict.items(), key = lambda item:item[1], reverse = True))
    obj_keep_dict = dict(sorted(obj_keep_dict.items(), key = lambda item:item[1], reverse = True))
    print(f'There are {len(rel_keep_dict)} kinds of relationships and {len(obj_keep_dict)} kinds of objects.')
    print(f'There are {triplets} triplets, {hoi_triplets} of which are hoi triplets.')

    return {"relationship_names": list(rel_keep_dict.keys()),
            "object_names": list(obj_keep_dict.keys()),
            "relationship_freq": rel_keep_dict,
            "object_freq": obj_keep_dict}
    

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


def fuse_multi_keep_names_freq(
            annos_list):
    '''
    This function fuses multiple keep_names_freq.json files into one
    in order to perform pre-training on mixed datasets.
    '''
    # keys in annos:
    # ["relationship_names", "object_names", "relationship_freq", "object_freq"]
    
    ### We should treat "relationship_freq" and "object_freq" as anchors
    ### since we need ordered "relationship_names" and "object_names".
    base_annos = annos_list[0]
    for annos in annos_list[1:]:
        for rel in annos["relationship_freq"].keys():
            if rel not in base_annos["relationship_freq"].keys():
                base_annos["relationship_freq"][rel] = annos["relationship_freq"][rel]
                # print(rel)
            else:
                base_annos["relationship_freq"][rel] += annos["relationship_freq"][rel]
        
        for obj in annos["object_freq"].keys():
            if obj not in base_annos["object_freq"].keys():
                base_annos["object_freq"][obj] = annos["object_freq"][obj]
            else:
                base_annos["object_freq"][obj] += annos["object_freq"][obj]
    
    ### We should rank according to frequencies to enable further usage.
    rel_keep_dict = dict(sorted(base_annos["relationship_freq"].items(), key = lambda item:item[1], reverse = True))
    obj_keep_dict = dict(sorted(base_annos["object_freq"].items(), key = lambda item:item[1], reverse = True))
    print('')
    print(f'There are {len(rel_keep_dict)} kinds of relationships and {len(obj_keep_dict)} kinds of objects.')
    print(f'There are {sum(rel_keep_dict.values())} relationships in the merged json file.')

    return {"relationship_names": list(rel_keep_dict.keys()),
            "object_names": list(obj_keep_dict.keys()),
            "relationship_freq": rel_keep_dict,
            "object_freq": obj_keep_dict}


class MixedRelDetection(torch.utils.data.Dataset):
    def __init__(self, img_set,
                       anno_file, 
                       transforms, 
                       num_queries, 
                       keep_names_freq_file = None,
                       dataset = [],
                       relation_threshold = 0.,
                       pair_overlap = False,
                       use_all_text_labels = False,
                       vg_folder = None, 
                       coco2017_folder = None, 
                       o365_folder = None,
                       hico_folder = None):
        '''
        The dataset list contains choices from ['vg', 'coco2017', 'o365']. 
        '''
        # for d in dataset:
        #     assert eval(f'{d}_folder') is not None
        #     eval(f'self.{d}_folder') = eval(f'{d}_folder')
        self.vg_folder = None
        self.coco2017_folder = None
        self.o365_folder = None
        if 'vg' in dataset:
            self.vg_folder = vg_folder
        if 'coco2017' in dataset:
            self.coco2017_folder = coco2017_folder
        if 'o365' in dataset:
            self.o365_folder = o365_folder
            with open(o365_folder / 'image_id_to_filepath.json', "r") as f:
                self.image_id_to_filepath_o365 = json.load(f)
        if 'hico' in dataset:
            self.hico_folder = hico_folder

        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self.img_set = img_set
        self._transforms = transforms
        self.num_queries = num_queries
        self.num_pairs = num_queries//2
        self.relation_threshold = relation_threshold
        self.pair_overlap = pair_overlap
        self.use_all_text_labels = use_all_text_labels
        print(f'We filter every image to have no more than {self.num_pairs} triplets.')

        ### Perform thresholding
        if self.relation_threshold > 0.:
            print(f"We have {sum([len(anno['relationships']) for anno in self.annotations])} relationship triplets before confidence thresholding.")
            assert self.relation_threshold > 0. and self.relation_threshold <= 1.
            new_annotations = []
            total_rel = 0
            for anno in self.annotations:
                new_annotations.append(anno)
                new_rels = []
                for rel in anno["relationships"]:
                    if "confidence" in rel.keys():  # It means that these are generated by the verb tagger.
                        if rel["confidence"] >= self.relation_threshold:
                            new_rels.append(rel)
                    else:  # It means that these are possibly from the VG dataset, so we keep all of them.
                        new_rels.append(rel)
                new_annotations[-1]["relationships"] = new_rels
                total_rel += len(new_rels)
            self.annotations = new_annotations
            print(f"We have {total_rel} relationship triplets after confidence thresholding.")
        
        ### Perform pair_overlap filtering
        if 'vg' not in dataset and self.pair_overlap:
            print(f"We have {sum([len(anno['relationships']) for anno in self.annotations])} relationship triplets before pair_overlap filtering.")
            new_annotations = []
            total_rel = 0
            for anno in self.annotations:
                new_annotations.append(anno)
                objs = anno['objects']
                objs_dict = {obj['object_id']:obj for obj in objs}
                new_rels = []
                for rel in anno["relationships"]:
                    if "overlap" in rel.keys():
                        if rel["overlap"] is True:
                            new_rels.append(rel)
                    else:  
                        # # It means that these are possibly from the VG dataset, so we keep all of them.
                        # new_rels.append(rel)
                        sub_obj = objs_dict[rel['subject_id']]
                        obj_obj = objs_dict[rel['object_id']]
                        if check_overlap([sub_obj["x"], sub_obj["y"], sub_obj["w"], sub_obj["h"]], 
                                         [obj_obj["x"], obj_obj["y"], obj_obj["w"], obj_obj["h"]]):
                            new_rels.append(rel)

                new_annotations[-1]["relationships"] = new_rels
                total_rel += len(new_rels)
            self.annotations = new_annotations
            print(f"We have {total_rel} relationship triplets pair_overlap filtering.")

        if keep_names_freq_file is not None:
            with open(keep_names_freq_file, "r") as f:
                vg_keep_names = json.load(f)
                self.relationship_freq = vg_keep_names["relationship_freq"]
                self.object_freq = vg_keep_names["object_freq"]
                self.relationship_names = vg_keep_names["relationship_names"]
                self.object_names = vg_keep_names["object_names"]
            print('We use the pre-calculated keep_name_freq file.')
        else:
            print('We will generate the keep_name_freq file online.')
        
        self.ids = list(range(len(self.annotations)))
        nonzero_rel_ids = []
        for i in self.ids:
            if len(self.annotations[i]["relationships"]):
                nonzero_rel_ids.append(i)
        print(f'{len(self.ids) - len(nonzero_rel_ids)}/{len(self.ids)} images do not have any relations.\n')
        self.ids = nonzero_rel_ids
    
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
        # if self.vg_folder is not None:
        #     print(self.vg_folder)
        # if self.coco2017_folder is not None:
        #     print(self.coco2017_folder)
        img_anno = self.annotations[self.ids[idx]]
        objects_anno = img_anno['objects'] # data type: list
        relationships_anno = img_anno['relationships'] # data type: list
        
        # if "dataset" in img_anno.keys():
        #     print(img_anno['image_id'], type(img_anno['image_id']), img_anno["dataset"])
        # else:
        #     print(img_anno['image_id'], type(img_anno['image_id']))

        if "dataset" in img_anno.keys():
            if img_anno["dataset"] == "coco2017":
                img_file_name = str(img_anno['image_id']).zfill(12) + '.jpg'
                if 'data_split' not in img_anno.keys():
                    img = Image.open(self.coco2017_folder / 'train2017' / img_file_name).convert('RGB')
                else:
                    img = Image.open(self.coco2017_folder / img_anno['data_split'] / img_file_name).convert('RGB')
            elif img_anno["dataset"] == "o365":
                img_file_name = self.image_id_to_filepath_o365[str(img_anno['image_id'])]
                img = Image.open(self.o365_folder / img_file_name).convert('RGB')
            elif img_anno["dataset"] == "hico":
                img_file_name = img_anno['image_id']
                img = Image.open(self.hico_folder / img_file_name).convert('RGB')
        else:
            img_file_name = str(img_anno['image_id']) + '.jpg'
            # print(img_file_name)
            img = Image.open(self.vg_folder / img_file_name).convert('RGB')
        
        w, h = img.size
        
        # make sure that #queries are more than #bboxes
        if self.img_set == 'pretrain' and len(relationships_anno) > self.num_pairs:
            relationships_anno = relationships_anno[:self.num_pairs]
        # collect coordinates and names for all bboxes
        boxes = []
        for cur_box in objects_anno:
            cur_box_cor = [cur_box['x'], cur_box['y'], cur_box['x'] + cur_box['w'], cur_box['y'] + cur_box['h']]
            boxes.append(cur_box_cor)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

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
    # root = Path(args.vg_path)
    # assert root.exists(), f'provided VG path {root} does not exist'

    # /Path/To/data/VG
    # if args.dataset_file == 'vg':
    #     dataset_list = ['vg']
    #     # PATHS = {'pretrain': (root / 'images' , root / 'annotations' / 'scene_graphs_preprocessv1.json')}
    #     # PATHS = {'pretrain': (root / 'images' , root / 'annotations' / 'scene_graphs_preprocessv2.json')}
    #     vg_img_folder = Path(args.vg_path) / 'images'
    if args.dataset_file == 'vg_coco2017_o365':
        ### Mixed dataset version 2 (Using ConcatDataset)
        assert image_set == 'pretrain'
        dataset_list = ['vg', 'coco2017', "o365"]
        vg_img_folder = Path(args.vg_path) / 'images'
        assert vg_img_folder.exists(), f'provided VG path {vg_img_folder} does not exist'
        coco2017_img_folder = Path(args.coco_path) # / 'train2017'
        assert coco2017_img_folder.exists(), f'provided COCO2017 path {coco2017_img_folder} does not exist'
        o365_img_folder = Path(args.o365_path) # / 'train2017'
        assert o365_img_folder.exists(), f'provided Objects365 path {o365_img_folder} does not exist'


        print("Annotation file we use: ", args.mixed_anno_file)
        print("Keep_names_freq file we use: ", args.keep_names_freq_file)

        O365dataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.o365_rel_anno_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['o365'],
                          vg_folder = None,
                          coco2017_folder = None,
                          o365_folder = o365_img_folder)
        
        Vgdataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.vg_rel_anno_file,
                        #   keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['vg'],
                          vg_folder = vg_img_folder,
                          coco2017_folder = None,
                          o365_folder = None)
        
        Cocodataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.coco_rel_anno_file,
                        #   keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['coco2017'],
                          vg_folder = None,
                          coco2017_folder = coco2017_img_folder,
                          o365_folder = None)
        
        dataset = ConcatDataset([O365dataset, Vgdataset, Cocodataset])

    elif args.dataset_file == 'vg_coco2017':
        ### Mixed dataset version 2 (Using ConcatDataset)
        assert image_set == 'pretrain'
        dataset_list = ['vg', 'coco2017']
        vg_img_folder = Path(args.vg_path) / 'images'
        assert vg_img_folder.exists(), f'provided VG path {vg_img_folder} does not exist'
        coco2017_img_folder = Path(args.coco_path) # / 'train2017'
        assert coco2017_img_folder.exists(), f'provided COCO2017 path {coco2017_img_folder} does not exist'

        print("Annotation file we use: ", args.mixed_anno_file)
        print("Keep_names_freq file we use: ", args.keep_names_freq_file)
        
        Vgdataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.vg_rel_anno_file,
                          keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['vg'],
                          vg_folder = vg_img_folder,
                          coco2017_folder = None,
                          o365_folder = None)
        
        Cocodataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.coco_rel_anno_file,
                          keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['coco2017'],
                          vg_folder = None,
                          coco2017_folder = coco2017_img_folder,
                          o365_folder = None)
        
        dataset = ConcatDataset([Vgdataset, Cocodataset])


    elif args.dataset_file == 'vg_coco2017_o365_hico':
        assert image_set == 'pretrain'
        vg_img_folder = Path(args.vg_path) / 'images'
        assert vg_img_folder.exists(), f'provided VG path {vg_img_folder} does not exist'
        coco2017_img_folder = Path(args.coco_path) # / 'train2017'
        assert coco2017_img_folder.exists(), f'provided COCO2017 path {coco2017_img_folder} does not exist'
        o365_img_folder = Path(args.o365_path) # / 'train2017'
        assert o365_img_folder.exists(), f'provided Objects365 path {o365_img_folder} does not exist'
        hico_img_folder = Path(args.hico_path) / 'images' / 'train2015'
        assert hico_img_folder.exists(), f'provided VG path {hico_img_folder} does not exist'

        O365dataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.o365_rel_anno_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['o365'],
                          vg_folder = None,
                          coco2017_folder = None,
                          o365_folder = o365_img_folder)

        Vgdataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.vg_rel_anno_file,
                        #   keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['vg'],
                          vg_folder = vg_img_folder,
                          coco2017_folder = None,
                          o365_folder = None)
        
        Cocodataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.coco_rel_anno_file,
                        #   keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['coco2017'],
                          vg_folder = None,
                          coco2017_folder = coco2017_img_folder,
                          o365_folder = None)

        Hicodataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.hico_rel_anno_file,
                        #   keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['hico'],
                          vg_folder = None,
                          coco2017_folder = None,
                          o365_folder = None,
                          hico_folder = hico_img_folder)

        dataset = ConcatDataset([Vgdataset, Cocodataset, O365dataset, Hicodataset])

    
    elif args.dataset_file == 'vg_coco2017_hico':
        assert image_set == 'pretrain'
        vg_img_folder = Path(args.vg_path) / 'images'
        assert vg_img_folder.exists(), f'provided VG path {vg_img_folder} does not exist'
        coco2017_img_folder = Path(args.coco_path) # / 'train2017'
        assert coco2017_img_folder.exists(), f'provided COCO2017 path {coco2017_img_folder} does not exist'
        hico_img_folder = Path(args.hico_path) / 'images' / 'train2015'
        assert hico_img_folder.exists(), f'provided VG path {hico_img_folder} does not exist'

        Vgdataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.vg_rel_anno_file,
                        #   keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['vg'],
                          vg_folder = vg_img_folder,
                          coco2017_folder = None,
                          o365_folder = None)
        
        Cocodataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.coco_rel_anno_file,
                        #   keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['coco2017'],
                          vg_folder = None,
                          coco2017_folder = coco2017_img_folder,
                          o365_folder = None)

        Hicodataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.hico_rel_anno_file,
                        #   keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['hico'],
                          vg_folder = None,
                          coco2017_folder = None,
                          o365_folder = None,
                          hico_folder = hico_img_folder)

        dataset = ConcatDataset([Vgdataset, Cocodataset, Hicodataset])
    
    elif args.dataset_file == 'vg_hico':

        assert image_set == 'pretrain'
        vg_img_folder = Path(args.vg_path) / 'images'
        assert vg_img_folder.exists(), f'provided VG path {vg_img_folder} does not exist'
        hico_img_folder = Path(args.hico_path) / 'images' / 'train2015'
        assert hico_img_folder.exists(), f'provided VG path {hico_img_folder} does not exist'

        Vgdataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.vg_rel_anno_file,
                        #   keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['vg'],
                          vg_folder = vg_img_folder,
                          coco2017_folder = None,
                          o365_folder = None)

        Hicodataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.hico_rel_anno_file,
                        #   keep_names_freq_file = args.vg_keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = ['hico'],
                          vg_folder = None,
                          coco2017_folder = None,
                          o365_folder = None,
                          hico_folder = hico_img_folder)

        dataset = ConcatDataset([Vgdataset, Hicodataset])

    elif args.dataset_file == 'coco2017':
        assert args.use_all_text_labels
        assert image_set == 'pretrain'
        dataset_list = ['coco2017']
        coco2017_img_folder = Path(args.coco_path) # / 'train2017'
        assert coco2017_img_folder.exists(), f'provided COCO2017 path {coco2017_img_folder} does not exist'

        print("Annotation file we use: ", args.mixed_anno_file)
        print("Keep_names_freq file we use: ", args.keep_names_freq_file)
        dataset = MixedRelDetection(
                          img_set = image_set,
                          anno_file = args.mixed_anno_file,
                          keep_names_freq_file = args.keep_names_freq_file,
                          transforms = make_vg_transforms(image_set), 
                          num_queries = args.num_queries, 
                          relation_threshold = args.relation_threshold,
                          pair_overlap = args.pair_overlap,
                          dataset = dataset_list,
                          use_all_text_labels = args.use_all_text_labels,
                          vg_folder = None,
                          coco2017_folder = coco2017_img_folder,
                          o365_folder = None)

        
    return dataset



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
    None
