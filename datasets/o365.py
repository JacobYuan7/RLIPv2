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
from PIL import Image

import datasets.transforms as T
import json
from datasets.coco import CocoDetection
import os


class O365Detection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, image_id_to_filepath):
        super(O365Detection, self).__init__(img_folder, ann_file, transforms, return_masks)
        with open(image_id_to_filepath, "r") as f:
            self.image_id_to_filepath = json.load(f)
    
    def _load_image(self, id: int) -> Image.Image:
        # print(self.coco.loadImgs(id)[0].keys())
        # dict_keys(['height', 'id', 'license', 'width', 'file_name', 'url'])
        # print(self.coco.loadImgs(id))
        path = self.image_id_to_filepath[str(self.coco.loadImgs(id)[0]["id"])]
        return Image.open(os.path.join(self.root, path)).convert("RGB")



class O365RelDetection(O365Detection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, image_id_to_filepath, vg_rel_texts_for_o365_images):
        super(O365RelDetection, self).__init__(img_folder, ann_file, transforms, return_masks, image_id_to_filepath)
        with open(image_id_to_filepath, "r") as f:
            self.image_id_to_filepath = json.load(f)
        
        # Read relation annotations of COCO images
        print(f"Candidate file: {vg_rel_texts_for_o365_images}.")
        with open(vg_rel_texts_for_o365_images, "r") as f:
            self.o365_img_rels = json.load(f)
        # Filter out imgs without any annotation of relation text.
        new_ids = []
        for one_id in self.ids:
            if str(one_id) in self.o365_img_rels.keys():
                new_ids.append(one_id)
        self.ids = new_ids
    
    def _load_image(self, id: int) -> Image.Image:
        # print(self.coco.loadImgs(id)[0].keys())
        # dict_keys(['height', 'id', 'license', 'width', 'file_name', 'url'])
        # print(self.coco.loadImgs(id))
        path = self.image_id_to_filepath[str(self.coco.loadImgs(id)[0]["id"])]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    
    def __getitem__(self, idx):
        """
        Output:
            when it is in training, we would output relation annotations like vg and hico;
            when it is in inference, we would only output object annotations to obtain pesudo relation labels.
        """
        img, target = super(O365RelDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        # target = {'image_id': image_id, 'annotations': target}
        # img, target = self.prepare(img, target)
        # if self._transforms is not None:
        #     img, target = self._transforms(img, target)

        rel_cands = self.o365_img_rels[str(image_id)]
        target['relation_candidates'] = rel_cands

        return img, target
    
def make_coco_transforms(image_set):
    '''
    Note that Normalize() will convert 'boxes' in the target from the format of 
    xyxy to cxcywh using the function box_xyxy_to_cxcywh()!!!
    '''

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
    elif image_set == 'val' or image_set == 'tagger' or image_set == 'tagger_val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    # elif image_set == 'tagger_inference':
    #     return T.Compose([
    #         normalize,
    #     ])
    else:
        assert False

    raise ValueError(f'unknown {image_set}')


    
def build(image_set, args):
    root = Path(args.o365_path)
    assert root.exists(), f'provided Objects365 path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root, root / "train" / 'zhiyuan_objv2_train.json'),
        "val": (root, root / "val" / 'zhiyuan_objv2_val.json'),
    }
    # We do not need paths like root / "train" or root / "val"

    img_folder, ann_file = PATHS[image_set]
    dataset = O365Detection(img_folder,
                            ann_file,
                            transforms=make_coco_transforms(image_set),
                            return_masks=args.masks,
                            image_id_to_filepath= root / 'image_id_to_filepath.json')
    return dataset

def build_o365rel(image_set, args):
    root = Path(args.o365_path)
    assert root.exists(), f'provided Objects365 path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "tagger": (root, root / "train" / 'zhiyuan_objv2_train.json'),
        "tagger_val": (root, root / "val" / 'zhiyuan_objv2_val.json'),
    }
    # We do not need paths like root / "train" or root / "val"

    img_folder, ann_file = PATHS[image_set]
    dataset = O365RelDetection(img_folder,
                               ann_file,
                               transforms=make_coco_transforms(image_set),
                               return_masks=args.masks,
                               image_id_to_filepath= root / 'image_id_to_filepath.json',
                               vg_rel_texts_for_o365_images=args.vg_rel_texts_for_o365_images)
    return dataset
