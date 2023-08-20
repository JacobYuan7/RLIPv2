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

import json
import sys
sys.path.append("..")
import transforms as T

def make_coco_transforms(image_set):

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

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

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

        return image, target

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

def load_hico_object_txt(file_path = '/Path/To/jacob/RLIP/datasets/hico_object_names.txt'):
    '''
    Output like 
        ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',]
    '''
    with open(file_path, 'r') as f:
        object_names = json.load(f)
    object_dict = {j:i for i,j in object_names.items()}
    return object_dict


def sel_vg_candidate_given_objects(obj_source = 'hico'):
    '''
    This function aims to select possible relation texts for a given object list from the VG dataset.
    '''
    if obj_source == 'hico':
        object_dict = load_hico_object_txt()
        object_names = list(object_dict.values())
        possible_pairs = list(permutations(object_names, 2))
        possible_pairs += [[o, o] for o in object_names]
        print(len(possible_pairs))
        print(object_names)

        paraphrase_file = "/Path/To/jacob/RLIP/datasets/priors/hico_obj_paraphrase.json"
        with open(paraphrase_file, 'r') as f:
            obj_paraphrase = json.load(f)
        print(obj_paraphrase)
    
    anno_file = '/Path/To/data/VG/annotations/scene_graphs_preprocessv1.json'
    # anno_file = '/Path/To/data/VG/annotations/scene_graphs_preprocess_greater5.json'
    # anno_file = '/Path/To/data/VG/annotations/scene_graphs_preprocess_greater20.json'
    # anno_file = '/Path/To/data/VG/annotations/scene_graphs_preprocess_greater100.json'
    with open(anno_file, 'r') as f:
        all_annotations = json.load(f)

    sub_obj_rel_cand = {o:{j:[] for j in object_names} for o in object_names}
    for idx, anno in enumerate(all_annotations):
        if idx%5000 == 0:
            print(f'Finishing processing {idx}/{len(all_annotations)}')

        obj_dict = {}
        for obj in anno['objects']:
            obj_dict[obj["object_id"]] = obj
        
        for rel in anno['relationships']:
            sub_obj_name = [obj_dict[rel["subject_id"]]["names"], 
                            obj_dict[rel["object_id"]]["names"]]
            for i in object_names:
                # if i in sub_obj_name[0]:
                if match_anchor_obj_with_free_form_texts(anchor_obj = i, free_form_texts = sub_obj_name[0], obj_paraphrase = obj_paraphrase):
                    for j in object_names:
                        # if j in sub_obj_name[1]:
                        if match_anchor_obj_with_free_form_texts(anchor_obj = j, free_form_texts = sub_obj_name[1], obj_paraphrase = obj_paraphrase):
                            if rel['predicate'] not in sub_obj_rel_cand[i][j]:
                                sub_obj_rel_cand[i][j].append(rel['predicate'])
    
    # print(sub_obj_rel_cand['toothbrush'])
    # sub_obj_rel_cand_list
    # for i in sub_obj_rel_cand.keys():
    #     for j in sub_obj_rel_cand[i].keys():

    ### Save to a file.
    # with open("/Path/To/data/coco2017/annotations/vg_rel_texts_for_hico_objects_greater100.json", "w") as f:
    with open("/Path/To/data/coco2017/annotations/vg_rel_texts_for_hico_objects_greater0_v3.json", "w") as f:
    # with open("/Path/To/data/coco2017/annotations/vg_rel_texts_for_hico_objects_greater5_v3.json", "w") as f:
        json.dump(sub_obj_rel_cand, f)

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


def match_anchor_obj_with_free_form_texts(
    anchor_obj,
    free_form_texts,
    obj_paraphrase,
):
    anchor_list = obj_paraphrase[anchor_obj]
    flag = False

    for anchor in anchor_list:
        if anchor in free_form_texts:
            flag = True
    return flag
    


def relation_candidate_selection_from_OD():
    '''
    This function selects relation texts for images from OD datasets.
    If one image has more than 100 pairs,
        we would seperate them into several groups with each group storing no more than 100 pairs.
    Format of the output:
        {image_id:[[[100 pairs of indices], [all rel texts for 100 pairs]],
                   [[leftover pairs of indices], [all rel texts for leftover pairs]]]}
    '''
    Coco = CocoDetection(img_folder = '/Path/To/data/coco2017/train2017',
                         ann_file = '/Path/To/data/coco2017/annotations/instances_train2017.json',
                         transforms=make_coco_transforms('val'),
                         return_masks=False)
    object_dict = load_hico_object_txt()
    num_queries = 200
    num_pairs = num_queries//2

    vg_rel_texts_for_hico_objects_path = '/Path/To/data/coco2017/annotations/vg_rel_texts_for_hico_objects_greater0_v3.json'
    with open(vg_rel_texts_for_hico_objects_path, 'r') as f:
        vg_rel_texts_for_hico_objects = json.load(f)

    rel_cand = {}
    start_t = time.time()
    for idx, coco_data in enumerate(Coco):
        if idx%5000 == 0:
            print(f'Finishing processing {idx}/{len(Coco)}')

        coco_img, coco_target = coco_data
        # coco_target: dict_keys(['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size'])
        num_obj = coco_target['boxes'].shape[0]
        labels = [int(l) for l in coco_target['labels']]
        text_labels = []
        for l in labels:
            assert l in object_dict.keys()
            text_labels.append(object_dict[l])

        possible_pairs = list(permutations(range(0, num_obj), 2))
        possible_pairs_text = [[text_labels[pair[0]], text_labels[pair[1]]] for pair in possible_pairs]
        num_possible_pairs = len(possible_pairs)

        possible_rel_text = []
        for pair in possible_pairs_text:
            possible_rel_text.append(vg_rel_texts_for_hico_objects[pair[0]][pair[1]])
        
        ### We filter out pairs with no relations annotated in VG. 
        ### (If a subject and an object do not have any possible relations in VG, then we delete this pair.) 
        zero_flag = [len(p) for p in possible_rel_text]
        new_possible_pairs = []
        new_possible_rel_text = []
        for idx, f in enumerate(zero_flag):
            if f > 0:
                new_possible_pairs.append(possible_pairs[idx])
                new_possible_rel_text.append(possible_rel_text[idx])
        # print(len(possible_pairs)-len(new_possible_pairs))
        # print(len(possible_rel_text)-len(new_possible_rel_text))
        possible_pairs = new_possible_pairs
        possible_rel_text = new_possible_rel_text
        assert len(possible_pairs) == len(possible_rel_text)

        ### Merge rel texts for groups
        num_groups = num_possible_pairs//num_pairs + 1
        rel_cand[int(coco_target['image_id'])] = []
        for i in range(0, num_groups):
            if i == num_groups-1:
                i_pairs = possible_pairs[i*num_pairs:]
                i_pair_texts = possible_rel_text[i*num_pairs:]
            else:
                i_pairs = possible_pairs[i*num_pairs:(i+1)*num_pairs]
                i_pair_texts = possible_rel_text[i*num_pairs:(i+1)*num_pairs]

            # Merge pair texts
            i_rel_texts = []
            for t in i_pair_texts:
                for k in t:
                    if k not in i_rel_texts:
                        i_rel_texts.append(k)
            
            rel_cand[int(coco_target['image_id'])].append([i_pairs, i_rel_texts])
        
        # if idx%5 == 0 and idx>0:
        #     break
    
    print(f'Processing time: {int(time.time()-start_t)}s.')
    with open("/Path/To/data/coco2017/annotations/vg_rel_texts_for_coco_images_greater0_v3.json", "w") as f:
        json.dump(rel_cand, f)
    
    # vg_rel_texts_for_coco_images_greater5_v3.json
    # Processing time: 11589s


def cal_max_cand_rels(top_num = 100):
    '''
    This file outputs the top N number of relation texts recorded in file like 'vg_rel_texts_for_coco_images_greater5.json'/'vg_rel_texts_for_coco_images_greater100_v2.json'.
    '''
    file = '/Path/To/data/coco2017/annotations/vg_rel_texts_for_coco_images_greater0_v3.json'
    with open(file, "r") as f:
        rels = json.load(f)

    num_rel_texts_list = []
    num_groups_list = []
    for img_id, img_rels in rels.items():
        num_groups_list.append(len(img_rels))
        for group_img_rels in img_rels:
            num_rel_texts_list.append(len(group_img_rels[1]))
        # break
    rel_list = sorted(num_rel_texts_list, reverse = True)
    groups_list = sorted(num_groups_list, reverse = True)
    print(rel_list[:100])
    print(groups_list[:100])

    ### How many different kinds of relationships are there in the candidate set?
    unique_rels = {}
    for img_id, img_rels in rels.items():
        for group_img_rels in img_rels:
            for rel in group_img_rels[1]:
                if rel in unique_rels.keys():
                    unique_rels[rel] += 1
                else:
                    unique_rels[rel] = 1
    print(unique_rels)
    print(f"There are {sum(unique_rels.values())} relationships in {len(unique_rels)} kinds.")


if __name__=="__main__":
    # sel_vg_candidate_given_objects()

    # relation_candidate_selection_from_OD()

    cal_max_cand_rels()


