# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import json
from pathlib import Path
from random import sample as sample
import random

def select_percentage(percentage = 100, 
                      save_json = False):
    assert percentage in [1, 10, 100]
    hoi_path = Path('/Path/To/data/hico_20160224_det')

    image_path = hoi_path / 'images' / 'train2015'
    anno_file = hoi_path / 'annotations' / 'trainval_hico.json'
    with open(anno_file, 'r') as f:
        all_annotations = json.load(f)

    valid_verb_ids = list(range(1, 118))
    len_anno = len(all_annotations)
    print('There are {:d} images in HICO-DET trainval set.'.format(len_anno))
    
    ### First we sample for every class, then we perform random sampling
    all_objs_verbs = False
    verb_image_dict = {v:[] for v in valid_verb_ids}
    ## Aggregate stat for every verb
    for idx, anno in enumerate(all_annotations):
        idx_rel = anno["hoi_annotation"]
        for r in idx_rel:
            verb_image_dict[r['category_id']].append(idx)
    # print(sum([len(v) for v in verb_image_dict.values()]))
    ## Sample for every verb
    percentage_img_list = []
    for v_idx, v_img_list in verb_image_dict.items():
        percentage_img_list += sample(v_img_list, 1)
    
    left_image_num = int(len_anno * percentage / 100.) - len(percentage_img_list)
    left_list = [i for i in range(0, len_anno) if i not in percentage_img_list]
    all_objs_verbs = False
    while (not all_objs_verbs):
        verb_list = []
        obj_list = []
        sample_list = sample(left_list, left_image_num)
        full_list = sample_list + percentage_img_list

        for anno_idx in full_list:
            anno = all_annotations[anno_idx]
            idx_obj = anno["annotations"]
            for o in idx_obj:
                if o['category_id'] not in obj_list:
                    obj_list.append(o['category_id'])
                
            idx_rel = anno["hoi_annotation"]
            for r in idx_rel:
                if r['category_id'] not in verb_list:
                    verb_list.append(r['category_id'])
        
        if len(verb_list) == 117 and len(obj_list) == 80:
            all_objs_verbs = True
        
        print(len(verb_list), len(obj_list))
    
    ## Save annos
    if save_json:
        save_anno_file = hoi_path / 'annotations' / 'trainval_hico_{:d}percent.json'.format(percentage)
        new_anno = [all_annotations[f] for f in full_list]
        with open(save_anno_file, 'w') as outfile:
            json.dump(new_anno, outfile)
        print('Successfully save to {}.'.format(save_anno_file))


    ### Direct random sampling
    # all_objs_verbs = False
    # while (not all_objs_verbs):
    #     verb_list = []
    #     obj_list = []
    #     sample_list = sample(list(range(0, len_anno)), int(len_anno * percentage / 100.))
    #     for anno_idx in sample_list:
    #         anno = all_annotations[anno_idx]
    #         idx_obj = anno["annotations"]
    #         for o in idx_obj:
    #             if o['category_id'] not in obj_list:
    #                 obj_list.append(o['category_id'])
                
    #         idx_rel = anno["hoi_annotation"]
    #         for r in idx_rel:
    #             if r['category_id'] not in verb_list:
    #                 verb_list.append(r['category_id'])
        
    #     if len(verb_list) == 117 and len(obj_list) == 80:
    #         all_objs_verbs = True
        
    #     print(len(verb_list), len(obj_list))


def select_zero_shot_annotations(zero_shot_setting, save_json = False):
    assert zero_shot_setting in ['UC-NF', 'UC-RF', 'UO']

    unseen_idx = None
    if zero_shot_setting == 'UC-RF':
        # UC-RF, short for Unseen combinations (rare first)
        unseen_idx = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418, 70, 416,
                    389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596, 345, 189,
                    205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229, 158, 195,
                    238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188, 216, 597,
                    77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104, 55, 50,
                    198, 168, 391, 192, 595, 136, 581]
    elif zero_shot_setting == 'UC-NF':
        # UC-NF, short for  Unseen combinations (non-rare first)
        unseen_idx = [38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75, 212, 472, 61,
                    457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479, 230, 385, 73,
                    159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338, 29, 594, 346,
                    456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191, 266, 304, 6, 572,
                    529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329, 246, 173, 506,
                    383, 93, 516, 64]
    elif zero_shot_setting == 'UO':
        # UO, short for Unseen objects
        unseen_idx = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                    126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                    294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                    338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                    429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                    463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                    537, 558, 559, 560, 561, 595, 596, 597, 598, 599]
    # Convert the indices to the range of 1 to 600 
    # because the range in "trainval_hico.json" is [1, 600]
    unseen_idx = [u+1 for u in unseen_idx]

    hoi_path = Path('/Path/To/data/hico_20160224_det')
    image_path = hoi_path / 'images' / 'train2015'
    anno_file = hoi_path / 'annotations' / 'trainval_hico.json'
    with open(anno_file, 'r') as f:
        all_annotations = json.load(f)
    
    hoi_sum = [0, 0]
    for anno in all_annotations:
        anno_hoi = anno["hoi_annotation"]
        hoi_sum[0] = hoi_sum[0] + len(anno_hoi)
        
        new_anno_hoi = []
        for hoi in anno_hoi:
            if hoi["hoi_category_id"] not in unseen_idx:
                new_anno_hoi.append(hoi)
        hoi_sum[1] += len(new_anno_hoi)
        anno["hoi_annotation"] = new_anno_hoi
    
    print("Hoi triplets before and after unseen filtering: ", hoi_sum)
    print("Check: ", sum([len(anno["hoi_annotation"])for anno in all_annotations]))

    if save_json:
        save_anno_file = hoi_path / 'annotations' / 'trainval_hico_{}.json'.format(zero_shot_setting)
        with open(save_anno_file, 'w') as outfile:
            json.dump(all_annotations, outfile)
        print('Successfully save to {}.'.format(save_anno_file))


def add_asymmetric_rel_noise(noise_ratio, save_json):
    hoi_path = Path('/Path/To/data/hico_20160224_det')
    image_path = hoi_path / 'images' / 'train2015'
    anno_file = hoi_path / 'annotations' / 'trainval_hico.json'
    with open(anno_file, 'r') as f:
        all_annotations = json.load(f)
    
    rel_num = 0
    flip_num = 0
    valid_verb_ids = list(range(1, 118))
    for idx, anno in enumerate(all_annotations):
        idx_rel = anno["hoi_annotation"]
        for one_rel in idx_rel:
            rel_num += 1
            if random.uniform(0, 1) < noise_ratio:
                # It means that we need to flip the relation 'category_id'
                flip_num += 1
                new_rel_cat = one_rel['category_id']
                while new_rel_cat == one_rel['category_id']:
                    new_rel_cat = random.choice(valid_verb_ids)
                print(one_rel['category_id'], new_rel_cat)
                one_rel['category_id'] = new_rel_cat
    print('rel_num:{:d}, flip_num:{:d}.'.format(rel_num, flip_num))
    
    if save_json:
        save_anno_file = hoi_path / 'annotations' / 'trainval_hico_{:d}relation_noise.json'.format(int(noise_ratio*100))
        with open(save_anno_file, 'w') as outfile:
            json.dump(all_annotations, outfile)
        print('Successfully save to {}.'.format(save_anno_file))


def transform_hico_to_vg_form():
    hico_path = '/Path/To/data/hico_20160224_det/annotations/trainval_hico.json'
    save_path = '/Path/To/data/hico_20160224_det/annotations/trainval_hico_vg_format.json'
    with open(hico_path, "r") as f:
        hico_anno = json.load(f)
    
    hico_verb_id_to_name = load_hico_verb_dict()
    hico_obj_id_to_name = load_hico_object_dict()
    print(hico_verb_id_to_name)
    print(hico_obj_id_to_name)

    global_relationship_id = 50000000
    global_object_id = 50000000

    hico_vg_form = []
    for anno in hico_anno:
        obj_anno = anno["annotations"]
        hoi_anno = anno["hoi_annotation"]
        objects_list = []
        for obj_idx, obj in enumerate(obj_anno):
            objects_list.append(
                {
                    "object_id": global_object_id,
                    "x": obj["bbox"][0],
                    "y": obj["bbox"][1],
                    "w": obj["bbox"][2] - obj["bbox"][0],
                    "h": obj["bbox"][3] - obj["bbox"][1],
                    "names": hico_obj_id_to_name[obj['category_id']],
                }
            )
            global_object_id += 1

        rels_list = []
        for one_hoi in hoi_anno:
            sub_obj = objects_list[one_hoi['subject_id']]
            obj_obj = objects_list[one_hoi['object_id']]
            rels_list.append({
                "relationship_id": global_relationship_id,
                "predicate": hico_verb_id_to_name[one_hoi['category_id']],
                "subject_id": sub_obj['object_id'],
                "object_id": obj_obj['object_id'],
            })
            global_relationship_id += 1
        
        one_img_anno = {}
        one_img_anno["image_id"] = anno['file_name']
        one_img_anno["dataset"] = 'hico'
        one_img_anno["data_split"] = 'train'
        one_img_anno["objects"] = objects_list
        one_img_anno["relationships"] = rels_list
        hico_vg_form.append(one_img_anno)
    
    if save_path:
        with open(save_path, 'w') as outfile:
            json.dump(hico_vg_form, outfile)
        print('Successfully save to {}.'.format(save_path))



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


def load_hico_verb_dict(file_path = '/Path/To/jacob/RLIP/datasets/hico_verb_names.txt'):
    '''
    Output like [['train'], ['boat'], ['traffic', 'light'], ['fire', 'hydrant']]
    '''
    verb_names = []
    verb_index = []
    for line in open(file_path,'r'):
        # verb_names.append(line.strip().split(' ')[-1])
        verb_names.append(' '.join(line.strip().split(' ')[-1].split('_')))
        verb_index.append(int(line.strip().split(' ')[0]))
    verb_dict = {id:name for id, name in zip(verb_index, verb_names)}
    return verb_dict

def load_hico_object_dict(file_path = '/Path/To/jacob/RLIP/datasets/hico_object_names.txt'):
    '''
    Output like {90: "toothbrush"}
    '''
    with open(file_path, 'r') as f:
        object_names_dict = json.load(f)
    object_dict = {id:name for name, id in object_names_dict.items()}
    return object_dict




if __name__ == '__main__':
    ### The following codes aim to generate annotations for few_shot_transfer on HICO-DET.
    # select_percentage(percentage = 1, save_json = False) # It takes about 10 mins to sample all objects.
    # select_percentage(percentage = 10, save_json = False) # It takes about 1 sec to sample all objects.

    ### The following codes aim to generate annotations for few_shot_transfer on HICO-DET.
    # select_zero_shot_annotations(zero_shot_setting = 'UO', save_json = True)
    # select_zero_shot_annotations(zero_shot_setting = 'UC-RF', save_json = True)
    # select_zero_shot_annotations(zero_shot_setting = 'UC-NF', save_json = True)

    ### The following codes aim to add relation label noise to trainval set on HICO-DET
    # add_asymmetric_rel_noise(noise_ratio = 0.1, save_json = True)
    # add_asymmetric_rel_noise(noise_ratio = 0.3, save_json = True)
    # add_asymmetric_rel_noise(noise_ratio = 0.5, save_json = True)
    
    # Check noise relations
    # hoi_noise_path = Path('/Path/To/data/hico_20160224_det') / 'annotations' / 'trainval_hico_30relation_noise.json'
    # hoi_path = Path('/Path/To/data/hico_20160224_det') / 'annotations' / 'trainval_hico.json'
    # with open(hoi_noise_path, 'r') as f:
    #     all_annotations_noise = json.load(f)
    # with open(hoi_path, 'r') as f:
    #     all_annotations = json.load(f)

    # rel_num = 0
    # flip_num = 0
    # for i, j in zip(all_annotations_noise, all_annotations):
    #     for m, n in zip(i["hoi_annotation"], j["hoi_annotation"]):
    #         rel_num += 1
    #         if m['category_id'] != n['category_id']:
    #             flip_num += 1
    # print('rel_num:{:d}, flip_num:{:d}.'.format(rel_num, flip_num))

    # Hoi triplets before and after unseen filtering:  [117871, 100590]
    # Check:  100590
    # Successfully save to /Path/To/data/hico_20160224_det/annotations/trainval_hico_UO.json.
    # Hoi triplets before and after unseen filtering:  [117871, 117526]
    # Check:  117526
    # Successfully save to /Path/To/data/hico_20160224_det/annotations/trainval_hico_UC-RF.json.
    # Hoi triplets before and after unseen filtering:  [117871, 35159]
    # Check:  35159

    # transform the hico annotations into vg format to support for pre-training.
    transform_hico_to_vg_form()