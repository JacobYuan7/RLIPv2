# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.hico_eval import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator
import torch.nn.functional as F
from typing import List


def cal_uniformity_for_hico_verbs():
    hico_verb_feature_bf_RLIP = np.load('/Path/To/data/hico_20160224_det/word_embedding/hico_verb_Original_RoBERTa.npz', allow_pickle = True)['hico_verb_dict'].item()
    hico_verb_feature_aft_RLIP = np.load('/Path/To/data/hico_20160224_det/word_embedding/hico_verb_RLIP-ParSe_COCO_VG.npz', allow_pickle = True)['hico_verb_dict'].item()

    VG_verbs = {'hold': 304, 'wear': 240, 'watch': 224, 'carry': 163, 'light': 110, 'sit on': 107, 'train': 86, 'hit': 79, 'eat': 71, 'sign': 54, 'open': 51, 'stand on': 50, 'swing': 48, 'catch': 39, 'cut': 35, 'make': 34, 'pull': 34, 'ride': 28, 'park': 25, 'fill': 22, 'control': 21, 'walk': 21, 'fly': 16, 'push': 16, 'throw': 16, 'block': 15, 'board': 15, 'wave': 13, 'point': 11, 'sit on a': 11, 'feed': 10, 'tie': 10, 'turn': 10, 'kick': 9, 'read': 9, 'lift': 8, 'paint': 7, 'cut with': 6, 'dry': 6, 'lie on': 6, 'pick': 6, 'set': 6, 'sit at': 6, 'wear a': 6, 'carry a': 5, 'hold a': 5, 'make a': 5, 'pick up': 5, 'row': 5, 'buy': 4, 'clean': 4, 'exit': 4, 'flush': 4, 'pack': 4, 'pet': 4, 'run': 4, 'blow': 3, 'drive': 3, 'flip': 3, 'hose': 3, 'lick': 3, 'load': 3, 'repair': 3, 'sail': 3, 'stand on a': 3, 'tag': 3, 'wash': 3, 'adjust': 2, 'chase': 2, 'hug': 2, 'inspect': 2, 'are light': 2, 'race': 2, 'ride a': 2, 'serve': 2, 'smell': 2, 'stand under': 2, 'stop at': 2, 'swing a': 2, 'type on': 2, 'check': 1, 'direct': 1, 'are dry': 1, 'eat at': 1, 'fill the': 1, 'greet': 1, 'herd': 1, 'hit a': 1, 'jump': 1, 'kiss': 1, 'launch': 1, 'lose': 1, 'operate': 1, 'are pull': 1, 'pull a': 1, 'release': 1, 'shear': 1, 'sign a': 1, 'sit at a': 1, 'slide': 1, 'stand on the': 1, 'stand under a': 1, 'stick': 1, 'straddle': 1, 'talk on': 1, 'watch a': 1, 'a wave': 1, 'assemble': 0, 'break': 0, 'brush with': 0, 'cook': 0, 'drag': 0, 'dribble': 0, 'drink with': 0, 'grind': 0, 'groom': 0, 'hop on': 0, 'hunt': 0, 'install': 0, 'lasso': 0, 'milk': 0, 'move': 0, 'no interaction': 0, 'pay': 0, 'peel': 0, 'pour': 0, 'scratch': 0, 'sip': 0, 'spin': 0, 'squeeze': 0, 'stab': 0, 'stir': 0, 'teach': 0, 'text on': 0, 'toast': 0, 'wield': 0, 'zip': 0}
    non_exist_verbs = [i for i,j in VG_verbs.items() if j == 0]
    hico_verb = load_hico_verb_txt()

    norm_hico_bf_RLIP = {}
    norm_hico_aft_RLIP = {}
    for i,j in hico_verb_feature_bf_RLIP.items():
        norm_hico_bf_RLIP[i] = F.normalize(j, p=2, dim=-1)
    for i,j in hico_verb_feature_aft_RLIP.items():
        norm_hico_aft_RLIP[i] = F.normalize(j, p=2, dim=-1)
    # m = hico_verb_feature_aft_RLIP['drink with'].unsqueeze(0)
    # print(torch.einsum('ab,cb->ac', m, m))

    all_hico_verbs_ft_bf_RLIP = {i:norm_hico_bf_RLIP[i].unsqueeze(0) for i in hico_verb}
    exist_verbs_ft_bf_RLIP = {i:norm_hico_bf_RLIP[i].unsqueeze(0) for i in hico_verb if i not in non_exist_verbs}
    non_exist_verbs_ft_bf_RLIP = {i:norm_hico_bf_RLIP[i].unsqueeze(0) for i in hico_verb if i in non_exist_verbs}
    
    all_hico_verbs_ft_aft_RLIP = {i:norm_hico_aft_RLIP[i].unsqueeze(0) for i in hico_verb}
    exist_verbs_ft_aft_RLIP = {i:norm_hico_aft_RLIP[i].unsqueeze(0) for i in hico_verb if i not in non_exist_verbs}
    non_exist_verbs_ft_aft_RLIP = {i:norm_hico_aft_RLIP[i].unsqueeze(0) for i in hico_verb if i in non_exist_verbs}
    
    print("Uniformity before RLIP, 117 all verbs: {}, 87 existing verbs: {}, 30 non-existing verbs: {}\n".format(cal_uniformity(all_hico_verbs_ft_bf_RLIP), cal_uniformity(exist_verbs_ft_bf_RLIP), cal_uniformity(non_exist_verbs_ft_bf_RLIP)))
    print("Uniformity after RLIP, 117 all verbs: {}, 87 existing verbs: {}, 30 non-existing verbs: {}\n".format(cal_uniformity(all_hico_verbs_ft_aft_RLIP), cal_uniformity(exist_verbs_ft_aft_RLIP), cal_uniformity(non_exist_verbs_ft_aft_RLIP)))
    

def cal_uniformity_alignment(relation_feature_path):
    verb_class_dict = np.load(relation_feature_path, allow_pickle = True)['verb_class_dict'].item()
    verb_class_tensot_dict = {}
    for verb_idx, ft_list in verb_class_dict.items():
        tensor_list = torch.stack([torch.from_numpy(ft) for ft in ft_list])
        # print(tensor_list.shape)
        verb_class_tensot_dict[verb_idx] = F.normalize(tensor_list, p=2, dim=-1)
    
    print(cal_uniformity(verb_class_tensot_dict), cal_alignment(verb_class_tensot_dict))

def cal_uniformity(feature_dict, t = 2):
    all_feature = torch.cat([j for i,j in feature_dict.items()], dim = 0)
    sq_dist = torch.pdist(all_feature, p = 2).pow(2)
    return sq_dist.mul(-t).exp().mean().log().item()

def cal_alignment(feature_dict, alpha = 2):
    all_feature_num = sum([j.shape[0] for i,j in feature_dict.items()])
    all_align = []

    # V1: Treat features as a sample
    # for i, j in feature_dict.items():
    #     all_align.append(torch.pdist(j, p = 2).pow(alpha).flatten())
    #     print(all_align[-1].shape)
    # return torch.cat(all_align).mean()
    # V2: Treat one class as a sample
    for i, j in feature_dict.items():
        all_align.append(torch.pdist(j, p = 2).pow(alpha).mean().item())
        print(all_align[-1])
    return sum(all_align)/len(all_align)


def load_hico_verb_txt(file_path = '/Path/To/jacob/OCN/datasets/hico_verb_names.txt') -> List[list]:
    '''
    Output like [['train'], ['boat'], ['traffic', 'light'], ['fire', 'hydrant']]
    '''
    verb_names = []
    for line in open(file_path,'r'):
        # verb_names.append(line.strip().split(' ')[-1])
        verb_names.append(' '.join(line.strip().split(' ')[-1].split('_')))
    return verb_names

if __name__ == "__main__":
    # # calculate uniformity and alignment for the visual features
    # relation_feature_path = '/Path/To/jacob/Uniformity/LSE_RQL_RPL_relation_feature_2.npz'
    # # relation_feature_path = '/Path/To/jacob/Uniformity/LSE_RQL_relation_feature_2.npz'
    # # relation_feature_path = '/Path/To/jacob/Uniformity/LSE_relation_feature_2.npz'
    # # relation_feature_path = '/Path/To/jacob/Uniformity/vanilla_relation_feature.npz'
    # cal_uniformity_alignment(relation_feature_path)

    # calculate uniformity and alignment for textual features
    cal_uniformity_for_hico_verbs()
