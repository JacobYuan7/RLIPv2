# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import numpy as np
from collections import defaultdict, OrderedDict
import os
import sys
import matplotlib.pyplot as plt
import argparse 
# import dill

class OISGGEvaluator():
    def __init__(self, preds, gts, correct_mat, topK = 50, use_corre_mat = False, args = None):
        '''
        correct_mat: [288, 30, 288]
        '''
        self.overlap_iou = 0.5
        self.max_rels = topK
        self.use_nms_filter = args.use_nms_filter
        self.thres_nms = args.thres_nms
        self.nms_alpha = args.nms_alpha
        self.nms_beta = args.nms_beta
        self.thres_nms_phr = args.thres_nms_phr
        self.use_corre_mat = use_corre_mat

        ### For relation detection
        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_triplets = []

        self.preds = []
        for index, img_preds in enumerate(preds):
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            rel_scores = img_preds['verb_scores'] # [100, 117]
            verb_labels = np.tile(np.arange(rel_scores.shape[1]), (rel_scores.shape[0], 1))
            # print(verb_labels)
            
            subject_ids = np.tile(img_preds['sub_ids'], (rel_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (rel_scores.shape[1], 1)).T

            rel_scores = rel_scores.ravel()  # [100*117,]
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
                if self.use_corre_mat:
                    subject_labels = np.array([bboxes[subject_id]['category_id'] for subject_id in subject_ids])
                    object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                    masks = correct_mat[subject_labels, verb_labels, object_labels]  # [11700, ]
                    rel_scores *= masks
                    # The above step filters the rels that are in the correct map, 
                    # otherwise the score will be multiplied to zero. 

                rels = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, rel_scores)]
                rels.sort(key=lambda k: (k.get('score', 0)), reverse=True)  # get(key, default)
                # Sort the list of dicts according to the value of the 'score'
                rels = rels[:self.max_rels]
            else:
                rels = []

            filename = gts[index]['filename']
            self.preds.append({
                'filename':filename,
                'predictions': bboxes,
                'rel_predictions': rels})
        
        ### Original 
        if self.use_nms_filter:
            print('Starting pairwise NMS...')
            self.preds = self.triplet_nms_filter(self.preds)
        self.generate_phrases(mode = 'prediction')
        if self.use_nms_filter:
            print('Starting phrase NMS...')
            self.preds = self.phrase_nms_filter(self.preds)

        ### New (generate phrases first and then perform phrase NMS)
        # self.generate_phrases(mode = 'prediction')
        # if self.use_nms_filter:
        #     print('Starting pairwise NMS...')
        #     self.preds = self.triplet_nms_filter(self.preds)
        #     print('Starting phrase NMS...')
        #     self.preds = self.phrase_nms_filter(self.preds)

        self.gts = []
        for img_gts in gts:
            filename = img_gts['filename']
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k != 'id' and k != 'filename'}
            self.gts.append({
                'filename':filename,
                'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_gts['boxes'], img_gts['labels'])],
                'rel_annotations': [{'subject_id': rel[0], 'object_id': rel[1], 'category_id': rel[2]} for rel in img_gts['rels']]
            })
            for rel in self.gts[-1]['rel_annotations']:
                triplet = (self.gts[-1]['annotations'][rel['subject_id']]['category_id'],
                           self.gts[-1]['annotations'][rel['object_id']]['category_id'],
                           rel['category_id'])

                if triplet not in self.gt_triplets:
                    self.gt_triplets.append(triplet)

                self.sum_gts[triplet] += 1
        
        print(f'OISGGEvaluator will evaluate on {len(self.gt_triplets)} kinds of triplets.')
        self.generate_phrases(mode = 'annotation')

        ### For phrase detection
        self.fp_phr = defaultdict(list)
        self.tp_phr = defaultdict(list)
        self.score_phr = defaultdict(list)

        # False Positive
        self.fp_dict = []

    def generate_phrases(self, mode):
        '''
        This function aims to merge the relation detection results/annotations to generate phrase detection results/annotations.
        
        Args:
            mode (string): it can be a choice from ['prediction', 'annotation']
        '''
        if mode == 'prediction':
            bbox_key = 'predictions'
            rel_key = 'rel_predictions'
            phrase_key = 'phrase_predictions'
            annos = self.preds
        elif mode == 'annotation':
            bbox_key = 'annotations'
            rel_key = 'rel_annotations'
            phrase_key = 'phrase_annotations'
            annos = self.gts
        else:
            assert False
        
        for anno in annos:
            bboxs = anno[bbox_key]
            rels = anno[rel_key]

            phrase_list = []
            for rel in rels:
                sub_bbox = bboxs[rel['subject_id']]['bbox'] # x1 y1 x2 y2
                obj_bbox = bboxs[rel['object_id']]['bbox']
                min_bounding_rect = np.array((min(sub_bbox[0], obj_bbox[0]), min(sub_bbox[1], obj_bbox[1]),
                                              max(sub_bbox[2], obj_bbox[2]), max(sub_bbox[3], obj_bbox[3])))
                phrase_category = (bboxs[rel['subject_id']]['category_id'],
                                   bboxs[rel['object_id']]['category_id'],
                                   rel['category_id'])
                phrase_list.append({
                    'bbox': min_bounding_rect,
                    'category_id': phrase_category,
                })
                if mode == 'prediction':
                    phrase_list[-1].update({'phrase_score': rel['score']})
            anno.update({phrase_key: phrase_list})

    def evaluate(self):
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            # print('pred_bboxes:'+ str(len(pred_bboxes))) # 200
            # print('gt_bboxes:' + str(len(gt_bboxes))) # num_of_gt boxes
            pred_rels = img_preds['rel_predictions']
            gt_rels = img_gts['rel_annotations']
            pred_phrases = img_preds['phrase_predictions']
            gt_phrases = img_gts['phrase_annotations']

            # if len(gt_bboxes) != 0:
            ### len(pred_rels) != 0 is used to defend against the situation in zero-shot eval.
            if len(gt_bboxes) != 0 and len(pred_rels) != 0:
                ### Compute fps and tps for relation detection           
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_rels, gt_rels, bbox_pairs, pred_bboxes, bbox_overlaps)
                ### Compute fps and tps for phrase detection
                phrase_pairs, phrase_overlaps = self.compute_iou_mat(gt_phrases, pred_phrases)
                self.compute_fptp_phrase(gt_phrases, pred_phrases, phrase_pairs, phrase_overlaps)
            else:
                ### Compute fps and tps for relation detection  
                for pred_rel in pred_rels:
                    triplet = [pred_bboxes[pred_rel['subject_id']]['category_id'],
                               pred_bboxes[pred_rel['object_id']]['category_id'], pred_rel['category_id']]
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_rel['score'])
                ### Compute fps and tps for phrase detection
                for pred_phrase in pred_phrases:
                    triplet = pred_phrase['category_id']
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp_phr[triplet].append(0)
                    self.fp_phr[triplet].append(1)
                    self.score_phr[triplet].append(pred_phrase['phrase_score'])

        # np.savez('test.npz', fp_dict = np.array(self.fp_dict))
        # print('Finish Recording fp samples...')
        
        map_rel = self.compute_map()
        map_phr = self.compute_map_phr()

        map = map_rel
        map.update(map_phr)
        return map
    
    def print_res(self, map):
        '''
        This function aims to print the final results.

        Args:
            map (dict): a dict containing the final result
        '''
        map.update({'score_wtd (mAP)': 0.2*map['max recall@50 (RelD)'] + 0.4*map['mAP@100 (RelD)'] + 0.4*map['mAP@100 (PhrD)']})
        map.update({'score_wtd (wmAP)': 0.2*map['max recall@50 (RelD)'] + 0.4*map['wmAP@100 (RelD)'] + 0.4*map['wmAP@100 (PhrD)']})
        print('--------------------')
        print(f"mAP@100 (RelD): {map['mAP@100 (RelD)']}, wmAP@100 (RelD):{map['wmAP@100 (RelD)']}, mean max recall@100 (RelD): {map['mean max recall@100 (RelD)']}, max recall@100 (RelD): {map['max recall@100 (RelD)']}, max recall@50 (RelD): {map['max recall@50 (RelD)']}, mean max recall@50 (RelD): {map['mean max recall@50 (RelD)']}")
        print(f"mAP@100 (PhrD): {map['mAP@100 (PhrD)']}, wmAP@100 (PhrD):{map['wmAP@100 (PhrD)']}, mean max recall@100 (PhrD): {map['mean max recall@100 (PhrD)']}, max recall@100 (PhrD): {map['max recall@100 (PhrD)']}")
        print(f"score_wtd (mAP): {map['score_wtd (mAP)']}, score_wtd (wmAP): {map['score_wtd (wmAP)']}")
        print('--------------------')

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        tp_all_triplet_sum = 0
        gts_all_triplet_sum = 0
        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                gts_all_triplet_sum += sum_gts # Test. This will lower the performance a bit sometimes, but gts should be computed.
                continue

            score = np.array(self.score[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[triplet] = self.voc_ap(rec, prec)
            # print(rec)
            max_recall[triplet] = np.amax(rec)

            ### Aim to compute the recall metric
            tp_all_triplet_sum += tp[-1]
            gts_all_triplet_sum += sum_gts
            # print(tp_all_triplet_sum/gts_all_triplet_sum)


        m_ap = np.mean(list(ap.values()))
        weighted_m_ap = sum([trip_ap*self.sum_gts[triplet]/sum(self.sum_gts.values()) for triplet, trip_ap in ap.items()])
        # m_ap_rare = np.mean(list(rare_ap.values()))
        # m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))
        max_recall = tp_all_triplet_sum/gts_all_triplet_sum

        # print('--------------------')
        # print('mAP (RelD): {}, mean max recall (RelD): {}, max recall (RelD): {}'.format(m_ap, m_max_recall, max_recall))
        # print('--------------------')

        return {f"mAP@{self.max_rels} (RelD)": m_ap,
                f"wmAP@{self.max_rels} (RelD)": weighted_m_ap,
                f"mean max recall@{self.max_rels} (RelD)": m_max_recall,
                f"max recall@{self.max_rels} (RelD)": max_recall}

    def compute_map_phr(self):
        ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        tp_all_triplet_sum = 0
        gts_all_triplet_sum = 0
        for triplet in self.gt_triplets:
            # print(triplet)
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp_phr[triplet]))
            fp = np.array((self.fp_phr[triplet]))
            if len(tp) == 0:
                # print('XXX')
                ap[triplet] = 0
                max_recall[triplet] = 0
                continue

            score = np.array(self.score_phr[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[triplet] = self.voc_ap(rec, prec)
            # print(rec)
            max_recall[triplet] = np.amax(rec)
            tp_all_triplet_sum += tp[-1]
            gts_all_triplet_sum += sum_gts
            # print(tp_all_triplet_sum/gts_all_triplet_sum)

        m_ap = np.mean(list(ap.values()))
        weighted_m_ap = sum([trip_ap*self.sum_gts[triplet]/sum(self.sum_gts.values()) for triplet, trip_ap in ap.items()])
        m_max_recall = np.mean(list(max_recall.values()))
        max_recall = tp_all_triplet_sum/gts_all_triplet_sum

        # print('--------------------')
        # print('mAP (PhrD): {}, mean max recall (PhrD): {}, max recall (PhrD): {}'.format(m_ap, m_max_recall, max_recall))
        # print('--------------------')

        return {f"mAP@{self.max_rels} (PhrD)": m_ap,
                f"wmAP@{self.max_rels} (PhrD)": weighted_m_ap,
                f"mean max recall@{self.max_rels} (PhrD)": m_max_recall,
                f"max recall@{self.max_rels} (PhrD)": max_recall}

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_rels, gt_rels, match_pairs, pred_bboxes, bbox_overlaps):
        pos_pred_ids = match_pairs.keys() # positive bbox pred_ids with iou>0.5
        vis_tag = np.zeros(len(gt_rels))
        pred_rels.sort(key=lambda k: (k.get('score', 0)), reverse=True)

        if len(pred_rels) != 0:
            for pred_rel in pred_rels:
                is_match = 0
                if len(match_pairs) != 0 and pred_rel['subject_id'] in pos_pred_ids and pred_rel['object_id'] in pos_pred_ids:
                    # this 'if' ensures the subject and object are rightly detected with iou>0.5 
                    pred_sub_ids = match_pairs[pred_rel['subject_id']]
                    pred_obj_ids = match_pairs[pred_rel['object_id']]
                    pred_sub_overlaps = bbox_overlaps[pred_rel['subject_id']]
                    pred_obj_overlaps = bbox_overlaps[pred_rel['object_id']]
                    pred_category_id = pred_rel['category_id']
                    max_overlap = 0
                    max_gt_rel = 0

                    for gt_rel in gt_rels:
                        # print(gt_rel) like {'subject_id': 0, 'object_id': 1, 'category_id': 76}
                        if gt_rel['subject_id'] in pred_sub_ids and gt_rel['object_id'] in pred_obj_ids \
                           and pred_category_id == gt_rel['category_id']: 
                            # one rel is right if bounding boxes for the sub and obj are >=0.5 
                            # and verb label is right
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_rel['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_rel['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_rel = gt_rel

                triplet = (pred_bboxes[pred_rel['subject_id']]['category_id'], pred_bboxes[pred_rel['object_id']]['category_id'],
                           pred_rel['category_id'])
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[gt_rels.index(max_gt_rel)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    vis_tag[gt_rels.index(max_gt_rel)] = 1 
                    # One pred rel for the most matched gt_rel
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                self.score[triplet].append(pred_rel['score'])

    def compute_fptp_phrase(self, gt_phrases, pred_phrases, phrase_pairs, phrase_overlaps):
        pos_pred_ids = phrase_pairs.keys() # positive bbox pred_ids with iou>0.5
        vis_tag = np.zeros(len(gt_phrases))
        pred_phrases.sort(key=lambda k: (k.get('phrase_score', 0)), reverse=True)

        if len(pred_phrases) != 0:
            for idx_pred, pred_phrase in enumerate(pred_phrases):
                is_match = 0
                if len(phrase_pairs) != 0 and idx_pred in pos_pred_ids:
                    pred_ids = phrase_pairs[idx_pred]
                    pred_overlaps = phrase_overlaps[idx_pred]
                    max_overlap = 0
                    max_gt_phr = 0

                    for gt_idx, gt_phrase in enumerate(gt_phrases):
                        if gt_idx in pred_ids:
                            is_match = 1
                            gt_overlap = pred_overlaps[pred_ids.index(gt_idx)]
                            if gt_overlap > max_overlap:
                                max_overlap = gt_overlap
                                max_gt_phr = gt_idx
                
                triplet = pred_phrase['category_id']
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[max_gt_phr] == 0:
                    self.fp_phr[triplet].append(0)
                    self.tp_phr[triplet].append(1)
                    vis_tag[max_gt_phr] = 1 
                    # One pred rel for the most matched gt_rel
                else:
                    self.fp_phr[triplet].append(1)
                    self.tp_phr[triplet].append(0)
                self.score_phr[triplet].append(pred_phrase['phrase_score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        # gt_bboxes, pred_bboxes
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov=iou_mat.copy()
        iou_mat[iou_mat>=self.overlap_iou] = 1
        iou_mat[iou_mat<self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat) # return gt index array and pred index array
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0: # if there is a matched pair
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict, match_pair_overlaps
        # dict like:
        # match_pairs_dict {pred_id: [gt_id], pred_id: [gt_id], ...}
        # match_pair_overlaps {pred_id: [gt_id], pred_id: [gt_id], ...} 
        # we may have many gt_ids for a specific pred_id, because we don't consider the class

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
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
    
    def triplet_nms_filter(self, preds):
        preds_filtered = []
        for img_preds in preds:
            pred_bboxes = img_preds['predictions']
            pred_rels = img_preds['rel_predictions']
            all_triplets = {}
            for index, pred_rel in enumerate(pred_rels):
                triplet = str(pred_bboxes[pred_rel['subject_id']]['category_id']) + '_' + \
                          str(pred_bboxes[pred_rel['object_id']]['category_id']) + '_' + str(pred_rel['category_id'])

                if triplet not in all_triplets:
                    all_triplets[triplet] = {'subs':[], 'objs':[], 'scores':[], 'indexes':[]}
                all_triplets[triplet]['subs'].append(pred_bboxes[pred_rel['subject_id']]['bbox'])
                all_triplets[triplet]['objs'].append(pred_bboxes[pred_rel['object_id']]['bbox'])
                all_triplets[triplet]['scores'].append(pred_rel['score'])
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
                'rel_predictions': list(np.array(img_preds['rel_predictions'])[all_keep_inds])
                })
            
            if 'phrase_predictions' in img_preds.keys():
                preds_filtered[-1].update({'phrase_predictions': img_preds['phrase_predictions']})
        return preds_filtered

    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        order = scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = np.power(sub_inter/sub_union, self.nms_alpha) * np.power(obj_inter / obj_union, self.nms_beta)
            inds = np.where(ovr <= self.thres_nms)[0]

            order = order[inds + 1]
        return keep_inds
    
    def phrase_nms_filter(self, preds):
        '''
        This function aims to filter phrases with the NMS algorithm.

        Args:
            preds (list): the list of prediction results
        '''
        # preds_filtered = []
        for img_preds in preds:

            pred_phrases = img_preds['phrase_predictions']
            all_triplets = {}
            for index, pred_phrase in enumerate(pred_phrases):
                triplet = pred_phrase['category_id']
                if triplet not in all_triplets:
                    all_triplets[triplet] = {'phrs':[], 'scores':[], 'indexes':[]}
                all_triplets[triplet]['phrs'].append(pred_phrase['bbox'])
                all_triplets[triplet]['scores'].append(pred_phrase['phrase_score'])
                all_triplets[triplet]['indexes'].append(index)
            
            all_keep_inds = []
            for triplet, values in all_triplets.items():
                phrs, scores = values['phrs'], values['scores']
                keep_inds = self.nms_cpu(phrs, scores, self.thres_nms_phr) # 测试一下这里的threshold设置多少比较好

                keep_inds = list(np.array(values['indexes'])[keep_inds])
                all_keep_inds.extend(keep_inds)

            img_preds.update({'phrase_predictions': list(np.array(img_preds['phrase_predictions'])[all_keep_inds])})

        return preds

    def nms_cpu(self, dets, scores, thresh):
        '''
        NMS copy paste from https://github.com/jwyang/faster-rcnn.pytorch/blob/f9d984d27b48a067b29792932bcb5321a39c1f09/lib/model/nms/nms_cpu.py
        '''
        dets = np.array(dets)
        scores = np.array(scores)
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        # scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order.item(0)
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
        # return torch.IntTensor(keep)


def dual_y_plot(title, X1, Y1, X2, Y2, style = 'plot'):
    '''
    :param title: tile , string
    :param X1: axis x1, list
    :param Y1: axis y1, list
    :param X2: axis x2, list
    :param Y2: axis y2, list
    :return: save the fig to 'title.png'
    '''
    fig, ax1 = plt.subplots(figsize=(12, 9))
    plt.title(title, fontsize=20)
    plt.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    plt.tick_params(axis='both', labelsize=14)
    if style == 'plot':
        plot1 = ax1.plot(X1, Y1, 'b', label='AP')
    elif style == 'scatter':
        plot1 = ax1.scatter(X1, Y1, s = 8., label='AP') # s = size
    ax1.set_ylabel('Recall', fontsize=18)
    ax1.set_ylim(0, 1)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax2 = ax1.twinx()
    if style == 'plot':
        plot2 = ax2.plot(X2, Y2, 'g', label='Recall')
    elif style == 'scatter':
        plot2 = ax2.scatter(X2, Y2, s = 8., label='Recall') # s = size
    # plot2 = ax2.plot(X2, Y2, 'g', label='AP')
    ax2.set_ylabel('AP', fontsize=18)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelsize=14)
    for tl in ax2.get_yticklabels():
        tl.set_color('g')
    #ax2.set_xlim(1966, 2014.15)
    lines = plot1 + plot2
    ax1.legend(lines, [l.get_label() for l in lines])
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 9))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 9))
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    fig.text(0.1, 0.02,
             'The original content: http://savvastjortjoglou.com/nba-draft-part02-visualizing.html\nPorter: MingYan',
             fontsize=10)
    plt.savefig(title + ".png")


def array2tuplelist(array):
    return [tuple(a) for a in array]


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # Rel eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    return parser

if __name__ == '__main__':
    # eval_stat = np.load('/mnt/data-nas/peizhi/jacob/eval_stat.npz', allow_pickle=True)
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    eval_stat = np.load('/mnt/data-nas/peizhi/jacob/eval_stat_best.npz', allow_pickle=True)
    evaluator = HICOEvaluator(eval_stat['preds'], 
                              eval_stat['gts'], 
                              eval_stat['subject_category_id'], 
                              array2tuplelist(eval_stat['rare_triplets']),
                              array2tuplelist(eval_stat['non_rare_triplets']), 
                              eval_stat['correct_mat'],
                              args = args)
    evaluator.evaluate()
    # stats = evaluator.evaluate_visualization()
    # 把compute mAP之前的参数都保存下来
    # stats = evaluator.evaluate_obj_verb_co()
