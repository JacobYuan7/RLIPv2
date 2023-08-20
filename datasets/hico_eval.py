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

class HICOEvaluator():
    def __init__(self, preds, gts, subject_category_id, rare_triplets, non_rare_triplets, correct_mat, args = None):
        '''
        correct_mat: [117, 80]
        '''
        self.overlap_iou = 0.5
        self.max_hois = 100
        self.use_nms_filter = args.use_nms_filter
        self.thres_nms = args.thres_nms
        self.nms_alpha = args.nms_alpha
        self.nms_beta = args.nms_beta

        self.rare_triplets = rare_triplets
        self.non_rare_triplets = non_rare_triplets

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_triplets = []

        self.preds = []
        for index, img_preds in enumerate(preds):
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores'] # [100, 117]
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            # print(verb_labels)
            
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()  # [100*117,]
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
                object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                masks = correct_mat[verb_labels, object_labels]  # [11700, ]
                hoi_scores *= masks
                # The above step filters the hois that are in the correct map, 
                # otherwise the score will be multiplied to zero. 

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                # len of hois: [11700,]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)  # get(key, default)
                # Sort the list of dicts according to the value of the 'score'
                hois = hois[:self.max_hois]
            else:
                hois = []

            filename = gts[index]['filename']
            self.preds.append({
                'filename':filename,
                'predictions': bboxes,
                'hoi_prediction': hois})
        
        if self.use_nms_filter:
            print('Starting NMS...')
            self.preds = self.triplet_nms_filter(self.preds)

        self.gts = []
        for img_gts in gts:
            filename = img_gts['filename']
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k != 'id' and k != 'filename'}
            self.gts.append({
                'filename':filename,
                'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_gts['boxes'], img_gts['labels'])],
                'hoi_annotation': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2]} for hoi in img_gts['hois']]
            })
            for hoi in self.gts[-1]['hoi_annotation']:
                triplet = (self.gts[-1]['annotations'][hoi['subject_id']]['category_id'],
                           self.gts[-1]['annotations'][hoi['object_id']]['category_id'],
                           hoi['category_id'])

                if triplet not in self.gt_triplets:
                    self.gt_triplets.append(triplet)

                self.sum_gts[triplet] += 1
        
        # False Positive
        self.fp_dict = []

    def evaluate(self):
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            # print('pred_bboxes:'+ str(len(pred_bboxes))) # 200
            # print('gt_bboxes:' + str(len(gt_bboxes))) # num_of_gt boxes
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            # print('pred_hois:'+ str(len(pred_hois))) # 100
            # print('gt_hois:' + str(len(gt_hois)))  # num_of_gt hois

            # if len(gt_bboxes) != 0:
            ### len(pred_hois) != 0 is used to defend against the situation in zero-shot eval.
            if len(gt_bboxes) != 0 and len(pred_hois) != 0:            
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                # self.record_fp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_hoi in pred_hois:
                    triplet = [pred_bboxes[pred_hoi['subject_id']]['category_id'],
                               pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id']]
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_hoi['score'])
        # np.savez('test.npz', fp_dict = np.array(self.fp_dict))
        # print('Finish Recording fp samples...')
        
        map = self.compute_map()
        return map
    
    def evaluate_visualization(self):
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            # print('pred_bboxes:'+ str(len(pred_bboxes))) # 200
            # print('gt_bboxes:' + str(len(gt_bboxes))) # num_of_gt boxes
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            # print('pred_hois:'+ str(len(pred_hois))) # 100
            # print('gt_hois:' + str(len(gt_hois)))  # num_of_gt hois
            if len(gt_bboxes) != 0:
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                # self.record_fp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_hoi in pred_hois:
                    triplet = [pred_bboxes[pred_hoi['subject_id']]['category_id'],
                               pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id']]
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_hoi['score'])
        # np.savez('test.npz', fp_dict = np.array(self.fp_dict))
        # print('Finish Recording fp samples...')
        
        map = self.recall_mAP_curve()
        return map

    def evaluate_obj_verb_co(self):
        co_matrices = np.zeros((80, 117))
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            # print('pred_bboxes:'+ str(len(pred_bboxes))) # 200
            # print('gt_bboxes:' + str(len(gt_bboxes))) # num_of_gt boxes
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            # print('pred_hois:'+ str(len(pred_hois))) # 100
            # print('gt_hois:' + str(len(gt_hois)))  # num_of_gt hois
            if len(gt_bboxes) != 0:
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                # self.record_fp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_hoi in pred_hois:
                    triplet = [pred_bboxes[pred_hoi['subject_id']]['category_id'],
                               pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id']]
                    if triplet not in self.gt_triplets:
                        continue
                    co_matrices[pred_bboxes[pred_hoi['object_id']]['category_id']][pred_hoi['category_id']]+=1
                    # self.tp[triplet].append(0)
                    # self.fp[triplet].append(1)
                    # self.score[triplet].append(pred_hoi['score'])
        # np.savez('test.npz', fp_dict = np.array(self.fp_dict))
        # print('Finish Recording fp samples...')
        print(co_matrices[1])
        return
    

    def recall_mAP_curve(self):
        triplet_ranking_ascend = np.load('datasets/priors/triplet_ranking_counts.npz', allow_pickle=True)['ranking_counts'].item()
        triplet_ranking_ascend = list(triplet_ranking_ascend.keys())
        ap_ranking = OrderedDict([(i, 0) for i in triplet_ranking_ascend])
        max_recall_ranking = OrderedDict([(i, 0) for i in triplet_ranking_ascend])

        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        rare_max_recall = defaultdict(lambda: 0)
        non_rare_max_recall = defaultdict(lambda: 0)

        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                if triplet in self.rare_triplets:
                    rare_ap[triplet] = 0
                    rare_max_recall[triplet] = 0
                elif triplet in self.non_rare_triplets:
                    non_rare_ap[triplet] = 0
                    non_rare_max_recall[triplet] = 0
                else:
                    print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
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
            max_recall[triplet] = np.amax(rec)
            if triplet in self.rare_triplets:
                rare_ap[triplet] = ap[triplet]
                rare_max_recall[triplet] = max_recall[triplet]
            elif triplet in self.non_rare_triplets:
                non_rare_ap[triplet] = ap[triplet]
                non_rare_max_recall[triplet] = max_recall[triplet]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
        
            ap_ranking[triplet] = ap[triplet]
            max_recall_ranking[triplet] = max_recall[triplet]
        
        m_ap = np.mean(list(ap.values()))
        m_ap_rare = np.mean(list(rare_ap.values()))
        m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_rare_max_recall = np.mean(list(rare_max_recall.values()))
        m_non_rare_max_recall = np.mean(list(non_rare_max_recall.values()))

        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP non-rare: {}  mean rare max recall: {} mean non-rare max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare, m_rare_max_recall, m_non_rare_max_recall))
        print('--------------------')

        # Plot 1
        ap_ranking = np.cumsum(np.array(list(ap_ranking.values())))
        ap_ranking = [ap_ranking[i]/20 if i==19 else (ap_ranking[i]-ap_ranking[i-20])/20 for i in range(19,600,20)]
        max_recall_ranking = np.cumsum(np.array(list(max_recall_ranking.values())))
        max_recall_ranking = [max_recall_ranking[i]/20 if i==19 else (max_recall_ranking[i]-max_recall_ranking[i-20])/20 for i in range(19,600,20)]
        dual_y_plot('Hois-recall-AP_samplesranking', list(range(20,601,20)), ap_ranking,
                                      list(range(20,601,20)), max_recall_ranking)
        
        # Plot 2
        ap_sort = sorted(ap.items(), key = lambda x:x[1])
        ap_sort = [i[1] for i in ap_sort]
        maxrecall_sort = sorted(max_recall.items(), key = lambda x:x[1])
        maxrecall_sort = [i[1] for i in maxrecall_sort]
        ap_sort = np.cumsum(np.array(ap_sort))
        ap_sort = [ap_sort[i]/20 if i==19 else (ap_sort[i]-ap_sort[i-20])/20 for i in range(19,600,20)]
        maxrecall_sort = np.cumsum(np.array(maxrecall_sort))
        maxrecall_sort = [maxrecall_sort[i]/20 if i==19 else (maxrecall_sort[i]-maxrecall_sort[i-20])/20 for i in range(19,600,20)]
        
        dual_y_plot('Hois-recall-AP_recallsranking', list(range(20,601,20)), ap_sort,
                                      list(range(20,601,20)), maxrecall_sort)

        return {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare, \
                'mean rare max recall': m_rare_max_recall, 'mean non-rare max recall': m_non_rare_max_recall}



    def compute_map(self):
        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                if triplet in self.rare_triplets:
                    rare_ap[triplet] = 0
                elif triplet in self.non_rare_triplets:
                    non_rare_ap[triplet] = 0
                else:
                    print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
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
            max_recall[triplet] = np.amax(rec)
            if triplet in self.rare_triplets:
                rare_ap[triplet] = ap[triplet]
            elif triplet in self.non_rare_triplets:
                non_rare_ap[triplet] = ap[triplet]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
        m_ap = np.mean(list(ap.values()))
        m_ap_rare = np.mean(list(rare_ap.values()))
        m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))

        # np.savez('/Path/To/jacob/RLIP/analysis/RLIP-ParSe_COCO_VG_NF_HOI_classwise_results.npz',
        #         ap = (list(ap.keys()), list(ap.values())),
        #         rare_ap = (list(rare_ap.keys()), list(rare_ap.values())),
        #         non_rare_ap = (list(non_rare_ap.keys()), list(non_rare_ap.values())))
        # print('Saving to ', '/Path/To/jacob/RLIP/analysis/RLIP-ParSeD_VG_NF_HOI_classwise_results.npz')

        print('--------------------')
        print('mAP: {} mAP rare (unseen): {}  mAP non-rare (seen): {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare, m_max_recall))
        print('--------------------')

        return {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare, 'mean max recall': m_max_recall}

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hois, gt_hois, match_pairs, pred_bboxes, bbox_overlaps):
        pos_pred_ids = match_pairs.keys() # positive bbox pred_ids with iou>0.5
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)

        if len(pred_hois) != 0:
            for pred_hoi in pred_hois:
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                    # this 'if' ensures the subject and object are rightly detected with iou>0.5 
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0

                    for gt_hoi in gt_hois:
                        # print(gt_hoi) like {'subject_id': 0, 'object_id': 1, 'category_id': 76}
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids \
                           and pred_category_id == gt_hoi['category_id']: 
                            # one hoi is right if bounding boxes for the sub and obj are >=0.5 
                            # and verb label is right
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi

                triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'],
                           pred_hoi['category_id'])
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    vis_tag[gt_hois.index(max_gt_hoi)] = 1 
                    # One pred hoi for the most matched gt_hoi 
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                self.score[triplet].append(pred_hoi['score'])

    
    def record_fp(self, pred_hois, gt_hois, match_pairs, pred_bboxes, bbox_overlaps):
        pos_pred_ids = match_pairs.keys() # positive bbox pred_ids with iou>0.5
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)

        # Dict for recording FP samples 
        fp_record = []

        if len(pred_hois) != 0:
            for pred_hoi in pred_hois:
                # print(pred_hoi)
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                    # this 'if' ensures the subject and object are rightly detected with iou>0.5 
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0

                    for gt_hoi in gt_hois:
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids\
                            and pred_category_id != gt_hoi['category_id'] and pred_hoi['score'] >= 1/117.:
                            # print(pred_category_id, gt_hoi['category_id'])
                            real_triplet = gt_hoi
                            predicted_triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'],
                                                 pred_hoi['category_id'])
                            if predicted_triplet in self.gt_triplets:
                                fp_record.append({'real_triplet': gt_hoi, 
                                                  'predicted_triplet': predicted_triplet,
                                                  'verb_score': pred_hoi['score'],
                                                  'predicted_bbox': [pred_bboxes[pred_hoi['subject_id']]['bbox'],
                                                                     pred_bboxes[pred_hoi['object_id']]['bbox']]}) # xyxy  0~w, 0~h
        self.fp_dict.append(fp_record)

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
    # HOI eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    return parser

if __name__ == '__main__':
    # eval_stat = np.load('/Path/To/jacob/eval_stat.npz', allow_pickle=True)
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    eval_stat = np.load('/Path/To/jacob/eval_stat_best.npz', allow_pickle=True)
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
