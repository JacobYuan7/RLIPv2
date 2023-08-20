# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcherHOI(nn.Module):

    def __init__(self, cost_obj_class: float = 1, cost_verb_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1, subject_class = False):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.subject_class = subject_class
        print('HungarianMatcherHOI matches with the subject class?', self.subject_class)
        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets, return_cost = False):
        if self.subject_class:
            bs, num_queries = outputs['pred_obj_logits'].shape[:2]

            out_sub_prob = outputs['pred_sub_logits'].flatten(0, 1).softmax(-1)
            out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)
            out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()
            out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
            out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)

            # Cat labels for different images to process the cost calculation jointly
            tgt_sub_labels = torch.cat([v['sub_labels'] for v in targets])
            tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])
            max_text_len = max([v['verb_labels'].shape[1] for v in targets])
            verb_labels = ()
            max_len_tensor = None
            for v in targets:
                if v['verb_labels'].shape[0] > 0:   # guard against squeeze_(dim = 0).unsqueeze_(dim = -1) operation
                    verb_labels += v['verb_labels'].split(1, dim = 0)
                elif v['verb_labels'].shape[0] == 0 and v['verb_labels'].shape[1] == max_text_len:
                    max_len_tensor = torch.zeros((1, v['verb_labels'].shape[1]), device = v['verb_labels'].device)
            if max_len_tensor is not None:
                verb_labels += (max_len_tensor,)
            for v in verb_labels:
                v.squeeze_(dim = 0).unsqueeze_(dim = -1)
            tgt_verb_labels = pad_sequence(verb_labels).squeeze_(dim = -1)
            if max_len_tensor is not None:
                tgt_verb_labels = tgt_verb_labels[:,:tgt_verb_labels.shape[1]-1]

            # tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])
            # tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
            tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])
            tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])

            cost_sub_class = -out_sub_prob[:, tgt_sub_labels]
            cost_obj_class = -out_obj_prob[:, tgt_obj_labels]

            # print(out_verb_prob.shape)  # [200, 117]
            # print(tgt_verb_labels_permute.shape)  # [117,3] [117,12] [117,5]....
            tgt_verb_labels_permute = tgt_verb_labels
            tgt_verb_labels = tgt_verb_labels.permute(1, 0)
            ## TEST POINT1
            if out_verb_prob.shape[1] != tgt_verb_labels_permute.shape[0]:
                if out_verb_prob.shape[1] - 1 == tgt_verb_labels_permute.shape[0]:
                    # Defend against the use of self.no_pred_embedding in ParSeDETRHOI
                    out_verb_prob = out_verb_prob[:, :out_verb_prob.shape[1] - 1]
                else:
                    print(out_verb_prob.shape, tgt_verb_labels_permute.shape)
                    for v in targets:
                        print(v['verb_labels'])
                        print(v['image_id'])
            cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                                (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                                (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                                ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2

            cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
            cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
            if cost_sub_bbox.shape[1] == 0:
                cost_bbox = cost_sub_bbox
            else:
                cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]

            cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
            cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                            cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
            if cost_sub_giou.shape[1] == 0:
                cost_giou = cost_sub_giou
            else:
                cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]
            
            
            # print('cost_obj_class.shape ' + str(cost_obj_class.shape))
            C = self.cost_obj_class * cost_obj_class + self.cost_obj_class * cost_sub_class +\
                self.cost_verb_class * cost_verb_class + \
                self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()
            # print("C.shape "+ str(C.shape))  [2, 100, 6] [2, 100, 5] ...
            # 6 and 5 in the previous example are the sum of object labels for this batch (size 2)

            sizes = [len(v['obj_labels']) for v in targets]
            # The split function splits the previously cat labels 
            #                         to perform hungarian matching for every images
            #           the index i is like an image index 
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            # indices: list of array tuples  [(), ()]
            # like [(array([ 5, 42, 51, 61]), array([2, 3, 0, 1])), (array([20]), array([0]))]
            if return_cost:
                cost_list = [cost_giou, (cost_sub_giou, cost_obj_giou), 
                             cost_bbox, (cost_sub_bbox, cost_obj_bbox), 
                             cost_verb_class, cost_sub_class, cost_obj_class]
                return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], cost_list
            else:
                return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            # returned indices: list of tensor tuples
            # like [(tensor([ 5, 42, 51, 61]), tensor([2, 3, 0, 1])), (tensor([20]), tensor([0]))]
        else:
            bs, num_queries = outputs['pred_obj_logits'].shape[:2]

            out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)
            out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()
            out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
            out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)

            # Cat labels for different images to process the cost calculation jointly
            tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])
            tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets]) # [X, 117]
            # print(tgt_verb_labels.shape)
            tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
            tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])
            tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])

            cost_obj_class = -out_obj_prob[:, tgt_obj_labels]

            # print(out_verb_prob.shape)  # [200, 117]
            # print(tgt_verb_labels_permute.shape)  # [117,3] [117,12] [117,5]....
            tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
            cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                                (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                                (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                                ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2

            cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
            cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
            if cost_sub_bbox.shape[1] == 0:
                cost_bbox = cost_sub_bbox
            else:
                cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]

            cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
            cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                            cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
            if cost_sub_giou.shape[1] == 0:
                cost_giou = cost_sub_giou
            else:
                cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]
            
            
            # print('cost_obj_class.shape ' + str(cost_obj_class.shape))
            C = self.cost_obj_class * cost_obj_class + self.cost_verb_class * cost_verb_class + \
                self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()
            # print("C.shape "+ str(C.shape))  [2, 100, 6] [2, 100, 5] ...
            # 6 and 5 in the previous example are the sum of object labels for this batch (size 2)

            sizes = [len(v['obj_labels']) for v in targets]
            # The split function splits the previously cat labels 
            #                         to perform hungarian matching for every images
            #           the index i is like an image index 
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            # indices: list of array tuples  [(), ()]
            # like [(array([ 5, 42, 51, 61]), array([2, 3, 0, 1])), (array([20]), array([0]))]
            if return_cost:
                cost_list = [cost_giou, (cost_sub_giou, cost_obj_giou), 
                             cost_bbox, (cost_sub_bbox, cost_obj_bbox), 
                             cost_verb_class, cost_obj_class]
                return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], cost_list
            else:
                return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            # returned indices: list of tensor tuples
            # like [(tensor([ 5, 42, 51, 61]), tensor([2, 3, 0, 1])), (tensor([20]), tensor([0]))]
            

def build_matcher(args):
    if args.hoi or args.sgg or args.cross_modal_pretrain:
        return HungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class, 
                                   cost_verb_class=args.set_cost_verb_class,
                                   cost_bbox=args.set_cost_bbox,
                                   cost_giou=args.set_cost_giou,
                                   subject_class = args.subject_class)
        # return HungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class, 
        #                            cost_verb_class=args.set_cost_verb_class,
        #                            cost_bbox=args.set_cost_bbox,
        #                            cost_giou=args.set_cost_giou,
        #                            cross_modal_matching = args.ParSeDETRHOI)
        #                         # if the model is ParSeDETRHOI, then we are using cross-modal matching to perform classifying
    else:
        return HungarianMatcher(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou)