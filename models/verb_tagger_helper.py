# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import torch
from util.misc import inverse_sigmoid
import numpy as np

def prepare_query(label_embeds, num_queries, targets, training, box_embed_func,
                  label_noise_scale = 0.2, box_noise_scale = 0.4):
                # label_noise_scale = 0., box_noise_scale = 0.):
                # verb_tagger = False,
                # query_embeds,
                # pos_neg_ratio = 1.):
    '''
    label_embeds: a tensor with the shape [batch_size, num_objs, 256]
                  This should be detached?
    num_queries: the number of subject and object queries 
    pos_neg_ratio: #positive sub-obj query pairs/#negative sub-obj query pairs
                   'positive' means that the subject and the object has a valid relation label.
    '''
    num_pairs = num_queries//2
    # pos_pairs = int(num_pairs/(pos_neg_ratio + 1)*pos_neg_ratio + 0.5)
    # neg_pairs = num_pairs - pos_pairs
    num_classes = label_embeds.shape[0]
    d_model = label_embeds.shape[-1]
    bs = len(targets)

    if training:
        ### Add Noise
        pair_num_list = [v["sub_boxes"].shape[0] for v in targets]
        pair_idx_list = np.cumsum(pair_num_list)
        pair_idx_list = [0,] + list(pair_idx_list)
        tgt_sub_bbox = torch.cat([v["sub_boxes"] for v in targets])  # cx, cy, w, h
        tgt_obj_bbox = torch.cat([v["obj_boxes"] for v in targets])  # cx, cy, w, h
        # print(tgt_bbox.shape[0], num_pairs * len(targets[0]))
        # tgt_bbox_obj = torch.cat([v["obj_boxes"] for v in targets])  # cx, cy, w, h
        # assert tgt_bbox.shape[0] == num_pairs * len(targets[0])

        # Change x1,y1,x2,y2 to x_c,y_c,w,h 
        # tgt_bbox_x_c = (tgt_bbox[:, 0] + tgt_bbox[:, 2])/2 
        # tgt_bbox_y_c = (tgt_bbox[:, 1] + tgt_bbox[:, 3])/2
        # tgt_bbox_w = tgt_bbox[:, 2] - tgt_bbox[:, 0]
        # tgt_bbox_h = tgt_bbox[:, 3] - tgt_bbox[:, 1]   # x_c, y_c, w, h
        # tgt_bbox_coco = torch.stack((tgt_bbox_x_c, tgt_bbox_y_c, tgt_bbox_w, tgt_bbox_h), dim = 1)
        
        # tgt_bbox_coco = tgt_bbox
        if box_noise_scale > 0.:
            # sub_box noise
            diff = torch.zeros_like(tgt_sub_bbox)
            diff[:, :2] = tgt_sub_bbox[:, 2:] / 2
            diff[:, 2:] = tgt_sub_bbox[:, 2:]
            tgt_sub_bbox += torch.mul((torch.rand_like(tgt_sub_bbox) * 2 - 1.0),
                                           diff).cuda() * box_noise_scale
            tgt_sub_bbox = tgt_sub_bbox.clamp(min=0.0, max=1.0)

            # obj_box noise
            diff = torch.zeros_like(tgt_obj_bbox)
            diff[:, :2] = tgt_obj_bbox[:, 2:] / 2
            diff[:, 2:] = tgt_obj_bbox[:, 2:]
            tgt_obj_bbox += torch.mul((torch.rand_like(tgt_obj_bbox) * 2 - 1.0),
                                           diff).cuda() * box_noise_scale
            tgt_obj_bbox = tgt_obj_bbox.clamp(min=0.0, max=1.0)

        tgt_sub_bbox_inv = inverse_sigmoid(tgt_sub_bbox)
        tgt_sub_bbox_query = box_embed_func(tgt_sub_bbox_inv)
        tgt_obj_bbox_inv = inverse_sigmoid(tgt_obj_bbox)
        tgt_obj_bbox_query = box_embed_func(tgt_obj_bbox_inv)

        tgt_sub_labels = torch.cat([v["sub_labels"] for v in targets])
        tgt_obj_labels = torch.cat([v["obj_labels"] for v in targets])
        label_bs_idx = torch.cat([torch.tensor([i,]*v["sub_labels"].shape[0], dtype=torch.int64) for i,v in enumerate(targets)]).to(label_embeds.device)
        if label_noise_scale > 0.:
            p_sub = torch.rand_like(tgt_sub_labels.float())
            chosen_indice_sub = torch.nonzero(p_sub < (label_noise_scale)).view(-1)  # usually half of bbox noise
            new_label_sub = torch.randint_like(chosen_indice_sub, 0, num_classes)  # randomly put a new one here
            tgt_sub_labels.scatter_(0, chosen_indice_sub, new_label_sub)

            p_obj = torch.rand_like(tgt_obj_labels.float())
            chosen_indice_obj = torch.nonzero(p_obj < (label_noise_scale)).view(-1)  # usually half of bbox noise
            new_label_obj = torch.randint_like(chosen_indice_obj, 0, num_classes)  # randomly put a new one here
            tgt_obj_labels.scatter_(0, chosen_indice_obj, new_label_obj)
        
        tgt_sub_label_query = label_embeds[label_bs_idx, tgt_sub_labels]
        tgt_obj_label_query = label_embeds[label_bs_idx, tgt_obj_labels]

        ### Remark: Compose the query and the key_padding_mask
        noised_query = []
        key_padding_mask = torch.ones((bs, num_queries)).to(device = label_embeds.device) > 0 # 'True' means that this position will not be attended to.
        for i in range(bs):
            zero_query = torch.zeros((num_queries, d_model*2)).to(device = label_embeds.device)
            i_box_query = torch.cat((tgt_sub_bbox_query[pair_idx_list[i]:pair_idx_list[i+1],:], \
                                     tgt_obj_bbox_query[pair_idx_list[i]:pair_idx_list[i+1],:]), dim = 0)
            i_label_query = torch.cat((tgt_sub_label_query[pair_idx_list[i]:pair_idx_list[i+1],:], \
                                       tgt_obj_label_query[pair_idx_list[i]:pair_idx_list[i+1],:]), dim = 0)

            attended_idx = list(range(0, pair_num_list[i])) + list(range(num_queries//2, num_queries//2 + pair_num_list[i]))
            ### Remark by Hangjie: IMPORTANT!!!
            ### The following code is to guard against the scenario when key_padding_mask is filled with 'True'.
            ### Thus, when there is no triplet, we will not fill key_padding_mask with Trues but leave the first mask to False.
            ### According to the calculation of all losses, this won't be calculated into the total loss,
            ### but this can avoid 'nan' of the text encoder.
            zero_query[attended_idx, :] = torch.cat([i_box_query, i_label_query], dim = 1)
            if len(attended_idx) == 0:
                attended_idx = [0,]
                # TODO: Test on 11-13-2022
                # attended_idx = [0, num_pairs]

            key_padding_mask[i, attended_idx] = False # True # False 
            noised_query.append(zero_query) # Make the dimension 512.

        noised_query = torch.stack(noised_query).to(device = label_embeds.device)
        
        ### Remark 1: Generate attn_mask to avoid information leakage (Set all Trues and find Falses) 
        # attn_mask = torch.ones((bs, num_queries, num_queries)).to(device = label_embeds.device) > 0 # Set to all Trues (do not attend)
        
        # TODO: test with attn_mask
        # 1. One mask
        # for i in range(bs):
        #     attn_mask[i,0,:] = False

        # 2. Diagonal 
        # for i in range(bs):
        #     for j in range(num_queries):
        #         attn_mask[i][j][j] = False # 再次核对DN-DETR的代码
        #         if j-1>=0:
        #             attn_mask[i][j][j-1] = False
        #         if j+1<num_queries:
        #             attn_mask[i][j][j+1] = False
        # print(attn_mask.sum(-1))
                
        # 3. Normal
        # for i in range(bs):
        #     sub_labels, obj_labels = targets[i]["sub_labels"], targets[i]["obj_labels"]
        #     sub_boxes, obj_boxes = targets[i]['sub_boxes'], targets[i]['obj_boxes']
        #     for t_s in range(len(sub_labels)):
        #         for s_s in range(len(sub_labels)):
        #             if (sub_labels[t_s] != sub_labels[s_s] or ~torch.eq(sub_boxes[t_s], sub_boxes[s_s]).all()) \
        #                 or t_s == s_s: 
        #             # If the labels and boxes are the same, then we can say they are the same box.
        #             # Another option is to only consider the index to decide whether they are the same box.
        #                 attn_mask[i][t_s][s_s] = False
        #         for s_o in range(len(obj_labels)):
        #             if sub_labels[t_s] != obj_labels[s_o] or ~torch.eq(sub_boxes[t_s], obj_boxes[s_o]).all():
        #                 attn_mask[i][t_s][num_pairs + s_o] = False
        #     for t_o in range(len(obj_labels)):
        #         for s_s in range(len(sub_labels)):
        #             if obj_labels[t_o] != sub_labels[s_s] or ~torch.eq(obj_boxes[t_o], sub_boxes[s_s]).all():
        #                 attn_mask[i][num_pairs + t_o][s_s] = False
        #         for s_o in range(len(obj_labels)):
        #             if (obj_labels[t_o] != obj_labels[s_o] or ~torch.eq(obj_boxes[t_o], obj_boxes[s_o]).all()) \
        #                 or t_o == s_o:
        #                 attn_mask[i][num_pairs + t_o][num_pairs + s_o] = False
            
        #     ### Note: We should consider situations when there is no pair.
        #     # print(sub_labels.shape)
        #     # if len(sub_labels) == 0:
        #     #     attn_mask[i][0][0] = False
        #     #     attn_mask[i][num_pairs + 0][num_pairs + 0] = False
        #     # print((~attn_mask[i]).sum())

        ### Remark 2: Generate attn_mask to avoid information leakage (Set all Falses and find Trues) 
        # Reverse Generation by iteration
        # attn_mask = torch.ones((bs, num_queries, num_queries)).to(device = label_embeds.device) < 0 # Set to all Falses (attend)
        # for i in range(bs):
        #     sub_labels, obj_labels = targets[i]["sub_labels"], targets[i]["obj_labels"]
        #     sub_boxes, obj_boxes = targets[i]['sub_boxes'], targets[i]['obj_boxes']
        #     for t_s in range(len(sub_labels)):
        #         for s_s in range(len(sub_labels)):
        #             if (sub_labels[t_s] == sub_labels[s_s] and torch.eq(sub_boxes[t_s], sub_boxes[s_s]).all()): 
        #             # If the labels and boxes are the same, then we can say they are the same box.
        #             # Another option is to only consider the index to decide whether they are the same box.
        #                 attn_mask[i][t_s][s_s] = True
        #         for s_o in range(len(obj_labels)):
        #             if sub_labels[t_s] != obj_labels[s_o] and torch.eq(sub_boxes[t_s], obj_boxes[s_o]).all():
        #                 attn_mask[i][t_s][num_pairs + s_o] = True
        #     for t_o in range(len(obj_labels)):
        #         for s_s in range(len(sub_labels)):
        #             if obj_labels[t_o] == sub_labels[s_s] and torch.eq(obj_boxes[t_o], sub_boxes[s_s]).all():
        #                 attn_mask[i][num_pairs + t_o][s_s] = True
        #         for s_o in range(len(obj_labels)):
        #             if (obj_labels[t_o] == obj_labels[s_o] and torch.eq(obj_boxes[t_o], obj_boxes[s_o]).all()):
        #                 attn_mask[i][num_pairs + t_o][num_pairs + s_o] = True
        #     for j in range(num_queries):
        #         attn_mask[i][j][j] = False # Make sure that they can attend to themselves
        # print('True ', (attn_mask[-1]).sum())
        # print('False ', (~attn_mask[-1]).sum())
        
        # Reverse Generation by tensor operation
        sub_labels = torch.empty((bs, num_pairs)).fill_(torch.inf).to(device = label_embeds.device)
        obj_labels = torch.empty((bs, num_pairs)).fill_(torch.inf).to(device = label_embeds.device)
        sub_boxes = torch.empty((bs, num_pairs, 4)).fill_(torch.inf).to(device = label_embeds.device)
        obj_boxes = torch.empty((bs, num_pairs, 4)).fill_(torch.inf).to(device = label_embeds.device)
        for i in range(bs):
            sub_labels[i,:pair_num_list[i]] = targets[i]["sub_labels"]
            obj_labels[i,:pair_num_list[i]] = targets[i]["obj_labels"]
            sub_boxes[i,:pair_num_list[i]] = targets[i]['sub_boxes']
            obj_boxes[i,:pair_num_list[i]] = targets[i]['obj_boxes']
        sub_obj_labels = torch.cat((sub_labels, obj_labels), dim = 1)
        sub_obj_boxes = torch.cat((sub_boxes, obj_boxes), dim = 1)

        labels_1 = sub_obj_labels.unsqueeze(dim = 2)
        labels_2 = sub_obj_labels.unsqueeze(dim = 1)
        labels1_not_inf = (labels_1 != torch.inf)
        labels2_not_inf = (labels_2 != torch.inf)
        labels_eq = torch.eq(labels_1, labels_2) & labels1_not_inf & labels2_not_inf

        boxes_1 = sub_obj_boxes.unsqueeze(dim = 2)
        boxes_2 = sub_obj_boxes.unsqueeze(dim = 1)
        boxes1_not_inf = (boxes_1 != torch.inf).all()
        boxes2_not_inf = (boxes_2 != torch.inf).all()
        boxes_eq = torch.eq(boxes_1, boxes_2).all(dim = -1) & boxes1_not_inf & boxes2_not_inf

        label_box_eq = labels_eq & boxes_eq
        
        eye_mask = torch.stack([torch.eye(num_queries).to(device = label_embeds.device) for i in range(bs)], dim = 0) > 0
        label_box_eq = label_box_eq & (~eye_mask)  
        attn_mask = label_box_eq

        ### Expand to (num_heads = 8)
        num_heads = 8
        attn_mask = attn_mask.unsqueeze(dim = 0).expand(num_heads, -1, -1, -1) # (num_heads, bs, num_queries, num_queries)
        attn_mask = attn_mask.transpose(0, 1).contiguous().view(-1, num_queries, num_queries) # (bs*num_heads, num_queries, num_queries)

        # attn_mask = None
        # key_padding_mask = None

        return noised_query, key_padding_mask, attn_mask
    else:
        ### In this case, we do not need to add noise.
        ### We just output the query with gt labels and boxes.
        pair_num_list = [v["sub_boxes"].shape[0] for v in targets]
        pair_idx_list = np.cumsum(pair_num_list)
        pair_idx_list = [0,] + list(pair_idx_list)

        tgt_sub_bbox = torch.cat([v["sub_boxes"] for v in targets])  # cx, cy, w, h
        # assert tgt_bbox.shape[0] == num_pairs * len(targets)
        tgt_sub_bbox_inv = inverse_sigmoid(tgt_sub_bbox)
        tgt_sub_bbox_query = box_embed_func(tgt_sub_bbox_inv)

        tgt_obj_bbox = torch.cat([v["obj_boxes"] for v in targets])  # cx, cy, w, h
        tgt_obj_bbox_inv = inverse_sigmoid(tgt_obj_bbox)
        tgt_obj_bbox_query = box_embed_func(tgt_obj_bbox_inv)

        tgt_sub_labels = torch.cat([v["sub_labels"] for v in targets])
        tgt_obj_labels = torch.cat([v["obj_labels"] for v in targets])
        label_bs_idx = torch.cat([torch.tensor([i,]*v["sub_labels"].shape[0], dtype=torch.int64) for i,v in enumerate(targets)]).to(label_embeds.device)
        if len(tgt_sub_labels)>1:
            if tgt_sub_labels.max() >= label_embeds.shape[1] or tgt_sub_labels.min() < -label_embeds.shape[1]:
                print(tgt_sub_labels)
                assert False
            if label_bs_idx.max() >= label_embeds.shape[0]:
                print(label_bs_idx)
                print(label_embeds.shape)
                assert False

        tgt_sub_label_query = label_embeds[label_bs_idx, tgt_sub_labels]
        tgt_obj_label_query = label_embeds[label_bs_idx, tgt_obj_labels]

        ### Compose the query
        no_noise_query = []
        key_padding_mask = torch.ones((bs, num_queries)).to(device = label_embeds.device) > 0 # 'True' means that this position will not be attended to.
        for i in range(bs):
            zero_query = torch.zeros((num_queries, d_model*2)).to(device = label_embeds.device)
            i_box_query = torch.cat((tgt_sub_bbox_query[pair_idx_list[i]:pair_idx_list[i+1],:], \
                                     tgt_obj_bbox_query[pair_idx_list[i]:pair_idx_list[i+1],:]), dim = 0)
            i_label_query = torch.cat((tgt_sub_label_query[pair_idx_list[i]:pair_idx_list[i+1],:], \
                                       tgt_obj_label_query[pair_idx_list[i]:pair_idx_list[i+1],:]), dim = 0)
            # no_noise_query.append(torch.cat([i_box_query, i_label_query], dim = 1))
            attended_idx = list(range(0, pair_num_list[i])) + list(range(num_queries//2, num_queries//2 + pair_num_list[i]))
            key_padding_mask[i, attended_idx] = False
            zero_query[attended_idx, :] = torch.cat([i_box_query, i_label_query], dim = 1)
            no_noise_query.append(zero_query)

        no_noise_query = torch.stack(no_noise_query).to(device = label_embeds.device)
        
        # ### Remark 2: Generate attn_mask to avoid information leakage (Set all Falses and find Trues)         
        # # Reverse Generation by tensor operation
        # sub_labels = torch.empty((bs, num_pairs)).fill_(torch.inf).to(device = label_embeds.device)
        # obj_labels = torch.empty((bs, num_pairs)).fill_(torch.inf).to(device = label_embeds.device)
        # sub_boxes = torch.empty((bs, num_pairs, 4)).fill_(torch.inf).to(device = label_embeds.device)
        # obj_boxes = torch.empty((bs, num_pairs, 4)).fill_(torch.inf).to(device = label_embeds.device)
        # for i in range(bs):
        #     sub_labels[i,:pair_num_list[i]] = targets[i]["sub_labels"]
        #     obj_labels[i,:pair_num_list[i]] = targets[i]["obj_labels"]
        #     sub_boxes[i,:pair_num_list[i]] = targets[i]['sub_boxes']
        #     obj_boxes[i,:pair_num_list[i]] = targets[i]['obj_boxes']
        # sub_obj_labels = torch.cat((sub_labels, obj_labels), dim = 1)
        # sub_obj_boxes = torch.cat((sub_boxes, obj_boxes), dim = 1)

        # labels_1 = sub_obj_labels.unsqueeze(dim = 2)
        # labels_2 = sub_obj_labels.unsqueeze(dim = 1)
        # labels1_not_inf = (labels_1 != torch.inf)
        # labels2_not_inf = (labels_2 != torch.inf)
        # labels_eq = torch.eq(labels_1, labels_2) & labels1_not_inf & labels2_not_inf

        # boxes_1 = sub_obj_boxes.unsqueeze(dim = 2)
        # boxes_2 = sub_obj_boxes.unsqueeze(dim = 1)
        # boxes1_not_inf = (boxes_1 != torch.inf).all()
        # boxes2_not_inf = (boxes_2 != torch.inf).all()
        # boxes_eq = torch.eq(boxes_1, boxes_2).all(dim = -1) & boxes1_not_inf & boxes2_not_inf

        # label_box_eq = labels_eq & boxes_eq
        
        # eye_mask = torch.stack([torch.eye(num_queries).to(device = label_embeds.device) for i in range(bs)], dim = 0) > 0
        # label_box_eq = label_box_eq & (~eye_mask)  
        # attn_mask = label_box_eq

        # ### Expand to (num_heads = 8)
        # num_heads = 8
        # attn_mask = attn_mask.unsqueeze(dim = 0).expand(num_heads, -1, -1, -1) # (num_heads, bs, num_queries, num_queries)
        # attn_mask = attn_mask.transpose(0, 1).contiguous().view(-1, num_queries, num_queries) # (bs*num_heads, num_queries, num_queries)        
        
        attn_mask = None
        return no_noise_query, key_padding_mask, attn_mask