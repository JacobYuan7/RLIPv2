# ------------------------------------------------------------------------
# RLIPv2: Fast Scaling of Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import argparse

import torch
from torch import nn


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--load_path', type=str, required=True,
    )
    parser.add_argument(
        '--save_path', type=str, required=True,
    )
    parser.add_argument(
        '--dataset', type=str, default='hico',
    )
    parser.add_argument(
        '--num_ref_points', type=int, default = 4, help="Number of verb classes"
    )
    parser.add_argument(
        '--with_box_refine', default=False, action='store_true'
    )
    parser.add_argument(
        '--drop_class_embed', default=False, action='store_true'
    )
    parser.add_argument(
        '--DETReg', default=False, action='store_true'
    )
    parser.add_argument(
        '--swin_backbone', default=False, action='store_true'
    )
    parser.add_argument('--SepDDETRHOIv3', action = 'store_true',
                        help='SepDDETRHOIv3: Fully disentangled decoding by DDETRHOI')
    

    args = parser.parse_args()

    return args


def main(args):
    ps = torch.load(args.load_path)

    obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
               82, 84, 85, 86, 87, 88, 89, 90]

    # For no pair
    obj_ids.append(91)  # This corresponds to the background class in DETR. 
                        # But Deformable DETR uses sigmoid + BCE, without a background class.
    background_class = nn.Linear(256, 1, bias = True) # .cuda()

    
    mmdet = 0
    if 'state_dict' in ps.keys():
        # it indicates that it's from mmdetection
        mmdet = 1
        ps['model'] = ps.pop('state_dict')
        new_mdoel = {}
        for i,j in ps['model'].items():
            if 'bbox_head' in i:
                new_mdoel[i.replace('bbox_head.', '')] = j
            else:
                new_mdoel[i] = j
        ps['model'] = new_mdoel

    # for i,j in ps['model'].items():
    #     if 'backbone' in i:
    #         print(i)
    #     if 'bbox' in i:
    #         print(i)
    #     if 'points' in i:
    #         print(i)

    if args.SepDDETRHOIv3:
        for i in list(ps['model'].keys()):
            if 'transformer.encoder' in i:
                ps['model'][i.replace('transformer.encoder', 'transformer.ho_encoder')] = ps['model'][i].clone()
            if 'transformer.decoder' in i:
                ps['model'][i.replace('transformer.decoder', 'transformer.ho_decoder')] = ps['model'][i].clone()
                ps['model'][i.replace('transformer.decoder', 'transformer.verb_decoder')] = ps['model'][i].clone()
        
        if not mmdet:
            for i in range(6):
                param_key = ['weight', 'bias']
                bbox_key = ['sub_bbox_embed.','obj_bbox_embed.']
                for j in range(3):
                    ps['model'][bbox_key[0] + str(i) + '.' + 'layers.' + str(j) + '.' +'weight'] = \
                            ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'weight']
                    ps['model'][bbox_key[1] + str(i) + '.' + 'layers.' + str(j) + '.' +'weight'] = \
                            ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'weight']
                    ps['model'][bbox_key[0] + str(i) + '.' + 'layers.' + str(j) + '.' +'bias'] = \
                            ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'bias']
                    ps['model'][bbox_key[1] + str(i) + '.' + 'layers.' + str(j) + '.' +'bias'] = \
                            ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'bias']

                if args.with_box_refine:
                    bbox_key_list = [['transformer.ho_decoder.sub_bbox_embed.','transformer.ho_decoder.obj_bbox_embed.'],
                                ['transformer.verb_decoder.sub_bbox_embed.','transformer.verb_decoder.obj_bbox_embed.']]
                    for bbox_key in bbox_key_list:
                    # bbox_key = bbox_key_list[0] if i <=2 else bbox_key_list[1]
                        if not args.DETReg:
                            for j in range(3):
                                ps['model'][bbox_key[0] + str(i) + '.' + 'layers.' + str(j) + '.' +'weight'] = \
                                        ps['model']['transformer.decoder.bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'weight']
                                ps['model'][bbox_key[1] + str(i) + '.' + 'layers.' + str(j) + '.' +'weight'] = \
                                        ps['model']['transformer.decoder.bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'weight']
                                ps['model'][bbox_key[0] + str(i) + '.' + 'layers.' + str(j) + '.' +'bias'] = \
                                        ps['model']['transformer.decoder.bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'bias']
                                ps['model'][bbox_key[1] + str(i) + '.' + 'layers.' + str(j) + '.' +'bias'] = \
                                        ps['model']['transformer.decoder.bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'bias']
                        else:
                            for j in range(3):
                                ps['model'][bbox_key[0] + str(i) + '.' + 'layers.' + str(j) + '.' +'weight'] = \
                                        ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'weight']
                                ps['model'][bbox_key[1] + str(i) + '.' + 'layers.' + str(j) + '.' +'weight'] = \
                                        ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'weight']
                                ps['model'][bbox_key[0] + str(i) + '.' + 'layers.' + str(j) + '.' +'bias'] = \
                                        ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'bias']
                                ps['model'][bbox_key[1] + str(i) + '.' + 'layers.' + str(j) + '.' +'bias'] = \
                                        ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'bias']
                        
                if not args.drop_class_embed:
                    class_key = ['obj_class_embed.']
                    ps['model'][class_key[0] + str(i) + '.' + 'weight'] = \
                            torch.cat((ps['model']['class_embed.' + str(i) + '.' +'weight'].clone(), background_class.weight.clone()), dim = 0)[obj_ids]
                    ps['model'][class_key[0] + str(i) + '.' + 'bias'] = \
                            torch.cat((ps['model']['class_embed.' + str(i) + '.' +'bias'].clone(), background_class.bias.clone()), dim = 0)[obj_ids]
        
        if not args.swin_backbone:
            ps['model']['transformer.reference_points_sub.weight'] = ps['model']['transformer.reference_points.weight']
            ps['model']['transformer.reference_points_sub.bias'] = ps['model']['transformer.reference_points.bias']
            ps['model']['transformer.reference_points_obj.weight'] = ps['model']['transformer.reference_points.weight']
            ps['model']['transformer.reference_points_obj.bias'] = ps['model']['transformer.reference_points.bias']

        if not args.swin_backbone:
            if not mmdet:
                query_dimension = ps['model']['query_embed.weight'].shape[1]
                ps['model']['query_embed.weight'] = torch.cat((ps['model']['query_embed.weight'], 
                                                            ps['model']['query_embed.weight'][:, query_dimension//2:]), dim = 1)
            else:
                query_dimension = ps['model']['query_embedding.weight'].shape[1]
                ps['model']['query_embedding.weight'] = torch.cat((ps['model']['query_embedding.weight'], 
                                                            ps['model']['query_embedding.weight'][:, query_dimension//2:]), dim = 1)
        else:
            ps['model']['query_embed.weight'] = torch.cat((ps['model']['query_embed.weight'], 
                                                           ps['model']['query_embed.weight'],
                                                           ps['model']['query_embed.weight']), dim = 1)

    else:
        for i in range(6):
            param_key = ['weight', 'bias']
            bbox_key = ['sub_bbox_embed.','obj_bbox_embed.']
            for j in range(3):
                ps['model'][bbox_key[0] + str(i) + '.' + 'layers.' + str(j) + '.' +'weight'] = \
                        ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'weight']
                ps['model'][bbox_key[1] + str(i) + '.' + 'layers.' + str(j) + '.' +'weight'] = \
                        ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'weight']
                ps['model'][bbox_key[0] + str(i) + '.' + 'layers.' + str(j) + '.' +'bias'] = \
                        ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'bias']
                ps['model'][bbox_key[1] + str(i) + '.' + 'layers.' + str(j) + '.' +'bias'] = \
                        ps['model']['bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'bias']
            
            # Test the use of 'transformer.decoder.sub_bbox_embed.' and 'transformer.decoder.obj_bbox_embed.'
            if args.with_box_refine:
                bbox_key = ['transformer.decoder.sub_bbox_embed.','transformer.decoder.obj_bbox_embed.']
                for j in range(3):
                    ps['model'][bbox_key[0] + str(i) + '.' + 'layers.' + str(j) + '.' +'weight'] = \
                            ps['model']['transformer.decoder.bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'weight']
                    ps['model'][bbox_key[1] + str(i) + '.' + 'layers.' + str(j) + '.' +'weight'] = \
                            ps['model']['transformer.decoder.bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'weight']
                    ps['model'][bbox_key[0] + str(i) + '.' + 'layers.' + str(j) + '.' +'bias'] = \
                            ps['model']['transformer.decoder.bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'bias']
                    ps['model'][bbox_key[1] + str(i) + '.' + 'layers.' + str(j) + '.' +'bias'] = \
                            ps['model']['transformer.decoder.bbox_embed.' + str(i) + '.' + 'layers.' + str(j) + '.' +'bias']

            class_key = ['obj_class_embed.']
            ps['model'][class_key[0] + str(i) + '.' + 'weight'] = \
                    torch.cat((ps['model']['class_embed.' + str(i) + '.' +'weight'].clone(), background_class.weight.clone()), dim = 0)[obj_ids]
            ps['model'][class_key[0] + str(i) + '.' + 'bias'] = \
                    torch.cat((ps['model']['class_embed.' + str(i) + '.' +'bias'].clone(), background_class.bias.clone()), dim = 0)[obj_ids]
            # print(type(ps['model']['class_embed.' + str(i) + '.' +'weight']))
            # print(ps['model']['class_embed.' + str(i) + '.' +'weight'].shape)

        if args.num_ref_points == 2:
            ps['model']['transformer.reference_points_subobj.weight'] = ps['model']['transformer.reference_points.weight']
            ps['model']['transformer.reference_points_subobj.bias'] = ps['model']['transformer.reference_points.bias']
        elif args.num_ref_points == 4:
            ps['model']['transformer.reference_points_subobj.weight'] = torch.cat((ps['model']['transformer.reference_points.weight'], ps['model']['transformer.reference_points.weight']), dim = 0)
            ps['model']['transformer.reference_points_subobj.bias'] = torch.cat((ps['model']['transformer.reference_points.bias'], ps['model']['transformer.reference_points.bias']), dim = 0)
    
        # ps['model']['sub_bbox_embed.layers.0.weight'] = ps['model']['bbox_embed.layers.0.weight'].clone()
        # ps['model']['sub_bbox_embed.layers.0.bias'] = ps['model']['bbox_embed.layers.0.bias'].clone()
        # ps['model']['sub_bbox_embed.layers.1.weight'] = ps['model']['bbox_embed.layers.1.weight'].clone()
        # ps['model']['sub_bbox_embed.layers.1.bias'] = ps['model']['bbox_embed.layers.1.bias'].clone()
        # ps['model']['sub_bbox_embed.layers.2.weight'] = ps['model']['bbox_embed.layers.2.weight'].clone()
        # ps['model']['sub_bbox_embed.layers.2.bias'] = ps['model']['bbox_embed.layers.2.bias'].clone()

        # ps['model']['obj_bbox_embed.layers.0.weight'] = ps['model']['bbox_embed.layers.0.weight'].clone()
        # ps['model']['obj_bbox_embed.layers.0.bias'] = ps['model']['bbox_embed.layers.0.bias'].clone()
        # ps['model']['obj_bbox_embed.layers.1.weight'] = ps['model']['bbox_embed.layers.1.weight'].clone()
        # ps['model']['obj_bbox_embed.layers.1.bias'] = ps['model']['bbox_embed.layers.1.bias'].clone()
        # ps['model']['obj_bbox_embed.layers.2.weight'] = ps['model']['bbox_embed.layers.2.weight'].clone()
        # ps['model']['obj_bbox_embed.layers.2.bias'] = ps['model']['bbox_embed.layers.2.bias'].clone()

        # ps['model']['obj_class_embed.weight'] = ps['model']['class_embed.weight'].clone()[obj_ids]
        # ps['model']['obj_class_embed.bias'] = ps['model']['class_embed.bias'].clone()[obj_ids]
        
        # print(ps['model']['class_embed.weight'].shape)

    if args.dataset == 'vcoco':
        for i in range(6):
            obj_weight = 'obj_class_embed.{:d}.weight'.format(i)
            obj_bias = 'obj_class_embed.{:d}.bias'.format(i)

            l = nn.Linear(ps['model'][obj_weight].shape[1], 1)
            l.to(ps['model'][obj_weight].device)
            ps['model'][obj_weight] = torch.cat((
                ps['model'][obj_weight][:-1], l.weight, ps['model'][obj_weight][[-1]]))
            ps['model'][obj_bias] = torch.cat(
                (ps['model'][obj_bias][:-1], l.bias, ps['model'][obj_bias][[-1]]))

    torch.save(ps, args.save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
