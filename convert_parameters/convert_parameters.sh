# SepDDETRHOIv3 with DETReg
# DETReg does not inocrporate the bbox refinement, thus we need to initialize the transformer,decoder.bbox_embed with bbox_embed
# python convert_parameters_DDETR.py \
#         --load_path /Path/To/params/DETReg_checkpoint_imagenet.pth \
#         --save_path /Path/To/params/DETReg_checkpoint_imagenet_SepDDETRHOIv3_converted.pth \
#         --SepDDETRHOIv3 \
#         --num_ref_points 2 \
#         --with_box_refine \
#         --DETReg \

# # SepDDETRHOIv3 with DETReg
# # DETReg does not inocrporate the bbox refinement, thus we need to initialize the transformer,decoder.bbox_embed with bbox_embed
# python convert_parameters_DDETR.py \
#         --load_path /Path/To/params/DETReg_full_coco_finetune.pth \
#         --save_path /Path/To/params/DETReg_full_coco_finetune_SepDDETRHOIv3_converted.pth \
#         --SepDDETRHOIv3 \
#         --num_ref_points 2 \
#         --with_box_refine \
#         --DETReg \


# SepDDETRHOIv3
# python convert_parameters_DDETR.py \
#         --load_path /Path/To/logs/DeformableDETR_vg_v1/checkpoint0049.pth \
#         --save_path /Path/To/params/r50_deformable_detr_vcoco_plus_iterative_bbox_refinement_SepDDETRHOIv3_vg_v1.pth \
#         --SepDDETRHOIv3 \
#         --with_box_refine \
#         --num_ref_points 2 \
#         --dataset vcoco \
# python convert_parameters_DDETR.py \
#         --load_path /Path/To/params/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
#         --save_path /Path/To/params/r50_deformable_detr_vcoco_plus_iterative_bbox_refinement-checkpoint_numrefpoints4_SepDDETRHOIv3_converted.pth \
#         --SepDDETRHOIv3 \
#         --with_box_refine \
#         --num_ref_points 2 \
#         --dataset vcoco \
# python convert_parameters_DDETR.py \
#         --load_path /Path/To/params/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
#         --save_path /Path/To/params/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_numrefpoints4_SepDDETRHOIv3_converted.pth \
#         --SepDDETRHOIv3 \
#         --with_box_refine \
#         --num_ref_points 2 \
# python convert_parameters_DDETR.py \
#         --load_path /Path/To/params/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
#         --save_path /Path/To/params/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_numrefpoints4_SepDDETRHOIv3_converted_no_obj_class.pth \
#         --SepDDETRHOIv3 \
#         --with_box_refine \
#         --num_ref_points 2 \
#         --drop_class_embed \
# python convert_parameters_DDETR.py \
#         --load_path /mnt/data-nas/fengtao/data_nas/model_zoo/mmdet/coco_vg/deformable_detr_vg_v2/latest.pth \
#         --save_path /Path/To/params/r50_deformable_detr_plus_iterative_bbox_refinement_SepDDETRHOIv3_vg_v2.pth \
#         --SepDDETRHOIv3 \
#         --with_box_refine \
#         --num_ref_points 2 \
# python convert_parameters_DDETR.py \
#         --load_path /Path/To/logs/DeformableDETR_vg_v1/checkpoint0049.pth \
#         --save_path /Path/To/params/r50_deformable_detr_plus_iterative_bbox_refinement_SepDDETRHOIv3_vg_v1.pth \
#         --SepDDETRHOIv3 \
#         --with_box_refine \
#         --num_ref_points 2 \
#         --drop_class_embed \

# R50 COCO detection pre-training
# python convert_parameters_DABDDETR.py \
#         --load_path /Path/To/params/DAB/dab_deformable/R50_v2/checkpoint.pth \
#         --save_path /Path/To/params/r50_dab_deformable_detr_plus_iterative_bbox_refinement-checkpoint_converted.pth \
#         --ParSeDABDDETR \
#         --with_box_refine \
#         --num_ref_points 2 \

# # R50 COCO+O365 detection pre-training
# python convert_parameters_DABDDETR.py \
#         --load_path /Path/To/logs/dab_deformable_detr/r50_cocoo365_lr141_dp0_mqs_lft_dab_deformable_detr_plus_iterative_bbox_refinement_36eps/checkpoint.pth \
#         --save_path /Path/To/params/r50_cocoo365_lr141_dp0_mqs_lft_dab_deformable_detr_plus_iterative_bbox_refinement_36eps_converted.pth \
#         --ParSeDABDDETR \
#         --with_box_refine \
#         --num_ref_points 2 \

# # Swin-T COCO detection pre-training
# python convert_parameters_DABDDETR.py \
#         --load_path /Path/To/logs/dab_deformable_detr/swin_tiny_drop_path0.2_dp0_mqs_lft_dab_deformable_detr_plus_iterative_bbox_refinement_50eps/checkpoint.pth \
#         --save_path /Path/To/params/swin_tiny_drop_path0.2_dp0_mqs_lft_dab_deformable_detr_plus_iterative_bbox_refinement_50eps_converted.pth \
#         --ParSeDABDDETR \
#         --with_box_refine \
#         --num_ref_points 2 \

# Swin-T COCO+O365 detection pre-training
# python convert_parameters_DABDDETR.py \
#         --load_path /Path/To/logs/dab_deformable_detr/swin_tiny_cocoo365_drop_path0.2_dp0_mqs_lft_dab_deformable_detr_plus_iterative_bbox_refinement_36eps/checkpoint.pth \
#         --save_path /Path/To/params/swin_tiny_cocoo365_drop_path0.2_dp0_mqs_lft_dab_deformable_detr_plus_iterative_bbox_refinement_36eps_converted.pth \
#         --ParSeDABDDETR \
#         --with_box_refine \
#         --num_ref_points 2 \

# # Swin-L COCO detection pre-training
# python convert_parameters_DABDDETR.py \
#         --load_path /Path/To/logs/dab_deformable_detr/swin_large_drop_path0.5_dp0_mqs_lft_dab_deformable_detr_plus_iterative_bbox_refinement_50eps/checkpoint.pth \
#         --save_path /Path/To/params/swin_large_drop_path0.5_dp0_mqs_lft_dab_deformable_detr_plus_iterative_bbox_refinement_50eps_converted.pth \
#         --ParSeDABDDETR \
#         --with_box_refine \
#         --num_ref_points 2 \

# Swin-L COCO+O365 detection pre-training
python convert_parameters_DABDDETR.py \
        --load_path /Path/To/logs/dab_deformable_detr/swin_large_cocoo365_bs64_lr141_drop_path0.5_dp0_mqs_lft_dab_deformable_detr_plus_iterative_bbox_refinement_36eps/checkpoint.pth \
        --save_path /Path/To/params/swin_large_cocoo365_bs64_lr141_drop_path0.5_dp0_mqs_lft_dab_deformable_detr_plus_iterative_bbox_refinement_36eps_converted.pth \
        --ParSeDABDDETR \
        --with_box_refine \
        --num_ref_points 2 \


# python convert_parameters_DDETR.py \
#         --load_path /Path/To/params/swin_tiny_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth \
#         --save_path /Path/To/params/swin_tiny_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps_SepDDETRHOIv3_converted.pth \
#         --SepDDETRHOIv3 \
#         --with_box_refine \
#         --num_ref_points 2 \
#         --swin_backbone \

# python convert_parameters_DDETR.py \
#         --load_path /Path/To/params/drop_path0.5_swin_large_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth \
#         --save_path /Path/To/params/drop_path0.5_swin_large_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps_SepDDETRHOIv3_converted.pth \
#         --SepDDETRHOIv3 \
#         --with_box_refine \
#         --num_ref_points 2 \
#         --swin_backbone \

# Below codes are used to convert to ParSeDDETR parameters. \
# We use "--SepDDETRHOIv3" because they use the same structure. We use "--drop_class_embed" because cross-modal pretraining does not require this. 
# python convert_parameters_DDETR.py \
#         --load_path /Path/To/logs/DeformableDETR_vg_v1_25ep/checkpoint0024.pth \
#         --save_path /Path/To/params/r50_deformable_detr_plus_iterative_bbox_refinement_ParSeDDETRHOI_vg_v1_25ep.pth \
#         --SepDDETRHOIv3 \
#         --with_box_refine \
#         --num_ref_points 2 \
#         --drop_class_embed \


# python convert_parameters.py \
#         --load_path params/detr-r50-e632da11.pth \
#         --save_path params/detr-r50-pre-hico.pth
# python convert_parameters.py \
#         --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#         --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted.pth \
# python convert_parameters.py \
#         --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#         --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted_encoder_decoder_objclass.pth \
# python convert_parameters.py \
#         --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#         --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted_encoder_decoder_subobjbbox.pth \
# python convert_parameters.py \
#         --load_path /Path/To/params/up-detr-pre-training-60ep-imagenet.pth \
#         --save_path /Path/To/params/up-detr-pre-training-60ep-imagenet_converted.pth \
# python convert_parameters.py \
#         --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#         --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted_SeqTransformer.pth \
# python convert_parameters.py \
#         --load_path /Path/To/params/detr-r50-e632da11.pth \
#         --save_path /Path/To/params/detr-r50-pre-hico_SeqTransformer.pth
# python convert_parameters.py \
#         --load_path /Path/To/params/detr-r50-e632da11.pth \
#         --save_path /Path/To/params/detr-r50-pre-hico_SepTransformer.pth
# python convert_parameters.py \
#          --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#          --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted_SepTransformer.pth \
# python convert_parameters.py \
#          --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#          --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted_SepTransformer_query300.pth \
# python convert_parameters.py \
#          --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#          --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted_SepTransformer_query400.pth \
# python convert_parameters.py \
#          --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#          --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted_SepTransformer_query200_withoutsharedecoder.pth \
# python convert_parameters.py \
#          --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#          --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted_SepTransformer_query128_withoutsharedecoder.pth \
# python convert_parameters.py \
#         --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#         --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted_SepTransformerv2_query128.pth
# python convert_parameters.py \
#         --load_path /Path/To/params/detr-r50-e632da11.pth \
#         --save_path /Path/To/params/detr-r50-pre-hico_SepTransformerv2_query200.pth \
# python convert_parameters.py \
#         --load_path /Path/To/params/detr-r101-2c7b67e5.pth \
#         --save_path /Path/To/params/detr-r101-pre-hico_SepTransformerv2_query128.pth \
# python convert_parameters.py \
#         --load_path /Path/To/params/detr-r101-2c7b67e5.pth \
#         --save_path /Path/To/params/detr-r101-pre-hico_SepTransformerv2_query200_no_obj_class.pth \
# python convert_parameters.py \
#         --load_path /Path/To/params/detr-r50-e632da11.pth \
#         --save_path /Path/To/params/detr-r50-pre-vcoco_SepTransformerv2_query200.pth \
#         --dataset vcoco \
# python convert_parameters.py \
#         --load_path /Path/To/params/detr-r101-2c7b67e5.pth \
#         --save_path /Path/To/params/detr-r101-pre-vcoco_SepTransformerv2_query200.pth \
#         --dataset vcoco \
# python convert_parameters.py \
#         --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#         --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted_SepTransformerv3_query128.pth
# python convert_parameters.py \
#         --load_path /Path/To/params/detr-r50-e632da11.pth \
#         --save_path /Path/To/params/detr-r50-pre-hico_CDN_query64.pth \
# python convert_parameters.py \
#         --load_path /Path/To/params/detr-r50-e632da11.pth \
#         --save_path /Path/To/params/detr-r50-pre-hico_CDN_query64_no_obj_class.pth
# python convert_parameters.py \
#         --load_path /Path/To/params/up-detr-coco-fine-tuned-300ep.pth \
#         --save_path /Path/To/params/up-detr-coco-fine-tuned-300ep_converted_CDN_query64.pth
# OCN public parameters
# python convert_parameters.py \
#         --load_path /Path/To/params/detr-r50-e632da11.pth \
#         --save_path /Path/To/params/detr-r50-pre-hico_OCN.pth \
# python convert_parameters.py \
#         --load_path /Path/To/params/detr-r50-e632da11.pth \
#         --save_path /Path/To/params/detr-r50-pre-vcoco_OCN.pth \
#         --dataset vcoco \

# python convert_parameters.py \
#         --load_path /data-nas/peizhi/jacob/params/detr-r101-2c7b67e5.pth \
#         --save_path /data-nas/peizhi/jacob/params/detr-r101-pre-hico.pth

# python convert_parameters.py \
#         --load_path /data-nas/peizhi/jacob/params/detr-r101-2c7b67e5.pth \
#         --save_path /data-nas/peizhi/params/detr-r101-pre-vcoco.pth \
#         --dataset vcoco

# python convert_vcoco_annotations.py \
#         --load_path data/v-coco/data \
#         --prior_path data/v-coco/prior.pickle \
#         --save_path data/v-coco/annotations



# pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"

# git clone --recursive https://github.com/s-gupta/v-coco.git