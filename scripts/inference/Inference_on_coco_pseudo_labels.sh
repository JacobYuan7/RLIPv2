# inference_on_custom_imgs_hico.py for RLIP-ParSe (RLIP_ParSe)
python inference_on_custom_imgs_pseudo_coco.py \
        --param_path /Path/To/logs/RLIP_PD_v2_verbtgtv5_RQL_LSE_RPL_COCO_20e_GLIP2aux_L1/checkpoint0019.pth \
        --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_pseudo-labels_SceneGraph_model_large_caption_nucleus10_thre20.json \
        --batch_size 1 \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --dec_layers 3 \
        --enc_layers 6 \
        --num_queries 200 \
        --use_no_obj_token \
        --backbone resnet50 \
        --RLIP_ParSeD_v2 \
        --with_box_refine \
        --num_feature_levels 4 \
        --dim_feedforward 1024 \
        --subject_class \
        --fusion_type GLIP_attn \
        --gating_mechanism VXAc \
        --verb_query_tgt_type vanilla_MBF \
        --fusion_interval 2 \
        --fusion_last_vis \
        --lang_aux_loss \


# # RLIPv2_ParSeDA
# # inference_on_custom_imgs_hico.py for RLIP-ParSe (RLIP_ParSe)
# python inference_on_custom_imgs_pseudo_coco.py \
#         --param_path /Path/To/logs/RLIP_PDA_v2_verbtgtv5_RQL_LSE_RPL_COCO_20e_GLIP2aux_L1/checkpoint0019.pth \
#         --save_path custom_imgs/result/custom_imgs.pickle \
#         --batch_size 1 \
#         --RLIP_ParSe \
#         --num_obj_classes 80 \
#         --num_verb_classes 117 \
#         --dec_layers 3 \
#         --enc_layers 6 \
#         --num_queries 128 \
#         --use_no_obj_token \
#         --backbone resnet50 \
#         --RLIP_ParSeDA_v2 \
#         --with_box_refine \
#         --num_feature_levels 4 \
#         --num_patterns 0 \
#         --pe_temperatureH 20 \
#         --pe_temperatureW 20 \
#         --dim_feedforward 2048 \
#         --dropout 0.0 \
#         --subject_class \
#         --use_no_obj_token \
#         --fusion_type GLIP_attn \
#         --gating_mechanism VXAc \
#         --verb_query_tgt_type vanilla_MBF \
#         --fusion_interval 2 \
#         --fusion_last_vis \
#         --lang_aux_loss \


