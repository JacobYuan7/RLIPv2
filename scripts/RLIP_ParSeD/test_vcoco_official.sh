# # ### RLIP-ParSeD (R50)
# # generate_vcoco_official for RLIP-ParSeD
# python generate_vcoco_official.py \
#         --param_path /Path/To/logs/RLIP_PD_v2_VCOCO_R50_VG_RQL_LSE_RPL_20e_L1_20e/checkpoint0019.pth \
#         --save_path /Path/To/jacob/VCOCO_pickle/RLIPv2/RLIP_PD_v2_VCOCO_R50_VG.pickle \
#         --hoi_path /Path/To/data/v-coco \
#         --batch_size 16 \
#         --num_obj_classes 81 \
#         --num_verb_classes 29 \
#         --dec_layers 3 \
#         --enc_layers 6 \
#         --num_queries 200 \
#         --num_feature_levels 4 \
#         --with_box_refine \
#         --RLIP_ParSeD_v2 \
#         --dim_feedforward 1024 \
#         --use_no_obj_token \
#         --fusion_type GLIP_attn \
#         --gating_mechanism VXAc \
#         --verb_query_tgt_type vanilla_MBF \
#         --fusion_interval 2 \
#         --fusion_last_vis \
#         --lang_aux_loss \
        
#         # scaling
#         # --param_path /Path/To/logs/RLIP_PD_v2_VCOCO_R50_VGCOCOO365_RQL_LSE_RPL_20e_L1_20e/checkpoint0019.pth \
#         # --save_path /Path/To/jacob/VCOCO_pickle/RLIPv2/RLIP_PD_v2_VCOCO_R50_VGCOCOO365.pickle \
#         # --param_path /Path/To/logs/RLIP_PD_v2_VCOCO_R50_VG_RQL_LSE_RPL_20e_L1_20e/checkpoint0019.pth \
#         # --save_path /Path/To/jacob/VCOCO_pickle/RLIPv2/RLIP_PD_v2_VCOCO_R50_VG.pickle \
        

# ### RLIP-ParSeD (SwinT)
# # generate_vcoco_official for RLIP-ParSeD
# python generate_vcoco_official.py \
#         --param_path /Path/To/logs/RLIP_PD_v2_VCOCO_SwinT_VGCOCOO365_RQL_LSE_RPL_20e_L1_20e/checkpoint0019.pth \
#         --save_path /Path/To/jacob/VCOCO_pickle/RLIPv2/RLIP_PD_v2_VCOCO_SwinT_VGCOCOO365.pickle \
#         --hoi_path /Path/To/data/v-coco \
#         --backbone swin_tiny \
#         --drop_path_rate 0.2 \
#         --batch_size 16 \
#         --num_obj_classes 81 \
#         --num_verb_classes 29 \
#         --dec_layers 3 \
#         --enc_layers 6 \
#         --num_queries 200 \
#         --num_feature_levels 4 \
#         --with_box_refine \
#         --RLIP_ParSeD_v2 \
#         --dim_feedforward 2048 \
#         --use_no_obj_token \
#         --fusion_type GLIP_attn \
#         --gating_mechanism VXAc \
#         --verb_query_tgt_type vanilla_MBF \
#         --fusion_interval 2 \
#         --fusion_last_vis \
#         --lang_aux_loss \


### RLIP-ParSeD (SwinL)
# generate_vcoco_official for RLIP-ParSeD
python generate_vcoco_official.py \
        --param_path /Path/To/logs/RLIP_PD_v2_VCOCO_SwinL_VGCOCOO365_RQL_LSE_RPL_20e_L1_20e/checkpoint0019.pth \
        --save_path /Path/To/jacob/VCOCO_pickle/RLIPv2/RLIP_PD_v2_VCOCO_SwinL_VGCOCOO365.pickle \
        --hoi_path /Path/To/data/v-coco \
        --backbone swin_large \
        --drop_path_rate 0.5 \
        --batch_size 16 \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --dec_layers 3 \
        --enc_layers 6 \
        --num_queries 200 \
        --num_feature_levels 4 \
        --with_box_refine \
        --RLIP_ParSeD_v2 \
        --dim_feedforward 2048 \
        --use_no_obj_token \
        --fusion_type GLIP_attn \
        --gating_mechanism VXAc \
        --verb_query_tgt_type vanilla_MBF \
        --fusion_interval 2 \
        --fusion_last_vis \
        --lang_aux_loss \