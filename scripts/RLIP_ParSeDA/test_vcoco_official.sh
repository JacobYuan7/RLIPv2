# ### RLIP-ParSeDA (R50)
# # generate_vcoco_official for RLIP-ParSeDA
# python generate_vcoco_official.py \
#         --param_path /Path/To/logs/RLIP_PDA_v2_VCOCO_R50_VGCOO365_COO365det_RQL_LSE_RPL_20e_L1_20e/checkpoint0019.pth \
#         --save_path /Path/To/jacob/VCOCO_pickle/RLIPv2/RLIP_PDA_v2_VCOCO_R50_VGCOCOO365.pickle \
#         --hoi_path /Path/To/data/v-coco \
#         --batch_size 16 \
#         --num_obj_classes 81 \
#         --num_verb_classes 29 \
#         --dec_layers 3 \
#         --enc_layers 6 \
#         --num_queries 200 \
#         --num_feature_levels 4 \
#         --with_box_refine \
#         --RLIP_ParSeDA_v2 \
#         --num_patterns 0 \
#         --pe_temperatureH 20 \
#         --pe_temperatureW 20 \
#         --dim_feedforward 2048 \
#         --dropout 0.0 \
#         --use_no_obj_token \
#         --fusion_type GLIP_attn \
#         --gating_mechanism VXAc \
#         --verb_query_tgt_type vanilla_MBF \
#         --fusion_interval 2 \
#         --fusion_last_vis \
#         --lang_aux_loss \
#         # --param_path /Path/To/logs/RLIP_PDA_v2_VCOCO_R50_VG_RQL_LSE_RPL_20e_L1_20e/checkpoint.pth \
#         # --save_path /Path/To/jacob/VCOCO_pickle/RLIPv2/RLIP_PDA_v2_VCOCO_R50_VG.pickle \
#         # --param_path /Path/To/logs/RLIP_PDA_v2_VCOCO_R50_VGCOCO_RQL_LSE_RPL_20e_L1_20e/checkpoint0019.pth \
#         # --save_path /Path/To/jacob/VCOCO_pickle/RLIPv2/RLIP_PDA_v2_VCOCO_R50_VGCOCO.pickle \
#         # --param_path /Path/To/logs/RLIP_PDA_v2_VCOCO_R50_VGCOO365_COO365det_RQL_LSE_RPL_20e_L1_20e/checkpoint0019.pth \
#         # --save_path /Path/To/jacob/VCOCO_pickle/RLIPv2/RLIP_PDA_v2_VCOCO_R50_VGCOCOO365.pickle \



# ### RLIP-ParSeDA (SwinT)
# # generate_vcoco_official for RLIP-ParSeDA
# python generate_vcoco_official.py \
#         --param_path /Path/To/logs/RLIP_PDA_v2_VCOCO_SwinT_VGCOCOO365_RQL_LSE_RPL_20e_L1_20e/checkpoint0019.pth \
#         --save_path /Path/To/jacob/VCOCO_pickle/RLIPv2/RLIP_PDA_v2_VCOCO_SwinT_VGCOCOO365.pickle \
#         --hoi_path /Path/To/data/v-coco \
#         --backbone swin_tiny \
#         --batch_size 16 \
#         --num_obj_classes 81 \
#         --num_verb_classes 29 \
#         --dec_layers 3 \
#         --enc_layers 6 \
#         --num_queries 200 \
#         --num_feature_levels 4 \
#         --with_box_refine \
#         --RLIP_ParSeDA_v2 \
#         --num_patterns 0 \
#         --pe_temperatureH 20 \
#         --pe_temperatureW 20 \
#         --dim_feedforward 2048 \
#         --dropout 0.0 \
#         --drop_path_rate 0.2 \
#         --use_no_obj_token \
#         --fusion_type GLIP_attn \
#         --gating_mechanism VXAc \
#         --verb_query_tgt_type vanilla_MBF \
#         --fusion_interval 2 \
#         --fusion_last_vis \
#         --lang_aux_loss \


### RLIP-ParSeDA (SwinL)
# generate_vcoco_official for RLIP-ParSeDA
python generate_vcoco_official.py \
        --param_path /Path/To/logs/RLIP_PDA_v2_VCOCO_SwinL_VGCOCOO365_RQL_LSE_RPL_20e_L1_20e/checkpoint0019.pth \
        --save_path /Path/To/jacob/VCOCO_pickle/RLIPv2/RLIP_PDA_v2_VCOCO_SwinL_VGCOCOO365.pickle \
        --hoi_path /Path/To/data/v-coco \
        --backbone swin_large \
        --batch_size 16 \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --dec_layers 3 \
        --enc_layers 6 \
        --num_queries 200 \
        --num_feature_levels 4 \
        --with_box_refine \
        --RLIP_ParSeDA_v2 \
        --num_patterns 0 \
        --pe_temperatureH 20 \
        --pe_temperatureW 20 \
        --dim_feedforward 2048 \
        --dropout 0.0 \
        --drop_path_rate 0.5 \
        --use_no_obj_token \
        --fusion_type GLIP_attn \
        --gating_mechanism VXAc \
        --verb_query_tgt_type vanilla_MBF \
        --fusion_interval 2 \
        --fusion_last_vis \
        --lang_aux_loss \