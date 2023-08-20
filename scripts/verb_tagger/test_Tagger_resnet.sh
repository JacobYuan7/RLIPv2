# Run the Verb Tagger on OD datasets
python generate_relations_using_verb_tagger.py \
        --param_path /Path/To/logs/RLIP_PD_v2_Tagger2_COCO_20e_bs64_GLIP2aux_L1/checkpoint0019.pth \
        --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_BLIP1_nucleus10_BLIPdense_thre05.json \
        --dataset_file coco \
        --coco_path /Path/To/data/coco2017 \
        --dec_layers 3 \
        --enc_layers 6 \
        --num_queries 200 \
        --backbone resnet50 \
        --dim_feedforward 1024 \
        --num_feature_levels 4 \
        --with_box_refine \
        --RLIP_ParSeD_v2 \
        --fusion_type GLIP_attn \
        --fusion_interval 2 \
        --fusion_last_vis \
        --verb_tagger \
        --batch_size 1 \
        --img_inference_batch 16 \
        --num_workers 4 \
        --use_no_obj_token \
        --lang_aux_loss \
        --subject_class \
        --relation_threshold 0.05 \
        --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP2_captions/SceneGraph_BLIP1_nucleus10_BLIPdense_trainval2017_Paraphrases_rel_texts_for_coco_images.json \
        
        # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_BLIP1_nucleus10_BLIP2_nucleus10_thre05.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP2_captions/SceneGraph_BLIP1_nucleus10_BLIP2_nucleus10_trainval2017_Paraphrases_rel_texts_for_coco_images.json \
        # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_coco_opt2.7b_nucleus10_thre05.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP2_captions/SceneGraph_coco_opt2.7b_nucleus10_trainval2017_Paraphrases_rel_texts_for_coco_images.json \
        # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_coco_opt2.7b_beam_thre05.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP2_captions/SceneGraph_coco_opt2.7b_beam_trainval2017_Paraphrases_rel_texts_for_coco_images.json \
        # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus20_thre05.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_caption_nucleus20_trainval2017_Paraphrases_rel_texts_for_coco_images.json \
        # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_nucleus5_thre05.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_nucleus5_trainval2017_Paraphrases_rel_texts_for_coco_images.json \
        # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_beam_thre05.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_caption_beam_trainval2017_Paraphrases_rel_texts_for_coco_images.json \
        # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_OracleCaps_Paraphrases_thre05.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/SceneGraph_OracleCaps_Paraphrases_rel_texts_for_coco_images.json \
        # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_OracleCaps_Overlap_Paraphrases.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/SceneGraph_OracleCaps_Overlap_Paraphrases_rel_texts_for_coco_images.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/vg_rel_texts_for_coco_images_greater5_v2.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/vg_rel_texts_for_coco_images_greater0_v3.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/vg_rel_texts_for_coco_images_greater5_v3.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/vg_rel_texts_for_coco_images_greater5_v2.json \
        
        # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_greater0_v3.json \

        # We must use --lang_aux_loss to maintain coherence of the code.
        # --ParSeDDETRHOI \
        # --use_nms_filter \

# # Run the Verb Tagger on OD datasets and generate many json files
# python generate_relations_using_verb_tagger.py \
#         --param_path /Path/To/logs/RLIP_PD_v2_Tagger2_COCO_20e_bs64_GLIP2aux_L1/checkpoint0019.pth \
#         --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e.json \
#         --dataset_file coco \
#         --coco_path /Path/To/data/coco2017 \
#         --dec_layers 3 \
#         --enc_layers 6 \
#         --num_queries 200 \
#         --backbone resnet50 \
#         --dim_feedforward 1024 \
#         --num_feature_levels 4 \
#         --with_box_refine \
#         --RLIP_ParSeD_v2 \
#         --fusion_type GLIP_attn \
#         --fusion_interval 2 \
#         --fusion_last_vis \
#         --verb_tagger \
#         --batch_size 1 \
#         --num_workers 4 \
#         --use_no_obj_token \
#         --lang_aux_loss \
#         --subject_class \
#         --save_keep_names_freq_path /Path/To/jacob/RLIP/datasets/priors/RLIPv2_train2017_threshold05_Tagger2_Noi24_20e_keep_names_freq.json \
#         --merge_rel_det_annos_path_1 /Path/To/data/VG/annotations/scene_graphs_preprocessv1.json \
#         --merge_rel_det_annos_path_2 /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e.json \
#         --merge_rel_det_annos_path_save /Path/To/data/mixed_datasets/vg_v1_train2017_threshold20_Tagger2_Noi24_20e.json \
#         --merge_keep_names_freq_path_1  /Path/To/jacob/RLIP/datasets/priors/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_keep_names_freq.json \
#         --merge_keep_names_freq_path_2  /Path/To/jacob/RLIP/datasets/vg_keep_names_v1_no_lias_freq.json \
#         --merge_keep_names_freq_path_save /Path/To/jacob/RLIP/datasets/priors/Merge_RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_vg_v1_no_alias.json \


#         # --merge_rel_det_annos_path_1 /Path/To/data/VG/annotations/scene_graphs_preprocessv1.json \
#         # --merge_rel_det_annos_path_2 /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold25_Tagger2_Noi24_20e.json \
#         # --merge_rel_det_annos_path_save /Path/To/data/mixed_datasets/vg_v1_train2017_threshold25_Tagger2_Noi24_20e.json \

#         # --merge_keep_names_freq_path_1  /Path/To/jacob/RLIP/datasets/priors/RLIPv2_train2017_threshold25_Tagger2_Noi24_20e_keep_names_freq.json \
#         # --merge_keep_names_freq_path_2  /Path/To/jacob/RLIP/datasets/vg_keep_names_v1_no_lias_freq.json \
#         # --merge_keep_names_freq_path_save /Path/To/jacob/RLIP/datasets/priors/Merge_RLIPv2_train2017_threshold25_Tagger2_Noi24_20e_vg_v1_no_alias.json \


### produce_stat_for_pseudo_labels/filtering pairwise relations
# python generate_relations_using_verb_tagger.py \
#         --param_path /Path/To/logs/RLIP_PD_v2_Tagger2_COCO_20e_bs64_GLIP2aux_L1/checkpoint0019.pth \
#         --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e.json \
#         --save_filtering_anno_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Top2_Tagger2_Noi24_20e.json \
#         --save_filtering_keep_names_freq_path /Path/To/jacob/RLIP/datasets/priors/RLIPv2_train2017_threshold20_Top2_Tagger2_Noi24_20e_keep_names_freq.json \
#         --vg_rel_texts_for_hico_objects_path /Path/To/data/coco2017/annotations/vg_rel_texts_for_hico_objects_greater5.json \
#         --save_filtering_ood_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Xood_Tagger2_Noi24_20e.json \
#         --dataset_file coco \
#         --coco_path /Path/To/data/coco2017 \
#         --dec_layers 3 \
#         --enc_layers 6 \
#         --num_queries 200 \
#         --backbone resnet50 \
#         --dim_feedforward 1024 \
#         --num_feature_levels 4 \
#         --with_box_refine \
#         --RLIP_ParSeD_v2 \
#         --fusion_type GLIP_attn \
#         --fusion_interval 2 \
#         --fusion_last_vis \
#         --verb_tagger \
#         --batch_size 1 \
#         --num_workers 4 \
#         --use_no_obj_token \
#         --lang_aux_loss \
#         --subject_class \
#         --relation_threshold 0.2 \