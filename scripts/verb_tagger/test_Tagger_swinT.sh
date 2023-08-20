# Run the Verb Tagger on OD datasets
# COCO train: 117266   3:44:14 (0.1147 s / it)
python generate_relations_using_verb_tagger.py \
        --param_path /Path/To/logs/RLIP_PD_v2_SwinT_Tagger2_Noi24_COCO_20e_GLIP2aux_L1/checkpoint0009.pth \
        --save_path /Path/To/data/coco2017/annotations/swin/RLIPv2_SwinT_10ep_trainval2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05.json \
        --dataset_file coco \
        --coco_path /Path/To/data/coco2017 \
        --dec_layers 3 \
        --enc_layers 6 \
        --num_queries 200 \
        --backbone swin_tiny \
        --dropout 0.0 \
        --drop_path_rate 0.2 \
        --dim_feedforward 2048 \
        --num_feature_levels 4 \
        --with_box_refine \
        --RLIP_ParSeD_v2 \
        --fusion_type GLIP_attn \
        --gating_mechanism VXAc \
        --verb_query_tgt_type vanilla_MBF \
        --fusion_interval 2 \
        --fusion_last_vis \
        --verb_tagger \
        --batch_size 1 \
        --img_inference_batch 8 \
        --num_workers 4 \
        --use_no_obj_token \
        --lang_aux_loss \
        --subject_class \
        --relation_threshold 0.05 \
        --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_caption_nucleus10_trainval2017_Paraphrases_rel_texts_for_coco_images.json \
        # --save_path /Path/To/data/coco2017/annotations/swin/RLIPv2_SwinT_trainval2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05.json \
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