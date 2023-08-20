# # Segment 4 val
# # Run the Verb Tagger on OD datasets
# python generate_relations_using_verb_tagger.py \
#         --param_path /Path/To/logs/RLIP_PD_v2_Tagger2_COCO_20e_bs64_GLIP2aux_L1/checkpoint0019.pth \
#         --save_path /Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_4_val_4.json \
#         --dataset_file o365_det \
#         --o365_path /Path/To/data/Objects365 \
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
#         --img_inference_batch 16 \
#         --num_workers 4 \
#         --use_no_obj_token \
#         --lang_aux_loss \
#         --subject_class \
#         --relation_threshold 0.05 \
#         --o365_segment 4 \
#         --vg_rel_texts_for_o365_images /Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_4_4_Paraphrases_rel_texts_for_o365_images.json \
#         # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_nucleus5_thre05.json \
#         # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_nucleus5_trainval2017_Paraphrases_rel_texts_for_coco_images.json \
#         # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_beam_thre05.json \
#         # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_caption_beam_trainval2017_Paraphrases_rel_texts_for_coco_images.json \



# Segment 4 train
# Run the Verb Tagger on OD datasets
python generate_relations_using_verb_tagger.py \
        --param_path /Path/To/logs/RLIP_PD_v2_Tagger2_COCO_20e_bs64_GLIP2aux_L1/checkpoint0019.pth \
        --save_path /Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_4_train_4.json \
        --dataset_file o365_det \
        --o365_path /Path/To/data/Objects365 \
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
        --o365_segment 4 \
        --vg_rel_texts_for_o365_images /Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_4_4_Paraphrases_rel_texts_for_o365_images.json \
        # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_nucleus5_thre05.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_nucleus5_trainval2017_Paraphrases_rel_texts_for_coco_images.json \
        # --save_path /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_beam_thre05.json \
        # --vg_rel_texts_for_coco_images /Path/To/data/coco2017/annotations/BLIP_captions/SceneGraph_model_large_caption_beam_trainval2017_Paraphrases_rel_texts_for_coco_images.json \


# Segment 3 train
# # Run the Verb Tagger on OD datasets
# python generate_relations_using_verb_tagger.py \
#         --param_path /Path/To/logs/RLIP_PD_v2_Tagger2_COCO_20e_bs64_GLIP2aux_L1/checkpoint0019.pth \
#         --save_path /Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_3_4.json \
#         --dataset_file o365_det \
#         --o365_path /Path/To/data/Objects365 \
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
#         --img_inference_batch 16 \
#         --num_workers 4 \
#         --use_no_obj_token \
#         --lang_aux_loss \
#         --subject_class \
#         --relation_threshold 0.05 \
#         --o365_segment 3 \
#         --vg_rel_texts_for_o365_images /Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_3_4_Paraphrases_rel_texts_for_o365_images.json \


# Segment 2 train
# # Run the Verb Tagger on OD datasets
# python generate_relations_using_verb_tagger.py \
#         --param_path /Path/To/logs/RLIP_PD_v2_Tagger2_COCO_20e_bs64_GLIP2aux_L1/checkpoint0019.pth \
#         --save_path /Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_2_4.json \
#         --dataset_file o365_det \
#         --o365_path /Path/To/data/Objects365 \
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
#         --img_inference_batch 16 \
#         --num_workers 4 \
#         --use_no_obj_token \
#         --lang_aux_loss \
#         --subject_class \
#         --relation_threshold 0.05 \
#         --o365_segment 2 \
#         --vg_rel_texts_for_o365_images /Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_2_4_Paraphrases_rel_texts_for_o365_images.json \


# Segment 1 train
# # Run the Verb Tagger on OD datasets
# python generate_relations_using_verb_tagger.py \
#         --param_path /Path/To/logs/RLIP_PD_v2_Tagger2_COCO_20e_bs64_GLIP2aux_L1/checkpoint0019.pth \
#         --save_path /Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_1_4.json \
#         --dataset_file o365_det \
#         --o365_path /Path/To/data/Objects365 \
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
#         --img_inference_batch 16 \
#         --num_workers 4 \
#         --use_no_obj_token \
#         --lang_aux_loss \
#         --subject_class \
#         --relation_threshold 0.05 \
#         --o365_segment 1 \
#         --vg_rel_texts_for_o365_images /Path/To/data/Objects365/BLIP_captions/SceneGraph_model_large_caption_nucleus10_o365trainval_1_4_Paraphrases_rel_texts_for_o365_images.json \