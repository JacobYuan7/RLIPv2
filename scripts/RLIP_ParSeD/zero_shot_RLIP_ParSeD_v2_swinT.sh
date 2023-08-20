# Zero-shot (NF): RLIP ParSeD v2
# batch_size是为了测试FPS改成1的, fusion_interval默认是2。
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
        --pretrained /Path/To/logs/RLIP_PD_v2_VGCOCOO365_SwinT_BLIP_Nu10_thr20_20e_GLIP2aux_L1/checkpoint0019.pth \
        --output_dir /Path/To/logs/ParSeDETRHOI_Random_enc6_dec33_query200_pnms \
        --dataset_file hico \
        --hoi_path /Path/To/data/hico_20160224_det \
        --hoi \
        --backbone swin_tiny \
        --drop_path_rate 0.2 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1 \
        --batch_size 16 \
        --num_workers 16 \
        --lr_drop 60 \
        --epochs 80 \
        --load_backbone supervised \
        --lr 1e-4 \
        --dec_layers 3 \
        --enc_layers 6 \
        --dim_feedforward 2048 \
        --num_feature_levels 4 \
        --with_box_refine \
        --lr_backbone 1e-5 \
        --text_encoder_lr 1e-5 \
        --num_queries 200 \
        --use_nms_filter \
        --RLIP_ParSeD_v2 \
        --schedule step \
        --subject_class \
        --use_no_obj_token \
        --obj_loss_type cross_entropy \
        --verb_loss_type focal \
        --negative_text_sampling 500 \
        --sampling_stategy freq \
        --pseudo_verb \
        --zero_shot_eval hico \
        --eval \
        --fusion_type GLIP_attn \
        --gating_mechanism VXAc \
        --verb_query_tgt_type vanilla_MBF \
        --fusion_interval 2 \
        --fusion_last_vis \
        --lang_aux_loss \


        ### Scaling experiments
        # --pretrained /Path/To/logs/RLIP_PD_v2_VG_SwinT_BLIP_Nu10_thr20_20e_GLIP2aux_L1/checkpoint0019.pth \
        # --pretrained /Path/To/logs/RLIP_PD_v2_VGCOCO_SwinT_BLIP_Nu10_thr20_20e_GLIP2aux_L1/checkpoint0019.pth \
        