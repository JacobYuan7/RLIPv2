# Zero-shot eval RLIP_ParSeDA_v2 on HICO
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
        --pretrained /Path/To/logs/RLIP_PDA_v2_SwinL_VGCOO365_BLIP_Nu10_thr20_RQL_LSE_RPL_bs64_20e/checkpoint0019.pth \
        --output_dir /Path/To/logs/IterativeDETRHOI_correction \
        --dataset_file hico \
        --hoi_path /Path/To/data/hico_20160224_det \
        --hoi \
        --load_backbone supervised \
        --backbone swin_large \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1 \
        --lr_drop 60 \
        --epochs 80 \
        --lr 1e-4 \
        --lr_backbone 1e-5 \
        --text_encoder_lr 1e-5 \
        --schedule step \
        --num_workers 16 \
        --batch_size 16 \
        --exponential_hyper 1 \
        --exponential_loss \
        --obj_loss_type cross_entropy \
        --verb_loss_type focal \
        --enc_layers 6 \
        --dec_layers 3 \
        --num_queries 200 \
        --use_nms_filter \
        --save_ckp \
        --RLIP_ParSeDA_v2 \
        --with_box_refine \
        --num_feature_levels 4 \
        --num_patterns 0 \
        --pe_temperatureH 20 \
        --pe_temperatureW 20 \
        --dim_feedforward 2048 \
        --dropout 0.0 \
        --drop_path_rate 0.5 \
        --subject_class \
        --use_no_obj_token \
        --sampling_stategy freq \
        --fusion_type GLIP_attn \
        --gating_mechanism VXAc \
        --verb_query_tgt_type vanilla_MBF \
        --fusion_interval 2 \
        --fusion_last_vis \
        --lang_aux_loss \
        --giou_verb_label \
        --zero_shot_eval hico \
        --eval \
        
        # VG
        # --pretrained /Path/To/logs/RLIP_PDA_v2_SwinL_VG_BLIP_Nu10_thr20_RQL_LSE_RPL_20e/checkpoint0019.pth \
        # VG + COCO
        # --pretrained /Path/To/logs/RLIP_PDA_v2_SwinL_VGCOCO_BLIP_Nu10_thr20_RQL_LSE_RPL_20e/checkpoint0019.pth \
        # VG + COCO + O365
        # --pretrained /Path/To/logs/RLIP_PDA_v2_SwinL_VGCOO365_BLIP_Nu10_thr20_RQL_LSE_RPL_bs64_20e/checkpoint.pth \