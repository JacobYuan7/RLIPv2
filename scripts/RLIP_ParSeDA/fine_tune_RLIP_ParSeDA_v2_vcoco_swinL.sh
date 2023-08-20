# python models/ops/setup.py build install;
# pip install -I transformers==4.5.1 --no-cache-dir --force-reinstall
# pip install -r requirements_RLIP_ParSe.txt;
# pip install pkgs/pycocotools-2.0.2.tar.gz;
# pip install submitit==1.3.0;
# pip install timm;
# export NCCL_DEBUG=INFO;
# export NCCL_IB_HCA=mlx5;
# export NCCL_IB_TC=136;
# export NCCL_IB_SL=5;
# export NCCL_IB_GID_INDEX=3;
# export TORCH_DISTRIBUTED_DETAIL=DEBUG;
# # Pay attention to the learning rate if channging #nodes
# # python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py \
    --pretrained /Path/To/logs/RLIP_PDA_v2_SwinL_VGCOO365_BLIP_Nu10_thr20_RQL_LSE_RPL_bs64_20e/checkpoint.pth \
    --output_dir /Path/To/logs/RLIP_PDA_v2_VCOCO_SwinL_VGCOCOO365_RQL_LSE_RPL_20e_L1_20e \
    --dataset_file vcoco \
    --hoi_path /Path/To/data/v-coco \
    --num_obj_classes 81 \
    --num_verb_classes 29 \
    --hoi \
    --load_backbone supervised \
    --backbone swin_large \
    --set_cost_bbox 2.5 \
    --set_cost_giou 1 \
    --bbox_loss_coef 2.5 \
    --giou_loss_coef 1 \
    --lr_drop 15 \
    --epochs 20 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --text_encoder_lr 1e-5 \
    --schedule step \
    --num_workers 4 \
    --batch_size 2 \
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
    --use_no_obj_token \
    --sampling_stategy freq \
    --fusion_type GLIP_attn \
    --gating_mechanism VXAc \
    --verb_query_tgt_type vanilla_MBF \
    --fusion_interval 2 \
    --fusion_last_vis \
    --lang_aux_loss \
    --subject_class \
    # --giou_verb_label \
    # --few_shot_transfer 1 \
    # --zero_shot_setting UC-NF \