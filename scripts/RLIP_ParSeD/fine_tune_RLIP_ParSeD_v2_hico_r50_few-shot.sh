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
# # few-shot: --few_shot_transfer 10, output_dir, batch_size = 32, epoch = 5, lr=1e-4, lr_backbone=1e-5, text_encoder_lr = 1e-5 
# #           num_queries = 200
# # python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK \
# #     --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py \
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --pretrained /Path/To/logs/RLIP_PD_v2_VGCOCOO365_R50_BLIP_Nu10_thr20_20e_GLIP2aux_L1/checkpoint0019.pth \
    --output_dir /Path/To/logs/RLIP_PD_v2_HICO_R50_VGCOO365_COdet_20e_L1_20e_few1 \
    --dataset_file hico \
    --hoi_path /Path/To/data/hico_20160224_det \
    --hoi \
    --load_backbone supervised \
    --backbone resnet50 \
    --set_cost_bbox 2.5 \
    --set_cost_giou 1 \
    --bbox_loss_coef 2.5 \
    --giou_loss_coef 1 \
    --lr_drop 7 \
    --epochs 10 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --text_encoder_lr 1e-5 \
    --schedule step \
    --num_workers 8 \
    --batch_size 8 \
    --obj_loss_type cross_entropy \
    --verb_loss_type focal \
    --enc_layers 6 \
    --dec_layers 3 \
    --num_queries 200 \
    --use_nms_filter \
    --save_ckp \
    --RLIP_ParSeD_v2 \
    --with_box_refine \
    --num_feature_levels 4 \
    --dim_feedforward 1024 \
    --use_no_obj_token \
    --sampling_stategy freq \
    --fusion_type GLIP_attn \
    --gating_mechanism VXAc \
    --verb_query_tgt_type vanilla_MBF \
    --fusion_interval 2 \
    --fusion_last_vis \
    --lang_aux_loss \
    --giou_verb_label \
    --few_shot_transfer 1 \
    # --subject_class \
    

    ### few1
    # --pretrained /Path/To/logs/RLIP_PD_v2_VGCOCOO365_R50_BLIP_Nu10_thr20_20e_GLIP2aux_L1/checkpoint0019.pth \
    # --output_dir /Path/To/logs/RLIP_PD_v2_HICO_R50_VGCOO365_COdet_20e_L1_20e_few1 \
    # --few_shot_transfer 1 \
    ### few10
    # --pretrained /Path/To/logs/RLIP_PD_v2_VGCOCOO365_R50_BLIP_Nu10_thr20_20e_GLIP2aux_L1/checkpoint0019.pth \
    # --output_dir /Path/To/logs/RLIP_PD_v2_HICO_R50_VGCOO365_COdet_20e_L1_20e_few10 \
    # --few_shot_transfer 10 \
