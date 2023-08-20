python models/ops/setup.py build install;
pip install -I transformers==4.5.1 --no-cache-dir --force-reinstall
pip install -r requirements_ParSeDETRHOI.txt;
pip install pkgs/pycocotools-2.0.2.tar.gz;
pip install submitit==1.3.0;
pip install timm;
export NCCL_DEBUG=INFO;
export NCCL_IB_HCA=mlx5;
export NCCL_IB_TC=136;
export NCCL_IB_SL=5;
export NCCL_IB_GID_INDEX=3;
export TORCH_DISTRIBUTED_DETAIL=DEBUG;
# python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py \
    --output_dir /Path/To/logs/RLIP_PD_v2_OISGGtrain_backboneInit_L1_20e \
    --dim_feedforward 1024 \
    --lr_drop 15 \
    --epochs 20 \
    --num_queries 200 \
    --enc_layers 6 \
    --dec_layers 3 \
    --dataset_file oi_sgg \
    --oi_sgg_path /Path/To/data/open-imagev6 \
    --sgg \
    --backbone resnet50 \
    --set_cost_bbox 2.5 \
    --set_cost_giou 1 \
    --bbox_loss_coef 2.5 \
    --giou_loss_coef 1 \
    --num_workers 8 \
    --batch_size 8 \
    --num_feature_levels 4 \
    --with_box_refine \
    --use_nms_filter \
    --load_backbone supervised \
    --lr 1.41e-4 \
    --lr_backbone 1.41e-5 \
    --text_encoder_lr 1.41e-5 \
    --RLIP_ParSeD_v2 \
    --use_no_obj_token \
    --obj_loss_type cross_entropy \
    --verb_loss_type focal \
    --save_ckp \
    --schedule step \
    --sampling_stategy freq \
    --giou_verb_label \
    --fusion_type GLIP_attn \
    --gating_mechanism VXAc \
    --verb_query_tgt_type vanilla_MBF \
    --fusion_interval 2 \
    --fusion_last_vis \
    --lang_aux_loss \
    --subject_class \
