# python models/ops/setup.py build install;
# pip install -I transformers==4.5.1 --no-cache-dir --force-reinstall
# pip install -r requirements_ParSeDETRHOI.txt;
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
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py \
    --pretrained /Path/To/params/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_numrefpoints4_SepDDETRHOIv3_converted.pth \
    --output_dir /Path/To/logs/RLIP_PD_v2_Tagger2_Noi36_COCO_20e_bs64_GLIP2aux_L1 \
    --dim_feedforward 1024 \
    --epochs 20 \
    --lr_drop 15 \
    --num_queries 200 \
    --enc_layers 6 \
    --dec_layers 3 \
    --dataset_file vg \
    --vg_path /Path/To/data/VG \
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
    --cross_modal_pretrain \
    --RLIP_ParSeD_v2 \
    --subject_class \
    --use_no_obj_token \
    --verb_loss_type focal \
    --save_ckp \
    --schedule step \
    --negative_text_sampling 500 \
    --obj_loss_type cross_entropy \
    --sampling_stategy freq \
    --fusion_type GLIP_attn \
    --fusion_interval 2 \
    --fusion_last_vis \
    --lang_aux_loss \
    --verb_tagger \
    --label_noise_scale 0.3 \
    --box_noise_scale 0.6 \


    # COCO + VG
    # --pretrained /Path/To/params/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_numrefpoints4_SepDDETRHOIv3_converted.pth \
    # --output_dir /Path/To/logs/ParSeD_CEFocal_BiasT_Freq500_PseuEuc03_COCO_bs128_50e \
    # --output_dir /Path/To/logs/RLIP_ParSeD_v2_RQL_LSE_RPLEuc03_COCO_20e_bs64lr20_NoFusion_2 \
    # --output_dir /Path/To/logs/RLIP_ParSeD_v2_RQL_LSE_RPLEuc03_COCO_50e_bs64lr14_GLIP2_L1 \