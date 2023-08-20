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
# # Pay attention to the batch_size if we want to use gradient_accumulation. 
# # (change  --batch_size  --iterative_paradigm 0,1 \ --gradient_strategy gradient_accumulation \ --output_dir )
# # python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
# # --fusion_type GLIP_attn \ 注意！
# # python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK \
# #     --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py \
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --pretrained /Path/To/params/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_numrefpoints4_SepDDETRHOIv3_converted.pth \
    --output_dir /Path/To/logs/RLIP_PD_v2_VGCOCOO365_R50_BLIP_Nu10_thr20_20e_GLIP2aux_L1 \
    --dim_feedforward 1024 \
    --epochs 20 \
    --lr_drop 15 \
    --num_queries 200 \
    --enc_layers 6 \
    --dec_layers 3 \
    --vg_path /Path/To/data/VG \
    --coco_path /Path/To/data/coco2017 \
    --o365_path /Path/To/data/Objects365 \
    --coco_rel_anno_file /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05.json \
    --vg_rel_anno_file /Path/To/data/VG/annotations/scene_graphs_preprocessv1.json \
    --o365_rel_anno_file /Path/To/data/Objects365/rel_annotations/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_1234_4.json \
    --dataset_file vg_coco2017_o365 \
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
    --obj_loss_type cross_entropy \
    --sampling_stategy freq \
    --giou_verb_label \
    --fusion_type GLIP_attn \
    --gating_mechanism VXAc \
    --verb_query_tgt_type vanilla_MBF \
    --fusion_interval 2 \
    --fusion_last_vis \
    --lang_aux_loss \
    --pseudo_verb \
    --negative_text_sampling 500 \
    --relation_threshold 0.20 \

    # --pair_overlap \
    # --iterative_paradigm 0,1 \
    # --gradient_strategy gradient_accumulation \
    # --use_all_text_labels \
