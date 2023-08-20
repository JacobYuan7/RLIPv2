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
# # MDETR: epoch, output_dir, fusion_type, verb_query_tgt_type
# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK \
#     --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py \
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --pretrained /Path/To/params/drop_path0.5_swin_large_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps_SepDDETRHOIv3_converted.pth \
    --output_dir /Path/To/logs/RLIP_PD_v2_SwinL_VGCOCO_BLIP_Nu10_thr20_RQL_LSE_RPL_20e \
    --vg_path /Path/To/data/VG \
    --coco_path /Path/To/data/coco2017 \
    --coco_rel_anno_file /Path/To/data/coco2017/annotations/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05.json \
    --vg_rel_anno_file /Path/To/data/VG/annotations/scene_graphs_preprocessv1.json \
    --dataset_file vg_coco2017 \
    --load_backbone supervised \
    --backbone swin_large \
    --drop_path_rate 0.5 \
    --dim_feedforward 2048 \
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
    --num_workers 2 \
    --batch_size 2 \
    --obj_loss_type cross_entropy \
    --verb_loss_type focal \
    --enc_layers 6 \
    --dec_layers 3 \
    --num_queries 200 \
    --use_nms_filter \
    --cross_modal_pretrain \
    --save_ckp \
    --RLIP_ParSeD_v2 \
    --with_box_refine \
    --num_feature_levels 4 \
    --subject_class \
    --use_no_obj_token \
    --negative_text_sampling 500 \
    --sampling_stategy freq \
    --fusion_type GLIP_attn \
    --gating_mechanism VXAc \
    --verb_query_tgt_type vanilla_MBF \
    --fusion_interval 2 \
    --fusion_last_vis \
    --lang_aux_loss \
    --giou_verb_label \
    --pseudo_verb \
    --relation_threshold 0.20 \

    