# Stage 2 (SMCA, no guide) on top of the VoxelNeXt LiDAR branch: stage 0
# livip3d_resnet50_lidar_only_voxelnext_detect_only.py -> stage 1 run
# s1_voxelnext_from_s0_b8 (0.075 m voxels, 171x171 BEV) -> this. The LiDAR
# branch below is that run's, frozen by fix_lidar; everything else is
# livip3d_resnet50_smca_noguide.py, so the comparison with s2_smca_noguide_b8
# (PointPillars) changes only the LiDAR backbone. debug is off here (it writes
# TensorBoard images to runs/bev_vis).
#
# LiViP3D stage 2, ablation: LiDAR-camera fusion (SMCA) WITHOUT the
# image-guided query initialisation -- TransFusion's "w/o Guide" row
# (Table 7: fusion +4.8 mAP, guide +1.6 mAP).
#
# Queries are still initialised from the stage-1 LiDAR heatmap, so the
# only new module is the SMCA layer (331k params, residual). Everything
# else matches livip3d_resnet50_lidar_img_guided_smca.py so the two runs
# are directly comparable.
_base_ = [
    '../../_base_/nus-3d.py',
    '../../_base_/default_runtime.py',
    '../../_base_/heatmap_tasks.py',
]
workflow = [('train', 1)]
plugin = True
plugin_dir = 'plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.075, 0.075, 0.2]
voxel_grid = [1365, 1365, 40]   # range / voxel_size, x y z

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
]
prediction_eval_classes = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='ViP3D',
    use_grid_mask=True,
    num_classes=7,
    num_query=300,
    bbox_coder=dict(
        type='DETRTrack3DCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=100,
        num_classes=7),
    # fix_feats would no_grad the whole image path, neck included. Freeze the
    # backbone via frozen_stages=4 instead, so the FPN can still adapt.
    fix_feats=False,
    fix_lidar=True,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    use_lidar=True,
    lidar_bev_channels=128,
    lidar_voxel_size=voxel_size,
    lidar_out_size_factor=8,
    heatmap_head=dict(
        type='LiViPHeatmapHead',
        in_channels=128,
        hidden_channels=64,
        class_names=class_names,
        tasks={{_base_.centerpoint_nusc_tasks}}),
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    # Mean of the points in each voxel, as VoxelNeXt's MeanVFE.
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='VoxelNeXtBackbone',
        in_channels=5,
        sparse_shape=[voxel_grid[2] + 1, voxel_grid[1], voxel_grid[0]],
        channels=[16, 32, 64, 128, 128],
        out_channels=128),
    # The middle encoder already returns the dense stride-8 BEV.
    pts_backbone=None,
    pts_neck=None,
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0, update_query_pos=True,
        fp_ratio=0.3, random_drop=0.1),
    mem_cfg=dict(
        memory_bank_type='MemoryBank',
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    img_backbone=dict(
        type='ResNet',
        with_cp=False,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,   # whole backbone frozen (stop grad + eval)
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    loss_cfg=dict(
        type='ClipMatcher',
        num_classes=7,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3DTrack',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        # BEV heatmap supervision for query initialisation (CenterPoint
        # alpha/gamma). loss_weight=1.0 is the weight this term had when
        # it was hard-coded in ViP3D._heatmap_loss.
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            loss_weight=1.0),
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=4,
        norm_cfg=dict(type='BN2d'),
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='TransFusionDetHead',
        num_classes=7,
        in_channels=256,
        num_cams=6,
        num_feature_levels=4,
        transformer=dict(
            type='TransFusionTransformer',
            decoder=dict(
                type='TransFusionTransformerDecoder',
                embed_dims=256,
                num_heads=8,
                ffn_dims=512,
                dropout=0.1,
                use_smca=True,
                lidar_bev_attn=dict(
                    type='LiDARBEVDeformCrossAtten',
                    embed_dims=256,
                    num_heads=8,
                    num_points=4,
                    bev_in_channels=128,
                    dropout=0.1,
                    pc_range=point_cloud_range),
                smca_attn=dict(
                    type='SMCACrossAtten',
                    embed_dims=256,
                    num_heads=8,
                    num_cams=6,
                    num_levels=4,
                    pc_range=point_cloud_range,
                    dropout=0.1,
                    sigma_scale=1.0,
                    feat_level=0),
            )),
        pc_range=point_cloud_range,
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
    ),
    debug=False,
    vis_interval=20,
    use_img_guided=False,  # ablation: SMCA fusion only
    use_smca=True,
    do_pred=True,
    relative_pred=True,
    agents_layer_0=True,
    add_branch=True,
    predictor=dict(
        hidden_size=128,
        laneGCN=True,
        decoder=dict(
            variety_loss=True,
            variety_loss_prob=True,
            hidden_size=128,
        ),
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=voxel_grid,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='GIoU3DCost', weight=0.0),
                pc_range=point_cloud_range)
        )
    ),
)

dataset_type = 'NuScenesTrackDatasetRadar'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='ResizeMultiViewKeepRatio', scale=(960, 544), keep_ratio=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        load_dim=5,
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='InstanceRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]
train_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=[
        'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img',
        'points', 'timestamp', 'l2g_r_mat', 'l2g_t',
        'pred_matrix', 'polyline_spans', 'mapping', 'instance_idx_2_labels']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='ResizeMultiViewKeepRatio', scale=(960, 544), keep_ratio=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        load_dim=5,
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]
test_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=[
        'gt_bboxes_3d', 'gt_labels_3d',
        'points', 'img', 'timestamp', 'l2g_r_mat', 'l2g_t',
        'pred_matrix', 'polyline_spans', 'mapping', 'instance_idx_2_labels']),
]

# Clips per GPU. The backbones run batched over them; tracking and prediction
# run clip by clip and the losses are averaged (ViP3D.forward_train). The LR
# is not rescaled with it.
samples_per_gpu = 1   # 8 GPUs -> effective batch 8, 3516 iters/epoch

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        num_frames_per_sample=3,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_tracking_infos_train.pkl',
        pipeline_single=train_pipeline,
        pipeline_post=train_pipeline_post,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        camera_types=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        do_pred=True),
    val=dict(
        type=dataset_type,
        pipeline_single=test_pipeline,
        pipeline_post=test_pipeline_post,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + 'nuscenes_tracking_infos_val.pkl',
        num_frames_per_sample=1,
        camera_types=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        do_pred=True),
    test=dict(
        type=dataset_type,
        pipeline_single=test_pipeline,
        pipeline_post=test_pipeline_post,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + 'nuscenes_tracking_infos_val.pkl',
        num_frames_per_sample=1,
        camera_types=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        do_pred=True))

optimizer = dict(
    type='AdamW',
    lr=3.5e-4,   # sqrt-scaled from 5e-4 at batch 16
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),          # frozen via frozen_stages=4
            'img_neck':     dict(lr_mult=0.1),          # pretrained DETR3D FPN
            # fpn_convs.3 can't load from DETR3D (add_extra_convs='on_input'),
            # so it starts random and needs the full rate. Longest key wins.
            'img_neck.fpn_convs.3': dict(lr_mult=1.0),
            # pts_voxel_encoder / pts_middle_encoder (VoxelNeXt) need no entry:
            # the model's fix_lidar=True already sets requires_grad=False on them.
            'heatmap_head': dict(lr_mult=0.1),  # trained in stage 1, fine-tune
            # img_bev_proj and img_hm_head are new here and use the base lr.
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=350,   # ~1.7% of the 21k steps in 6 epochs at batch 8
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

total_epochs = 6   # TransFusion's stage 2 length
# in-training DistEvalHook crashes on mmcv 1.7.2 + torch 2.1
evaluation = dict(interval=99999)
runner = dict(type='EpochBasedRunner', max_epochs=6)

find_unused_parameters = True
# Best stage 1 epoch (EPA 0.403 with dedup). Carries the frozen DETR3D image
# backbone and neck too, so nothing else needs loading. The SMCA layer is new
# here and starts from its init.
load_from = 'work_dirs/perun/non-augmented/s1_voxelnext_from_s0_b8/epoch_8.pth'
# fp16 = dict(loss_scale='dynamic')
