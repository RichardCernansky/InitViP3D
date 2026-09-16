# Stage 1 (LiDAR-only), augmented, with the prediction head on. Identical to
# livip3d_resnet50_lidar_only_detect_only.py (same augmentation, init, schedule
# and fade) plus trajectory prediction, configured as in the non-augmented S1
# run. Prediction targets and lanes follow the augmentation (see
# NuScenesTrackDatasetRadar._move_augmented_box / _augment_lane).
_base_ = ['./livip3d_resnet50_lidar_only_detect_only.py']

model = dict(
    do_pred=True,
    relative_pred=True,
    add_branch=True,
    agents_layer_0=True,
    predictor=dict(
        hidden_size=128,
        laneGCN=True,
        decoder=dict(hidden_size=128, variety_loss=True, variety_loss_prob=True)))

pred_keys = ['pred_matrix', 'polyline_spans', 'mapping', 'instance_idx_2_labels']

data = dict(
    train=dict(
        dataset=dict(
            do_pred=True,
            pipeline_post=[
                dict(type='FormatBundle3DTrack'),
                dict(type='Collect3D', keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'points',
                    'timestamp', 'l2g_r_mat', 'l2g_t'] + pred_keys),
            ])),
    val=dict(
        do_pred=True,
        pipeline_post=[
            dict(type='FormatBundle3DTrack'),
            dict(type='Collect3D', keys=[
                'gt_bboxes_3d', 'gt_labels_3d', 'points', 'timestamp',
                'l2g_r_mat', 'l2g_t'] + pred_keys),
        ]),
    test=dict(
        do_pred=True,
        pipeline_post=[
            dict(type='FormatBundle3DTrack'),
            dict(type='Collect3D', keys=[
                'gt_bboxes_3d', 'gt_labels_3d', 'points', 'timestamp',
                'l2g_r_mat', 'l2g_t'] + pred_keys),
        ]))
