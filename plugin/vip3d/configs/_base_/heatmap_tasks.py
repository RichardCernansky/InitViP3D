# Class groupings for LiViPHeatmapHead: one head per group, each predicting
# that group's classes. A config picks one and passes it as heatmap_head.tasks.
#
# Reference it through mmcv's base interpolation, since a plain name from a
# _base_ file is not in scope in the config that inherits it:
#     tasks={{_base_.centerpoint_nusc_tasks}}

# CenterPoint's nuScenes grouping. Use this to load CenterPoint's pretrained
# per-task heatmap weights: the head keeps each group's full width, so the
# shapes still line up. Groups holding none of the model's classes are not
# built (barrier), and extra classes inside a kept group are predicted and
# then dropped (construction_vehicle, traffic_cone).
centerpoint_nusc_tasks = [
    ['car'],
    ['truck', 'construction_vehicle'],
    ['bus', 'trailer'],
    ['barrier'],
    ['motorcycle', 'bicycle'],
    ['pedestrian', 'traffic_cone'],
]
