import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import HEADS

@HEADS.register_module()
class LiViPHeatmapHead(BaseModule):
    """Class heatmap over the BEV grid, whose peaks initialise object queries.

    A shared conv reduces the BEV features to `hidden_channels`, then one head
    per task predicts that task's classes. The output is regrouped into
    `class_names` order, so the grouping is free to differ from the model's
    class list: a task may hold classes the model does not detect (they are
    predicted and dropped), and a task holding none of them is not built at
    all -- that is how CenterPoint's barrier task disappears here.

    Args:
        in_channels (int): channels of the BEV feature map.
        class_names (list[str]): the model's classes, in its own order. The
            output channels follow this order.
        tasks (list[list[str]]): class names per task head. The groupings live
            in configs/_base_/heatmap_tasks.py; `[class_names]` gives a single
            head over all classes.
        hidden_channels (int): width of the shared conv. Default: 64.
        init_bias (float): bias of each final conv, so training starts from a
            low probability per cell as in CenterPoint. Default: -2.19
            (p ~ 0.1). Overwritten afterwards when a checkpoint is loaded.
    """

    def __init__(self,
                 in_channels,
                 class_names,
                 tasks,
                 hidden_channels=64,
                 init_bias=-2.19,
                 init_cfg=None):
        super(LiViPHeatmapHead, self).__init__(init_cfg)
        self.class_names = list(class_names)
        self.init_bias = init_bias

        tasks = [list(t) for t in tasks]
        # Drop tasks this model has no class in; the rest keep their full width
        # so their weights still line up with the checkpoint they came from.
        self.tasks = [t for t in tasks if any(n in t for n in self.class_names)]

        # (task index, channel) supplying each class, in class_names order.
        self.layout = []
        for name in self.class_names:
            for task_idx, task in enumerate(self.tasks):
                if name in task:
                    self.layout.append((task_idx, task.index(name)))
                    break
            else:
                raise ValueError(
                    f"class '{name}' is in no task of {self.tasks}")

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, len(task), 3, padding=1),
            ) for task in self.tasks
        ])

    def init_weights(self):
        super(LiViPHeatmapHead, self).init_weights()
        for head in self.task_heads:
            nn.init.constant_(head[-1].bias, self.init_bias)

    def forward(self, bev_feat):
        """bev_feat [B, in_channels, H, W] -> [B, len(class_names), H, W].

        Pre-sigmoid logits. Each head runs once and is then sliced, since
        several classes can come from the same head.
        """
        shared = self.shared_conv(bev_feat)
        outs = [head(shared) for head in self.task_heads]
        return torch.cat(
            [outs[task_idx][:, c:c + 1] for task_idx, c in self.layout], dim=1)
