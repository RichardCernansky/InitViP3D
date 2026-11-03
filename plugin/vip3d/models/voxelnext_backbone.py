"""VoxelNeXt sparse 3D backbone (Chen et al., CVPR 2023), as a middle encoder.

Ported from VoxelResBackBone8xVoxelNeXt in
https://github.com/JIA-Lab-research/VoxelNeXt
(pcdet/models/backbones_3d/spconv_backbone_voxelnext.py). Only the backbone is
taken: its stride-8 sparse BEV output is densified, so the rest of LiViP3D
(heatmap head, top-k query init, BEV deformable attention) is unchanged. The
fully sparse VoxelNeXt head is not used.

Uses spconv 2.x when installed and falls back to the spconv 1.x copy that ships
with mmdet3d otherwise. Both give the same layers; 2.x is faster, so latency
numbers should be measured with it.
"""
from functools import partial

import torch
import torch.nn as nn

from mmdet3d.models.builder import MIDDLE_ENCODERS

try:
    import spconv.pytorch as spconv
    SPCONV_BACKEND = 'spconv2'
except ImportError:
    from mmdet3d.ops import spconv
    SPCONV_BACKEND = 'mmdet3d-spconv1'


def replace_feature(x, features):
    # spconv 2.x forbids setting .features, 1.x has no replace_feature.
    if hasattr(x, 'replace_feature'):
        return x.replace_feature(features)
    x.features = features
    return x


def post_act_block(in_channels, out_channels, kernel_size, norm_fn,
                   stride=1, padding=0, indice_key=None):
    """Strided sparse conv -> BN -> ReLU (the 'spconv' case in OpenPCDet)."""
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, bias=False,
                            indice_key=indice_key),
        norm_fn(out_channels),
        nn.ReLU(),
    )


class SparseBasicBlock(spconv.SparseModule):
    """Residual block of two submanifold 3x3x3 convs."""

    def __init__(self, inplanes, planes, norm_fn, indice_key=None):
        super(SparseBasicBlock, self).__init__()
        # bias=True next to BN is in the original code; kept for fidelity.
        self.conv1 = spconv.SubMConv3d(inplanes, planes, 3, padding=1,
                                       bias=True, indice_key=indice_key)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(planes, planes, 3, padding=1,
                                       bias=True, indice_key=indice_key)
        self.bn2 = norm_fn(planes)

    def forward(self, x):
        identity = x.features
        out = self.conv1(x)
        out = replace_feature(out, self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))
        out = replace_feature(out, self.relu(out.features + identity))
        return out


@MIDDLE_ENCODERS.register_module()
class VoxelNeXtBackbone(nn.Module):
    """Sparse voxels -> dense BEV map at stride 8.

    Six stages (strides 1, 2, 4, 8, 16, 32). Stages 5 and 6 are projected back
    onto the stride-8 grid and merged with stage 4, the z axis is summed away,
    and two sparse 2D convs produce the output.

    Args:
        in_channels (int): voxel feature channels (5 for x, y, z, intensity,
            time lag from HardSimpleVFE).
        sparse_shape (list[int]): voxel grid as [z, y, x]; z is the number of
            z voxels + 1, as in OpenPCDet.
        channels (list[int]): widths of stages 1-5; stage 6 reuses the last.
        out_channels (int): BEV output channels.

    Forward returns a list with one dense tensor [B, out_channels, H, W], the
    same form the SECONDFPN neck returns, so ViP3D reads it unchanged.
    """

    def __init__(self,
                 in_channels=5,
                 sparse_shape=(41, 1365, 1365),
                 channels=(16, 32, 64, 128, 128),
                 out_channels=128,
                 kernel_sizes=(3, 3, 3, 3)):
        super(VoxelNeXtBackbone, self).__init__()
        self.sparse_shape = list(sparse_shape)
        self.spconv_backend = SPCONV_BACKEND
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        c = channels

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, c[0], 3, padding=1, bias=False,
                              indice_key='subm1'),
            norm_fn(c[0]),
            nn.ReLU(),
        )

        def stage(cin, cout, k, idx):
            return spconv.SparseSequential(
                post_act_block(cin, cout, k, norm_fn, stride=2,
                               padding=k // 2, indice_key=f'spconv{idx}'),
                SparseBasicBlock(cout, cout, norm_fn, indice_key=f'res{idx}'),
                SparseBasicBlock(cout, cout, norm_fn, indice_key=f'res{idx}'),
            )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(c[0], c[0], norm_fn, indice_key='res1'),
            SparseBasicBlock(c[0], c[0], norm_fn, indice_key='res1'),
        )
        self.conv2 = stage(c[0], c[1], kernel_sizes[0], 2)
        self.conv3 = stage(c[1], c[2], kernel_sizes[1], 3)
        self.conv4 = stage(c[2], c[3], kernel_sizes[2], 4)
        self.conv5 = stage(c[3], c[4], kernel_sizes[3], 5)
        self.conv6 = stage(c[4], c[4], kernel_sizes[3], 6)

        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv2d(c[3], out_channels, 3, stride=1, padding=1,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(out_channels),
            nn.ReLU(),
        )
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channels, out_channels, 3, stride=1,
                              padding=1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
        )

    @staticmethod
    def _bev_out(features, indices, spatial_shape, batch_size):
        """Drop z: sum the features of voxels that share a (b, y, x) cell."""
        indices_2d = indices[:, [0, 2, 3]]
        indices_unique, inv = torch.unique(indices_2d, dim=0,
                                           return_inverse=True)
        features_unique = features.new_zeros(
            (indices_unique.shape[0], features.shape[1]))
        features_unique.index_add_(0, inv, features)
        return spconv.SparseConvTensor(
            features_unique, indices_unique.int().contiguous(),
            list(spatial_shape[1:]), batch_size)

    def forward(self, voxel_features, coors, batch_size):
        """
        Args:
            voxel_features (Tensor): [N, in_channels]
            coors (Tensor): [N, 4] as (batch, z, y, x), as voxelize() returns.
            batch_size (int | Tensor)
        """
        batch_size = int(batch_size)
        x = spconv.SparseConvTensor(voxel_features, coors.int().contiguous(),
                                    self.sparse_shape, batch_size)
        x = self.conv_input(x)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        # Put stages 5 and 6 on the stride-8 grid of stage 4. The original
        # scales x_conv5/6.indices in place; new tensors here, same result.
        scale5 = x_conv5.indices.new_tensor([1, 2, 2, 2])
        scale6 = x_conv6.indices.new_tensor([1, 4, 4, 4])
        indices = torch.cat([x_conv4.indices,
                             x_conv5.indices * scale5,
                             x_conv6.indices * scale6])
        features = torch.cat([x_conv4.features, x_conv5.features,
                              x_conv6.features])

        out = self._bev_out(features, indices, x_conv4.spatial_shape,
                            batch_size)
        out = self.conv_out(out)
        out = self.shared_conv(out)
        return [out.dense()]
