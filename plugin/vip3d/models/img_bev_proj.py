import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class ImageGuidedBEVProjection(BaseModule):
    """Project image features onto the BEV plane by per-camera cross-attention.

    TransFusion Sec. 3.6 / Fig. 4, the image-guided query initialisation:
      1. Collapse each image's height axis by max -> one column per pixel
         column, on the observation that a BEV location maps to an image
         column by camera geometry and a column holds at most one object.
      2. Per-camera cross-attention with the BEV cells as queries and that
         camera's columns as keys/values.
      3. Average over cameras -> F_LC, which feeds a second heatmap.

    Args:
        bev_channels (int): channels of the BEV feature map, in and out.
        img_channels (int): channels of the image features. Default: 256
            (the FPN's out_channels).
        embed_dims (int): width of the attention. Default: 256.
        num_heads (int): attention heads. Default: 8.
        num_cams (int): cameras to build an attention for. Each gets its own,
            so this must be at least the number of cameras fed in; extra ones
            simply go unused. Default: 6.
        feat_level (int): which FPN level to read. Default: 0 (finest).
    """

    def __init__(self,
                 bev_channels,
                 img_channels=256,
                 embed_dims=256,
                 num_heads=8,
                 num_cams=6,
                 feat_level=0,
                 init_cfg=None):
        super(ImageGuidedBEVProjection, self).__init__(init_cfg)
        self.num_cams = num_cams
        self.feat_level = feat_level
        self.bev_proj = nn.Linear(bev_channels, embed_dims)
        self.img_proj = nn.Linear(img_channels, embed_dims)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
            for _ in range(num_cams)
        ])
        self.out_proj = nn.Linear(embed_dims, bev_channels)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, bev_feat, img_feats):
        """
        Args:
            bev_feat:  [B, C_bev, H, W]
            img_feats: list of [B, N_cams, C_img, H_img, W_img], one per level
        Returns:
            F_LC: [B, C_bev, H, W]
        """
        B, C_bev, H, W = bev_feat.shape
        img = img_feats[self.feat_level]
        _, N_cams, C_img, H_img, W_img = img.shape

        # Collapse height axis via max → [B, N_cams, W_img, C_img] then project
        img_cols = img.max(dim=3).values           # [B, N_cams, C_img, W_img]
        img_cols = img_cols.permute(0, 1, 3, 2)    # [B, N_cams, W_img, C_img]
        img_cols = self.img_proj(img_cols)          # [B, N_cams, W_img, D]

        # BEV as queries → [B, H*W, D]
        bev_q = self.bev_proj(bev_feat.permute(0, 2, 3, 1).reshape(B, H * W, C_bev))

        # Per-camera attention, average across cameras
        n = min(N_cams, self.num_cams)
        cam_outs = []
        for i in range(n):
            kv_i = img_cols[:, i]                          # [B, W_img, D]
            attn_i, _ = self.cross_attns[i](bev_q, kv_i, kv_i)
            cam_outs.append(attn_i)
        attn_avg = torch.stack(cam_outs, dim=0).mean(dim=0)  # [B, H*W, D]
        attn_avg = self.norm(bev_q + attn_avg)

        F_LC = self.out_proj(attn_avg).reshape(B, H, W, C_bev).permute(0, 3, 1, 2)
        return F_LC
