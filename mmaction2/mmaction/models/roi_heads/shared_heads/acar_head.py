# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

# Note: All these heads take 5D Tensors as input (N, C, T, H, W)

class HR2O_NL(nn.Module):
    def __init__(self, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(HR2O_NL, self).__init__()

        self.hidden_dim = hidden_dim

        padding = kernel_size // 2
        self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)

        self.conv = nn.Conv2d(
            hidden_dim, hidden_dim,
            1 if mlp_1x1 else kernel_size,
            padding=0 if mlp_1x1 else padding,
            bias=False
        )
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        query = self.conv_q(x).unsqueeze(1)
        key = self.conv_k(x).unsqueeze(0)
        att = (query * key).sum(2) / (self.hidden_dim ** 0.5)
        att = nn.Softmax(dim=1)(att)
        value = self.conv_v(x)
        virt_feats = (att.unsqueeze(2) * value).sum(1)

        virt_feats = self.norm(virt_feats)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)
        virt_feats = self.dp(virt_feats)

        x = x + virt_feats
        return x


class ACARHead(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 depth = 3,
                 **kwargs):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, bias=False)

        layers = []
        for _ in range(self.depth):
            layers.append(HR2O_NL(out_channels, 3, False))
        self.acar = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def init_weights(self, **kwargs):
        """Weight Initialization for ACRNHead."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)

    def forward(self, x, feat, rois, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The extracted RoI feature.
            feat (torch.Tensor): The context feature.
            rois (torch.Tensor): The regions of interest.

        Returns:
            torch.Tensor: The RoI features that have interacted with context
                feature.
        """
        # We use max pooling by default
        x = self.max_pool(x)   
        h, w = feat.shape[-2:]
        x_tile = x.repeat(1, 1, 1, h, w)
        roi_inds = rois[:, 0].type(torch.long)
        roi_gfeat = feat[roi_inds]
        new_feat = torch.cat([roi_gfeat, x_tile], dim=1).squeeze(2)
        new_feat = self.conv1(new_feat)
        new_feat = nn.functional.relu(new_feat)
        new_feat = self.conv2(new_feat)
        new_feat = nn.functional.relu(new_feat)

        new_feat = self.acar(new_feat)
        new_feat = self.gap(new_feat).unsqueeze(2)
        new_feat = torch.cat([x, new_feat], dim=1)
        return new_feat
