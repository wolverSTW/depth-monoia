
"""
Intrinsic-Guided Gated Depth Fusion for Depth-MonoIA.

Inputs
------
rgb_features: [B, 256, H, W]
depth_features: [B, 384, Hd, Wd]
intrinsic_embedding: [B, 256]

Output
------
fused_features: [B, 256, H, W]

The module preserves the original RGB feature shape.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntrinsicGuidedDepthFusion(nn.Module):

    def __init__(
        self,
        rgb_channels=256,
        depth_channels=384,
        intrinsic_dim=256,
    ):
        super().__init__()

        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.intrinsic_dim = intrinsic_dim

        # Convert depth features into MonoIA feature space.
        self.depth_projection = nn.Conv2d(
            in_channels=depth_channels,
            out_channels=rgb_channels,
            kernel_size=1,
            bias=False,
        )

        # Convert the intrinsic embedding into a
        # spatially broadcastable conditioning vector.
        self.intrinsic_projection = nn.Linear(
            intrinsic_dim,
            rgb_channels,
        )

        # Learn where depth features are useful.
        self.gate = nn.Sequential(
            nn.Conv2d(
                rgb_channels * 3,
                rgb_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        # Initially preserve the original RGB pathway.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        rgb_features,
        depth_features,
        intrinsic_embedding,
    ):
        if rgb_features.ndim != 4:
            raise ValueError(
                "rgb_features must be BCHW"
            )

        if depth_features.ndim != 4:
            raise ValueError(
                "depth_features must be BCHW"
            )

        batch, channels, height, width = (
            rgb_features.shape
        )

        if channels != self.rgb_channels:
            raise ValueError(
                "Incorrect RGB feature channels"
            )

        if depth_features.shape[0] != batch:
            raise ValueError(
                "RGB and depth batch sizes differ"
            )

        if depth_features.shape[1] != self.depth_channels:
            raise ValueError(
                "Incorrect depth feature channels"
            )

        if intrinsic_embedding.shape != (
            batch,
            self.intrinsic_dim,
        ):
            raise ValueError(
                "Incorrect intrinsic embedding shape"
            )

        # Project depth feature channels.
        depth = self.depth_projection(
            depth_features
        )

        # Match the RGB feature resolution.
        depth = F.interpolate(
            depth,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        # Generate intrinsic conditioning.
        intrinsic = self.intrinsic_projection(
            intrinsic_embedding
        )

        intrinsic = intrinsic[:, :, None, None]

        intrinsic = intrinsic.expand(
            -1,
            -1,
            height,
            width,
        )

        # Combine all three information sources.
        gate_input = torch.cat(
            [
                rgb_features,
                depth,
                intrinsic,
            ],
            dim=1,
        )

        gate = self.gate(gate_input)

        # Residual fusion.
        fused_features = (
            rgb_features
            + self.alpha * gate * depth
        )

        return fused_features
