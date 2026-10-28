
"""
Frozen Depth Anything V2 feature extractor.

Research project: Depth-MonoIA
Encoder: ViT-S (DINOv2)
Output: High-level feature map [B, 384, H/14, W/14]

This module does not modify the original MonoIA model.
"""

import torch
from torch import nn

from depth_anything_v2.dpt import DepthAnythingV2


class DepthFeatureExtractor(nn.Module):
    def __init__(
        self,
        checkpoint_path=None,
        encoder="vits",
    ):
        super().__init__()

        if encoder != "vits":
            raise ValueError(
                "Only ViT-S is currently supported."
            )

        self.encoder_name = encoder

        self.model = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
        )

        if checkpoint_path is not None:
            state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            self.model.load_state_dict(
                state,
                strict=True,
            )
        else:
            raise ValueError(
                "A pretrained Depth Anything V2 "
                "checkpoint is required."
            )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.model.eval()

        self.feature_channels = 384
        self.patch_size = 14

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self


    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4:
            raise ValueError("Expected BCHW image tensor")

        if images.shape[1] != 3:
            raise ValueError("Expected 3 RGB channels")

        height, width = images.shape[-2:]

        # Images are already RGB and ImageNet-normalized
        # by the MonoIA KITTI data pipeline.
        patch_h = max(
            self.patch_size,
            round(height / self.patch_size)
            * self.patch_size,
        )
        patch_w = max(
            self.patch_size,
            round(width / self.patch_size)
            * self.patch_size,
        )

        if (patch_h, patch_w) != (height, width):
            images = torch.nn.functional.interpolate(
                images,
                size=(patch_h, patch_w),
                mode="bilinear",
                align_corners=False,
            )

        layer_ids = self.model.intermediate_layer_idx[
            self.encoder_name
        ]

        features = self.model.pretrained.get_intermediate_layers(
            images,
            layer_ids,
            reshape=True,
        )

        return features[-1]