# Baseline Provenance

## Project

Depth-MonoIA

## Baseline

Towards Intrinsic-Aware Monocular 3D Object Detection
CVPR 2026

## Source

https://github.com/alanzhangcs/MonoIA

## Source Commit

PASTE_VERIFIED_MONOIA_COMMIT_SHA_HERE

## Depth Foundation Model

https://github.com/DepthAnything/Depth-Anything-V2

## Depth Source Commit

PASTE_VERIFIED_DEPTH_ANYTHING_COMMIT_SHA_HERE

## Baseline Modification Policy

The original MonoIA implementation must be preserved
for reproducible baseline comparison.

All Depth-MonoIA modifications must be documented.

## Dataset

KITTI 3D Object Detection Benchmark

## Current Status

Repository setup only.
Full training not performed.
AP3D results not yet generated.

## Depth Anything V2 Checkpoint

- Architecture: Depth Anything V2 Small (ViT-S)
- Checkpoint: depth_anything_v2_vits.pth
- Source: https://huggingface.co/depth-anything/Depth-Anything-V2-Small
- SHA-256: 715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378
- Checkpoint size: approximately 99.22 MB
- Strict loading: PASS
- Total parameters: 24,785,089
- Trainable encoder parameters: 0
- Encoder evaluation mode: PASS
