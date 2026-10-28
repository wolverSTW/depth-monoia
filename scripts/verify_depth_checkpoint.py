
"""
Depth Anything V2 pretrained checkpoint verification.

Checks:
1. Real pretrained checkpoint loading
2. Frozen parameters
3. Evaluation mode
4. Frozen state after train() call

No KITTI training or benchmark evaluation.
"""

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

DEPTH_REPO = ROOT / "external" / "Depth-Anything-V2"
sys.path.insert(0, str(DEPTH_REPO))

EXTRACTOR_PATH = (
    ROOT / "MonoIA/lib/models/monoia/depth_extractor.py"
)

CHECKPOINT_PATH = (
    ROOT
    / "weights"
    / "depth_anything_v2"
    / "depth_anything_v2_vits.pth"
)

assert CHECKPOINT_PATH.is_file(), (
    f"Checkpoint missing: {CHECKPOINT_PATH}"
)

spec = importlib.util.spec_from_file_location(
    "depth_extractor",
    EXTRACTOR_PATH,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

DepthFeatureExtractor = module.DepthFeatureExtractor

print("Loading Depth Anything V2 ViT-S checkpoint...")

extractor = DepthFeatureExtractor(
    checkpoint_path=str(CHECKPOINT_PATH),
    encoder="vits",
)

print("PASS: Strict pretrained checkpoint loading")

total_params = sum(
    p.numel() for p in extractor.parameters()
)

trainable_params = sum(
    p.numel()
    for p in extractor.parameters()
    if p.requires_grad
)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)

assert trainable_params == 0, (
    "Depth encoder must remain frozen"
)

assert not extractor.model.training, (
    "Depth encoder must be in evaluation mode"
)

print("PASS: All encoder parameters frozen")
print("PASS: Encoder is in evaluation mode")

# Verify that calling train() does not unfreeze
# or switch the pretrained model into training mode.
extractor.train()

assert not extractor.model.training
assert all(
    not p.requires_grad
    for p in extractor.parameters()
)

print("PASS: Encoder remains frozen after train()")
print("PASS: Depth checkpoint runtime verification")
