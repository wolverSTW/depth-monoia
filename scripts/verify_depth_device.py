
"""
Verify Depth Anything V2 device, dtype and frozen state.

Uses the authentic pretrained checkpoint.
Does not run KITTI inference or benchmark evaluation.
"""

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

DEPTH_REPO = ROOT / "external" / "Depth-Anything-V2"

CHECKPOINT = (
    ROOT
    / "weights"
    / "depth_anything_v2"
    / "depth_anything_v2_vits.pth"
)

EXTRACTOR_FILE = (
    ROOT
    / "MonoIA"
    / "lib"
    / "models"
    / "monoia"
    / "depth_extractor.py"
)

assert DEPTH_REPO.is_dir(), "Depth Anything V2 repository missing"
assert CHECKPOINT.is_file(), "Pretrained checkpoint missing"

sys.path.insert(0, str(DEPTH_REPO))

spec = importlib.util.spec_from_file_location(
    "depth_extractor",
    EXTRACTOR_FILE,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

extractor = module.DepthFeatureExtractor(
    checkpoint_path=str(CHECKPOINT),
    encoder="vits",
)

device = torch.device("cpu")
extractor = extractor.to(device)

params = list(extractor.parameters())

assert params, "No model parameters found"

assert all(
    parameter.device == device
    for parameter in params
), "Parameters are on different devices"

print("PASS: All parameters on CPU")

dtypes = {parameter.dtype for parameter in params}

print("Parameter dtypes:", dtypes)

assert dtypes == {torch.float32}, (
    f"Unexpected parameter dtypes: {dtypes}"
)

print("PASS: Pretrained weights use float32")

assert all(
    not parameter.requires_grad
    for parameter in params
), "Frozen encoder contains trainable parameters"

print("PASS: All pretrained parameters frozen")

extractor.train()

assert not extractor.model.training, (
    "Frozen encoder unexpectedly entered training mode"
)

print("PASS: Encoder remains in evaluation mode")

print("PASS: Depth device/dtype verification")
