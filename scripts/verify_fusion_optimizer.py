
"""
Verify that trainable depth-fusion parameters
are included exactly once in the original
MonoIA optimizer.

This checks optimizer wiring only.
It does not train a model or report metrics.
"""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name, path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


fusion_module = load_module(
    "depth_fusion",
    ROOT / "MonoIA/lib/models/monoia/depth_fusion.py",
)

optimizer_module = load_module(
    "optimizer_helper",
    ROOT / "MonoIA/lib/helpers/optimizer_helper.py",
)


Fusion = fusion_module.IntrinsicGuidedDepthFusion
build_optimizer = optimizer_module.build_optimizer


# Isolated module-level optimizer wiring check.
# No input images or fabricated training results.
class FusionParameterContainer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.depth_fusions = torch.nn.ModuleList([
            Fusion() for _ in range(4)
        ])


model = FusionParameterContainer()

cfg_optimizer = {
    "type": "adamw",
    "lr": 0.0002,
    "weight_decay": 0.0001,
}

optimizer = build_optimizer(cfg_optimizer, model)

optimizer_parameters = [
    param
    for group in optimizer.param_groups
    for param in group["params"]
]

optimizer_ids = [id(p) for p in optimizer_parameters]

assert len(optimizer_ids) == len(set(optimizer_ids)), (
    "Duplicate optimizer parameters found"
)

missing = []

for name, param in model.named_parameters():
    if param.requires_grad and id(param) not in optimizer_ids:
        missing.append(name)

assert not missing, (
    f"Trainable parameters missing: {missing}"
)

expected = sum(
    param.numel()
    for param in model.parameters()
    if param.requires_grad
)

found = sum(
    param.numel()
    for param in optimizer_parameters
    if param.requires_grad
)

print("Trainable parameters in model:", expected)
print("Trainable parameters in optimizer:", found)
print("Optimizer groups:", len(optimizer.param_groups))

assert expected == found

print("PASS: All fusion parameters included in optimizer")
print("PASS: No duplicate optimizer parameters")
