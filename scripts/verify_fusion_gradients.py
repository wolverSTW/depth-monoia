
"""
Verify gradient propagation in IntrinsicGuidedDepthFusion.

Engineering verification only:
- No KITTI training
- No benchmark scores
- No model-performance claims
"""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

MODULE_PATH = (
    ROOT / "MonoIA/lib/models/monoia/depth_fusion.py"
)

spec = importlib.util.spec_from_file_location(
    "depth_fusion",
    MODULE_PATH,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

Fusion = module.IntrinsicGuidedDepthFusion


def check_gradients(fusion, expected_nonzero):
    results = {}

    for name, parameter in fusion.named_parameters():
        gradient = parameter.grad

        if gradient is None:
            status = "NONE"
            norm = 0.0
        else:
            assert torch.isfinite(gradient).all(), (
                f"Non-finite gradient: {name}"
            )
            norm = gradient.norm().item()
            status = "NONZERO" if norm > 0 else "ZERO"

        results[name] = norm
        print(f"{name}: {status}, grad_norm={norm:.8f}")

    for name in expected_nonzero:
        assert results[name] > 0, (
            f"Expected nonzero gradient for {name}"
        )


def main():
    torch.manual_seed(444)

    fusion = Fusion()
    fusion.train()

    # Synthetic tensors are used ONLY to verify
    # differentiation, not research performance.
    rgb = torch.randn(2, 256, 8, 16)
    depth = torch.randn(2, 384, 4, 8)
    intrinsic = torch.randn(2, 256)

    print("\nSTEP 1: Initial alpha = 0")

    result = fusion(rgb, depth, intrinsic)

    assert result.shape == rgb.shape
    assert torch.allclose(result, rgb)

    loss = result.square().mean()
    loss.backward()

    check_gradients(
        fusion,
        expected_nonzero=["alpha"],
    )

    print("PASS: Initial residual identity")
    print("PASS: Alpha receives gradients")

    print("\nSTEP 2: Gradient flow with nonzero alpha")

    fusion.zero_grad(set_to_none=True)

    with torch.no_grad():
        fusion.alpha.fill_(0.1)

    result = fusion(rgb, depth, intrinsic)
    loss = result.square().mean()
    loss.backward()

    check_gradients(
        fusion,
        expected_nonzero=[
            "alpha",
            "depth_projection.weight",
            "intrinsic_projection.weight",
            "intrinsic_projection.bias",
            "gate.0.weight",
            "gate.0.bias",
        ],
    )

    print("PASS: All fusion components receive gradients")


if __name__ == "__main__":
    main()
