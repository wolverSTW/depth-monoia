
"""Verify baseline/enhanced Depth Anything V2 dependency rules."""

import copy
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.depth_dependency_provenance import (
    EXPECTED_VITS_SHA256,
    build_depth_dependency_provenance,
)


CONFIG_PATH = ROOT / "MonoIA/config/monoia_local.yaml"
CHECKPOINT_RELATIVE = (
    "weights/depth_anything_v2/depth_anything_v2_vits.pth"
)


def expect_rejection(action, expected_exception, message):
    try:
        action()
    except expected_exception:
        print(f"PASS: {message}")
    else:
        raise AssertionError(message)


def main():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    # Baseline must not depend on external depth files.
    baseline = copy.deepcopy(cfg)
    baseline["model"]["depth_fusion"]["enabled"] = False
    baseline["model"]["depth_fusion"]["checkpoint_path"] = None

    baseline_result = build_depth_dependency_provenance(
        baseline,
        ROOT,
    )

    assert baseline_result == {
        "schema_version": 1,
        "enabled": False,
    }

    print("PASS: Baseline does not require depth dependencies")

    # Enhanced mode must use the real, verified checkpoint.
    enhanced = copy.deepcopy(cfg)
    enhanced["model"]["depth_fusion"]["enabled"] = True
    enhanced["model"]["depth_fusion"][
        "checkpoint_path"
    ] = CHECKPOINT_RELATIVE

    first = build_depth_dependency_provenance(
        enhanced,
        ROOT,
    )

    second = build_depth_dependency_provenance(
        enhanced,
        ROOT,
    )

    assert first == second
    assert first["enabled"] is True
    assert first["checkpoint_sha256"] == EXPECTED_VITS_SHA256
    assert first["source_file_count"] > 0

    print("PASS: Enhanced dependency manifest deterministic")
    print("PASS: Official ViT-S checkpoint SHA256 verified")
    print(
        "PASS: External Python source files hashed:",
        first["source_file_count"],
    )

    missing = copy.deepcopy(enhanced)
    missing["model"]["depth_fusion"]["checkpoint_path"] = None

    expect_rejection(
        lambda: build_depth_dependency_provenance(
            missing,
            ROOT,
        ),
        ValueError,
        "Missing enhanced checkpoint path rejected",
    )

    invalid_encoder = copy.deepcopy(enhanced)
    invalid_encoder["model"]["depth_fusion"]["encoder"] = "unknown"

    expect_rejection(
        lambda: build_depth_dependency_provenance(
            invalid_encoder,
            ROOT,
        ),
        ValueError,
        "Unsupported depth encoder rejected",
    )

    invalid_path = copy.deepcopy(enhanced)
    invalid_path["model"]["depth_fusion"][
        "checkpoint_path"
    ] = "weights/nonexistent_depth_checkpoint.pth"

    expect_rejection(
        lambda: build_depth_dependency_provenance(
            invalid_path,
            ROOT,
        ),
        FileNotFoundError,
        "Missing pretrained file rejected",
    )

    print("PASS: Depth dependency provenance verification")


if __name__ == "__main__":
    main()