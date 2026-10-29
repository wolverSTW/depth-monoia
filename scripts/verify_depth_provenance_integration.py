
"""Verify depth dependency inclusion in experiment provenance.

Infrastructure verification only; no training or benchmark scores.
"""

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_provenance import (
    assert_provenance_matches,
    build_provenance,
)
from scripts.depth_dependency_provenance import (
    build_depth_dependency_provenance,
)
from scripts.source_fingerprint import (
    build_source_fingerprint,
)

CONFIG_PATH = ROOT / "MonoIA/config/monoia_local.yaml"
IMAGESETS = ROOT / "MonoIA/data/KITTIDataset/ImageSets"
WEIGHTS = "weights/depth_anything_v2/depth_anything_v2_vits.pth"


def build(cfg, source):
    depth = build_depth_dependency_provenance(cfg, ROOT)

    experiment = build_provenance(
        cfg=cfg,
        effective_seed=444,
        imagesets_dir=IMAGESETS,
        source_manifest=source,
        depth_manifest=depth,
    )

    return experiment, depth


def expect_rejection(operation, message):
    try:
        operation()
    except (ValueError, TypeError):
        print(f"PASS: {message}")
    else:
        raise AssertionError(message)


def main():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        base_cfg = yaml.safe_load(file)

    source = build_source_fingerprint(ROOT)

    baseline_cfg = copy.deepcopy(base_cfg)
    baseline_cfg["model"]["depth_fusion"]["enabled"] = False
    baseline_cfg["model"]["depth_fusion"]["checkpoint_path"] = None

    baseline, baseline_depth = build(baseline_cfg, source)

    assert baseline_depth == {
        "schema_version": 1,
        "enabled": False,
    }
    assert baseline["depth_dependency"] == baseline_depth
    assert_provenance_matches(baseline, baseline)

    print("PASS: Baseline depth dependency stored correctly")

    enhanced_cfg = copy.deepcopy(base_cfg)
    enhanced_cfg["model"]["depth_fusion"]["enabled"] = True
    enhanced_cfg["model"]["depth_fusion"]["encoder"] = "vits"
    enhanced_cfg["model"]["depth_fusion"]["checkpoint_path"] = WEIGHTS

    enhanced, enhanced_depth = build(enhanced_cfg, source)

    assert enhanced["depth_dependency"] == enhanced_depth
    assert enhanced_depth["enabled"] is True
    assert enhanced_depth["source_file_count"] == 38
    assert_provenance_matches(enhanced, enhanced)

    print("PASS: Enhanced dependency manifest stored correctly")

    assert baseline["identity_sha256"] != enhanced["identity_sha256"]

    expect_rejection(
        lambda: assert_provenance_matches(baseline, enhanced),
        "Baseline and enhanced checkpoint identities differ",
    )

    enhanced_again, _ = build(enhanced_cfg, source)
    assert enhanced == enhanced_again

    print("PASS: Enhanced experiment identity deterministic")

    # A dependency for the wrong model mode must be rejected.
    expect_rejection(
        lambda: build_provenance(
            cfg=baseline_cfg,
            effective_seed=444,
            imagesets_dir=IMAGESETS,
            source_manifest=source,
            depth_manifest=enhanced_depth,
        ),
        "Incorrect model-mode dependency rejected",
    )

    # Corrupt the nested external-source manifest.
    corrupted = copy.deepcopy(enhanced_depth)
    target = next(iter(corrupted["source_files"]))
    corrupted["source_files"][target] = "0" * 64

    expect_rejection(
        lambda: build_provenance(
            cfg=enhanced_cfg,
            effective_seed=444,
            imagesets_dir=IMAGESETS,
            source_manifest=source,
            depth_manifest=corrupted,
        ),
        "Corrupted external-source dependency rejected",
    )

    missing_weights = copy.deepcopy(enhanced_cfg)
    missing_weights["model"]["depth_fusion"][
        "checkpoint_path"
    ] = "weights/nonexistent_depth_checkpoint.pth"

    try:
        build(missing_weights, source)
    except FileNotFoundError:
        print("PASS: Enhanced experiment requires actual weights")
    else:
        raise AssertionError("Missing pretrained weights accepted")

    print("PASS: Depth provenance integration verification")


if __name__ == "__main__":
    main()