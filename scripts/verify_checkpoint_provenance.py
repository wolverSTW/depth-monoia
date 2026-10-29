
"""Verify checkpoint provenance under schema v3."""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_provenance import (
    assert_provenance_matches,
)
from scripts.provenance_test_fixture import (
    build_test_provenance,
    load_test_config,
)


def expect_rejection(saved, current, label):
    try:
        assert_provenance_matches(saved, current)
    except ValueError:
        print(f"PASS: {label}")
    else:
        raise AssertionError(f"Expected rejection: {label}")


def main():
    cfg = load_test_config()

    original = build_test_provenance(cfg, seed=444)
    repeated = build_test_provenance(
        copy.deepcopy(cfg), seed=444
    )

    assert original == repeated
    assert original["schema_version"] == 3
    assert "source" in original
    assert "depth_dependency" in original

    assert_provenance_matches(original, repeated)
    print("PASS: Identical schema v3 provenance accepted")

    changed_model = copy.deepcopy(cfg)
    changed_model["model"]["depth_fusion"]["enabled"] = True

    # Model mismatch without requiring a real enhanced run.
    # Change a model configuration field unrelated to dependency mode.
    changed_model = copy.deepcopy(cfg)
    changed_model["model"]["depth_fusion"]["encoder"] = "different"

    changed_model_provenance = build_test_provenance(
        changed_model, seed=444
    )
    expect_rejection(
        original,
        changed_model_provenance,
        "Changed model configuration rejected",
    )

    changed_seed_provenance = build_test_provenance(
        cfg, seed=123
    )
    expect_rejection(
        original,
        changed_seed_provenance,
        "Changed effective seed rejected",
    )

    changed_optimizer = copy.deepcopy(cfg)
    changed_optimizer["optimizer"]["lr"] = 0.0001

    changed_optimizer_provenance = build_test_provenance(
        changed_optimizer, seed=444
    )
    expect_rejection(
        original,
        changed_optimizer_provenance,
        "Changed optimizer configuration rejected",
    )

    expect_rejection(
        None,
        original,
        "Missing provenance rejected",
    )

    print("PASS: Checkpoint provenance helper verification")


if __name__ == "__main__":
    main()