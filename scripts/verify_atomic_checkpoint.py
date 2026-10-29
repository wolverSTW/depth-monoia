
"""
Verify atomic checkpoint save/load behavior.

This is an infrastructure check.
No KITTI dataset, mock benchmark results, or model training.
"""

import importlib.util
import tempfile
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAVE_HELPER = (
    ROOT
    / "MonoIA"
    / "lib"
    / "helpers"
    / "save_helper.py"
)

spec = importlib.util.spec_from_file_location(
    "save_helper",
    SAVE_HELPER,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory(
    prefix="depth_monoia_checkpoint_"
) as directory:

    checkpoint_base = Path(directory) / "checkpoint"
    checkpoint_file = Path(directory) / "checkpoint.pth"

    # Engineering metadata only; no model metrics.
    first_state = {
        "epoch": 1,
        "model_state": None,
        "optimizer_state": None,
        "best_result": 0.0,
        "best_epoch": 0,
    }

    module.save_checkpoint(
        first_state,
        str(checkpoint_base),
    )

    assert checkpoint_file.is_file()

    loaded = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=True,
    )

    assert loaded["epoch"] == 1

    print("PASS: Initial checkpoint saved and loaded")

    # Verify replacing an existing checkpoint.
    second_state = dict(first_state)
    second_state["epoch"] = 2

    module.save_checkpoint(
        second_state,
        str(checkpoint_base),
    )

    loaded = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=True,
    )

    assert loaded["epoch"] == 2

    print("PASS: Existing checkpoint replaced correctly")

    # The temporary file should not remain after success.
    temporary_files = list(
        Path(directory).glob(".checkpoint_*.tmp")
    )

    assert not temporary_files

    print("PASS: No temporary checkpoint files remain")

print("PASS: Atomic checkpoint round-trip verification")