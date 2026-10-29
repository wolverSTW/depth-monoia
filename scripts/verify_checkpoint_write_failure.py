
"""Verify checkpoint preservation after a simulated write failure.

Infrastructure verification only; no KITTI data or model training.
"""

import importlib.util
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HELPER_PATH = ROOT / "MonoIA/lib/helpers/save_helper.py"

spec = importlib.util.spec_from_file_location(
    "save_helper_failure_check",
    HELPER_PATH,
)

helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    base = directory / "checkpoint"
    destination = directory / "checkpoint.pth"

    original_state = {"epoch": 1}
    new_state = {"epoch": 2}

    # Step 1: Save an existing valid checkpoint.
    helper.save_checkpoint(original_state, str(base))

    assert destination.is_file()

    original_bytes = destination.read_bytes()

    loaded = torch.load(
        destination,
        map_location="cpu",
        weights_only=True,
    )

    assert loaded["epoch"] == 1

    print("PASS: Original checkpoint created")

    # Step 2: Force a failure during the next checkpoint write.
    real_torch_save = torch.save

    def failing_save(state, file, *args, **kwargs):
        # Simulate a partially written temporary file.
        file.write(b"INCOMPLETE_CHECKPOINT")
        raise OSError("Simulated checkpoint write failure")

    failure_observed = False

    with patch.object(
        helper.torch,
        "save",
        side_effect=failing_save,
    ):
        try:
            helper.save_checkpoint(new_state, str(base))
        except OSError as exc:
            assert str(exc) == (
                "Simulated checkpoint write failure"
            )
            failure_observed = True

    assert failure_observed, (
        "Expected checkpoint write failure was not raised"
    )

    print("PASS: Write failure propagated correctly")

    # Step 3: Verify the old checkpoint was not overwritten.
    assert destination.read_bytes() == original_bytes, (
        "Existing checkpoint changed after failed write"
    )

    preserved = torch.load(
        destination,
        map_location="cpu",
        weights_only=True,
    )

    assert preserved["epoch"] == 1

    print("PASS: Original checkpoint preserved")

    # Step 4: Verify temporary-file cleanup.
    remaining_temp_files = list(
        directory.glob(".checkpoint_*.tmp")
    )

    assert not remaining_temp_files, (
        f"Temporary files remain: {remaining_temp_files}"
    )

    print("PASS: Failed-write temporary file cleaned")

    # Step 5: Verify a subsequent legitimate save succeeds.
    helper.save_checkpoint(new_state, str(base))

    recovered = torch.load(
        destination,
        map_location="cpu",
        weights_only=True,
    )

    assert recovered["epoch"] == 2

    print("PASS: Subsequent checkpoint save succeeds")

print("PASS: Checkpoint write-failure recovery verification")