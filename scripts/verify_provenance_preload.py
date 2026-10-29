
"""Verify provenance rejection before model state is modified.

Infrastructure verification only; no KITTI model training.
"""

import copy
import importlib.util
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.provenance_test_fixture import (
    build_test_provenance,
)


HELPER_PATH = ROOT / "MonoIA/lib/helpers/save_helper.py"

spec = importlib.util.spec_from_file_location(
    "save_helper_provenance_test",
    HELPER_PATH,
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

with (ROOT / "MonoIA/config/monoia_local.yaml").open(
    encoding="utf-8"
) as file:
    cfg = yaml.safe_load(file)

imagesets = ROOT / "MonoIA/data/KITTIDataset/ImageSets"

original_provenance = build_test_provenance(
    cfg,
    seed=444,
)

changed_cfg = copy.deepcopy(cfg)
changed_cfg["optimizer"]["lr"] = 0.0001

mismatched_provenance = build_test_provenance(
    changed_cfg,
    seed=444,
)


random.seed(444)
np.random.seed(444)
torch.manual_seed(444)

saved_model = torch.nn.Linear(2, 1)

with tempfile.TemporaryDirectory() as directory:
    checkpoint_base = str(Path(directory) / "checkpoint")

    checkpoint = helper.get_checkpoint_state(
        model=saved_model,
        epoch=1,
        include_rng_state=True,
    )
    checkpoint["provenance"] = original_provenance

    helper.save_checkpoint(checkpoint, checkpoint_base)

    target_model = torch.nn.Linear(2, 1)

    before_weights = {
        key: value.clone()
        for key, value in target_model.state_dict().items()
    }

    before_rng = helper.capture_rng_state()

    try:
        helper.load_checkpoint(
            model=target_model,
            optimizer=None,
            filename=checkpoint_base + ".pth",
            map_location="cpu",
            logger=None,
            restore_rng=True,
            expected_provenance=mismatched_provenance,
        )
    except ValueError:
        print("PASS: Provenance mismatch rejected")
    else:
        raise AssertionError("Mismatched checkpoint was accepted")

    for key, original in before_weights.items():
        assert torch.equal(
            original,
            target_model.state_dict()[key],
        ), f"Model was modified: {key}"

    print("PASS: Model unchanged after rejection")

    after_rng = helper.capture_rng_state()

    assert before_rng["python"] == after_rng["python"]
    assert torch.equal(
        before_rng["numpy"]["keys"],
        after_rng["numpy"]["keys"],
    )
    assert torch.equal(
        before_rng["torch_cpu"],
        after_rng["torch_cpu"],
    )

    print("PASS: RNG unchanged after rejection")

    helper.load_checkpoint(
        model=target_model,
        optimizer=None,
        filename=checkpoint_base + ".pth",
        map_location="cpu",
        logger=None,
        expected_provenance=original_provenance,
    )

    for key, original in saved_model.state_dict().items():
        assert torch.equal(
            original,
            target_model.state_dict()[key],
        )

    print("PASS: Matching provenance accepted")

print("PASS: Checkpoint provenance preload verification")