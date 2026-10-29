
"""Verify RNG restoration through the real checkpoint loader."""

import importlib.util
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HELPER = ROOT / "MonoIA/lib/helpers/save_helper.py"

spec = importlib.util.spec_from_file_location(
    "save_helper_rng_resume",
    HELPER,
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

random.seed(444)
np.random.seed(444)
torch.manual_seed(444)

state = module.get_checkpoint_state(
    epoch=7,
    include_rng_state=True,
)

with tempfile.TemporaryDirectory() as directory:
    base = str(Path(directory) / "checkpoint")
    module.save_checkpoint(state, base)

    expected_python = [random.random() for _ in range(5)]
    expected_numpy = np.random.random(5)
    expected_torch = torch.rand(5)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    epoch, _, _ = module.load_checkpoint(
        model=None,
        optimizer=None,
        filename=base + ".pth",
        map_location="cpu",
        logger=None,
        restore_rng=True,
    )

    assert epoch == 7

    assert [random.random() for _ in range(5)] == expected_python
    print("PASS: Python RNG restored by load_checkpoint")

    assert np.array_equal(np.random.random(5), expected_numpy)
    print("PASS: NumPy RNG restored by load_checkpoint")

    assert torch.equal(torch.rand(5), expected_torch)
    print("PASS: PyTorch CPU RNG restored by load_checkpoint")

print("PASS: Training resume RNG integration")