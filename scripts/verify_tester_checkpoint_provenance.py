
"""Verify production Tester checkpoint provenance enforcement.

CPU-only infrastructure verification.
Not a KITTI benchmark or model-training experiment.
"""

import ast
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

ROOT = Path(__file__).resolve().parents[1]
MONOIA_DIR = ROOT / "MonoIA"

for directory in (ROOT, MONOIA_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.checkpoint_provenance import (
    assert_provenance_matches,
)
from scripts.provenance_test_fixture import (
    build_test_provenance,
)

import yaml


def load_production_test_method():
    """Extract and execute the actual Tester.test method."""
    path = ROOT / "MonoIA/lib/helpers/tester_helper.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    tester_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Tester"
    )

    test_method = next(
        node for node in tester_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test"
    )

    module = ast.Module(
        body=[test_method],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    from lib.helpers.save_helper import load_checkpoint

    namespace = {
        "os": __import__("os"),
        "load_checkpoint": load_checkpoint,
    }

    exec(compile(module, str(path), "exec"), namespace)
    return namespace["test"]


def main():
    cfg_path = ROOT / "MonoIA/config/monoia_local.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    provenance = build_test_provenance(cfg, seed=444)

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        checkpoint = directory / "checkpoint_best.pth"

        model = torch.nn.Linear(2, 1)
        initial_state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

        torch.save(
            {
                "epoch": 1,
                "model_state": initial_state,
                "provenance": provenance,
            },
            checkpoint,
        )

        test_method = load_production_test_method()

        def build_tester(expected):
            return SimpleNamespace(
                checkpoint_path=str(checkpoint),
                output_dir=str(directory),
                expected_provenance=expected,
                model=model,
                device=torch.device("cpu"),
                logger=MagicMock(),
                calculate_flops=MagicMock(),
                inference=MagicMock(),
                evaluate=MagicMock(return_value=0.0),
                dataloader=SimpleNamespace(
                    dataset=SimpleNamespace(split="val")
                ),
            )

        # Case 1: missing expected provenance
        tester = build_tester(None)

        try:
            test_method(tester)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Tester accepted missing expected provenance"
            )

        tester.inference.assert_not_called()
        print("PASS: Missing expected provenance rejected")

        # Case 2: mismatched training provenance
        changed = dict(provenance)
        changed["identity_sha256"] = "0" * 64

        tester = build_tester(changed)

        try:
            test_method(tester)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Tester accepted mismatched provenance"
            )

        tester.inference.assert_not_called()

        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, initial_state[name])

        print("PASS: Mismatched provenance rejected")
        print("PASS: Model weights preserved after rejection")

        # Case 3: matching provenance
        tester = build_tester(provenance)
        result = test_method(tester)

        tester.inference.assert_called_once()
        tester.evaluate.assert_called_once()
        assert result == 0.0

        print("PASS: Matching provenance accepted")
        print("PASS: Validation evaluation path executed")

    print("PASS: Tester checkpoint provenance runtime verification")


if __name__ == "__main__":
    main()
