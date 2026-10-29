
"""CPU engineering test of production KITTI prediction isolation.

Does not train a model or produce KITTI benchmark results.
"""

import ast
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
TESTER_PATH = ROOT / "MonoIA/lib/helpers/tester_helper.py"


class DatasetFixture:
    split = "val"
    idx_list = ["000001", "000002"]

    def __len__(self):
        return len(self.idx_list)

    def eval(self, results_dir, logger):
        return results_dir


def load_production_tester_class():
    """Compile the actual Tester class without CUDA-only imports."""
    tree = ast.parse(TESTER_PATH.read_text(encoding="utf-8"))

    original_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Tester"
    )

    module = ast.Module(body=[original_class], type_ignores=[])
    ast.fix_missing_locations(module)

    import numpy as np
    import torch

    namespace = {
        "os": os,
        "json": json,
        "uuid": uuid,
        "datetime": datetime,
        "timezone": timezone,
        "Path": Path,
        "np": np,
        "torch": torch,
    }

    exec(compile(module, str(TESTER_PATH), "exec"), namespace)
    return namespace["Tester"]


def main():
    Tester = load_production_tester_class()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = DatasetFixture()

        def make_tester():
            tester = Tester.__new__(Tester)
            tester.output_dir = str(root)
            tester.prediction_dir = None
            tester.dataset_type = "KITTI"
            tester.dataloader = SimpleNamespace(dataset=dataset)
            tester.class_name = ["Car"]
            tester.logger = MagicMock()
            tester.checkpoint_path = str(
                root / "checkpoint_best.pth"
            )
            tester.expected_provenance = {
                "schema_version": 3,
                "identity_sha256": "a" * 64,
            }
            return tester

        # Two images: one detection and one empty prediction.
        results = {
            1: [[0, 1.0, 2.0, 3.0]],
            2: [],
        }

        first = make_tester()
        first.save_results(results)

        first_dir = Path(first.prediction_dir)

        assert first_dir.is_dir()
        assert (first_dir / "000001.txt").is_file()
        assert (first_dir / "000002.txt").is_file()
        assert (first_dir / "000002.txt").read_text() == ""

        first._verify_prediction_files()

        print("PASS: Complete KITTI prediction files created")
        print("PASS: Empty detection file preserved")

        metadata_path = first_dir.parent / "metadata.json"

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        assert metadata["prediction_count"] == 2
        assert metadata["split"] == "val"
        assert metadata["provenance_sha256"] == "a" * 64

        print("PASS: Evaluation metadata created")

        second = make_tester()
        second.save_results(results)

        assert first.prediction_dir != second.prediction_dir
        assert first_dir.is_dir()

        print("PASS: Evaluation runs use distinct directories")
        print("PASS: Previous predictions remain untouched")

        # Missing image ID must fail.
        incomplete = make_tester()

        try:
            incomplete.save_results({1: results[1]})
        except ValueError as error:
            assert "Incomplete KITTI predictions" in str(error)
        else:
            raise AssertionError(
                "Incomplete KITTI prediction set was accepted"
            )

        print("PASS: Incomplete predictions rejected")

        # Unexpected file must block evaluator.
        extra = first_dir / "999999.txt"
        extra.write_text("", encoding="utf-8")

        try:
            first.evaluate()
        except ValueError as error:
            assert "prediction file mismatch" in str(error)
        else:
            raise AssertionError(
                "Evaluator accepted unexpected prediction file"
            )

        extra.unlink()

        print("PASS: Unexpected prediction file rejected")

        result = first.evaluate()

        assert result == str(first_dir)

        print("PASS: Evaluator uses isolated prediction directory")

    print("PASS: KITTI prediction isolation runtime verification")


if __name__ == "__main__":
    main()