
"""
Verify zero-score best checkpoint creation and provenance.

Infrastructure verification only:
- Tests the real Trainer.train() checkpoint control flow.
- Checks that a first zero validation score creates a best file.
- Checks latest/best checkpoint metadata and provenance.
- Does not train MonoIA or generate KITTI benchmark results.
"""

import ast
import importlib.util
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import tqdm
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
from scripts.source_fingerprint import build_source_fingerprint



SAVE_HELPER_PATH = (
    ROOT / "MonoIA/lib/helpers/save_helper.py"
)
TRAINER_PATH = (
    ROOT / "MonoIA/lib/helpers/trainer_helper.py"
)
CONFIG_PATH = (
    ROOT / "MonoIA/config/monoia_local.yaml"
)
IMAGESETS_DIR = (
    ROOT / "MonoIA/data/KITTIDataset/ImageSets"
)


def make_test_provenance():
    """Build complete provenance from real project inputs."""
    config_path = ROOT / "MonoIA/config/monoia_local.yaml"
    imagesets_dir = ROOT / "MonoIA/data/KITTIDataset/ImageSets"

    with config_path.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    source_manifest = build_source_fingerprint(ROOT)

    depth_manifest = build_depth_dependency_provenance(
        cfg=cfg,
        repository_root=ROOT,
    )

    return build_provenance(
        cfg=cfg,
        effective_seed=444,
        imagesets_dir=imagesets_dir,
        source_manifest=source_manifest,
        depth_manifest=depth_manifest,
    )




def import_save_helper():
    """Import the production checkpoint helper."""
    spec = importlib.util.spec_from_file_location(
        "save_helper_zero_score_test",
        SAVE_HELPER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot load save_helper.py"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def extract_training_loop(helper):
    """Extract the actual Trainer.train() method."""
    source = TRAINER_PATH.read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    trainer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Trainer"
    )

    train_method = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "train"
    )

    test_class = ast.ClassDef(
        name="TrainingLoopUnderTest",
        bases=[],
        keywords=[],
        body=[train_method],
        decorator_list=[],
    )

    module_ast = ast.fix_missing_locations(
        ast.Module(
            body=[test_class],
            type_ignores=[],
        )
    )

    namespace = {
        "os": os,
        "np": np,
        "tqdm": tqdm,
        "save_checkpoint": helper.save_checkpoint,
        "get_checkpoint_state": (
            helper.get_checkpoint_state
        ),
    }

    exec(
        compile(
            module_ast,
            str(TRAINER_PATH),
            "exec",
        ),
        namespace,
    )

    return namespace["TrainingLoopUnderTest"]


class ZeroScoreTester:
    """Controlled zero-score validation for checkpoint logic."""

    def __init__(self):
        self.inference_calls = 0
        self.evaluate_calls = 0

    def inference(self):
        self.inference_calls += 1

    def evaluate(self):
        self.evaluate_calls += 1

        # Control value, NOT a real KITTI AP score.
        return 0.0


def read_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


def verify_provenance(state, expected):
    """Fail if provenance is missing or mismatched."""
    assert "provenance" in state, (
        "Checkpoint has no provenance field"
    )

    assert_provenance_matches(
        saved=state["provenance"],
        current=expected,
    )


def main():
    torch.manual_seed(444)
    np.random.seed(444)

    helper = import_save_helper()
    TrainingLoopUnderTest = extract_training_loop(
        helper
    )
    provenance = make_test_provenance()

    with tempfile.TemporaryDirectory(
        prefix="monoia_zero_score_"
    ) as directory:

        trainer = TrainingLoopUnderTest()

        trainer.model = torch.nn.Linear(2, 1)

        trainer.optimizer = torch.optim.SGD(
            trainer.model.parameters(),
            lr=0.01,
        )

        trainer.lr_scheduler = (
            torch.optim.lr_scheduler.LambdaLR(
                trainer.optimizer,
                lr_lambda=lambda epoch: 1.0,
            )
        )

        trainer.warmup_lr_scheduler = None

        trainer.cfg = {
            "max_epoch": 2,
            "save_frequency": 1,
            "save_all": False,
        }

        trainer.output_dir = directory
        trainer.logger = logging.getLogger(
            "zero_score_checkpoint"
        )
        trainer.tester = ZeroScoreTester()

        trainer.epoch = 0
        trainer.best_result = 0.0
        trainer.best_epoch = 0
        trainer.experiment_provenance = provenance

        # Actual optimizer step, without detection training.
        def engineering_optimizer_step(epoch):
            trainer.optimizer.zero_grad()

            for parameter in trainer.model.parameters():
                parameter.grad = torch.ones_like(
                    parameter
                )

            trainer.optimizer.step()

        trainer.train_one_epoch = (
            engineering_optimizer_step
        )

        trainer.train()

        directory_path = Path(directory)

        latest_path = (
            directory_path / "checkpoint.pth"
        )
        best_path = (
            directory_path / "checkpoint_best.pth"
        )

        assert latest_path.is_file(), (
            "Missing latest checkpoint"
        )
        assert best_path.is_file(), (
            "Missing best checkpoint for zero score"
        )

        print(
            "PASS: Best checkpoint created for zero score"
        )

        latest = read_checkpoint(latest_path)
        best = read_checkpoint(best_path)

        # Latest checkpoint must represent epoch 2.
        assert latest["epoch"] == 2
        assert latest["best_epoch"] == 1
        assert latest["best_result"] == 0.0

        print(
            "PASS: Latest checkpoint metadata consistent"
        )

        # First validation established the best checkpoint.
        assert best["epoch"] == 1
        assert best["best_epoch"] == 1
        assert best["best_result"] == 0.0

        print(
            "PASS: Zero-score best checkpoint preserved"
        )

        assert trainer.tester.inference_calls == 2
        assert trainer.tester.evaluate_calls == 2

        print(
            "PASS: Both zero-score validation calls executed"
        )

        # Verify model and training states are present.
        for name, state in {
            "latest": latest,
            "best": best,
        }.items():
            assert state["model_state"] is not None, name
            assert state["optimizer_state"] is not None, name
            assert state["lr_scheduler_state"] is not None, name
            assert state["rng_state"] is not None, name

        print(
            "PASS: Latest and best contain complete "
            "training-state fields"
        )

        # Check actual provenance values saved in both files.
        verify_provenance(
            latest,
            provenance,
        )
        verify_provenance(
            best,
            provenance,
        )

        print(
            "PASS: Latest and best checkpoint provenance verified"
        )

        # Check the in-memory Trainer metadata too.
        assert trainer.epoch == 2
        assert trainer.best_epoch == 1
        assert trainer.best_result == 0.0

        print(
            "PASS: Trainer zero-score metadata consistent"
        )

        # save_all=False must not create epoch snapshots.
        assert not list(
            directory_path.glob("checkpoint_epoch_*.pth")
        )

        print(
            "PASS: save_all=False produces no epoch snapshots"
        )

    print(
        "PASS: Zero-score checkpoint control-flow verification"
    )


if __name__ == "__main__":
    main()
