
"""
Verify Trainer.train() checkpoint consistency and provenance.

Infrastructure verification only:
- Uses the actual Trainer.train() control flow.
- Checks latest, best, and epoch checkpoint files.
- Checks checkpoint metadata and provenance.
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
        "checkpoint_helper_consistency",
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
    """
    Extract the real Trainer.train() method.

    Avoid importing CUDA-dependent model components.
    """
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


class ValidationControlFlow:
    """Controlled validation values for infrastructure testing."""

    def __init__(self):
        # These are control values, NOT KITTI AP scores.
        self.values = iter([0.25, 0.10])
        self.inference_calls = 0
        self.evaluate_calls = 0

    def inference(self):
        self.inference_calls += 1

    def evaluate(self):
        self.evaluate_calls += 1
        return next(self.values)


def read_checkpoint(path):
    """Load an infrastructure test checkpoint safely."""
    if not path.is_file():
        raise FileNotFoundError(path)

    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


def verify_checkpoint_provenance(state, expected):
    """Validate provenance stored inside a checkpoint."""
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
        prefix="monoia_checkpoint_consistency_"
    ) as directory:

        trainer = TrainingLoopUnderTest()

        # Small engineering module, not the MonoIA model.
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
            "save_all": True,
        }

        trainer.output_dir = directory
        trainer.logger = logging.getLogger(
            "checkpoint_consistency"
        )
        trainer.tester = ValidationControlFlow()

        trainer.epoch = 0
        trainer.best_result = 0.0
        trainer.best_epoch = 0
        trainer.experiment_provenance = provenance

        # Real optimizer steps, no detection training.
        def verification_optimizer_step(epoch):
            trainer.optimizer.zero_grad()

            for parameter in trainer.model.parameters():
                parameter.grad = torch.ones_like(
                    parameter
                )

            trainer.optimizer.step()

        trainer.train_one_epoch = (
            verification_optimizer_step
        )

        # Execute the real Trainer.train() control flow.
        trainer.train()

        directory_path = Path(directory)

        latest_path = (
            directory_path / "checkpoint.pth"
        )
        best_path = (
            directory_path / "checkpoint_best.pth"
        )
        epoch1_path = (
            directory_path / "checkpoint_epoch_1.pth"
        )
        epoch2_path = (
            directory_path / "checkpoint_epoch_2.pth"
        )

        for checkpoint_path in (
            latest_path,
            best_path,
            epoch1_path,
            epoch2_path,
        ):
            assert checkpoint_path.is_file(), (
                f"Missing checkpoint: {checkpoint_path.name}"
            )

        print(
            "PASS: Latest, best and epoch checkpoints created"
        )

        latest = read_checkpoint(latest_path)
        best = read_checkpoint(best_path)
        epoch1 = read_checkpoint(epoch1_path)
        epoch2 = read_checkpoint(epoch2_path)

        # Latest checkpoint should represent epoch 2.
        assert latest["epoch"] == 2
        assert latest["best_epoch"] == 1
        assert latest["best_result"] == 0.25

        print(
            "PASS: Latest checkpoint metadata consistent"
        )

        # Best was achieved during epoch 1 and must remain.
        assert best["epoch"] == 1
        assert best["best_epoch"] == 1
        assert best["best_result"] == 0.25

        print(
            "PASS: Best checkpoint preserved after non-improvement"
        )

        assert epoch1["epoch"] == 1
        assert epoch1["best_epoch"] == 1
        assert epoch1["best_result"] == 0.25

        assert epoch2["epoch"] == 2
        assert epoch2["best_epoch"] == 1
        assert epoch2["best_result"] == 0.25

        print(
            "PASS: Epoch snapshots have correct metadata"
        )

        assert trainer.tester.inference_calls == 2
        assert trainer.tester.evaluate_calls == 2

        print(
            "PASS: Validation executed for both epochs"
        )

        # Verify complete checkpoint fields are present.
        for name, state in {
            "latest": latest,
            "best": best,
            "epoch1": epoch1,
            "epoch2": epoch2,
        }.items():
            assert state["model_state"] is not None, name
            assert state["optimizer_state"] is not None, name
            assert state["lr_scheduler_state"] is not None, name
            assert state["rng_state"] is not None, name

        print(
            "PASS: All checkpoints contain model, "
            "optimizer, scheduler and RNG states"
        )

        # Verify provenance in latest and best.
        verify_checkpoint_provenance(
            latest,
            provenance,
        )
        verify_checkpoint_provenance(
            best,
            provenance,
        )

        print(
            "PASS: Latest and best checkpoint provenance verified"
        )

        # Verify provenance in both epoch snapshots.
        verify_checkpoint_provenance(
            epoch1,
            provenance,
        )
        verify_checkpoint_provenance(
            epoch2,
            provenance,
        )

        print(
            "PASS: Epoch snapshot provenance verified"
        )

        # Trainer's in-memory metadata should agree.
        assert trainer.epoch == 2
        assert trainer.best_epoch == 1
        assert trainer.best_result == 0.25

        print(
            "PASS: Trainer in-memory metadata consistent"
        )

    print(
        "PASS: Checkpoint consistency control-flow verification"
    )


if __name__ == "__main__":
    main()
