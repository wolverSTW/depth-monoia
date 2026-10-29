
"""
Verify complete custom AdamW optimizer state restoration.

Infrastructure verification only.
No KITTI data, MonoIA training, or benchmark metrics.
"""

import importlib.util
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


save_helper = load_module(
    "save_helper_state_check",
    "MonoIA/lib/helpers/save_helper.py",
)

optimizer_helper = load_module(
    "optimizer_helper_state_check",
    "MonoIA/lib/helpers/optimizer_helper.py",
)

scheduler_helper = load_module(
    "scheduler_helper_state_check",
    "MonoIA/lib/helpers/scheduler_helper.py",
)

with (ROOT / "MonoIA/config/monoia_local.yaml").open(
    encoding="utf-8"
) as file:
    cfg = yaml.safe_load(file)


def build_components():
    model = torch.nn.Linear(2, 1)

    optimizer = optimizer_helper.build_optimizer(
        cfg["optimizer"], model
    )

    scheduler, warmup = scheduler_helper.build_lr_scheduler(
        cfg["lr_scheduler"],
        optimizer,
        last_epoch=-1,
    )

    return model, optimizer, scheduler, warmup


def assert_state_equal(expected, actual, path="state"):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert torch.equal(expected, actual), (
            f"Tensor mismatch: {path}"
        )

    elif isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert expected.keys() == actual.keys(), (
            f"Key mismatch: {path}"
        )

        for key in expected:
            assert_state_equal(
                expected[key],
                actual[key],
                f"{path}.{key}",
            )

    elif isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual), path
        assert len(expected) == len(actual), path

        for index, (left, right) in enumerate(
            zip(expected, actual)
        ):
            assert_state_equal(
                left,
                right,
                f"{path}[{index}]",
            )

    else:
        assert expected == actual, (
            f"Value mismatch: {path}"
        )


torch.manual_seed(444)

model, optimizer, scheduler, warmup = build_components()

assert warmup is None, (
    "This verification expects warmup=False"
)

# Exercise the actual MonoIA custom optimizer.
for step in range(5):
    optimizer.zero_grad()

    for parameter in model.parameters():
        parameter.grad = torch.full_like(
            parameter,
            float(step + 1),
        )

    optimizer.step()
    scheduler.step()

expected_optimizer_state = optimizer.state_dict()
expected_scheduler_state = scheduler.state_dict()

assert expected_optimizer_state["state"], (
    "Optimizer state is empty"
)

with tempfile.TemporaryDirectory() as directory:
    base = str(Path(directory) / "checkpoint")
    path = base + ".pth"

    checkpoint = save_helper.get_checkpoint_state(
        model=model,
        optimizer=optimizer,
        epoch=5,
        best_result=0.0,
        best_epoch=0,
        lr_scheduler=scheduler,
        warmup_lr_scheduler=warmup,
    )

    save_helper.save_checkpoint(checkpoint, base)

    restored_model, restored_optimizer, restored_scheduler, restored_warmup = (
        build_components()
    )

    epoch, _, _ = save_helper.load_checkpoint(
        model=restored_model,
        optimizer=restored_optimizer,
        filename=path,
        map_location="cpu",
        logger=None,
        lr_scheduler=restored_scheduler,
        warmup_lr_scheduler=restored_warmup,
    )

    assert epoch == 5

    assert_state_equal(
        expected_optimizer_state,
        restored_optimizer.state_dict(),
        "optimizer",
    )

    print("PASS: Complete optimizer state restored")

    assert_state_equal(
        expected_scheduler_state,
        restored_scheduler.state_dict(),
        "scheduler",
    )

    print("PASS: Complete scheduler state restored")

    assert_state_equal(
        model.state_dict(),
        restored_model.state_dict(),
        "model",
    )

    print("PASS: Complete model state restored")

    # Verify continuation after resuming.
    for step in range(5, 10):
        for current_model, current_optimizer, current_scheduler in (
            (model, optimizer, scheduler),
            (restored_model, restored_optimizer, restored_scheduler),
        ):
            current_optimizer.zero_grad()

            for parameter in current_model.parameters():
                parameter.grad = torch.full_like(
                    parameter,
                    float(step + 1),
                )

            current_optimizer.step()
            current_scheduler.step()

        assert_state_equal(
            model.state_dict(),
            restored_model.state_dict(),
            "continued_model",
        )

        assert_state_equal(
            optimizer.state_dict(),
            restored_optimizer.state_dict(),
            "continued_optimizer",
        )

    print("PASS: Optimizer continuation matches")
    print("PASS: Complete checkpoint state verification")