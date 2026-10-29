
"""
Verify MonoIA optimizer and LR scheduler checkpoint restoration.

Infrastructure verification only:
- Uses a tiny PyTorch parameter module
- Uses the repository's real custom AdamW and LambdaLR
- Does not train MonoIA or generate KITTI benchmark results
"""

import importlib.util
import tempfile
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


save_helper = load_module(
    "monoia_save_helper",
    "MonoIA/lib/helpers/save_helper.py",
)

optimizer_helper = load_module(
    "monoia_optimizer_helper",
    "MonoIA/lib/helpers/optimizer_helper.py",
)

scheduler_helper = load_module(
    "monoia_scheduler_helper",
    "MonoIA/lib/helpers/scheduler_helper.py",
)

config_path = ROOT / "MonoIA/config/monoia_local.yaml"

with config_path.open(encoding="utf-8") as file:
    cfg = yaml.safe_load(file)


def build_components():
    model = torch.nn.Linear(2, 1)

    optimizer = optimizer_helper.build_optimizer(
        cfg["optimizer"],
        model,
    )

    lr_scheduler, warmup_scheduler = (
        scheduler_helper.build_lr_scheduler(
            cfg["lr_scheduler"],
            optimizer,
            last_epoch=-1,
        )
    )

    return model, optimizer, lr_scheduler, warmup_scheduler


def read_lrs(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


torch.manual_seed(444)

model, optimizer, scheduler, warmup = build_components()

assert warmup is None, (
    "This verification expects warmup=False in monoia_local.yaml"
)

# Exercise the actual custom optimizer using explicit engineering
# gradients, not KITTI samples or detection training.
for _ in range(3):
    optimizer.zero_grad()

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    optimizer.step()
    scheduler.step()

saved_lrs = read_lrs(optimizer)
saved_scheduler_state = scheduler.state_dict()
saved_optimizer_state = optimizer.state_dict()

with tempfile.TemporaryDirectory() as directory:
    checkpoint_base = str(Path(directory) / "checkpoint")
    checkpoint_path = checkpoint_base + ".pth"

    state = save_helper.get_checkpoint_state(
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_result=0.0,
        best_epoch=0,
        lr_scheduler=scheduler,
        warmup_lr_scheduler=warmup,
    )

    save_helper.save_checkpoint(state, checkpoint_base)

    assert Path(checkpoint_path).is_file()

    print("PASS: Optimizer/scheduler checkpoint saved")

    restored_model, restored_optimizer, restored_scheduler, restored_warmup = (
        build_components()
    )

    epoch, best_result, best_epoch = save_helper.load_checkpoint(
        model=restored_model,
        optimizer=restored_optimizer,
        filename=checkpoint_path,
        map_location="cpu",
        logger=None,
        lr_scheduler=restored_scheduler,
        warmup_lr_scheduler=restored_warmup,
    )

    assert epoch == 3

    assert read_lrs(restored_optimizer) == saved_lrs, (
        "Optimizer learning rates were not restored"
    )

    assert restored_scheduler.state_dict() == saved_scheduler_state, (
        "LR scheduler state mismatch"
    )

    assert (
        restored_optimizer.state_dict()["param_groups"]
        == saved_optimizer_state["param_groups"]
    ), "Optimizer parameter-group state mismatch"

    for original, restored in zip(
        model.parameters(),
        restored_model.parameters(),
    ):
        assert torch.equal(original, restored), (
            "Model parameter state mismatch"
        )

    print("PASS: Model weights restored")
    print("PASS: Optimizer parameter groups restored")
    print("PASS: LR scheduler state restored")
    print("PASS: Epoch restored")

    # Verify that both schedulers continue with identical LR values.
    for _ in range(3):
        scheduler.step()
        restored_scheduler.step()

        assert read_lrs(optimizer) == read_lrs(restored_optimizer), (
            "Learning-rate trajectories diverged"
        )

    print("PASS: Learning-rate continuation matches")

print("PASS: MonoIA optimizer/scheduler resume round-trip")