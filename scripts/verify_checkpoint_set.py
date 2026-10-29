
"""Check checkpoint-set metadata consistency.

Infrastructure validation only; no KITTI training or evaluation.
"""

import argparse
from pathlib import Path

import torch


def read_checkpoint(path):
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state, dict):
        raise ValueError(f"Invalid checkpoint: {path}")

    required = {
        "epoch",
        "best_epoch",
        "best_result",
        "model_state",
        "optimizer_state",
        "lr_scheduler_state",
        "rng_state",
    }

    missing = required - state.keys()
    if missing:
        raise ValueError(
            f"{path.name}: missing keys {sorted(missing)}"
        )

    epoch = state["epoch"]
    best_epoch = state["best_epoch"]

    if type(epoch) is not int or type(best_epoch) is not int:
        raise ValueError("Checkpoint epoch fields must be integers")

    if epoch < 1 or not 0 <= best_epoch <= epoch:
        raise ValueError("Invalid checkpoint epoch metadata")

    for key in (
        "model_state",
        "optimizer_state",
        "lr_scheduler_state",
        "rng_state",
    ):
        if state[key] is None:
            raise ValueError(f"{path.name}: missing {key}")

    return state


def verify_checkpoint_set(directory):
    directory = Path(directory).resolve()

    latest_path = directory / "checkpoint.pth"
    best_path = directory / "checkpoint_best.pth"

    if not latest_path.is_file():
        raise FileNotFoundError(latest_path)

    latest = read_checkpoint(latest_path)
    print("PASS: Latest checkpoint readable and structured")

    if not best_path.is_file():
        raise FileNotFoundError(
            "Best checkpoint missing: " + str(best_path)
        )

    best = read_checkpoint(best_path)
    print("PASS: Best checkpoint readable and structured")

    if latest["best_epoch"] != best["best_epoch"]:
        raise ValueError(
            "Best epoch mismatch: "
            f"latest={latest['best_epoch']}, "
            f"best={best['best_epoch']}"
        )

    if latest["best_result"] != best["best_result"]:
        raise ValueError(
            "Best result metadata mismatch"
        )

    if best["epoch"] != latest["best_epoch"]:
        raise ValueError(
            "Best checkpoint epoch disagrees with latest metadata"
        )

    print("PASS: Latest/best metadata consistent")
    print("PASS: Checkpoint-set metadata audit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_directory",
        type=Path,
        help="Directory containing checkpoint.pth and checkpoint_best.pth",
    )
    args = parser.parse_args()
    verify_checkpoint_set(args.run_directory)