import os
import random

import numpy as np
import torch
import torch.nn as nn
from scripts.checkpoint_provenance import assert_provenance_matches



def capture_rng_state():
    """Capture Python, NumPy, PyTorch CPU and CUDA RNG states."""
    numpy_state = np.random.get_state()

    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [s.cpu().clone() for s in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }

    return state


def restore_rng_state(state):
    """Restore RNG states saved by capture_rng_state()."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}

    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Incomplete RNG checkpoint state")

    random.setstate(state["python"])

    numpy_state = state["numpy"]

    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32,copy=True,),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )

    torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_states = state["torch_cuda"]

    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state, "
                "but CUDA is not available."
            )

        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from checkpoint RNG state."
            )

        torch.cuda.set_rng_state_all(
            [s.cpu() for s in cuda_states]
        )



def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu



def get_checkpoint_state(
    model=None,
    optimizer=None,
    epoch=None,
    best_result=None,
    best_epoch=None,
    lr_scheduler=None,
    warmup_lr_scheduler=None,
    include_rng_state=False,
    provenance=None,
):
    optim_state = (
        optimizer.state_dict()
        if optimizer is not None
        else None
    )

    if model is None:
        model_state = None
    elif isinstance(model, torch.nn.DataParallel):
        model_state = model_state_to_cpu(
            model.module.state_dict()
        )
    else:
        model_state = model_state_to_cpu(
            model.state_dict()
        )

    return {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optim_state,
        "best_result": best_result,
        "best_epoch": best_epoch,
        "lr_scheduler_state": (
            lr_scheduler.state_dict()
            if lr_scheduler is not None
            else None
        ),
        "warmup_lr_scheduler_state": (
            warmup_lr_scheduler.state_dict()
            if warmup_lr_scheduler is not None
            else None
        ),
        "rng_state": capture_rng_state() if include_rng_state else None,
        "provenance": provenance,
    }




def save_checkpoint(state, filename):
    """
    Atomically save a PyTorch checkpoint.

    The checkpoint is first written to a temporary file
    in the same directory, then replaces the destination.

    Existing checkpoint format is preserved.
    """
    import tempfile

    destination = os.path.abspath(f"{filename}.pth")
    directory = os.path.dirname(destination)

    os.makedirs(directory, exist_ok=True)

    fd, temporary_path = tempfile.mkstemp(
        prefix=".checkpoint_",
        suffix=".tmp",
        dir=directory,
    )

    try:
        with os.fdopen(fd, "wb") as file:
            torch.save(state, file)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, destination)

    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)




def load_checkpoint(
    model,
    optimizer,
    filename,
    map_location,
    logger=None,
    lr_scheduler=None,
    warmup_lr_scheduler=None,
    restore_rng=False,
    expected_provenance=None,
):
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    if logger is not None:
        logger.info("Loading checkpoint: %s", filename)

    # Load only checkpoints from trusted sources.
    checkpoint = torch.load(
        filename,
        map_location=map_location,
        weights_only=True,
    )

    # Validate experiment identity before modifying any training state.
    if expected_provenance is not None:
        assert_provenance_matches(
            saved=checkpoint.get("provenance"),
            current=expected_provenance,
        )


    epoch = checkpoint.get("epoch", -1)
    best_result = checkpoint.get("best_result", 0.0)
    best_epoch = checkpoint.get("best_epoch", 0)

    if model is not None:
        model_state = checkpoint.get("model_state")
        if model_state is None:
            raise ValueError("Checkpoint has no model state")
        model.load_state_dict(model_state, strict=True)

    if optimizer is not None:
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is None:
            raise ValueError("Checkpoint has no optimizer state")
        optimizer.load_state_dict(optimizer_state)

    if lr_scheduler is not None:
        scheduler_state = checkpoint.get("lr_scheduler_state")
        if scheduler_state is None:
            raise ValueError(
                "Checkpoint has no LR scheduler state. "
                "Exact scheduler resume is not supported "
                "for this legacy checkpoint."
            )
        lr_scheduler.load_state_dict(scheduler_state)

    if warmup_lr_scheduler is not None:
        warmup_state = checkpoint.get(
            "warmup_lr_scheduler_state"
        )
        if warmup_state is None:
            raise ValueError(
                "Checkpoint has no warmup scheduler state"
            )
        warmup_lr_scheduler.load_state_dict(warmup_state)

    # Restore RNG only when explicitly resuming training.
    if restore_rng:
        rng_state = checkpoint.get("rng_state")

        if rng_state is None:
            raise ValueError(
                "Checkpoint contains no RNG state. "
                "Reproducible resume cannot be guaranteed."
            )

        restore_rng_state(rng_state)

    if logger is not None:
        logger.info(
            "Checkpoint loaded at epoch %s", epoch
        )

    return epoch, best_result, best_epoch
