
"""
Windows-compatible DataLoader worker seeding verification.

Infrastructure test only.
No KITTI data or benchmark results.
"""

import ast
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    get_worker_info,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "MonoIA/lib/helpers/dataloader_helper.py"


def verify_production_worker_source():
    """Verify the production worker seeding implementation."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    worker_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "my_worker_init_fn"
    )

    source = ast.get_source_segment(
        SOURCE.read_text(encoding="utf-8"),
        worker_fn,
    )

    assert "torch.initial_seed()" in source
    assert "np.random.seed(worker_seed)" in source
    assert "random.seed(worker_seed)" in source

    print("PASS: Production worker seeding source verified")


# Importable module-level function required by Windows spawn.
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


class WorkerRNGDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        worker = get_worker_info()

        return (
            worker.id if worker else -1,
            np.random.randint(0, 2**31),
            random.randint(0, 2**31 - 1),
        )


def collect(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        WorkerRNGDataset(),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    rows = []

    for batch in loader:
        rows.append(
            tuple(int(value.item()) for value in batch)
        )

    return rows


def main():
    verify_production_worker_source()

    first = collect(444)
    second = collect(444)

    assert first == second, (
        "Worker RNG sequences differ across identical runs"
    )

    print("PASS: Fixed seed reproduces worker RNG sequences")

    worker_ids = {row[0] for row in first}

    assert worker_ids == {0, 1}
    print("PASS: Both workers executed")

    worker_zero = [
        row[1:] for row in first if row[0] == 0
    ]

    worker_one = [
        row[1:] for row in first if row[0] == 1
    ]

    assert worker_zero != worker_one
    print("PASS: Worker RNG sequences differ")

    print("PASS: DataLoader worker seeding verification")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
