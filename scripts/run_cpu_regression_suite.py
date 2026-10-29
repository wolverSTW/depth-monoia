
"""Run CPU engineering regression tests with fail-fast behavior.

This suite verifies infrastructure only.
It does not train MonoIA or report KITTI benchmark metrics.
"""

import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Allow direct execution: python scripts/run_cpu_regression_suite.py
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TESTS = (
    "verify_kitti_splits.py",
    "verify_depth_dependency_provenance.py",
    "verify_source_fingerprint.py",
    "verify_checkpoint_provenance.py",
    "verify_source_provenance_integration.py",
    "verify_depth_provenance_integration.py",
    "verify_provenance_preload.py",
    "verify_trainer_resume_provenance.py",
    "verify_atomic_checkpoint.py",
    "verify_checkpoint_write_failure.py",
    "verify_checkpoint_consistency.py",
    "verify_zero_score_best.py",
    "verify_optimizer_state.py",
    "verify_scheduler_resume.py",
    "verify_rng_resume.py",
    "verify_worker_seeding.py",
    "verify_tester_checkpoint_provenance.py",
    "verify_kitti_prediction_isolation.py",
)


def main():
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"

    python = sys.executable
    print("Python:", python, flush=True)
    print("Repository:", ROOT, flush=True)
    print("Total tests:", len(TESTS) + 1, flush=True)

    started = time.perf_counter()
    passed = []

    for index, filename in enumerate(TESTS, start=1):
        path = ROOT / "scripts" / filename

        if not path.is_file():
            print(
                f"FAIL: Missing script: {filename}",
                flush=True,
            )
            return 1

        print(
            f"\n[{index}/{len(TESTS) + 1}] {filename}",
            flush=True,
        )

        # Compile before execution.
        compile_result = subprocess.run(
            [python, "-m", "py_compile", str(path)],
            cwd=str(ROOT),
            env=environment,
            check=False,
        )

        if compile_result.returncode != 0:
            print(
                f"FAIL: Syntax check: {filename}",
                flush=True,
            )
            return 1

        # Each script runs in a fresh process.
        result = subprocess.run(
            [
                python,
                "-W",
                "error::UserWarning",
                str(path),
            ],
            cwd=str(ROOT),
            env=environment,
            check=False,
        )

        if result.returncode != 0:
            print(
                f"\nFAIL: {filename}",
                flush=True,
            )
            print(
                f"Exit code: {result.returncode}",
                flush=True,
            )
            print(
                f"Completed before failure: {len(passed)}",
                flush=True,
            )
            return 1

        passed.append(filename)

        print(
            f"PASS: {filename}",
            flush=True,
        )

    # Checkpoint-set verifier requires an existing run directory.
    # Create a real serialized infrastructure checkpoint set,
    # then execute the production verifier against it.
    print(
        f"\n[{len(TESTS) + 1}/{len(TESTS) + 1}] "
        "verify_checkpoint_set.py",
        flush=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="monoia_checkpoint_set_"
    ) as directory:
        from MonoIA.lib.helpers.save_helper import (
            get_checkpoint_state,
            save_checkpoint,
        )
        from scripts.provenance_test_fixture import (
            build_test_provenance,
            load_test_config,
        )

        cfg = load_test_config()
        provenance = build_test_provenance(cfg, seed=444)

        # CPU-only checkpoint infrastructure fixture.
        # This is NOT a MonoIA training model or KITTI result.
        import torch

        model = torch.nn.Linear(2, 1)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
        )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0,
        )

        state = get_checkpoint_state(
            model=model,
            optimizer=optimizer,
            epoch=1,
            best_result=0.0,
            best_epoch=1,
            lr_scheduler=lr_scheduler,
            include_rng_state=True,
            provenance=provenance,
        )

        save_checkpoint(
            state,
            str(Path(directory) / "checkpoint"),
        )

        save_checkpoint(
            state,
            str(Path(directory) / "checkpoint_best"),
        )

        result = subprocess.run(
            [
                python,
                "-W",
                "error::UserWarning",
                str(ROOT / "scripts/verify_checkpoint_set.py"),
                directory,
            ],
            cwd=str(ROOT),
            env=environment,
            check=False,
        )

        if result.returncode != 0:
            print(
                "FAIL: verify_checkpoint_set.py",
                flush=True,
            )
            return 1

        passed.append("verify_checkpoint_set.py")

        print(
            "PASS: verify_checkpoint_set.py",
            flush=True,
        )


    elapsed = time.perf_counter() - started

    print("\n" + "=" * 54, flush=True)
    print(
        f"PASS: {len(passed)}/{len(TESTS) + 1} CPU engineering tests",
        flush=True,
    )
    print(
        f"Elapsed: {elapsed:.2f} seconds",
        flush=True,
    )
    print(
        "PASS: CPU regression suite",
        flush=True,
    )
    print("=" * 54, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())