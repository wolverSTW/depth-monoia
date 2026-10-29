
"""
Verify production Trainer.__init__ provenance protection.

Infrastructure verification only.
No KITTI model training or benchmark metrics.
"""

import ast
import copy
import importlib.util
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_provenance import (
    assert_provenance_matches,
    build_provenance,
    fingerprint,
)

from scripts.source_fingerprint import (
    assert_source_fingerprint_matches,
    build_source_fingerprint,
    sha256_json,
)

from scripts.depth_dependency_provenance import (
    build_depth_dependency_provenance,
)


SAVE_PATH = ROOT / "MonoIA/lib/helpers/save_helper.py"
TRAINER_PATH = ROOT / "MonoIA/lib/helpers/trainer_helper.py"
CONFIG_PATH = ROOT / "MonoIA/config/monoia_local.yaml"
IMAGESETS = ROOT / "MonoIA/data/KITTIDataset/ImageSets"


def import_save_helper():
    spec = importlib.util.spec_from_file_location(
        "save_helper_trainer_resume",
        SAVE_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import save_helper.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_trainer_init(helper):
    source = TRAINER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    trainer_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Trainer"
    )

    init_method = next(
        node for node in trainer_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "__init__"
    )

    isolated_class = ast.ClassDef(
        name="TrainerUnderTest",
        bases=[],
        keywords=[],
        body=[init_method],
        decorator_list=[],
    )

    isolated_module = ast.fix_missing_locations(
        ast.Module(
            body=[isolated_class],
            type_ignores=[],
        )
    )

    namespace = {
        "os": os,
        "torch": torch,
        "Path": Path,
        "__file__": str(TRAINER_PATH),
        "load_checkpoint": helper.load_checkpoint,
        "assert_provenance_matches": assert_provenance_matches,
        "assert_source_fingerprint_matches": (
            assert_source_fingerprint_matches
        ),
        "build_source_fingerprint": build_source_fingerprint,
    }

    exec(
        compile(isolated_module, str(TRAINER_PATH), "exec"),
        namespace,
    )

    return namespace["TrainerUnderTest"]


def build_components():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0,
    )
    return model, optimizer, scheduler


def make_trainer(
    trainer_class,
    provenance,
    resume_path,
    output_root,
):
    model, optimizer, scheduler = build_components()

    cfg = {
        "save_path": str(output_root),
        "pretrain_model": None,
        "resume_model": False,
        "resume_checkpoint": resume_path,
    }

    if provenance is not None:
        cfg["experiment_provenance"] = provenance

    trainer = trainer_class(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        train_loader=None,
        test_loader=None,
        lr_scheduler=scheduler,
        warmup_lr_scheduler=None,
        logger=logging.getLogger("trainer_resume_test"),
        loss=None,
        model_name="verification_run",
    )

    return trainer


def main():
    torch.manual_seed(444)

    helper = import_save_helper()
    TrainerUnderTest = extract_trainer_init(helper)

    with CONFIG_PATH.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    source_manifest = build_source_fingerprint(ROOT)

    depth_manifest = build_depth_dependency_provenance(
        cfg=cfg,
        repository_root=ROOT,
    )

    correct_provenance = build_provenance(
        cfg=cfg,
        effective_seed=444,
        imagesets_dir=IMAGESETS,
        source_manifest=source_manifest,
        depth_manifest=depth_manifest,
    )

    changed_cfg = copy.deepcopy(cfg)
    changed_cfg["optimizer"]["lr"] = 0.0001

    wrong_provenance = build_provenance(
        cfg=changed_cfg,
        effective_seed=444,
        imagesets_dir=IMAGESETS,
        source_manifest=source_manifest,
        depth_manifest=depth_manifest,
    )

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        checkpoint_base = folder / "checkpoint"
        checkpoint_file = folder / "checkpoint.pth"

        saved_model, saved_optimizer, saved_scheduler = (
            build_components()
        )

        state = helper.get_checkpoint_state(
            model=saved_model,
            optimizer=saved_optimizer,
            epoch=3,
            best_result=0.0,
            best_epoch=1,
            lr_scheduler=saved_scheduler,
            include_rng_state=True,
            provenance=correct_provenance,
        )

        helper.save_checkpoint(
            state,
            str(checkpoint_base),
        )

        assert checkpoint_file.is_file()
        print("PASS: Provenance-aware checkpoint created")

        # Case 1: No provenance supplied to Trainer.
        try:
            make_trainer(
                TrainerUnderTest,
                provenance=None,
                resume_path=str(checkpoint_file),
                output_root=folder,
            )
        except ValueError as error:
            assert "provenance" in str(error).lower()
            print("PASS: Missing Trainer provenance rejected")
        else:
            raise AssertionError(
                "Trainer accepted missing provenance"
            )

        # Case 2: Mismatched provenance must fail before
        # modifying the model or RNG.
        target_model, target_optimizer, target_scheduler = (
            build_components()
        )

        original_weights = {
            key: value.detach().clone()
            for key, value in target_model.state_dict().items()
        }

        original_optimizer = copy.deepcopy(
            target_optimizer.state_dict()
        )
        original_scheduler = copy.deepcopy(
            target_scheduler.state_dict()
        )
        original_rng = helper.capture_rng_state()

        bad_cfg = {
            "save_path": str(folder),
            "pretrain_model": None,
            "resume_model": False,
            "resume_checkpoint": str(checkpoint_file),
            "experiment_provenance": wrong_provenance,
        }

        try:
            TrainerUnderTest(
                cfg=bad_cfg,
                model=target_model,
                optimizer=target_optimizer,
                train_loader=None,
                test_loader=None,
                lr_scheduler=target_scheduler,
                warmup_lr_scheduler=None,
                logger=logging.getLogger("wrong_provenance"),
                loss=None,
                model_name="verification_run",
            )
        except ValueError as error:
            assert "provenance" in str(error).lower()
            print("PASS: Resume provenance mismatch rejected")
        else:
            raise AssertionError(
                "Trainer accepted mismatched provenance"
            )

        for key, value in original_weights.items():
            assert torch.equal(
                value,
                target_model.state_dict()[key],
            ), f"Model changed after rejection: {key}"

        assert (
            target_optimizer.state_dict()
            == original_optimizer
        )
        assert (
            target_scheduler.state_dict()
            == original_scheduler
        )

        current_rng = helper.capture_rng_state()
        assert original_rng["python"] == current_rng["python"]
        assert torch.equal(
            original_rng["numpy"]["keys"],
            current_rng["numpy"]["keys"],
        )
        assert torch.equal(
            original_rng["torch_cpu"],
            current_rng["torch_cpu"],
        )

        print(
            "PASS: Model, optimizer, scheduler and RNG "
            "unchanged after rejection"
        )

        # Case 3: Missing source fingerprint must be rejected.
        missing_source = copy.deepcopy(correct_provenance)
        missing_source.pop("source")

        # Recompute the outer identity to prove that missing
        # source is rejected even with a consistent outer hash.
        missing_source_payload = {
            key: value
            for key, value in missing_source.items()
            if key != "identity_sha256"
        }

        missing_source["identity_sha256"] = fingerprint(
            missing_source_payload
        )

        try:
            make_trainer(
                TrainerUnderTest,
                provenance=missing_source,
                resume_path=str(checkpoint_file),
                output_root=folder,
            )
        except ValueError as error:
            assert "source" in str(error).lower()
            print("PASS: Missing source fingerprint rejected")
        else:
            raise AssertionError(
                "Trainer accepted missing source fingerprint"
            )

        # Case 4: Corrupted source manifest must be rejected.
        corrupted_source = copy.deepcopy(correct_provenance)

        target_file = (
            "MonoIA/lib/helpers/trainer_helper.py"
        )

        corrupted_source["source"]["files"][
            target_file
        ] = "0" * 64

        try:
            make_trainer(
                TrainerUnderTest,
                provenance=corrupted_source,
                resume_path=str(checkpoint_file),
                output_root=folder,
            )
        except ValueError:
            print("PASS: Corrupted source manifest rejected")
        else:
            raise AssertionError(
                "Trainer accepted corrupted source manifest"
            )

        # Case 5: A different internally valid source
        # fingerprint must not resume the saved checkpoint.
        changed_source = copy.deepcopy(source_manifest)

        changed_source["files"][target_file] = "0" * 64
        changed_source["identity_sha256"] = sha256_json(
            changed_source["files"]
        )

        changed_source_provenance = build_provenance(
            cfg=cfg,
            effective_seed=444,
            imagesets_dir=IMAGESETS,
            source_manifest=changed_source,
            depth_manifest=depth_manifest,
        )

        try:
            make_trainer(
                TrainerUnderTest,
                provenance=changed_source_provenance,
                resume_path=str(checkpoint_file),
                output_root=folder,
            )
        except ValueError as error:
            assert "source fingerprint mismatch" in str(error).lower(), (
                f"Unexpected rejection reason: {error}"
            )
            print("PASS: Changed source identity rejected")
        else:
            raise AssertionError(
                "Trainer accepted changed source identity"
            )


        # Case 3: Matching provenance should resume.
        trainer = make_trainer(
            TrainerUnderTest,
            provenance=correct_provenance,
            resume_path=str(checkpoint_file),
            output_root=folder,
        )

        assert trainer.epoch == 3
        assert trainer.best_epoch == 1
        assert trainer.best_result == 0.0

        for key, value in saved_model.state_dict().items():
            assert torch.equal(
                value,
                trainer.model.state_dict()[key],
            )

        print("PASS: Matching provenance resumes epoch 3")
        print("PASS: Resumed model weights restored")

    print("PASS: Trainer resume provenance runtime verification")


if __name__ == "__main__":
    main()