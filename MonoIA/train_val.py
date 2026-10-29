"""MonoIA training/evaluation entry point with pre-initialization resume checks.

Run from the MonoIA directory (the official workflow), or invoke with a
configuration path relative to the caller or to MonoIA.
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import torch
import yaml

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.checkpoint_provenance import (
    build_provenance,
    verify_resume_checkpoint_provenance,
)
from scripts.depth_dependency_provenance import build_depth_dependency_provenance
from scripts.source_fingerprint import build_source_fingerprint


def parse_args():
    parser = argparse.ArgumentParser(
        description="MonoIA monocular 3D object detection"
    )
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument(
        "-e", "--evaluate_only", action="store_true", help="Evaluation only"
    )
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Existing run's checkpoint.pth (training resume only)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Existing thesis checkpoint for evaluation-only mode",
    )
    return parser.parse_args()


def resolve_config_path(value):
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and not candidate.is_file():
        candidate = BASE_DIR / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Configuration file not found: {candidate}")
    return candidate


def prepare_run(cfg, args):

    base_output_dir = Path(cfg["trainer"]["save_path"]).expanduser().resolve()

    if args.evaluate_only:
        if args.resume:
            raise ValueError(
                "--resume cannot be combined with --evaluate_only"
            )

        if not args.checkpoint:
            raise ValueError(
                "Evaluation-only requires --checkpoint"
            )

        checkpoint_path = Path(args.checkpoint).expanduser().resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Evaluation checkpoint not found: {checkpoint_path}"
            )

        if checkpoint_path.name != "checkpoint_best.pth":
            raise ValueError(
                "This evaluation profile requires checkpoint_best.pth"
            )

        output_path = checkpoint_path.parent

        if output_path.parent != base_output_dir:
            raise ValueError(
                "Evaluation checkpoint must belong to a run directory "
                "directly under trainer.save_path"
            )

        cfg["tester"]["checkpoint_path"] = str(checkpoint_path)
        cfg["trainer"]["resume_checkpoint"] = None

        return output_path, output_path.name

    if args.checkpoint:
        raise ValueError(
            "--checkpoint is only supported with --evaluate_only"
        )

    if args.resume:
        if cfg["trainer"].get("pretrain_model"):
            raise ValueError("pretrain_model and --resume cannot be combined")

        resume_checkpoint = Path(args.resume).expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        if resume_checkpoint.name != "checkpoint.pth":
            raise ValueError("Resume requires checkpoint.pth (not best/epoch checkpoint)")

        output_path = resume_checkpoint.parent
        if output_path.parent != base_output_dir:
            raise ValueError(
                "Resume checkpoint must be in a run directory directly "
                "under trainer.save_path"
            )
        model_name = output_path.name
        cfg["trainer"]["resume_checkpoint"] = str(resume_checkpoint)
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
        model_name = f"{cfg['model_name']}{args.seed}_{timestamp}_pid{os.getpid()}"
        output_path = base_output_dir / model_name
        cfg["trainer"]["resume_checkpoint"] = None

    return output_path, model_name


def build_training_provenance(cfg, seed):
    dataset_cfg = cfg["dataset"]

    if dataset_cfg.get("type") != "KITTI":
        raise ValueError(
            "Expected KITTI dataset"
        )

    train_split = dataset_cfg.get("train_split")
    val_split = dataset_cfg.get("test_split")

    if train_split != "train" or val_split != "val":
        raise ValueError(
            "This provenance profile requires train/val splits"
        )

    # Require explicit model mode: never silently treat an enhanced
    # checkpoint as a baseline (or the reverse).
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("Missing or invalid model configuration")

    depth_cfg = model_cfg.get("depth_fusion")
    if not isinstance(depth_cfg, dict):
        raise ValueError(
            "Model configuration must explicitly define "
            "model.depth_fusion.enabled"
        )

    if type(depth_cfg.get("enabled")) is not bool:
        raise ValueError("model.depth_fusion.enabled must be boolean")

    dataset_root = Path(dataset_cfg["root_dir"]).expanduser()

    if not dataset_root.is_absolute():
        dataset_root = BASE_DIR / dataset_root

    imagesets_dir = dataset_root.resolve() / "ImageSets"

    source_manifest = build_source_fingerprint(ROOT_DIR)

    depth_manifest = build_depth_dependency_provenance(
        cfg=cfg,
        repository_root=ROOT_DIR,
    )

    provenance = build_provenance(
        cfg=cfg,
        effective_seed=seed,
        imagesets_dir=str(imagesets_dir),
        source_manifest=source_manifest,
        depth_manifest=depth_manifest,
    )

    if "source" not in provenance or "depth_dependency" not in provenance:
        raise RuntimeError(
            "Required source or depth dependency provenance is missing"
        )

    return provenance, source_manifest


def main():
    args = parse_args()

    config_path = resolve_config_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")

    # Original MonoIA paths are relative to its working directory.
    os.chdir(BASE_DIR)

    output_path, model_name = prepare_run(cfg, args)

    # Perform training provenance and resume rejection BEFORE dataset/model setup.
    # The same strict train/val experiment profile is required
    # for training and thesis checkpoint evaluation.
    provenance, source_manifest = build_training_provenance(
        cfg, args.seed
    )

    cfg["trainer"]["experiment_provenance"] = provenance

    if args.evaluate_only:
        verify_resume_checkpoint_provenance(
            checkpoint_path=cfg["tester"]["checkpoint_path"],
            current_provenance=provenance,
        )

    elif args.resume:
        verify_resume_checkpoint_provenance(
            checkpoint_path=cfg["trainer"]["resume_checkpoint"],
            current_provenance=provenance,
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for production MonoIA training/evaluation. "
            "Run on the Linux GPU instance, not this CPU-only laptop."
        )

    gpu_ids = [int(part.strip()) for part in cfg["trainer"]["gpu_ids"].split(",")]
    if not gpu_ids or any(idx < 0 for idx in gpu_ids):
        raise ValueError("trainer.gpu_ids must contain nonnegative CUDA device IDs")
    for gpu_id in gpu_ids:
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"GPU device {gpu_id} not visible; {torch.cuda.device_count()} CUDA devices found"
            )

    # Keep CUDA-dependent KITTI/Numba imports out of CPU-side preflight.
    # This preserves CLI help and checkpoint rejection without a CUDA driver.
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.model_helper import build_model
    from lib.helpers.optimizer_helper import build_optimizer
    from lib.helpers.scheduler_helper import build_lr_scheduler
    from lib.helpers.tester_helper import Tester
    from lib.helpers.trainer_helper import Trainer
    from lib.helpers.utils_helper import create_logger, set_random_seed

    set_random_seed(seed=args.seed)

    output_path.mkdir(parents=True, exist_ok=True)
    log_name = "train.log." + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = create_logger(str(output_path / log_name))
    logger.info("Run directory: %s", output_path)
    logger.info("Resume checkpoint: %s", cfg["trainer"]["resume_checkpoint"])
    logger.info("Experiment provenance SHA256: %s", provenance["identity_sha256"])
    logger.info("Training source SHA256: %s", source_manifest["identity_sha256"])
    logger.info("Training source file count: %d", source_manifest["file_count"])
    if args.resume:
        logger.info("PASS: Early resume provenance verification")
    if args.evaluate_only:
        logger.info("PASS: Early evaluation provenance verification")
        logger.info("Evaluation checkpoint: %s", cfg["tester"]["checkpoint_path"])

    train_loader, test_loader = build_dataloader(cfg["dataset"])
    model, loss = build_model(cfg["model"])
    if len(gpu_ids) == 1:
        model = model.to(torch.device("cuda", gpu_ids[0]))
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(
            torch.device("cuda", gpu_ids[0])
        )

    if args.evaluate_only:
        logger.info("###################  Evaluation Only  ##################")
        # Tester repeats strict provenance verification before loading weights.
        tester = Tester(
            cfg=cfg["tester"],
            model=model,
            dataloader=test_loader,
            logger=logger,
            train_cfg=cfg["trainer"],
            model_name=model_name,
            expected_provenance=provenance,
        )
        tester.test()
        return

    optimizer = build_optimizer(cfg["optimizer"], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(
        cfg["lr_scheduler"], optimizer, last_epoch=-1
    )
    trainer = Trainer(
        cfg=cfg["trainer"], model=model, optimizer=optimizer,
        train_loader=train_loader, test_loader=test_loader,
        lr_scheduler=lr_scheduler, warmup_lr_scheduler=warmup_lr_scheduler,
        logger=logger, loss=loss, model_name=model_name,
    )
    tester = Tester(
        cfg=cfg["tester"],
        model=trainer.model,
        dataloader=test_loader,
        logger=logger,
        train_cfg=cfg["trainer"],
        model_name=model_name,
        expected_provenance=provenance,
    )
    if cfg["dataset"]["test_split"] != "test":
        trainer.tester = tester

    logger.info("###################  Training  ##################")
    logger.info("Batch Size: %d", cfg["dataset"]["batch_size"])
    logger.info("Learning Rate: %f", cfg["optimizer"]["lr"])
    trainer.train()

    if cfg["dataset"]["test_split"] != "test":
        logger.info("###################  Testing  ##################")
        logger.info("Batch Size: %d", cfg["dataset"]["batch_size"])
        logger.info("Split: %s", cfg["dataset"]["test_split"])
        tester.test()


if __name__ == "__main__":
    main()
