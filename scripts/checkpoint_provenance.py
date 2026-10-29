
"""Build deterministic checkpoint configuration provenance.

This module does not train, evaluate, or load a model.
"""

import hashlib
import json
from pathlib import Path
from scripts.source_fingerprint import (
    assert_source_fingerprint_matches,
)


# SCHEMA_VERSION = 1
# SCHEMA_VERSION = 2
SCHEMA_VERSION = 3

CONFIG_SECTIONS = (
    "dataset",
    "model",
    "optimizer",
    "lr_scheduler",
)

TRAINER_FIELDS = (
    "max_epoch",
    "use_dn",
    "scalar",
    "label_noise_scale",
    "box_noise_scale",
    "num_patterns",
)


def sha256_file(path):
    """Hash the exact bytes of an existing file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def fingerprint(value):
    """Stable SHA256 for JSON-compatible configuration data."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def build_provenance(
    cfg,
    effective_seed,
    imagesets_dir,
    source_manifest=None,
    depth_manifest=None,
):
    """Construct provenance from actual run settings and split files."""
    if type(effective_seed) is not int:
        raise TypeError("effective_seed must be an integer")

    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dictionary")

    for section in CONFIG_SECTIONS:
        if section not in cfg:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    if "trainer" not in cfg:
        raise ValueError("Missing trainer configuration")

    dataset_cfg = cfg["dataset"]

    if dataset_cfg.get("type") != "KITTI":
        raise ValueError("Expected KITTI dataset")

    train_split = dataset_cfg.get("train_split")
    val_split = dataset_cfg.get("test_split")

    if train_split != "train" or val_split != "val":
        raise ValueError(
            "This provenance profile requires train/val splits"
        )

    imagesets_dir = Path(imagesets_dir)

    train_file = imagesets_dir / f"{train_split}.txt"
    val_file = imagesets_dir / f"{val_split}.txt"

    trainer_config = {
        field: cfg["trainer"].get(field)
        for field in TRAINER_FIELDS
    }

    config_fingerprints = {
        section: fingerprint(cfg[section])
        for section in CONFIG_SECTIONS
    }

    config_fingerprints["trainer_training"] = fingerprint(
        trainer_config
    )

    # Schema v3 requires both source and depth dependency identity.
    if source_manifest is None:
        raise ValueError(
            "Schema v3 requires source_manifest"
        )

    if depth_manifest is None:
        raise ValueError(
            "Schema v3 requires depth_manifest"
        )

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "effective_seed": effective_seed,
        "configuration": config_fingerprints,
        "splits": {
            "train_sha256": sha256_file(train_file),
            "val_sha256": sha256_file(val_file),
        },
    }

    if source_manifest is not None:
        assert_source_fingerprint_matches(
            source_manifest,
            source_manifest,
        )
        provenance["source"] = source_manifest

    if depth_manifest is not None:
        if not isinstance(depth_manifest, dict):
            raise TypeError("depth_manifest must be a dictionary")

        expected_enabled = cfg["model"]["depth_fusion"]["enabled"]

        if type(expected_enabled) is not bool:
            raise ValueError("depth_fusion.enabled must be boolean")

        if depth_manifest.get("schema_version") != 1:
            raise ValueError("Unsupported depth dependency schema")

        if depth_manifest.get("enabled") is not expected_enabled:
            raise ValueError(
                "Depth dependency manifest does not match model mode"
            )


        if not expected_enabled:
            expected_disabled = {
                "schema_version": 1,
                "enabled": False,
            }

            if depth_manifest != expected_disabled:
                raise ValueError(
                    "Invalid baseline depth dependency manifest"
                )

        else:
            source_files = depth_manifest.get("source_files")

            if not isinstance(source_files, dict) or not source_files:
                raise ValueError("Missing external depth source manifest")

            if depth_manifest.get("source_file_count") != len(source_files):
                raise ValueError("Invalid external depth source count")

            if depth_manifest.get("source_identity_sha256") != fingerprint(
                source_files
            ):
                raise ValueError("External depth source checksum mismatch")

            if depth_manifest.get("encoder") != "vits":
                raise ValueError("Unsupported depth dependency encoder")

            if not isinstance(
                depth_manifest.get("checkpoint_sha256"), str
            ):
                raise ValueError("Missing depth checkpoint hash")

            payload = {
                key: value
                for key, value in depth_manifest.items()
                if key != "identity_sha256"
            }

            if depth_manifest.get("identity_sha256") != fingerprint(payload):
                raise ValueError("Depth dependency checksum mismatch")


        if not expected_enabled:
            expected_disabled = {
                "schema_version": 1,
                "enabled": False,
            }

            if depth_manifest != expected_disabled:
                raise ValueError(
                    "Invalid baseline depth dependency manifest"
                )

        else:
            source_files = depth_manifest.get("source_files")

            if not isinstance(source_files, dict) or not source_files:
                raise ValueError("Missing external depth source manifest")

            if depth_manifest.get("source_file_count") != len(source_files):
                raise ValueError("Invalid external depth source count")

            if depth_manifest.get("source_identity_sha256") != fingerprint(
                source_files
            ):
                raise ValueError("External depth source checksum mismatch")

            if depth_manifest.get("encoder") != "vits":
                raise ValueError("Unsupported depth dependency encoder")

            if not isinstance(
                depth_manifest.get("checkpoint_sha256"), str
            ):
                raise ValueError("Missing depth checkpoint hash")

            payload = {
                key: value
                for key, value in depth_manifest.items()
                if key != "identity_sha256"
            }

            if depth_manifest.get("identity_sha256") != fingerprint(payload):
                raise ValueError("Depth dependency checksum mismatch")

        provenance["depth_dependency"] = depth_manifest

    # Final experiment identity must include all provenance fields.
    provenance["identity_sha256"] = fingerprint(provenance)

    return provenance


def assert_provenance_matches(saved, current):
    """Fail closed when checkpoint provenance differs."""
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint provenance is missing")

    if saved.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Checkpoint provenance schema version mismatch"
        )

    saved_identity = saved.get("identity_sha256")
    current_identity = current.get("identity_sha256")

    if not isinstance(saved_identity, str):
        raise ValueError("Missing checkpoint identity hash")

    # Detect accidental mutation of the checkpoint metadata.
    saved_payload = {
        key: value
        for key, value in saved.items()
        if key != "identity_sha256"
    }

    if fingerprint(saved_payload) != saved_identity:
        raise ValueError(
            "Checkpoint provenance integrity mismatch"
        )

    if saved_identity != current_identity:
        raise ValueError(
            "Checkpoint and current experiment provenance differ"
        )

    return True

def verify_resume_checkpoint_provenance(
    checkpoint_path,
    current_provenance,
):
    """
    Validate resume checkpoint provenance before model/DataLoader
    initialization.

    This does not restore model, optimizer, scheduler or RNG state.
    Full checkpoint restoration remains the Trainer's responsibility.
    """
    from pathlib import Path
    import torch

    checkpoint_path = Path(checkpoint_path).resolve(strict=True)

    if not checkpoint_path.is_file():
        raise ValueError(
            f"Resume checkpoint is not a file: {checkpoint_path}"
        )

    # Keep CPU-only metadata verification independent of CUDA.
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Resume checkpoint must contain a dictionary")

    saved_provenance = checkpoint.get("provenance")

    assert_provenance_matches(
        saved=saved_provenance,
        current=current_provenance,
    )

    return True