
"""Audit Depth Anything V2 dependencies for experiment provenance.

Baseline mode does not require Depth Anything V2 files.
Enhanced mode requires verified pretrained weights and source.
"""

import hashlib
from pathlib import Path

from scripts.checkpoint_provenance import fingerprint, sha256_file


EXPECTED_VITS_SHA256 = (
    "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
)

SOURCE_RELATIVE = "external/Depth-Anything-V2"

REQUIRED_SOURCE_FILES = (
    "depth_anything_v2/dpt.py",
    "depth_anything_v2/dinov2.py",
)


def build_depth_dependency_provenance(cfg, repository_root):
    """Build a deterministic dependency record for baseline/enhanced runs."""
    root = Path(repository_root).resolve()

    depth_cfg = cfg["model"]["depth_fusion"]
    enabled = depth_cfg["enabled"]

    if type(enabled) is not bool:
        raise ValueError("depth_fusion.enabled must be a boolean")

    if not enabled:
        return {
            "schema_version": 1,
            "enabled": False,
        }

    encoder = depth_cfg.get("encoder")

    if encoder != "vits":
        raise ValueError(
            f"Unsupported Depth Anything V2 encoder: {encoder}"
        )

    configured_path = depth_cfg.get("checkpoint_path")

    if not configured_path:
        raise ValueError(
            "Depth fusion enabled but checkpoint_path is missing"
        )

    checkpoint = Path(configured_path)

    # Relative paths are explicitly resolved from repository root.
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint

    checkpoint = checkpoint.resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Depth Anything V2 checkpoint missing: {checkpoint}"
        )

    weight_hash = sha256_file(checkpoint)

    if weight_hash != EXPECTED_VITS_SHA256:
        raise ValueError(
            "Depth Anything V2 ViT-S checkpoint SHA256 mismatch"
        )

    source_root = root / SOURCE_RELATIVE

    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Depth Anything V2 source missing: {source_root}"
        )

    source_files = {}

    # Include all Python source files, not just two entrypoints.
    for path in source_root.rglob("*.py"):
        if path.is_file():
            relative = path.relative_to(source_root).as_posix()
            source_files[relative] = sha256_file(path)

    for required in REQUIRED_SOURCE_FILES:
        if required not in source_files:
            raise FileNotFoundError(
                f"Missing Depth Anything V2 source: {required}"
            )

    source_files = dict(sorted(source_files.items()))

    record = {
        "schema_version": 1,
        "enabled": True,
        "encoder": encoder,
        "checkpoint_sha256": weight_hash,
        "source_file_count": len(source_files),
        "source_files": source_files,
        "source_identity_sha256": fingerprint(source_files),
    }

    record["identity_sha256"] = fingerprint(record)

    return record