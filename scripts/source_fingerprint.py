
"""Deterministic SHA256 manifest of training source files.

Includes modified/uncommitted source files.
Does not modify the Git repository or training outputs.
"""

import hashlib
import json
from pathlib import Path


# Relative to the repository root.
SOURCE_DIRECTORIES = (
    "MonoIA/lib",
    "MonoIA/config",
)

SOURCE_FILES = (
    "MonoIA/train_val.py",
    "MonoIA/train.sh",
    "scripts/checkpoint_provenance.py",
    "scripts/source_fingerprint.py",
)

SOURCE_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    ".cu",
    ".cpp",
    ".h",
    ".cuh",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_json(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def collect_source_files(root):
    root = Path(root).resolve()
    paths = set()

    for relative_dir in SOURCE_DIRECTORIES:
        directory = root / relative_dir

        if not directory.is_dir():
            raise FileNotFoundError(
                f"Missing source directory: {directory}"
            )

        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS:
                paths.add(path)

    for relative_file in SOURCE_FILES:
        path = root / relative_file

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required source file: {path}"
            )

        paths.add(path)

    return sorted(
        paths,
        key=lambda path: path.relative_to(root).as_posix(),
    )


def build_source_fingerprint(root):
    """Fingerprint actual source bytes, independent of Git status."""
    root = Path(root).resolve()
    source_files = collect_source_files(root)

    if not source_files:
        raise ValueError("No source files found")

    manifest = {}

    for path in source_files:
        relative_path = path.relative_to(root).as_posix()
        manifest[relative_path] = sha256_file(path)

    return {
        "schema_version": 1,
        "file_count": len(manifest),
        "files": manifest,
        "identity_sha256": sha256_json(manifest),
    }


def assert_source_fingerprint_matches(saved, current):
    if not isinstance(saved, dict):
        raise ValueError("Missing source fingerprint")

    if not isinstance(current, dict):
        raise ValueError("Missing current source fingerprint")

    for source in (saved, current):
        if source.get("schema_version") != 1:
            raise ValueError("Unsupported source schema")

        files = source.get("files")

        if not isinstance(files, dict) or not files:
            raise ValueError("Invalid source file manifest")

        if source.get("file_count") != len(files):
            raise ValueError("Invalid source file count")

        if source.get("identity_sha256") != sha256_json(files):
            raise ValueError("Source manifest integrity mismatch")

    if saved["identity_sha256"] != current["identity_sha256"]:
        raise ValueError("Training source fingerprint mismatch")

    return True