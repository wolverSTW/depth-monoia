
"""Verify deterministic source fingerprints and change detection."""

import copy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_fingerprint import (
    assert_source_fingerprint_matches,
    build_source_fingerprint,
)


def main():
    first = build_source_fingerprint(ROOT)
    second = build_source_fingerprint(ROOT)

    assert first == second
    assert_source_fingerprint_matches(first, second)

    print("PASS: Source fingerprint deterministic")
    print(
        f"PASS: {first['file_count']} actual source files hashed"
    )

    expected_files = (
        "MonoIA/lib/helpers/save_helper.py",
        "MonoIA/lib/helpers/trainer_helper.py",
        "MonoIA/lib/helpers/dataloader_helper.py",
        "MonoIA/lib/helpers/optimizer_helper.py",
        "MonoIA/lib/models/monoia/monoia.py",
        "MonoIA/train_val.py",
        "MonoIA/train.sh",
        "scripts/checkpoint_provenance.py",
        "scripts/source_fingerprint.py",
    )

    for relative_path in expected_files:
        assert relative_path in first["files"], (
            f"Missing critical source: {relative_path}"
        )

    print("PASS: Critical training source files included")

    changed = copy.deepcopy(first)
    target = "MonoIA/lib/helpers/trainer_helper.py"

    changed["files"][target] = "0" * 64

    # Recalculate identity to simulate a changed source manifest.
    from scripts.source_fingerprint import sha256_json

    changed["identity_sha256"] = sha256_json(
        changed["files"]
    )

    try:
        assert_source_fingerprint_matches(first, changed)
    except ValueError:
        print("PASS: Changed source fingerprint rejected")
    else:
        raise AssertionError("Source mismatch was accepted")

    corrupted = copy.deepcopy(first)
    corrupted["files"][target] = "0" * 64

    try:
        assert_source_fingerprint_matches(corrupted, first)
    except ValueError:
        print("PASS: Corrupted source manifest rejected")
    else:
        raise AssertionError("Corrupted manifest was accepted")

    # Test actual file-byte modification in a temporary source tree.
    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(directory)

        for relative in first["files"]:
            source = ROOT / relative
            destination = temporary_root / relative
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            destination.write_bytes(source.read_bytes())

        before = build_source_fingerprint(temporary_root)

        modified_file = (
            temporary_root
            / "MonoIA/lib/helpers/trainer_helper.py"
        )

        with modified_file.open("ab") as file:
            file.write(b"\n# fingerprint verification change\n")

        after = build_source_fingerprint(temporary_root)

        assert before["identity_sha256"] != (
            after["identity_sha256"]
        )

        try:
            assert_source_fingerprint_matches(before, after)
        except ValueError:
            print("PASS: Actual source-byte change detected")
        else:
            raise AssertionError("Byte change was not detected")

    print("PASS: Source fingerprint verification")


if __name__ == "__main__":
    main()