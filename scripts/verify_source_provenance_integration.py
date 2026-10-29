
"""Verify source provenance under required schema v3 manifests."""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_provenance import (
    assert_provenance_matches,
    build_provenance,
    fingerprint,
)
from scripts.depth_dependency_provenance import (
    build_depth_dependency_provenance,
)
from scripts.provenance_test_fixture import (
    build_test_provenance,
    load_test_config,
    IMAGESETS_DIR,
)
from scripts.source_fingerprint import (
    build_source_fingerprint,
    sha256_json,
)


def main():
    cfg = load_test_config()
    source = build_source_fingerprint(ROOT)

    first = build_test_provenance(
        cfg, source_manifest=source
    )
    second = build_test_provenance(
        cfg, source_manifest=build_source_fingerprint(ROOT)
    )

    assert first == second
    assert first["schema_version"] == 3
    assert first["source"] == source
    assert_provenance_matches(first, second)

    print("PASS: Source manifest stored in provenance")
    print("PASS: Source-aware provenance deterministic")

    changed_source = copy.deepcopy(source)
    target = "MonoIA/lib/helpers/trainer_helper.py"
    assert target in changed_source["files"]

    changed_source["files"][target] = "0" * 64
    changed_source["identity_sha256"] = sha256_json(
        changed_source["files"]
    )

    changed = build_test_provenance(
        cfg, source_manifest=changed_source
    )

    try:
        assert_provenance_matches(first, changed)
    except ValueError:
        print("PASS: Changed training source rejected")
    else:
        raise AssertionError("Changed source was accepted")

    corrupted = copy.deepcopy(source)
    corrupted["files"][target] = "0" * 64

    try:
        build_test_provenance(
            cfg, source_manifest=corrupted
        )
    except ValueError:
        print("PASS: Corrupted source manifest rejected")
    else:
        raise AssertionError("Corrupted source was accepted")

    depth = build_depth_dependency_provenance(cfg, ROOT)

    try:
        build_provenance(
            cfg=cfg,
            effective_seed=444,
            imagesets_dir=IMAGESETS_DIR,
            depth_manifest=depth,
        )
    except ValueError:
        print("PASS: Missing source identity rejected")
    else:
        raise AssertionError("Missing source was accepted")

    print("PASS: Source provenance integration verification")


if __name__ == "__main__":
    main()