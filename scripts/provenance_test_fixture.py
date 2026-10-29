
"""Real project provenance inputs for infrastructure regression tests.

No fake KITTI data and no benchmark measurements.
"""

from pathlib import Path

import yaml

from scripts.checkpoint_provenance import build_provenance
from scripts.depth_dependency_provenance import (
    build_depth_dependency_provenance,
)
from scripts.source_fingerprint import build_source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "MonoIA/config/monoia_local.yaml"
IMAGESETS_DIR = ROOT / "MonoIA/data/KITTIDataset/ImageSets"


def load_test_config():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_test_provenance(cfg, seed=444, source_manifest=None):
    if source_manifest is None:
        source_manifest = build_source_fingerprint(ROOT)

    depth_manifest = build_depth_dependency_provenance(
        cfg=cfg,
        repository_root=ROOT,
    )

    return build_provenance(
        cfg=cfg,
        effective_seed=seed,
        imagesets_dir=IMAGESETS_DIR,
        source_manifest=source_manifest,
        depth_manifest=depth_manifest,
    )