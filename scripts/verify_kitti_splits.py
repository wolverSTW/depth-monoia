
"""Verify KITTI ImageSets provenance and split isolation.

This checks authentic split index files.
It does not claim that KITTI images or labels are present.
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGESETS = ROOT / "MonoIA/data/KITTIDataset/ImageSets"

EXPECTED = {
    "train": {
        "count": 3712,
        "sha256": "b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb",
    },
    "val": {
        "count": 3769,
        "sha256": "657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86",
    },
    "trainval": {
        "count": 7481,
        "sha256": "4b8593d264ff20f21bbe2c9238181b7ab893c1f1429cd254423dc0b800f9d866",
    },
    "test": {
        "count": 7518,
        "sha256": "a48fd4a14ec1dcff3b5f22ffa132efb9cb034d1370f370afcb4045d241786c6d",
    },
}


def read_split(name):
    path = IMAGESETS / f"{name}.txt"

    if not path.is_file():
        raise FileNotFoundError(path)

    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    expected = EXPECTED[name]

    assert digest == expected["sha256"], (
        f"{name}: SHA256 mismatch: {digest}"
    )

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == expected["count"], (
        f"{name}: incorrect image count"
    )

    assert all(
        len(index) == 6 and index.isdigit()
        for index in lines
    ), f"{name}: invalid KITTI image ID"

    assert len(lines) == len(set(lines)), (
        f"{name}: duplicate image IDs"
    )

    print(
        f"PASS: {name} — "
        f"{len(lines)} IDs, SHA256 verified"
    )

    return set(lines)


def main():
    train = read_split("train")
    val = read_split("val")
    trainval = read_split("trainval")
    test = read_split("test")

    assert train.isdisjoint(val), (
        "Train/validation leakage detected"
    )
    print("PASS: Train/validation splits are disjoint")

    assert train | val == trainval, (
        "Trainval does not equal train union validation"
    )
    print("PASS: Trainval matches train + validation")

    assert len(test) == 7518
    print("PASS: Test split count verified")

    print("PASS: KITTI ImageSets provenance verification")


if __name__ == "__main__":
    main()