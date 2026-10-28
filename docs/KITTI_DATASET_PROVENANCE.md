# KITTI Dataset Provenance

## Benchmark

KITTI 3D Object Detection

## Dataset Split Source

Repository: https://github.com/sshaoshuai/PointRCNN

Directory:
data/KITTI/ImageSets/

The split files were obtained from the PointRCNN repository.

MonoIA's evaluation documentation references the historical
MV3D/Chen split archive. Exact byte-for-byte equivalence with
that historical archive has not yet been independently verified.

## Split Statistics

| Split    | Number of IDs | SHA-256                                                          |
| -------- | ------------: | ---------------------------------------------------------------- |
| train    |          3712 | b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb |
| val      |          3769 | 657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86 |
| trainval |          7481 | 4b8593d264ff20f21bbe2c9238181b7ab893c1f1429cd254423dc0b800f9d866 |
| test     |          7518 | a48fd4a14ec1dcff3b5f22ffa132efb9cb034d1370f370afcb4045d241786c6d |

## Completed Checks

- Correct split counts: PASS
- Duplicate IDs: NONE
- Train/validation overlap: NONE
- Train + validation union equals trainval: PASS
- Complete labeled ID coverage: PASS
- Complete test ID coverage: PASS

## Dataset Location

MonoIA/data/KITTIDataset/ImageSets/

## Pending Checks

- Official KITTI image archive verification
- Calibration file verification
- Annotation file verification
- Dataset loader runtime verification
- Full model training
- Official KITTI evaluation

No benchmark results have been generated at this stage.
