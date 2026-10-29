"""MonoIA evaluation with isolated KITTI prediction artifacts.

This module does not produce benchmark claims unless real KITTI evaluation runs.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import tqdm

from lib.helpers.benchmark import flop_count
from lib.helpers.decode_helper import decode_detections, extract_dets_from_outputs
from lib.helpers.save_helper import load_checkpoint


class Tester(object):
    def __init__(
        self,
        cfg,
        model,
        dataloader,
        logger,
        train_cfg=None,
        model_name="monocop",
        expected_provenance=None,
    ):
        if not isinstance(train_cfg, dict) or not train_cfg.get("save_path"):
            raise ValueError("Tester requires train_cfg.save_path")
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.abspath(
            os.path.join(train_cfg["save_path"], model_name)
        )
        self.dataset_type = cfg.get("type", "KITTI")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name
        self.checkpoint_path = cfg.get("checkpoint_path")
        self.expected_provenance = expected_provenance
        self.prediction_dir = None

    def test(self):
        checkpoint_path = self.checkpoint_path or os.path.join(
            self.output_dir, "checkpoint_best.pth"
        )
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Evaluation checkpoint not found: {checkpoint_path}"
            )
        if not isinstance(self.expected_provenance, dict):
            raise ValueError(
                "Evaluation requires explicit checkpoint provenance. "
                "Unverified or official external checkpoints require "
                "a separate verified evaluation policy."
            )
        if self.expected_provenance.get("schema_version") != 3:
            raise ValueError("Evaluation requires schema v3 provenance")

        load_checkpoint(
            model=self.model,
            optimizer=None,
            filename=checkpoint_path,
            map_location=self.device,
            logger=self.logger,
            expected_provenance=self.expected_provenance,
        )
        self.model.to(self.device)

        self.prediction_dir = None
        self.logger.info("Evaluation checkpoint: %s", checkpoint_path)
        self.calculate_flops()
        self.inference()

        if self.dataloader.dataset.split == "test":
            self.logger.info(
                "KITTI test split: predictions saved for official leaderboard "
                "submission; local AP evaluation skipped."
            )
            return None
        return self.evaluate()

    def calculate_flops(self):
        model = self.model
        model.eval()
        warmup_steps = 5
        measure_steps = 20
        infer_times = []
        sample_counts = []
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {total_params / 1e6:.2f} M")

        with torch.no_grad():
            for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
                if batch_idx >= warmup_steps + measure_steps:
                    break
                inputs = inputs.to(self.device, dtype=torch.float32)
                calibs = calibs.to(self.device, dtype=torch.float32)
                img_sizes = info["img_size"].to(self.device, dtype=torch.long)
                if batch_idx == 0:
                    gflops = flop_count(model, (inputs, calibs, targets, img_sizes))
                    print(f"GFLOPs (first batch): {sum(gflops.values()):.2f}")
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                start = time.perf_counter()
                _ = model(inputs, calibs, targets, img_sizes)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                if batch_idx >= warmup_steps:
                    infer_times.append(time.perf_counter() - start)
                    sample_counts.append(int(inputs.shape[0]))

        if infer_times:
            total_seconds = float(np.sum(infer_times))
            total_samples = sum(sample_counts)
            print(f"Mean model latency: {1000 * total_seconds / total_samples:.2f} ms/image")
            print(f"Model throughput: {total_samples / total_seconds:.2f} images/s")
        else:
            self.logger.warning("Insufficient batches for timed benchmark")

    def inference(self):
        self.model.eval()
        results = {}
        with torch.no_grad():
            progress_bar = tqdm.tqdm(
                self.dataloader, total=len(self.dataloader),
                leave=True, desc="Evaluation Progress"
            )
            for inputs, calibs, targets, info in progress_bar:
                inputs = inputs.to(self.device)
                calibs = calibs.to(self.device)
                img_sizes = info["img_size"].to(self.device)
                outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=0)
                dets = extract_dets_from_outputs(
                    outputs=outputs, K=self.max_objs, topk=self.cfg["topk"]
                ).detach().cpu().numpy()
                calib_objects = [
                    self.dataloader.dataset.get_calib(index)
                    for index in info["img_id"]
                ]
                cpu_info = {
                    key: value.detach().cpu().numpy()
                    for key, value in info.items()
                }
                decoded = decode_detections(
                    dets=dets,
                    info=cpu_info,
                    calibs=calib_objects,
                    cls_mean_size=self.dataloader.dataset.cls_mean_size,
                    threshold=self.cfg.get("threshold", 0.2),
                )
                overlap = set(results).intersection(decoded)
                if overlap:
                    raise ValueError(f"Duplicate decoded image IDs: {sorted(overlap)[:5]}")
                results.update(decoded)
        self.logger.info("Saving predictions to %s", self.prediction_dir)
        self.save_results(results)

    def _expected_kitti_ids(self):
        dataset = self.dataloader.dataset
        if self.dataset_type != "KITTI":
            raise ValueError("Strict prediction isolation currently supports KITTI only")
        if not hasattr(dataset, "idx_list"):
            raise ValueError("KITTI dataset has no idx_list for completeness audit")
        ids = [int(value) for value in dataset.idx_list]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate KITTI IDs in dataset split")
        if len(ids) != len(dataset):
            raise ValueError("KITTI idx_list length differs from dataset length")
        return set(ids)

    def save_results(self, results):
        # Fresh unique location on every inference. Old outputs remain untouched.
        if self.prediction_dir is not None:
            raise RuntimeError("Tester already has a prediction directory")
        eval_root = Path(self.output_dir) / "evaluations"
        eval_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        eval_dir = eval_root / f"{timestamp}_{uuid.uuid4().hex[:12]}"
        eval_dir.mkdir(parents=False, exist_ok=False)
        self.prediction_dir = str(eval_dir / "data")
        expected = self._expected_kitti_ids()
        normalized = {}
        for key, rows in results.items():
            image_id = int(key)
            if image_id in normalized:
                raise ValueError(f"Duplicate normalized KITTI image ID {image_id}")
            normalized[image_id] = rows
        actual = set(normalized)
        if actual != expected:
            raise ValueError(
                f"Incomplete KITTI predictions: missing={len(expected - actual)}, "
                f"unexpected={len(actual - expected)}"
            )
        output_dir = Path(self.prediction_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        for image_id in sorted(expected):
            output_path = output_dir / f"{image_id:06d}.txt"
            with output_path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in normalized[image_id]:
                    class_id = int(row[0])
                    if class_id < 0 or class_id >= len(self.class_name):
                        raise ValueError(f"Invalid class ID: {class_id}")
                    handle.write(f"{self.class_name[class_id]} 0.0 0")
                    for value in row[1:]:
                        handle.write(f" {float(value):.2f}")
                    handle.write("\n")
        self._verify_prediction_files()
        metadata = {
            "checkpoint_path": os.path.abspath(
                self.checkpoint_path or os.path.join(self.output_dir, "checkpoint_best.pth")
            ),
            "provenance_sha256": self.expected_provenance["identity_sha256"],
            "split": self.dataloader.dataset.split,
            "dataset_type": self.dataset_type,
            "prediction_count": len(expected),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (eval_dir / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8"
        )

    def _verify_prediction_files(self):
        expected = {f"{image_id:06d}.txt" for image_id in self._expected_kitti_ids()}
        directory = Path(self.prediction_dir) if self.prediction_dir else None
        if directory is None or not directory.is_dir():
            raise RuntimeError("Prediction directory is missing")
        actual = {item.name for item in directory.iterdir() if item.is_file()}
        if actual != expected:
            raise ValueError(
                f"KITTI prediction file mismatch: missing={len(expected - actual)}, "
                f"unexpected={len(actual - expected)}"
            )
        self.logger.info("Verified complete prediction set: %d files", len(expected))

    def evaluate(self):
        if self.dataloader.dataset.split == "test":
            raise ValueError("KITTI test split cannot be evaluated locally")
        self._verify_prediction_files()
        return self.dataloader.dataset.eval(
            results_dir=self.prediction_dir, logger=self.logger
        )