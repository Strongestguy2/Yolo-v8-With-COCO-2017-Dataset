from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import load_config


def _write_sample(image_path: Path, label_path: Path, image_size: int, class_id: int, box: tuple[float, float, float, float]) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    cx, cy, bw, bh = box
    x1 = int(round((cx - bw / 2) * image_size))
    y1 = int(round((cy - bh / 2) * image_size))
    x2 = int(round((cx + bw / 2) * image_size))
    y2 = int(round((cy + bh / 2) * image_size))

    image = np.full((image_size, image_size, 3), 96, dtype=np.uint8)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), thickness=-1)
    cv2.imwrite(str(image_path), image)
    label_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")


def ensure_tiny_detection_dataset(root: str | Path, image_size: int = 96) -> Path:
    data_root = Path(root)
    samples = {
        "train": [
            ("train_0", 0, (0.50, 0.50, 0.42, 0.42)),
        ],
        "val": [
            ("val_0", 0, (0.52, 0.48, 0.38, 0.38)),
        ],
    }

    for split, items in samples.items():
        for stem, class_id, box in items:
            image_path = data_root / "images" / split / f"{stem}.jpg"
            label_path = data_root / "labels" / split / f"{stem}.txt"
            _write_sample(image_path, label_path, image_size, class_id, box)

    return data_root


def build_smoke_config(
    base_config: str | Path = "configs/smoke.yaml",
    data_root: str | Path = "data/smoke_local_yolo",
    image_size: int = 96,
    steps: int = 1,
) -> dict[str, Any]:
    cfg = load_config(base_config)
    smoke_root = ensure_tiny_detection_dataset(data_root, image_size=image_size)

    cfg["device"] = "auto"
    cfg["model"]["num_classes"] = 3
    cfg["model"]["image_size"] = image_size
    cfg["model"]["pretrained_backbone"] = False
    cfg["model"]["fpn_channels"] = 16
    cfg["model"]["head_channels"] = 16
    cfg["model"]["freeze_backbone_epochs"] = 0

    cfg["data"]["root"] = str(smoke_root)
    cfg["data"]["state_path"] = str(smoke_root.parent / "smoke_local_state.json")
    cfg["data"]["train_samples"] = 1
    cfg["data"]["val_samples"] = 1

    cfg["train"]["run_name"] = "smoke_local"
    cfg["train"]["output_dir"] = "outputs/runs"
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = 1
    cfg["train"]["val_batch_size"] = 1
    cfg["train"]["workers"] = 0
    cfg["train"]["pin_memory"] = False
    cfg["train"]["drop_last"] = False
    cfg["train"]["grad_accum"] = 1
    cfg["train"]["lr"] = 0.001
    cfg["train"]["min_lr_ratio"] = 0.1
    cfg["train"]["weight_decay"] = 0.0
    cfg["train"]["warmup_steps"] = 0
    cfg["train"]["total_steps"] = steps
    cfg["train"]["amp"] = False
    cfg["train"]["grad_clip_norm"] = 5.0
    cfg["train"]["checkpoint_steps"] = max(1, steps // 2)
    cfg["train"]["checkpoint_minutes"] = 60
    cfg["train"]["val_max_batches"] = 1
    return cfg