from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import COCO_CLASSES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class LetterboxInfo:
    ratio: float
    pad_x: int
    pad_y: int
    original_width: int
    original_height: int
    image_size: int


def letterbox_image(image: np.ndarray, image_size: int, color: int = 114) -> tuple[np.ndarray, LetterboxInfo]:
    h, w = image.shape[:2]
    ratio = min(image_size / h, image_size / w)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))
    pad_x = (image_size - new_w) // 2
    pad_y = (image_size - new_h) // 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((image_size, image_size, 3), color, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, LetterboxInfo(ratio, pad_x, pad_y, w, h, image_size)


def transform_boxes_to_letterbox(boxes: torch.Tensor, info: LetterboxInfo) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    out = boxes.clone().float()
    out[:, [0, 2]] = out[:, [0, 2]] * info.ratio + info.pad_x
    out[:, [1, 3]] = out[:, [1, 3]] * info.ratio + info.pad_y
    return out.clamp_(0, info.image_size)


def invert_letterbox_boxes(boxes: torch.Tensor, info: LetterboxInfo) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    out = boxes.clone().float()
    out[:, [0, 2]] = (out[:, [0, 2]] - info.pad_x) / info.ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - info.pad_y) / info.ratio
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, info.original_width)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, info.original_height)
    return out


def augment_hsv_rgb(image: np.ndarray, hgain: float, sgain: float, vgain: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + (random.random() * 2 - 1) * hgain * 179) % 179
    hsv[..., 1] *= 1 + (random.random() * 2 - 1) * sgain
    hsv[..., 2] *= 1 + (random.random() * 2 - 1) * vgain
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def parse_yolo_label_file(
    label_path: str | Path,
    original_width: int,
    original_height: int,
    num_classes: int,
    strict: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(label_path)
    labels: list[int] = []
    boxes: list[list[float]] = []
    if not path.exists():
        return torch.zeros(0, dtype=torch.long), torch.zeros((0, 4), dtype=torch.float32)

    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError("expected 5 fields")
            cls_f, cx, cy, bw, bh = map(float, parts)
            cls = int(cls_f)
            if cls != cls_f or cls < 0 or cls >= num_classes:
                raise ValueError(f"class {cls_f} out of range")
            if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in (cx, cy, bw, bh)):
                raise ValueError("normalized coordinates must be in [0, 1]")
            x1 = (cx - bw / 2) * original_width
            y1 = (cy - bh / 2) * original_height
            x2 = (cx + bw / 2) * original_width
            y2 = (cy + bh / 2) * original_height
            if x2 <= x1 or y2 <= y1:
                raise ValueError("box has non-positive area")
        except ValueError:
            if strict:
                raise ValueError(f"invalid label at {path}:{line_no}: {raw}") from None
            continue
        labels.append(cls)
        boxes.append([x1, y1, x2, y2])

    if not labels:
        return torch.zeros(0, dtype=torch.long), torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(labels, dtype=torch.long), torch.tensor(boxes, dtype=torch.float32)


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        image_size: int,
        num_classes: int,
        augment: bool = False,
        normalize: bool = True,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        hflip_p: float = 0.5,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.augment = augment
        self.normalize = normalize
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.hflip_p = hflip_p
        if not self.image_dir.exists():
            raise FileNotFoundError(f"image directory not found: {self.image_dir}")
        self.image_files = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.image_files[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            image = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            return self._normalize(tensor), self._empty_target()

        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        labels, boxes = parse_yolo_label_file(label_path, w, h, self.num_classes)

        if self.augment:
            image = augment_hsv_rgb(image, self.hsv_h, self.hsv_s, self.hsv_v)
            if random.random() < self.hflip_p:
                image = np.ascontiguousarray(image[:, ::-1])
                if boxes.numel() > 0:
                    x1 = boxes[:, 0].clone()
                    x2 = boxes[:, 2].clone()
                    boxes[:, 0] = w - x2
                    boxes[:, 2] = w - x1

        image, info = letterbox_image(image, self.image_size)
        boxes = transform_boxes_to_letterbox(boxes, info)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) if boxes.numel() else torch.zeros(0, dtype=torch.bool)
        boxes = boxes[valid]
        labels = labels[valid]

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        return self._normalize(tensor), {"boxes": boxes, "labels": labels}

    def _empty_target(self) -> dict[str, torch.Tensor]:
        return {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros(0, dtype=torch.long)}

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return tensor
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    images, targets = zip(*batch)
    batch_indices: list[torch.Tensor] = []
    boxes: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for i, target in enumerate(targets):
        n = len(target["labels"])
        boxes.append(target["boxes"])
        labels.append(target["labels"])
        batch_indices.append(torch.full((n,), i, dtype=torch.long))
    return torch.stack(list(images), 0), {
        "boxes": torch.cat(boxes, 0) if boxes else torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.cat(labels, 0) if labels else torch.zeros(0, dtype=torch.long),
        "batch_index": torch.cat(batch_indices, 0) if batch_indices else torch.zeros(0, dtype=torch.long),
    }


def validate_yolo_dataset(root: str | Path, num_classes: int) -> dict[str, Any]:
    data_root = Path(root)
    result: dict[str, Any] = {"root": str(data_root), "splits": {}, "errors": []}
    for split in ("train", "val"):
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        split_result = {"images": 0, "label_files": 0, "boxes": 0, "missing_labels": 0}
        if not image_dir.exists():
            result["errors"].append(f"missing image directory: {image_dir}")
            result["splits"][split] = split_result
            continue
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            split_result["images"] += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                split_result["missing_labels"] += 1
                continue
            split_result["label_files"] += 1
            raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if raw is None:
                result["errors"].append(f"failed to decode image: {image_path}")
                continue
            labels, boxes = parse_yolo_label_file(label_path, raw.shape[1], raw.shape[0], num_classes, strict=True)
            split_result["boxes"] += len(labels)
            if boxes.numel() and ((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])).any():
                result["errors"].append(f"invalid box area in: {label_path}")
        result["splits"][split] = split_result
    return result


def write_dataset_yaml(root: str | Path, num_classes: int) -> None:
    import yaml

    names = {i: COCO_CLASSES[i] for i in range(num_classes)}
    payload = {
        "path": str(Path(root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    with (Path(root) / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    aug_cfg = cfg.get("augmentation", {})
    root = Path(data_cfg["root"])
    image_size = int(cfg["model"]["image_size"])
    num_classes = int(cfg["model"]["num_classes"])
    workers = int(train_cfg.get("workers", 0))
    train_ds = YoloDetectionDataset(
        root / "images" / "train",
        root / "labels" / "train",
        image_size=image_size,
        num_classes=num_classes,
        augment=True,
        hsv_h=float(aug_cfg.get("hsv_h", 0.015)),
        hsv_s=float(aug_cfg.get("hsv_s", 0.7)),
        hsv_v=float(aug_cfg.get("hsv_v", 0.4)),
        hflip_p=float(aug_cfg.get("hflip_p", 0.5)),
    )
    val_ds = YoloDetectionDataset(
        root / "images" / "val",
        root / "labels" / "val",
        image_size=image_size,
        num_classes=num_classes,
        augment=False,
    )
    common = {
        "num_workers": workers,
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
        "collate_fn": collate_fn,
        "persistent_workers": workers > 0,
    }
    return (
        DataLoader(
            train_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            drop_last=bool(train_cfg.get("drop_last", False)),
            **common,
        ),
        DataLoader(val_ds, batch_size=int(train_cfg.get("val_batch_size", train_cfg["batch_size"])), shuffle=False, **common),
    )


def count_images(root: str | Path, split: str) -> int:
    image_dir = Path(root) / "images" / split
    if not image_dir.exists():
        return 0
    return sum(1 for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
