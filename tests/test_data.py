from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from yolo_lab.data import (
    YoloDetectionDataset,
    collate_fn,
    invert_letterbox_boxes,
    letterbox_image,
    parse_yolo_label_file,
    transform_boxes_to_letterbox,
)


def test_parse_yolo_label_filters_invalid_lines(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.25 0.25",
                "99 0.5 0.5 0.2 0.2",
                "1 nope 0.5 0.2 0.2",
                "2 1.5 0.5 0.2 0.2",
            ]
        ),
        encoding="utf-8",
    )
    labels, boxes = parse_yolo_label_file(label, original_width=200, original_height=100, num_classes=3)
    assert labels.tolist() == [0]
    assert boxes.shape == (1, 4)
    assert boxes[0].tolist() == pytest.approx([75, 37.5, 125, 62.5])


def test_parse_yolo_label_strict_raises(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("3 0.5 0.5 0.2 0.2", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_yolo_label_file(label, 100, 100, num_classes=3, strict=True)


def test_letterbox_roundtrip_boxes() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    _, info = letterbox_image(image, 128)
    boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    transformed = transform_boxes_to_letterbox(boxes, info)
    restored = invert_letterbox_boxes(transformed, info)
    assert torch.allclose(restored, boxes, atol=1e-4)


def test_dataset_and_collate_with_empty_labels(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    img = np.full((32, 48, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(image_dir / "a.jpg"), img)
    cv2.imwrite(str(image_dir / "b.jpg"), img)
    (label_dir / "a.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    ds = YoloDetectionDataset(image_dir, label_dir, image_size=64, num_classes=3, augment=False)
    assert len(ds) == 2
    batch = collate_fn([ds[0], ds[1]])
    images, targets = batch
    assert images.shape == (2, 3, 64, 64)
    assert targets["boxes"].shape == (1, 4)
    assert targets["labels"].tolist() == [1]
    assert targets["batch_index"].tolist() == [0]
