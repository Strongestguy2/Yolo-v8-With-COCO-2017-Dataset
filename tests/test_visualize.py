from __future__ import annotations

from pathlib import Path

import cv2
import torch

from yolo_lab.visualize import HEADER_HEIGHT, save_comparison_image, tensor_to_rgb


def test_tensor_to_rgb_preserves_unit_range_test_images() -> None:
    image = torch.zeros(3, 4, 5)
    image[0] = 1.0

    rgb = tensor_to_rgb(image)

    assert rgb.shape == (4, 5, 3)
    assert rgb[0, 0].tolist() == [255, 0, 0]


def test_save_comparison_image_writes_labeled_panel(tmp_path: Path) -> None:
    output_path = tmp_path / "preview.png"
    image = torch.zeros(3, 32, 32)
    boxes = torch.tensor([[4.0, 6.0, 28.0, 26.0]])
    labels = torch.tensor([0])

    written = save_comparison_image(
        output_path,
        image=image,
        target_boxes=boxes,
        target_labels=labels,
        prediction_boxes=boxes,
        prediction_labels=labels,
        prediction_scores=torch.tensor([0.75]),
        class_names=["sample"],
    )

    assert written == output_path
    decoded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (32 + HEADER_HEIGHT, 64)
