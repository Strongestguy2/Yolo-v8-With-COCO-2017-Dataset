from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD

HEADER_HEIGHT = 28
PREDICTION_COLOR = (0, 220, 80)
TARGET_COLOR = (0, 120, 255)


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    tensor = image.detach().float().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"expected CHW image tensor, got shape {tuple(tensor.shape)}")
    if tensor.numel() and (float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0):
        tensor = tensor * IMAGENET_STD + IMAGENET_MEAN
    tensor = tensor.clamp(0.0, 1.0)
    rgb = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(rgb)


def label_text(label: int, class_names: list[str] | None = None) -> str:
    if class_names is not None and 0 <= label < len(class_names):
        return class_names[label]
    return str(label)


def draw_boxes(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str] | None = None,
    scores: torch.Tensor | None = None,
    color: tuple[int, int, int] = (0, 220, 80),
) -> np.ndarray:
    output = image.copy()
    if boxes.numel() == 0:
        return output

    height, width = output.shape[:2]
    boxes_cpu = boxes.detach().float().cpu()
    labels_cpu = labels.detach().long().cpu()
    scores_cpu = scores.detach().float().cpu() if scores is not None else None

    for idx, box in enumerate(boxes_cpu):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = label_text(int(labels_cpu[idx]), class_names)
        if scores_cpu is not None:
            text = f"{text} {float(scores_cpu[idx]):.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        while font_scale > 0.25:
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            if text_width <= max(1, width - 4):
                break
            font_scale -= 0.05

        text_x = min(x1, max(0, width - text_width - 4))
        text_y = max(20, y1 - 5)
        if text_y + baseline > height - 1:
            text_y = max(text_height + 2, height - baseline - 1)
        cv2.rectangle(output, (text_x - 2, text_y - text_height - 2), (text_x + text_width + 2, text_y + baseline), color, -1)
        cv2.putText(output, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    return output


def add_header(image: np.ndarray, title: str, color: tuple[int, int, int]) -> np.ndarray:
    panel = np.full((image.shape[0] + HEADER_HEIGHT, image.shape[1], 3), 32, dtype=np.uint8)
    panel[HEADER_HEIGHT:, :] = image
    cv2.rectangle(panel, (0, 0), (image.shape[1], HEADER_HEIGHT), color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    scale = 0.55
    while scale > 0.3:
        (text_width, _), _ = cv2.getTextSize(title, font, scale, thickness)
        if text_width <= max(1, image.shape[1] - 12):
            break
        scale -= 0.05
    cv2.putText(panel, title, (6, 20), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return panel


def select_image_targets(targets: dict[str, torch.Tensor], image_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_index = targets.get("batch_index")
    if batch_index is None or batch_index.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros(0, dtype=torch.long)
    mask = batch_index == image_index
    return targets["boxes"][mask], targets["labels"][mask]


def build_comparison_panel(
    image: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    prediction_boxes: torch.Tensor,
    prediction_labels: torch.Tensor,
    prediction_scores: torch.Tensor,
    class_names: list[str] | None = None,
) -> np.ndarray:
    rgb = tensor_to_rgb(image)
    predicted = draw_boxes(rgb, prediction_boxes, prediction_labels, class_names=class_names, scores=prediction_scores, color=PREDICTION_COLOR)
    actual = draw_boxes(rgb, target_boxes, target_labels, class_names=class_names, color=TARGET_COLOR)
    predicted = add_header(predicted, "Predictions", PREDICTION_COLOR)
    actual = add_header(actual, "Targets", TARGET_COLOR)
    return np.concatenate([predicted, actual], axis=1)


def save_comparison_image(
    output_path: str | Path,
    image: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    prediction_boxes: torch.Tensor,
    prediction_labels: torch.Tensor,
    prediction_scores: torch.Tensor,
    class_names: list[str] | None = None,
) -> Path:
    output = build_comparison_panel(
        image=image,
        target_boxes=target_boxes,
        target_labels=target_labels,
        prediction_boxes=prediction_boxes,
        prediction_labels=prediction_labels,
        prediction_scores=prediction_scores,
        class_names=class_names,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write preview image: {path}")
    return path
