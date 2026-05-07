from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from .constants import COCO_CLASSES
from .data import IMAGENET_MEAN, IMAGENET_STD, LetterboxInfo, invert_letterbox_boxes, letterbox_image
from .loss import decode_ltrb


def decode_predictions(
    outputs: dict[str, Any],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 100,
    image_size: int = 640,
) -> list[torch.Tensor]:
    predictions: list[torch.Tensor] = []
    batch_size = outputs["obj"][0].shape[0]
    for b in range(batch_size):
        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for level, stride in enumerate(outputs["strides"]):
            obj = outputs["obj"][level][b].sigmoid().flatten()
            cls_prob = outputs["cls"][level][b].sigmoid().permute(1, 2, 0).reshape(-1, outputs["cls"][level].shape[1])
            cls_score, cls_label = cls_prob.max(dim=1)
            score = obj * cls_score
            keep = score >= conf_threshold
            if not keep.any():
                continue
            boxes = decode_ltrb(outputs["box"][level], stride)[b].reshape(-1, 4)
            all_boxes.append(boxes[keep].clamp(0, image_size))
            all_scores.append(score[keep])
            all_labels.append(cls_label[keep])
        if not all_boxes:
            predictions.append(torch.zeros((0, 6), device=outputs["obj"][0].device))
            continue
        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        keep_idx = batched_nms(boxes, scores, labels, iou_threshold)[:max_det]
        predictions.append(torch.cat((boxes[keep_idx], scores[keep_idx, None], labels[keep_idx, None].float()), dim=1))
    return predictions


def load_image_tensor(path: str | Path, image_size: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray, LetterboxInfo]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxed, info = letterbox_image(rgb, image_size)
    tensor = torch.from_numpy(np.ascontiguousarray(boxed)).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0).to(device), rgb, info


def annotate_image(
    rgb: np.ndarray,
    predictions: torch.Tensor,
    info: LetterboxInfo,
    class_names: list[str] | None = None,
) -> np.ndarray:
    class_names = class_names or COCO_CLASSES
    output = rgb.copy()
    if predictions.numel() == 0:
        return output
    boxes = invert_letterbox_boxes(predictions[:, :4].cpu(), info)
    scores = predictions[:, 4].cpu()
    labels = predictions[:, 5].long().cpu()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 80), 2)
        text = f"{name} {float(score):.2f}"
        cv2.putText(output, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 2)
    return output


def save_annotated(path: str | Path, rgb: np.ndarray) -> Path:
    src = Path(path)
    out = src.with_name(f"{src.stem}_pred{src.suffix}")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out), bgr)
    return out
