from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss


def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1 - prob) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_factor * (1 - pt).pow(gamma) * bce


def make_centers(h: int, w: int, stride: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack(((xs + 0.5) * stride, (ys + 0.5) * stride), dim=-1)


def decode_ltrb(distances: torch.Tensor, stride: int) -> torch.Tensor:
    b, _, h, w = distances.shape
    centers = make_centers(h, w, stride, distances.device).view(1, h, w, 2)
    d = distances.permute(0, 2, 3, 1) * stride
    x1 = centers[..., 0] - d[..., 0]
    y1 = centers[..., 1] - d[..., 1]
    x2 = centers[..., 0] + d[..., 2]
    y2 = centers[..., 1] + d[..., 3]
    return torch.stack((x1, y1, x2, y2), dim=-1).view(b, h, w, 4)


class YoloLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int,
        strides: list[int],
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        box_weight: float = 5.0,
        l1_weight: float = 0.25,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        small_box: int = 64,
        medium_box: int = 192,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = strides
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.l1_weight = l1_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.small_box = small_box
        self.medium_box = medium_box

    def forward(self, outputs: dict[str, Any], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        cls_logits = outputs["cls"]
        obj_logits = outputs["obj"]
        box_pred = outputs["box"]
        target_maps = self._build_targets(outputs, targets)

        obj_loss = torch.zeros((), device=obj_logits[0].device)
        cls_loss = torch.zeros((), device=obj_logits[0].device)
        box_loss = torch.zeros((), device=obj_logits[0].device)
        l1_loss = torch.zeros((), device=obj_logits[0].device)
        pos_total = 0

        for level, stride in enumerate(self.strides):
            obj_t = target_maps[level]["obj"]
            cls_t = target_maps[level]["cls"]
            box_t = target_maps[level]["box"]
            pos = obj_t[:, 0] > 0.5
            pos_count = int(pos.sum().item())
            pos_total += pos_count

            obj_loss = obj_loss + focal_bce_with_logits(obj_logits[level], obj_t, self.focal_alpha, self.focal_gamma).mean()

            if pos_count == 0:
                continue

            cls_p = cls_logits[level].permute(0, 2, 3, 1)[pos]
            cls_target = cls_t.permute(0, 2, 3, 1)[pos]
            cls_loss = cls_loss + focal_bce_with_logits(cls_p, cls_target, self.focal_alpha, self.focal_gamma).mean()

            pred_dist = box_pred[level].permute(0, 2, 3, 1)[pos]
            target_dist = box_t.permute(0, 2, 3, 1)[pos]
            l1_loss = l1_loss + F.smooth_l1_loss(pred_dist, target_dist, reduction="mean")

            pred_boxes = decode_ltrb(box_pred[level], stride)[pos]
            target_boxes = target_maps[level]["xyxy"].permute(0, 2, 3, 1)[pos]
            box_loss = box_loss + generalized_box_iou_loss(pred_boxes, target_boxes, reduction="mean")

        total = self.obj_weight * obj_loss + self.cls_weight * cls_loss + self.box_weight * box_loss + self.l1_weight * l1_loss
        stats = {
            "loss": float(total.detach().item()),
            "loss_obj": float(obj_loss.detach().item()),
            "loss_cls": float(cls_loss.detach().item()),
            "loss_box": float(box_loss.detach().item()),
            "loss_l1": float(l1_loss.detach().item()),
            "num_pos": float(pos_total),
            "num_gt": float(len(targets["labels"])),
        }
        return total, stats

    def _target_level(self, box: torch.Tensor) -> int:
        max_side = float(torch.max(box[2:] - box[:2]).item())
        if max_side < self.small_box:
            return 0
        if max_side < self.medium_box:
            return 1
        return 2

    def _build_targets(self, outputs: dict[str, Any], targets: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        batch_size = outputs["obj"][0].shape[0]
        device = outputs["obj"][0].device
        maps: list[dict[str, torch.Tensor]] = []
        for level, obj in enumerate(outputs["obj"]):
            _, _, h, w = obj.shape
            maps.append(
                {
                    "obj": torch.zeros((batch_size, 1, h, w), device=device),
                    "cls": torch.zeros((batch_size, self.num_classes, h, w), device=device),
                    "box": torch.zeros((batch_size, 4, h, w), device=device),
                    "xyxy": torch.zeros((batch_size, 4, h, w), device=device),
                    "area": torch.full((batch_size, 1, h, w), float("inf"), device=device),
                }
            )

        boxes = targets["boxes"].to(device)
        labels = targets["labels"].to(device)
        batch_idx = targets["batch_index"].to(device)
        for i in range(len(labels)):
            box = boxes[i].clamp(0, self.image_size)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            label = int(labels[i].item())
            level = self._target_level(box)
            stride = self.strides[level]
            _, _, h, w = outputs["obj"][level].shape
            cx = (box[0] + box[2]) * 0.5
            cy = (box[1] + box[3]) * 0.5
            gx = int(torch.clamp(torch.floor(cx / stride), 0, w - 1).item())
            gy = int(torch.clamp(torch.floor(cy / stride), 0, h - 1).item())
            b = int(batch_idx[i].item())
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area >= maps[level]["area"][b, 0, gy, gx]:
                continue
            center_x = (gx + 0.5) * stride
            center_y = (gy + 0.5) * stride
            ltrb = torch.tensor(
                [center_x - box[0], center_y - box[1], box[2] - center_x, box[3] - center_y],
                device=device,
                dtype=torch.float32,
            ).clamp(min=0.01) / stride
            maps[level]["obj"][b, 0, gy, gx] = 1.0
            maps[level]["cls"][b, :, gy, gx] = 0.0
            maps[level]["cls"][b, label, gy, gx] = 1.0
            maps[level]["box"][b, :, gy, gx] = ltrb
            maps[level]["xyxy"][b, :, gy, gx] = box
            maps[level]["area"][b, 0, gy, gx] = area
        return maps
