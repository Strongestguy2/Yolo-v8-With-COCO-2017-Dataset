from __future__ import annotations

import torch

from yolo_lab.infer import decode_predictions
from yolo_lab.loss import YoloLoss
from yolo_lab.model import build_model


def tiny_cfg() -> dict:
    return {
        "model": {
            "num_classes": 3,
            "image_size": 128,
            "backbone": "resnet34",
            "pretrained_backbone": False,
            "fpn_channels": 16,
            "head_channels": 16,
            "freeze_backbone_epochs": 0,
        }
    }


def test_model_forward_cpu_shapes() -> None:
    model = build_model(tiny_cfg(), pretrained=False)
    outputs = model(torch.randn(1, 3, 128, 128))
    assert len(outputs["cls"]) == 3
    assert outputs["cls"][0].shape[:2] == (1, 3)
    assert outputs["obj"][0].shape[1] == 1
    assert outputs["box"][0].shape[1] == 4


def test_one_train_step_cpu() -> None:
    cfg = tiny_cfg()
    model = build_model(cfg, pretrained=False)
    criterion = YoloLoss(num_classes=3, image_size=128, strides=[8, 16, 32])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    targets = {
        "boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0]]),
        "labels": torch.tensor([1]),
        "batch_index": torch.tensor([0]),
    }
    outputs = model(torch.randn(1, 3, 128, 128))
    loss, stats = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert stats["num_pos"] == 1


def test_decode_predictions_shape() -> None:
    model = build_model(tiny_cfg(), pretrained=False)
    outputs = model(torch.randn(1, 3, 128, 128))
    preds = decode_predictions(outputs, conf_threshold=0.0, max_det=5, image_size=128)
    assert len(preds) == 1
    assert preds[0].shape[1] == 6
    assert len(preds[0]) <= 5


def test_cuda_amp_step_if_available() -> None:
    if not torch.cuda.is_available():
        return
    cfg = tiny_cfg()
    model = build_model(cfg, pretrained=False).cuda()
    criterion = YoloLoss(num_classes=3, image_size=128, strides=[8, 16, 32]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    targets = {
        "boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0]], device="cuda"),
        "labels": torch.tensor([1], device="cuda"),
        "batch_index": torch.tensor([0], device="cuda"),
    }
    with torch.autocast(device_type="cuda", enabled=True):
        loss, _ = criterion(model(torch.randn(1, 3, 128, 128, device="cuda")), targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert torch.isfinite(loss.detach().cpu())
