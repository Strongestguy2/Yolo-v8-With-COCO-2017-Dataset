from __future__ import annotations

from pathlib import Path

import torch

from yolo_lab.checkpoint import checkpoint_payload, load_training_checkpoint, save_training_checkpoint
from yolo_lab.engine import lr_factor
from yolo_lab.model import build_model


def test_checkpoint_roundtrip_restores_step(tmp_path: Path) -> None:
    cfg = {
        "model": {
            "num_classes": 3,
            "image_size": 128,
            "backbone": "resnet34",
            "pretrained_backbone": False,
            "fpn_channels": 16,
            "head_channels": 16,
        }
    }
    model = build_model(cfg, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: lr_factor(s, 2, 10, 0.05))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "last.pt"
    payload = checkpoint_payload(model, optimizer, scheduler, scaler, cfg, epoch=2, global_step=7, best_metric=1.5)
    save_training_checkpoint(path, **payload)

    restored = build_model(cfg, pretrained=False)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lr_lambda=lambda s: lr_factor(s, 2, 10, 0.05))
    restored_scaler = torch.amp.GradScaler("cpu", enabled=False)
    ckpt = load_training_checkpoint(path, restored, restored_optimizer, restored_scheduler, restored_scaler)
    assert ckpt["epoch"] == 2
    assert ckpt["global_step"] == 7
    assert ckpt["best_metric"] == 1.5
