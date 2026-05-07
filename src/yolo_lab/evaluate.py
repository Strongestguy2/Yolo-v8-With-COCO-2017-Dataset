from __future__ import annotations

from typing import Any

import torch
from tqdm.auto import tqdm

from .infer import decode_predictions


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        moved = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        outputs = model(images)
        _, stats = criterion(outputs, moved)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if was_training:
        model.train()
    if batches == 0:
        return {"loss": float("inf")}
    return {key: value / batches for key, value in totals.items()}


@torch.no_grad()
def evaluate_prediction_counts(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    conf_threshold: float,
    max_batches: int = 10,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    images_seen = 0
    predictions_seen = 0
    for images, _ in tqdm(loader, desc="predict-val", leave=False):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = decode_predictions(outputs, conf_threshold=conf_threshold)
        images_seen += len(preds)
        predictions_seen += sum(len(p) for p in preds)
        if images_seen >= max_batches * loader.batch_size:
            break
    if was_training:
        model.train()
    return {"avg_predictions_per_image": predictions_seen / max(1, images_seen)}
