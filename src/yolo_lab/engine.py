from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler
from tqdm.auto import tqdm

from .checkpoint import checkpoint_payload, load_training_checkpoint, save_training_checkpoint
from .config import save_config
from .data import build_dataloaders
from .evaluate import evaluate_loss
from .loss import YoloLoss
from .model import build_model
from .utils import Stopwatch, get_device, set_seed


def train(cfg: dict[str, Any], resume: bool = False, max_steps: int | None = None, max_hours: float | None = None) -> Path:
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "auto")))
    run_dir = resolve_run_dir(cfg)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    model.apply_freeze_schedule(0)
    criterion = build_criterion(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.05)),
    )
    total_steps = int(cfg["train"].get("total_steps") or max(1, len(train_loader) * int(cfg["train"]["epochs"])))
    warmup_steps = int(cfg["train"].get("warmup_steps", min(1000, max(1, total_steps // 10))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_factor(step, warmup_steps, total_steps, float(cfg["train"].get("min_lr_ratio", 0.05))),
    )
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best_metric: float | None = None
    last_path = checkpoint_dir / "last.pt"
    if resume and last_path.exists():
        ckpt = load_training_checkpoint(last_path, model, optimizer, scheduler, scaler, device)
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_metric = ckpt.get("best_metric")
        print(f"Resumed {last_path} at epoch={start_epoch}, step={global_step}")

    log_path = run_dir / "train_log.csv"
    ensure_log_header(log_path)
    stopwatch = Stopwatch()
    steps_this_run = 0
    last_checkpoint_time = time.time()
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    checkpoint_steps = int(cfg["train"].get("checkpoint_steps", 100))
    checkpoint_minutes = float(cfg["train"].get("checkpoint_minutes", 15))
    max_grad_norm = float(cfg["train"].get("grad_clip_norm", 10.0))

    try:
        for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
            model.apply_freeze_schedule(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg['train']['epochs']}")
            for batch_idx, (images, targets) in enumerate(pbar):
                if max_steps is not None and steps_this_run >= max_steps:
                    save_last(last_path, model, optimizer, scheduler, scaler, cfg, epoch, global_step, best_metric)
                    return run_dir
                if stopwatch.exceeded_hours(max_hours):
                    save_training_checkpoint(
                        checkpoint_dir / "safe_stop.pt",
                        **checkpoint_payload(model, optimizer, scheduler, scaler, cfg, epoch, global_step, best_metric),
                    )
                    return run_dir

                step_start = time.time()
                images = images.to(device, non_blocking=True)
                moved_targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss, stats = criterion(outputs, moved_targets)
                    scaled_loss = loss / grad_accum

                scaler.scale(scaled_loss).backward()
                optimizer_stepped = False
                if (batch_idx + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_stepped = (not use_amp) or scaler.get_scale() >= scale_before
                    if optimizer_stepped:
                        scheduler.step()
                        global_step += 1
                        steps_this_run += 1
                else:
                    grad_norm = torch.tensor(0.0)

                row = {
                    "time": int(time.time()),
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "lr": scheduler.get_last_lr()[0],
                    "step_time": time.time() - step_start,
                    "grad_norm": float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                    "optimizer_step": int(optimizer_stepped),
                    **stats,
                }
                append_log(log_path, row)
                pbar.set_postfix(loss=f"{stats['loss']:.3f}", pos=int(stats["num_pos"]), lr=f"{scheduler.get_last_lr()[0]:.2e}")

                should_checkpoint = global_step > 0 and (
                    global_step % checkpoint_steps == 0 or (time.time() - last_checkpoint_time) >= checkpoint_minutes * 60
                )
                if should_checkpoint:
                    save_last(last_path, model, optimizer, scheduler, scaler, cfg, epoch, global_step, best_metric)
                    last_checkpoint_time = time.time()

            val_stats = evaluate_loss(model, criterion, val_loader, device, max_batches=int(cfg["train"].get("val_max_batches", 20)))
            metric = val_stats["loss"]
            if best_metric is None or metric < best_metric:
                best_metric = metric
                save_training_checkpoint(
                    checkpoint_dir / "best.pt",
                    **checkpoint_payload(model, optimizer, scheduler, scaler, cfg, epoch + 1, global_step, best_metric),
                )
            save_last(last_path, model, optimizer, scheduler, scaler, cfg, epoch + 1, global_step, best_metric)
    except KeyboardInterrupt:
        save_training_checkpoint(
            checkpoint_dir / "safe_stop.pt",
            **checkpoint_payload(model, optimizer, scheduler, scaler, cfg, start_epoch, global_step, best_metric),
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            save_training_checkpoint(
                checkpoint_dir / "safe_stop.pt",
                **checkpoint_payload(model, optimizer, scheduler, scaler, cfg, start_epoch, global_step, best_metric),
            )
        raise
    return run_dir


def build_criterion(cfg: dict[str, Any]) -> YoloLoss:
    loss_cfg = cfg.get("loss", {})
    return YoloLoss(
        num_classes=int(cfg["model"]["num_classes"]),
        image_size=int(cfg["model"]["image_size"]),
        strides=[8, 16, 32],
        obj_weight=float(loss_cfg.get("obj_weight", 1.0)),
        cls_weight=float(loss_cfg.get("cls_weight", 1.0)),
        box_weight=float(loss_cfg.get("box_weight", 5.0)),
        l1_weight=float(loss_cfg.get("l1_weight", 0.25)),
        focal_alpha=float(loss_cfg.get("focal_alpha", 0.25)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
    )


def lr_factor(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-6, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def resolve_run_dir(cfg: dict[str, Any]) -> Path:
    run_name = str(cfg["train"].get("run_name", "custom_yolo"))
    return Path(cfg["train"].get("output_dir", "outputs/runs")) / run_name


def save_last(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    cfg: dict[str, Any],
    epoch: int,
    global_step: int,
    best_metric: float | None,
) -> None:
    save_training_checkpoint(path, **checkpoint_payload(model, optimizer, scheduler, scaler, cfg, epoch, global_step, best_metric))


LOG_FIELDS = [
    "time",
    "epoch",
    "batch",
    "global_step",
    "lr",
    "step_time",
    "grad_norm",
    "optimizer_step",
    "loss",
    "loss_obj",
    "loss_cls",
    "loss_box",
    "loss_l1",
    "num_pos",
    "num_gt",
]


def ensure_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()


def append_log(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        writer.writerow(row)
