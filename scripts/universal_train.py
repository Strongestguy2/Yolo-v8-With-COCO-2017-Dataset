from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from yolo_lab.config import load_config, set_cfg
from yolo_lab.engine import train
from yolo_lab.prepare import prepare_coco
from yolo_lab.smoke import build_smoke_config
from yolo_lab.runs import (
    create_branch_from_checkpoint,
    load_current_run_config,
    resolve_mode2_config,
    run_dir_from_cfg,
    save_current_run_config,
    select_checkpoint_via_dialog,
)


def parse_value(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in text or "e" in lowered:
            return float(text)
        return int(text)
    except ValueError:
        return text


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use key=value format: {override}")
        dotted, raw = override.split("=", 1)
        set_cfg(cfg, dotted.strip(), parse_value(raw))
    return cfg


def dataset_ready(root: str | Path) -> bool:
    root_path = Path(root)
    train_dir = root_path / "images" / "train"
    val_dir = root_path / "images" / "val"
    has_train = train_dir.exists() and any(train_dir.iterdir())
    has_val = val_dir.exists() and any(val_dir.iterdir())
    return has_train and has_val


def main() -> None:
    parser = argparse.ArgumentParser(description="One command to prepare data, train, resume, and log previews.")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=2, help="1 = smoke, 2 = auto resume, 3 = branch from a selected checkpoint")
    parser.add_argument("--config", default="configs/coco_resnet34.yaml")
    parser.add_argument("--smoke", action="store_true", help="Alias for --mode 1.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to branch from in mode 3.")
    parser.add_argument("--branch-name", default=None, help="Optional custom branch name in mode 3.")
    parser.add_argument("--download-increment", type=int, default=1000)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--prepare-if-missing", action="store_true", default=True)
    parser.add_argument("--no-prepare-if-missing", dest="prepare_if_missing", action="store_false")
    parser.add_argument("--new-session", action="store_true", help="Ignore current run pointers and start from the selected config.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values using dotted key=value syntax.")
    args = parser.parse_args()

    mode = 1 if args.smoke else args.mode
    if mode == 1:
        cfg = build_smoke_config(args.config)
    elif mode == 2 and not args.new_session:
        current_config = load_current_run_config()
        cfg = load_config(current_config) if current_config is not None and current_config.exists() else load_config(args.config)
    else:
        cfg = load_config(args.config)

    cfg = apply_overrides(cfg, args.overrides)
    cfg.setdefault("train", {})
    cfg["train"].setdefault("vis_every_images", 100)
    cfg["train"].setdefault("vis_max_images", 2)
    cfg["train"].setdefault("vis_conf_threshold", 0.25)
    cfg["train"].setdefault("vis_iou_threshold", 0.45)

    if mode == 1:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cfg["train"]["output_dir"] = "outputs/runs/smoke_reports"
        cfg["train"]["run_name"] = f"smoke_{timestamp}"
        cfg["train"]["vis_every_images"] = 1
        cfg["train"]["vis_max_images"] = 1
        cfg["train"]["save_checkpoints"] = False
        cfg["train"]["resume"] = False
        cfg["train"]["epochs"] = 1
        cfg["train"]["amp"] = False
        cfg = apply_overrides(cfg, args.overrides)

    if mode == 2 and not args.new_session:
        cfg = resolve_mode2_config(cfg)

    if mode == 3:
        checkpoint = Path(args.checkpoint) if args.checkpoint else select_checkpoint_via_dialog(run_dir_from_cfg(cfg) / "checkpoints")
        if checkpoint is None:
            raise RuntimeError("no checkpoint selected for mode 3")
        cfg, branch_run_dir, main_run_dir = create_branch_from_checkpoint(cfg, checkpoint, branch_name=args.branch_name)
        print(f"Branch created: {branch_run_dir}")
        print(f"Main run: {main_run_dir}")

    if mode in {2, 3}:
        cfg = apply_overrides(cfg, args.overrides)

    data_root = cfg["data"]["root"]
    if mode == 1:
        print(f"Smoke dataset ready at {data_root}")
    elif args.prepare_if_missing and not dataset_ready(data_root):
        prepare_result = prepare_coco(cfg, download_increment=args.download_increment, force=args.force_prepare)
        print(f"Dataset: {prepare_result}")
    elif args.download_increment is not None or args.force_prepare:
        prepare_result = prepare_coco(cfg, download_increment=args.download_increment, force=args.force_prepare)
        print(f"Dataset: {prepare_result}")

    run_dir = train(
        cfg,
        resume=args.resume if mode != 1 else False,
        max_steps=args.max_steps,
        max_hours=args.max_hours,
        save_checkpoints=bool(cfg["train"].get("save_checkpoints", True)),
    )
    if mode in {2, 3} and bool(cfg["train"].get("save_checkpoints", True)):
        save_current_run_config(run_dir / "config.yaml")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
