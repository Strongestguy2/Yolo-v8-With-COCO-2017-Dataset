#!/usr/bin/env python3
"""
Debug script to test smoke training with better visualization.
This runs a quick smoke test with larger images and more steps to make visuals readable.

Usage:
  python scripts/smoke_debug.py                    # Run with defaults (256px, 5 steps)
  python scripts/smoke_debug.py --image-size 512  # Run with 512px images
  python scripts/smoke_debug.py --steps 10        # Run for 10 steps
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from yolo_lab.config import load_config
from yolo_lab.engine import train
from yolo_lab.smoke import build_smoke_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run smoke test with larger images and more steps for debugging visuals."
    )
    parser.add_argument("--image-size", type=int, default=256, help="Image size for smoke dataset (default: 256)")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps (default: 5)")
    parser.add_argument("--no-checkpoints", action="store_true", help="Don't save checkpoints")
    args = parser.parse_args()

    print(f"🎬 Smoke debug test: {args.image_size}px, {args.steps} steps")

    # Build smoke config with custom parameters
    cfg = build_smoke_config(
        base_config="configs/smoke.yaml",
        image_size=args.image_size,
        steps=args.steps,
    )

    # Add visualization settings
    cfg.setdefault("train", {})
    cfg["train"]["vis_every_images"] = 1
    cfg["train"]["vis_max_images"] = 1
    cfg["train"]["vis_conf_threshold"] = 0.25  # Filter low-confidence predictions
    cfg["train"]["vis_iou_threshold"] = 0.45
    
    # Override batch size to loop dataset multiple times to get N steps
    # With train_samples=1 and batch_size=1, we only get 1 batch per epoch
    # So create more epochs to get more steps
    num_epochs_needed = max(1, args.steps)
    cfg["train"]["epochs"] = num_epochs_needed
    
    # Add class names for better labeling
    cfg["model"]["class_names"] = ["person", "car", "dog"]

    print(f"\n📊 Config:")
    print(f"  - Image size: {cfg['model']['image_size']}")
    print(f"  - Steps: {cfg['train']['total_steps']}")
    print(f"  - Vis every: {cfg['train']['vis_every_images']} images")
    print(f"  - Model: ResNet34 backbone")
    print()

    # Train
    run_dir = train(
        cfg,
        resume=False,
        save_checkpoints=not args.no_checkpoints,
    )

    print(f"\n✅ Training complete!")
    print(f"📁 Output directory: {run_dir}")
    print(f"🖼️  Visuals saved to: {run_dir}/visuals/")
    print(f"📊 Log saved to: {run_dir}/train_log.csv")


if __name__ == "__main__":
    main()
