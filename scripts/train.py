from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from yolo_lab.config import load_config
from yolo_lab.engine import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or resume the custom YOLO-style detector.")
    parser.add_argument("--config", default="configs/coco_resnet34.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = train(cfg, resume=args.resume, max_steps=args.max_steps, max_hours=args.max_hours)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
