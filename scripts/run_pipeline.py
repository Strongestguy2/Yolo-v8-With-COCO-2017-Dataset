from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from yolo_lab.config import load_config
from yolo_lab.engine import train
from yolo_lab.prepare import prepare_coco


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data then train/resume in one rerunnable command.")
    parser.add_argument("--config", default="configs/coco_resnet34.yaml")
    parser.add_argument("--download-increment", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_result = prepare_coco(cfg, download_increment=args.download_increment, force=args.force_download)
    print(f"Dataset: {data_result}")
    run_dir = train(cfg, resume=not args.no_resume, max_steps=args.max_steps, max_hours=args.max_hours)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
