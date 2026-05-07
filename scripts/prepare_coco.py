from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from yolo_lab.config import load_config
from yolo_lab.prepare import prepare_coco


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a staged COCO YOLO-format dataset.")
    parser.add_argument("--config", default="configs/coco_resnet34.yaml")
    parser.add_argument("--download-increment", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = prepare_coco(cfg, download_increment=args.download_increment, force=args.force)
    print(result)


if __name__ == "__main__":
    main()
