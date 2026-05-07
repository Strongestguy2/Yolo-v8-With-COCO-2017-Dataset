from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from yolo_lab.checkpoint import load_training_checkpoint
from yolo_lab.config import load_config
from yolo_lab.infer import annotate_image, decode_predictions, load_image_tensor, save_annotated
from yolo_lab.model import build_model
from yolo_lab.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("--config", default="configs/coco_resnet34.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(str(cfg.get("device", "auto")))
    model = build_model(cfg, pretrained=False).to(device).eval()
    load_training_checkpoint(args.weights, model, device=device)
    tensor, rgb, info = load_image_tensor(args.source, int(cfg["model"]["image_size"]), device)
    with torch.no_grad():
        preds = decode_predictions(model(tensor), conf_threshold=args.conf, iou_threshold=args.iou, image_size=int(cfg["model"]["image_size"]))[0]
    print(preds.detach().cpu())
    if args.save:
        out = save_annotated(Path(args.source), annotate_image(rgb, preds, info))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
