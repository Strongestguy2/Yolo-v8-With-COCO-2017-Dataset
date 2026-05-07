from __future__ import annotations

import argparse
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from yolo_lab.checkpoint import load_training_checkpoint
from yolo_lab.config import load_config
from yolo_lab.data import build_dataloaders
from yolo_lab.engine import build_criterion
from yolo_lab.evaluate import evaluate_loss, evaluate_prediction_counts
from yolo_lab.model import build_model
from yolo_lab.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the validation split.")
    parser.add_argument("--config", default="configs/coco_resnet34.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(str(cfg.get("device", "auto")))
    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg, pretrained=False).to(device)
    load_training_checkpoint(args.weights, model, device=device)
    criterion = build_criterion(cfg).to(device)
    print(evaluate_loss(model, criterion, val_loader, device, max_batches=args.max_batches))
    print(evaluate_prediction_counts(model, val_loader, device, conf_threshold=args.conf, max_batches=args.max_batches))


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
