from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from yolo_lab.checkpoint import load_training_checkpoint
from yolo_lab.config import load_config
from yolo_lab.data import IMAGENET_MEAN, IMAGENET_STD, letterbox_image
from yolo_lab.infer import annotate_image, decode_predictions
from yolo_lab.model import build_model
from yolo_lab.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Gradio upload-image demo.")
    parser.add_argument("--config", default="configs/coco_resnet34.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is required for the demo. Install requirements.txt first.") from exc

    cfg = load_config(args.config)
    device = get_device(str(cfg.get("device", "auto")))
    image_size = int(cfg["model"]["image_size"])
    model = build_model(cfg, pretrained=False).to(device).eval()
    load_training_checkpoint(args.weights, model, device=device)

    def predict(image: np.ndarray) -> np.ndarray:
        if image is None:
            return np.zeros((image_size, image_size, 3), dtype=np.uint8)
        rgb = image.astype(np.uint8)
        boxed, info = letterbox_image(rgb, image_size)
        tensor = torch.from_numpy(np.ascontiguousarray(boxed)).permute(2, 0, 1).float() / 255.0
        tensor = ((tensor - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = decode_predictions(model(tensor), conf_threshold=0.25, image_size=image_size)[0]
        return annotate_image(rgb, preds, info)

    app = gr.Interface(fn=predict, inputs=gr.Image(type="numpy", label="Upload image"), outputs=gr.Image(type="numpy"))
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
