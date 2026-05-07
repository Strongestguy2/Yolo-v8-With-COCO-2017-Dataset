# Custom YOLO-Style Detector

Script-first PyTorch object detection project with no Ultralytics dependency. The default configuration trains a custom YOLO-style head with a TorchVision ResNet ImageNet backbone on staged COCO data.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Main Workflow

Run this repeatedly through the week. It increases the local COCO subset, validates labels, resumes training, and checkpoints progress.

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --config configs\coco_resnet34.yaml --download-increment 1000 --max-hours 2
```

Fast smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --config configs\smoke.yaml --download-increment 20 --max-steps 5
.\.venv\Scripts\python.exe scripts\train.py --config configs\smoke.yaml --resume --max-steps 2
```

## Inference And Demo

```powershell
.\.venv\Scripts\python.exe scripts\predict.py --config configs\coco_resnet34.yaml --weights outputs\runs\coco_resnet34\checkpoints\best.pt --source path\to\image.jpg --save
.\.venv\Scripts\python.exe scripts\demo.py --config configs\coco_resnet34.yaml --weights outputs\runs\coco_resnet34\checkpoints\best.pt
```

## Data Choice

Use staged local COCO data for the main training path. Hugging Face streaming is useful for very large datasets, but local staged data is simpler and safer for exact validation, shuffling, checkpoint resume, and week-long interrupted training.
