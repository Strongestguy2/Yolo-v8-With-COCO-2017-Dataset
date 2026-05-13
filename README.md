# Custom YOLO-Style Detector

Script-first PyTorch object detection project with no Ultralytics dependency. The default configuration trains a custom YOLO-style head with a TorchVision ResNet ImageNet backbone on staged COCO data.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Main Workflow

Use one command for the common path:

```powershell
.\.venv\Scripts\python.exe scripts\universal_train.py
```

If you want a very fast local smoke run with no dataset download, use:

```powershell
.\.venv\Scripts\python.exe scripts\universal_train.py --smoke --max-steps 1
```

The universal script will prepare COCO automatically when the dataset is missing, resumes by default if a checkpoint exists, and saves progress on Ctrl+C. It also accepts dotted `--set key=value` overrides for almost every training knob, including preview frequency.

## GUI Control Panel

The panel is a lightweight Gradio app that runs next to training instead of inside the hot training step. It can start or stop training, show the newest loss curve from `train_log.csv`, display the latest preview image from `visuals`, and clean project-local Python/test caches.

```powershell
.\.venv\Scripts\python.exe scripts\panel.py
```

It opens `http://127.0.0.1:7860` automatically. Use `Start / Resume` to continue an existing run, or `Start New Session` to create a fresh timestamped run folder. Use the preview controls to choose how often training writes visual checks; a larger interval keeps disk and UI overhead lower.

Modes:

```powershell
.\.venv\Scripts\python.exe scripts\universal_train.py --mode 1
.\.venv\Scripts\python.exe scripts\universal_train.py --mode 2
.\.venv\Scripts\python.exe scripts\universal_train.py --mode 3
```

Mode 1 is a non-saving smoke test for sharing logs. Mode 2 automatically resumes from the latest savepoint or active branch. Mode 3 opens a file picker so you can choose a checkpoint, branch from it, and keep the original run intact.

The lower-level commands still exist if you want them:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --config configs\coco_resnet34.yaml --download-increment 1000 --max-hours 2
.\.venv\Scripts\python.exe scripts\smoke_train.py --max-steps 1
.\.venv\Scripts\python.exe scripts\promote_branch.py --pick
```

## Inference And Demo

```powershell
.\.venv\Scripts\python.exe scripts\predict.py --config configs\coco_resnet34.yaml --weights outputs\runs\coco_resnet34\checkpoints\best.pt --source path\to\image.jpg --save
.\.venv\Scripts\python.exe scripts\demo.py --config configs\coco_resnet34.yaml --weights outputs\runs\coco_resnet34\checkpoints\best.pt
```

## Data Choice

Use staged local COCO data for the main training path. Hugging Face streaming is useful for very large datasets, but local staged data is simpler and safer for exact validation, shuffling, checkpoint resume, and week-long interrupted training.
