from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .constants import COCO_CLASSES
from .data import count_images, validate_yolo_dataset, write_dataset_yaml
from .utils import atomic_json_dump, json_load


def prepare_coco(cfg: dict[str, Any], download_increment: int | None = None, force: bool = False) -> dict[str, Any]:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    state_path = Path(data_cfg.get("state_path", "data/state.json"))
    num_classes = int(cfg["model"]["num_classes"])
    target_train = int(data_cfg.get("train_samples", 1000))
    target_val = int(data_cfg.get("val_samples", 200))
    state = json_load(state_path, {})
    current_target = int(state.get("prepared_train_samples", count_images(root, "train")))
    if download_increment is not None:
        target_train = min(target_train, current_target + int(download_increment))

    if not force and current_target >= target_train and count_images(root, "val") >= target_val:
        write_dataset_yaml(root, num_classes)
        validation = validate_yolo_dataset(root, num_classes)
        if not validation["errors"]:
            return {"root": str(root), "prepared_train_samples": current_target, "validation": validation, "changed": False}

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise RuntimeError("FiftyOne is required for COCO preparation. Install requirements.txt first.") from exc

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    seed = int(data_cfg.get("seed", cfg.get("seed", 42)))
    _export_split(fo, foz, root, split="train", export_split="train", max_samples=target_train, seed=seed)
    _export_split(fo, foz, root, split="validation", export_split="val", max_samples=target_val, seed=seed)
    write_dataset_yaml(root, num_classes)
    validation = validate_yolo_dataset(root, num_classes)
    if validation["errors"]:
        raise RuntimeError(f"dataset validation failed: {validation['errors'][:5]}")

    new_state = {
        "dataset": "coco-2017",
        "prepared_train_samples": target_train,
        "prepared_val_samples": target_val,
        "root": str(root.resolve()),
    }
    atomic_json_dump(new_state, state_path)
    return {"root": str(root), "prepared_train_samples": target_train, "validation": validation, "changed": True}


def _export_split(fo: Any, foz: Any, root: Path, split: str, export_split: str, max_samples: int, seed: int) -> None:
    dataset_name = f"yolo_lab_coco_{export_split}_{max_samples}_{seed}"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],
        max_samples=max_samples,
        shuffle=True,
        seed=seed,
        dataset_name=dataset_name,
    )
    try:
        dataset.export(
            export_dir=str(root),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split=export_split,
            classes=COCO_CLASSES,
        )
    finally:
        fo.delete_dataset(dataset.name)
