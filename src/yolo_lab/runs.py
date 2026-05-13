from __future__ import annotations

import copy
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from .config import load_config, save_config
from .utils import atomic_json_dump, json_load


def sanitize_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return cleaned or "branch"


def run_dir_from_cfg(cfg: dict[str, Any]) -> Path:
    return Path(cfg["train"].get("output_dir", "outputs/runs")) / str(cfg["train"].get("run_name", "custom_yolo"))


def active_branch_path(main_run_dir: str | Path) -> Path:
    return Path(main_run_dir) / "active_branch.json"


def current_run_pointer_path() -> Path:
    return Path("outputs") / "runs" / "current_run.json"


def branch_metadata(branch_run_dir: str | Path, source_checkpoint: str | Path, main_run_dir: str | Path) -> dict[str, str]:
    return {
        "branch_run_dir": str(Path(branch_run_dir).resolve()),
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "main_run_dir": str(Path(main_run_dir).resolve()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save_active_branch(main_run_dir: str | Path, metadata: dict[str, str]) -> Path:
    path = active_branch_path(main_run_dir)
    atomic_json_dump(metadata, path)
    return path


def load_active_branch(main_run_dir: str | Path) -> dict[str, str] | None:
    path = active_branch_path(main_run_dir)
    if not path.exists():
        return None
    data = json_load(path, {})
    return data if isinstance(data, dict) and data.get("branch_run_dir") else None


def clear_active_branch(main_run_dir: str | Path) -> None:
    path = active_branch_path(main_run_dir)
    if path.exists():
        path.unlink()


def save_current_run_config(config_path: str | Path) -> Path:
    path = current_run_pointer_path()
    payload = {"config_path": str(Path(config_path).resolve())}
    atomic_json_dump(payload, path)
    return path


def load_current_run_config() -> Path | None:
    path = current_run_pointer_path()
    if not path.exists():
        return None
    data = json_load(path, {})
    config_path = data.get("config_path") if isinstance(data, dict) else None
    return Path(config_path) if config_path else None


def copy_checkpoint(src: str | Path, dst: str | Path) -> Path:
    source = Path(src)
    destination = Path(dst)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def create_branch_from_checkpoint(
    base_cfg: dict[str, Any],
    checkpoint_path: str | Path,
    branch_name: str | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    try:
        ckpt_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        ckpt_payload = {}

    main_run_dir = checkpoint.parent.parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = sanitize_name(branch_name or checkpoint.stem)
    branch_run_dir = main_run_dir / "branches" / f"{stem}_{timestamp}"
    branch_checkpoint_dir = branch_run_dir / "checkpoints"
    branch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    copy_checkpoint(checkpoint, branch_checkpoint_dir / "last.pt")
    if checkpoint.name != "best.pt":
        source_best = checkpoint.parent / "best.pt"
        if source_best.exists():
            copy_checkpoint(source_best, branch_checkpoint_dir / "best.pt")

    cfg_source = ckpt_payload.get("cfg") if isinstance(ckpt_payload, dict) else None
    cfg = copy.deepcopy(cfg_source if isinstance(cfg_source, dict) else base_cfg)
    cfg["train"]["output_dir"] = str(branch_run_dir.parent)
    cfg["train"]["run_name"] = branch_run_dir.name
    cfg["train"]["save_checkpoints"] = True
    cfg["_branch"] = {
        "main_run_dir": str(main_run_dir.resolve()),
        "branch_run_dir": str(branch_run_dir.resolve()),
        "source_checkpoint": str(checkpoint.resolve()),
    }
    save_config(cfg, branch_run_dir / "config.yaml")
    save_active_branch(main_run_dir, branch_metadata(branch_run_dir, checkpoint, main_run_dir))
    return cfg, branch_run_dir, main_run_dir


def resolve_mode2_config(cfg: dict[str, Any]) -> dict[str, Any]:
    current_config = load_current_run_config()
    if current_config is not None and current_config.exists():
        return load_config(current_config)

    main_run_dir = run_dir_from_cfg(cfg)
    branch = load_active_branch(main_run_dir)
    if not branch:
        return cfg
    branch_run_dir = Path(branch["branch_run_dir"])
    branch_cfg_path = branch_run_dir / "config.yaml"
    if branch_cfg_path.exists():
        return load_config(branch_cfg_path)
    derived = dict(cfg)
    derived["train"] = dict(cfg.get("train", {}))
    derived["train"]["output_dir"] = str(branch_run_dir.parent)
    derived["train"]["run_name"] = branch_run_dir.name
    return derived


def select_checkpoint_via_dialog(initial_dir: str | Path | None = None) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            title="Select checkpoint to branch from",
            initialdir=str(initial_dir or Path.cwd()),
            filetypes=[("PyTorch checkpoints", "*.pt"), ("All files", "*.*")],
        )
        return Path(selected) if selected else None
    finally:
        root.destroy()


def select_branch_checkpoint_via_dialog(initial_dir: str | Path | None = None) -> Path | None:
    return select_checkpoint_via_dialog(initial_dir)


def promote_branch(branch_checkpoint: str | Path) -> tuple[Path, Path]:
    checkpoint = Path(branch_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    branch_run_dir = checkpoint.parent.parent
    main_run_dir = branch_run_dir.parent.parent
    main_checkpoint_dir = main_run_dir / "checkpoints"
    main_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    copy_checkpoint(checkpoint, main_checkpoint_dir / "last.pt")
    branch_best = branch_run_dir / "checkpoints" / "best.pt"
    if branch_best.exists():
        copy_checkpoint(branch_best, main_checkpoint_dir / "best.pt")

    branch_config = branch_run_dir / "config.yaml"
    if branch_config.exists():
        shutil.copy2(branch_config, main_run_dir / "config.yaml")

    clear_active_branch(main_run_dir)
    save_current_run_config(main_run_dir / "config.yaml")
    return main_run_dir, branch_run_dir