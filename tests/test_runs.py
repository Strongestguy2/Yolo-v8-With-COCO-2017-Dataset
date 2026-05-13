from __future__ import annotations

from pathlib import Path

from yolo_lab.config import load_config
from yolo_lab.runs import (
    create_branch_from_checkpoint,
    current_run_pointer_path,
    promote_branch,
    resolve_mode2_config,
    run_dir_from_cfg,
)


def write_fake_checkpoint(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(marker, encoding="utf-8")


def freeze_current_run_pointer() -> str | None:
    pointer = current_run_pointer_path()
    if not pointer.exists():
        return None
    backup = pointer.read_text(encoding="utf-8")
    pointer.unlink()
    return backup


def restore_current_run_pointer(backup: str | None) -> None:
    pointer = current_run_pointer_path()
    if pointer.exists():
        pointer.unlink()
    if backup is not None:
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(backup, encoding="utf-8")


def test_create_branch_and_resolve_mode2(tmp_path: Path) -> None:
    backup = freeze_current_run_pointer()
    cfg = load_config("configs/smoke.yaml")
    cfg["train"]["output_dir"] = str(tmp_path / "outputs" / "runs")
    cfg["train"]["run_name"] = "main_run"

    try:
        main_run_dir = run_dir_from_cfg(cfg)
        main_checkpoint = main_run_dir / "checkpoints" / "last.pt"
        write_fake_checkpoint(main_checkpoint, "main")
        write_fake_checkpoint(main_run_dir / "checkpoints" / "best.pt", "main-best")

        branch_cfg, branch_run_dir, resolved_main = create_branch_from_checkpoint(cfg, main_checkpoint, branch_name="branch_a")

        assert resolved_main == main_run_dir
        assert branch_run_dir.exists()
        assert (branch_run_dir / "checkpoints" / "last.pt").read_text(encoding="utf-8") == "main"
        assert (main_run_dir / "active_branch.json").exists()

        next_cfg = resolve_mode2_config(cfg)
        assert next_cfg["train"]["output_dir"] == str(branch_run_dir.parent)
        assert next_cfg["train"]["run_name"] == branch_run_dir.name
        assert branch_cfg["train"]["output_dir"] == str(branch_run_dir.parent)
    finally:
        restore_current_run_pointer(backup)


def test_promote_branch_copies_back_to_main(tmp_path: Path) -> None:
    backup = freeze_current_run_pointer()
    cfg = load_config("configs/smoke.yaml")
    cfg["train"]["output_dir"] = str(tmp_path / "outputs" / "runs")
    cfg["train"]["run_name"] = "main_run"

    try:
        main_run_dir = run_dir_from_cfg(cfg)
        main_checkpoint = main_run_dir / "checkpoints" / "last.pt"
        write_fake_checkpoint(main_checkpoint, "main")
        write_fake_checkpoint(main_run_dir / "checkpoints" / "best.pt", "main-best")

        _, branch_run_dir, _ = create_branch_from_checkpoint(cfg, main_checkpoint, branch_name="branch_b")
        branch_checkpoint = branch_run_dir / "checkpoints" / "last.pt"
        write_fake_checkpoint(branch_checkpoint, "branch")
        write_fake_checkpoint(branch_run_dir / "checkpoints" / "best.pt", "branch-best")

        promoted_main, promoted_branch = promote_branch(branch_checkpoint)

        assert promoted_main == main_run_dir
        assert promoted_branch == branch_run_dir
        assert (main_run_dir / "checkpoints" / "last.pt").read_text(encoding="utf-8") == "branch"
        assert (main_run_dir / "checkpoints" / "best.pt").read_text(encoding="utf-8") == "branch-best"
        assert not (main_run_dir / "active_branch.json").exists()
    finally:
        restore_current_run_pointer(backup)