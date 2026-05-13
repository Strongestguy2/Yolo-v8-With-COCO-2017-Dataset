from __future__ import annotations

from pathlib import Path

from yolo_lab.engine import train
from yolo_lab.smoke import build_smoke_config, ensure_tiny_detection_dataset


def test_tiny_smoke_train_runs(tmp_path: Path) -> None:
    data_root = ensure_tiny_detection_dataset(tmp_path / "smoke_data", image_size=96)
    cfg = build_smoke_config(data_root=data_root)
    cfg["train"]["output_dir"] = str(tmp_path / "runs")
    cfg["train"]["run_name"] = "smoke_test"
    cfg["train"]["vis_every_images"] = 1
    cfg["train"]["vis_max_images"] = 1

    run_dir = train(cfg, resume=False, max_steps=1)

    assert run_dir == Path(tmp_path / "runs" / "smoke_test")
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "train_log.csv").exists()
    assert any(run_dir.joinpath("visuals").glob("*.png"))
    log_text = (run_dir / "train_log.csv").read_text(encoding="utf-8")
    assert "loss" in log_text