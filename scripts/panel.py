from __future__ import annotations

import argparse
import csv
import html
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import add_src_to_path

add_src_to_path()

import gradio as gr
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "runs"
PANEL_DIR = ROOT / "outputs" / "panel"
STOP_FILE = PANEL_DIR / "stop_requested.txt"
LOG_FIELDS = [
    "time",
    "epoch",
    "batch",
    "global_step",
    "lr",
    "step_time",
    "loss",
    "loss_obj",
    "loss_cls",
    "loss_box",
    "num_pos",
    "num_gt",
]
MODE_TO_ID = {"Smoke test": 1, "Auto resume": 2, "Branch from checkpoint": 3}

PROCESS: subprocess.Popen[str] | None = None
PROCESS_LOG_HANDLE: Any | None = None
PROCESS_LOG_PATH: Path | None = None
PROCESS_CMD: list[str] = []
PREFERRED_RUN_LABEL: str | None = None
PROCESS_TARGET_STEPS: int | None = None


def relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def resolve_user_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def list_config_files() -> list[str]:
    return [relative(path) for path in sorted((ROOT / "configs").glob("*.yaml"))]


def marker_mtime(run_dir: Path) -> float:
    markers = [
        run_dir / "train_log.csv",
        run_dir / "config.yaml",
        run_dir / "checkpoints",
        run_dir / "visuals",
    ]
    times = [path.stat().st_mtime for path in markers if path.exists()]
    return max(times) if times else run_dir.stat().st_mtime


def list_run_dirs() -> list[str]:
    if not OUTPUT_ROOT.exists():
        return [PREFERRED_RUN_LABEL] if PREFERRED_RUN_LABEL else []

    candidates: dict[Path, float] = {}
    for pattern in ("train_log.csv", "config.yaml"):
        for marker in OUTPUT_ROOT.rglob(pattern):
            candidates[marker.parent] = max(candidates.get(marker.parent, 0.0), marker.stat().st_mtime)
    for marker_name in ("checkpoints", "visuals"):
        for marker in OUTPUT_ROOT.rglob(marker_name):
            if marker.is_dir():
                candidates[marker.parent] = max(candidates.get(marker.parent, 0.0), marker.stat().st_mtime)
    if PREFERRED_RUN_LABEL:
        preferred = resolve_user_path(PREFERRED_RUN_LABEL)
        candidates.setdefault(preferred, time.time())

    sorted_dirs = sorted(candidates, key=lambda path: candidates[path], reverse=True)
    return [relative(path) for path in sorted_dirs]


def selected_run_dir(run_label: str | None) -> Path | None:
    if not run_label:
        runs = list_run_dirs()
        return resolve_user_path(runs[0]) if runs else None
    path = resolve_user_path(run_label)
    if path.exists() or run_label == PREFERRED_RUN_LABEL:
        return path
    return None


def read_rows(run_dir: Path | None) -> list[dict[str, str]]:
    if run_dir is None:
        return []
    log_path = run_dir / "train_log.csv"
    if not log_path.exists():
        return []
    with log_path.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
        return list(csv.DictReader(handle))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def config_for_path(config_path: str | Path) -> dict[str, Any]:
    return read_yaml(resolve_user_path(config_path))


def run_config(run_dir: Path | None) -> dict[str, Any]:
    return read_yaml(run_dir / "config.yaml") if run_dir is not None else {}


def configured_total_steps(cfg: dict[str, Any]) -> int | None:
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    raw = train_cfg.get("total_steps") if isinstance(train_cfg, dict) else None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def sanitize_run_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return cleaned or "run"


def to_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def latest_preview(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    visual_dir = run_dir / "visuals"
    if not visual_dir.exists():
        return None
    images = sorted(visual_dir.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    return str(images[0]) if images else None


def empty_plot(message: str) -> Figure:
    fig = Figure(figsize=(7, 3), dpi=120)
    ax = fig.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def loss_plot(rows: list[dict[str, str]]) -> Figure:
    if not rows:
        return empty_plot("No loss data yet")

    fig = Figure(figsize=(7, 3), dpi=120)
    ax = fig.subplots()
    x_values: list[float] = []
    for index, row in enumerate(rows):
        x_values.append(to_float(row.get("global_step")) or float(index + 1))

    for field, linewidth in (("loss", 2.0), ("loss_box", 1.1), ("loss_obj", 1.1), ("loss_cls", 1.1)):
        y_values = [to_float(row.get(field)) for row in rows]
        points = [(x, y) for x, y in zip(x_values, y_values) if y is not None]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, label=field, linewidth=linewidth)

    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def tail_table(rows: list[dict[str, str]], limit: int = 12) -> list[list[str]]:
    return [[row.get(field, "") for field in LOG_FIELDS] for row in rows[-limit:]]


def process_state() -> str:
    global PROCESS_LOG_HANDLE
    if PROCESS is None:
        return "idle"
    exit_code = PROCESS.poll()
    if exit_code is None:
        return f"running pid={PROCESS.pid}"
    if PROCESS_LOG_HANDLE is not None:
        PROCESS_LOG_HANDLE.close()
        PROCESS_LOG_HANDLE = None
    return f"last run exited with code {exit_code}"


def read_process_log(limit: int = 80) -> str:
    path = PROCESS_LOG_PATH
    if path is None or not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-limit:])


def summarize_run(run_dir: Path | None, rows: list[dict[str, str]]) -> str:
    state = process_state()
    lines = [f"Training: {state}"]
    if PROCESS_LOG_PATH is not None:
        lines.append(f"Process log: {relative(PROCESS_LOG_PATH)}")
    if run_dir is None:
        lines.append("Run: none")
        return "\n\n".join(lines)

    lines.append(f"Run: {relative(run_dir)}")
    if not run_dir.exists():
        lines.append("Status: waiting for training to create this folder")
        return "\n\n".join(lines)
    if rows:
        latest = rows[-1]
        loss = latest.get("loss", "?")
        step = latest.get("global_step", "?")
        epoch = latest.get("epoch", "?")
        lr = latest.get("lr", "?")
        lines.append(f"Latest: epoch {epoch}, step {step}, loss {loss}, lr {lr}")
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        total_mb = sum(path.stat().st_size for path in checkpoints) / (1024 * 1024)
        lines.append(f"Checkpoints: {len(checkpoints)} files, {total_mb:.1f} MB")
    preview = latest_preview(run_dir)
    if preview:
        lines.append(f"Preview: {relative(preview)}")
    return "\n\n".join(lines)


def progress_html(run_dir: Path | None, rows: list[dict[str, str]]) -> str:
    latest_step = 0
    if rows:
        latest_step = int(to_float(rows[-1].get("global_step")) or 0)

    total_steps = configured_total_steps(run_config(run_dir))
    if PROCESS is not None and PROCESS.poll() is None and PREFERRED_RUN_LABEL and run_dir is not None:
        if relative(run_dir) == PREFERRED_RUN_LABEL and PROCESS_TARGET_STEPS:
            total_steps = PROCESS_TARGET_STEPS

    escaped_run = html.escape(relative(run_dir)) if run_dir is not None else "none"
    if total_steps:
        percent = min(100.0, max(0.0, latest_step / total_steps * 100.0))
        return (
            '<div style="padding:10px 12px;border:1px solid #d7d7d7;border-radius:6px;">'
            f'<div style="font-weight:600;margin-bottom:6px;">Progress: {percent:.1f}%</div>'
            f'<progress value="{latest_step}" max="{total_steps}" style="width:100%;height:18px;"></progress>'
            f'<div style="margin-top:6px;color:#555;">step {latest_step} / {total_steps} · {escaped_run}</div>'
            "</div>"
        )
    return (
        '<div style="padding:10px 12px;border:1px solid #d7d7d7;border-radius:6px;">'
        f'<div style="font-weight:600;">Progress: step {latest_step}</div>'
        f'<div style="margin-top:6px;color:#555;">No total step target found · {escaped_run}</div>'
        "</div>"
    )


def refresh_dashboard(run_label: str | None = None) -> tuple[Any, str, str, Figure, str | None, list[list[str]], str]:
    runs = list_run_dirs()
    training_running = PROCESS is not None and PROCESS.poll() is None
    if training_running and PREFERRED_RUN_LABEL:
        selected = PREFERRED_RUN_LABEL
    else:
        selected = run_label if run_label in runs else (runs[0] if runs else None)
    run_dir = selected_run_dir(selected)
    rows = read_rows(run_dir)
    return (
        gr.update(choices=runs, value=selected),
        summarize_run(run_dir, rows),
        progress_html(run_dir, rows),
        loss_plot(rows),
        latest_preview(run_dir),
        tail_table(rows),
        read_process_log(),
    )


def split_overrides(raw: str | None) -> list[str]:
    if not raw:
        return []
    overrides: list[str] = []
    for line in raw.replace(";", "\n").splitlines():
        value = line.strip()
        if value:
            overrides.append(value)
    return overrides


def append_override(cmd: list[str], key: str, value: Any) -> None:
    cmd.extend(["--set", f"{key}={value}"])


def start_training(
    mode: str,
    config_path: str,
    resume: bool,
    max_steps: float | None,
    max_hours: float | None,
    vis_every_images: float,
    vis_max_images: float,
    checkpoint_path: str,
    branch_name: str,
    overrides_text: str,
    new_session: bool = False,
) -> str:
    global PROCESS, PROCESS_CMD, PROCESS_LOG_HANDLE, PROCESS_LOG_PATH, PREFERRED_RUN_LABEL, PROCESS_TARGET_STEPS

    if PROCESS is not None and PROCESS.poll() is None:
        return f"Training is already running with pid={PROCESS.pid}."

    mode_id = MODE_TO_ID[mode]
    if mode_id == 3 and not checkpoint_path.strip():
        return "Branch mode needs a checkpoint path."

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    if STOP_FILE.exists():
        STOP_FILE.unlink()
    if PROCESS_LOG_HANDLE is not None:
        PROCESS_LOG_HANDLE.close()
        PROCESS_LOG_HANDLE = None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    PROCESS_LOG_PATH = PANEL_DIR / f"train_{timestamp}.log"
    cfg = config_for_path(config_path)
    base_train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    base_run_name = sanitize_run_name(str(base_train_cfg.get("run_name", "custom_yolo")))
    output_dir = str(base_train_cfg.get("output_dir", "outputs/runs"))
    run_name_override: str | None = None
    expected_run_dir: Path | None = None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "universal_train.py"),
        "--mode",
        str(mode_id),
        "--config",
        str(resolve_user_path(config_path)),
    ]

    if mode_id == 1:
        run_name_override = f"smoke_{timestamp}"
        output_dir = "outputs/runs/smoke_reports"
        expected_run_dir = resolve_user_path(output_dir) / run_name_override
    elif new_session and mode_id == 2:
        run_name_override = f"{base_run_name}_{timestamp}"
        expected_run_dir = resolve_user_path(output_dir) / run_name_override
        cmd.append("--new-session")
        resume = False

    if not resume:
        cmd.append("--no-resume")
    if max_steps is not None and max_steps > 0:
        cmd.extend(["--max-steps", str(int(max_steps))])
    if max_hours is not None and max_hours > 0:
        cmd.extend(["--max-hours", str(float(max_hours))])
    if checkpoint_path.strip():
        cmd.extend(["--checkpoint", str(resolve_user_path(checkpoint_path.strip()))])
    if branch_name.strip():
        cmd.extend(["--branch-name", branch_name.strip()])

    if run_name_override is not None:
        append_override(cmd, "train.output_dir", output_dir)
        append_override(cmd, "train.run_name", run_name_override)
    append_override(cmd, "train.vis_every_images", max(0, int(vis_every_images)))
    append_override(cmd, "train.vis_max_images", max(1, int(vis_max_images)))
    append_override(cmd, "train.stop_file", STOP_FILE)
    for override in split_overrides(overrides_text):
        cmd.extend(["--set", override])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)

    PROCESS_LOG_HANDLE = PROCESS_LOG_PATH.open("a", encoding="utf-8", buffering=1)
    PROCESS_LOG_HANDLE.write(" ".join(f'"{part}"' if " " in part else part for part in cmd) + "\n\n")
    PROCESS = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=PROCESS_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )
    PROCESS_CMD = cmd
    PREFERRED_RUN_LABEL = relative(expected_run_dir) if expected_run_dir is not None else None
    PROCESS_TARGET_STEPS = int(max_steps) if max_steps is not None and max_steps > 0 else configured_total_steps(cfg)
    lines = [f"Started training with pid={PROCESS.pid}.", f"Log: {relative(PROCESS_LOG_PATH)}"]
    if PREFERRED_RUN_LABEL:
        lines.append(f"Expected run: {PREFERRED_RUN_LABEL}")
    return "\n".join(lines)


def start_new_training(
    mode: str,
    config_path: str,
    resume: bool,
    max_steps: float | None,
    max_hours: float | None,
    vis_every_images: float,
    vis_max_images: float,
    checkpoint_path: str,
    branch_name: str,
    overrides_text: str,
) -> str:
    return start_training(
        mode,
        config_path,
        False,
        max_steps,
        max_hours,
        vis_every_images,
        vis_max_images,
        checkpoint_path,
        branch_name,
        overrides_text,
        new_session=True,
    )


def request_stop() -> str:
    if PROCESS is None or PROCESS.poll() is not None:
        return "No running training process."
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text(f"stop requested at {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    return "Stop requested. Training will save safe_stop.pt at the next batch boundary."


def cache_dirs() -> list[Path]:
    roots = [ROOT / ".pytest_cache", ROOT / ".ruff_cache", ROOT / ".mypy_cache"]
    roots.extend(path for path in ROOT.rglob("__pycache__") if ".venv" not in path.parts and ".git" not in path.parts)
    return sorted({path for path in roots if path.exists()})


def clean_project_caches() -> str:
    deleted: list[str] = []
    errors: list[str] = []
    for path in cache_dirs():
        try:
            shutil.rmtree(path)
            deleted.append(relative(path))
        except OSError as exc:
            errors.append(f"{relative(path)}: {exc}")
    if not deleted and not errors:
        return "No project cache directories found."
    message = [f"Deleted {len(deleted)} project cache directories."]
    if deleted:
        message.append("\n".join(deleted[:20]))
    if errors:
        message.append("Errors:\n" + "\n".join(errors))
    return "\n\n".join(message)


def smoke_run_dirs() -> list[Path]:
    runs: list[Path] = []
    smoke_reports = OUTPUT_ROOT / "smoke_reports"
    if smoke_reports.exists():
        runs.extend(path for path in smoke_reports.glob("smoke_*") if path.is_dir())
    for name in ("smoke_local", "smoke_test"):
        candidate = OUTPUT_ROOT / name
        if candidate.exists() and candidate.is_dir():
            runs.append(candidate)
    return sorted(set(runs))


def clear_smoke_runs(confirmed: bool = False) -> str:
    if not confirmed:
        return "Delete cancelled."
    time.sleep(2)
    deleted: list[str] = []
    errors: list[str] = []
    for run_dir in smoke_run_dirs():
        try:
            shutil.rmtree(run_dir)
            deleted.append(relative(run_dir))
        except OSError as exc:
            errors.append(f"{relative(run_dir)}: {exc}")

    if not deleted and not errors:
        return "No smoke runs found to delete."

    message = [f"Deleted {len(deleted)} smoke run directories."]
    if deleted:
        message.append("\n".join(deleted[:50]))
    if errors:
        message.append("Errors:\n" + "\n".join(errors))
    return "\n\n".join(message)


def actual_run_dirs() -> list[Path]:
    if not OUTPUT_ROOT.exists():
        return []
    excluded = {"smoke_reports", "smoke_local", "smoke_test"}
    runs: list[Path] = []
    for path in OUTPUT_ROOT.iterdir():
        if not path.is_dir():
            continue
        if path.name in excluded or path.name.startswith("smoke_"):
            continue
        runs.append(path)
    return sorted(runs)


def clear_actual_runs(confirmed: bool = False) -> str:
    global PREFERRED_RUN_LABEL
    if not confirmed:
        return "Delete cancelled."
    if PROCESS is not None and PROCESS.poll() is None:
        return "Stop training before deleting actual runs."
    time.sleep(2)

    deleted: list[str] = []
    errors: list[str] = []
    for run_dir in actual_run_dirs():
        try:
            shutil.rmtree(run_dir)
            deleted.append(relative(run_dir))
        except OSError as exc:
            errors.append(f"{relative(run_dir)}: {exc}")

    pointer = OUTPUT_ROOT / "current_run.json"
    if pointer.exists():
        try:
            pointer.unlink()
            deleted.append(relative(pointer))
        except OSError as exc:
            errors.append(f"{relative(pointer)}: {exc}")
    PREFERRED_RUN_LABEL = None

    if not deleted and not errors:
        return "No actual training runs found to delete."

    message = [f"Deleted {len(deleted)} actual training run/history items."]
    if deleted:
        message.append("\n".join(deleted[:50]))
    if errors:
        message.append("Errors:\n" + "\n".join(errors))
    return "\n\n".join(message)


def build_app() -> gr.Blocks:
    configs = list_config_files()
    default_config = "configs/coco_resnet34.yaml" if "configs/coco_resnet34.yaml" in configs else (configs[0] if configs else "")
    runs = list_run_dirs()
    default_run = runs[0] if runs else None

    with gr.Blocks(title="YOLO Lab Panel") as app:
        gr.Markdown("# YOLO Lab Panel")
        delete_confirmed = gr.Checkbox(value=False, visible=False)
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                mode = gr.Radio(list(MODE_TO_ID), value="Auto resume", label="Mode")
                config = gr.Dropdown(configs, value=default_config, label="Config")
                resume = gr.Checkbox(value=True, label="Resume")
                with gr.Row():
                    max_steps = gr.Number(value=0, precision=0, label="Max steps")
                    max_hours = gr.Number(value=0, label="Max hours")
                with gr.Row():
                    vis_every = gr.Number(value=100, precision=0, label="Preview every images")
                    vis_max = gr.Number(value=2, precision=0, label="Preview count")
                checkpoint = gr.Textbox(label="Branch checkpoint path", placeholder="outputs/runs/.../checkpoints/last.pt")
                branch = gr.Textbox(label="Branch name")
                overrides = gr.Textbox(label="Extra overrides", lines=5, placeholder="train.lr=0.0005\ntrain.batch_size=4")
                with gr.Row():
                    start = gr.Button("Start / Resume")
                    start_new = gr.Button("Start New Session", variant="primary")
                    stop = gr.Button("Stop")
                with gr.Row():
                    clean = gr.Button("Clean Project Caches")
                    clear_smoke = gr.Button("Clear Smoke Runs", variant="stop")
                clear_actual = gr.Button("Clear Actual Training Runs", variant="stop")
                action_status = gr.Textbox(label="Action status", lines=5, interactive=False)
            with gr.Column(scale=2, min_width=520):
                run_select = gr.Dropdown(runs, value=default_run, label="Run")
                status = gr.Markdown()
                progress = gr.HTML(label="Progress")
                loss = gr.Plot(label="Loss")
                preview = gr.Image(type="filepath", label="Latest preview")
                history = gr.Dataframe(headers=LOG_FIELDS, label="Recent training rows", interactive=False)
                process_log = gr.Textbox(label="Process log tail", lines=12, interactive=False)

        outputs = [run_select, status, progress, loss, preview, history, process_log]
        refresh_inputs = [run_select]
        app.load(refresh_dashboard, inputs=refresh_inputs, outputs=outputs)
        timer = gr.Timer(value=5)
        timer.tick(refresh_dashboard, inputs=refresh_inputs, outputs=outputs)
        run_select.change(refresh_dashboard, inputs=refresh_inputs, outputs=outputs)
        start.click(
            start_training,
            inputs=[mode, config, resume, max_steps, max_hours, vis_every, vis_max, checkpoint, branch, overrides],
            outputs=[action_status],
        ).then(refresh_dashboard, outputs=outputs)
        start_new.click(
            start_new_training,
            inputs=[mode, config, resume, max_steps, max_hours, vis_every, vis_max, checkpoint, branch, overrides],
            outputs=[action_status],
        ).then(refresh_dashboard, outputs=outputs)
        stop.click(request_stop, outputs=[action_status]).then(refresh_dashboard, inputs=refresh_inputs, outputs=outputs)
        clean.click(clean_project_caches, outputs=[action_status])
        clear_smoke.click(
            clear_smoke_runs,
            inputs=[delete_confirmed],
            outputs=[action_status],
            js="() => confirm('Delete all smoke runs? Deletion starts after a 2 second pause.')",
        ).then(
            refresh_dashboard,
            inputs=refresh_inputs,
            outputs=outputs,
        )
        clear_actual.click(
            clear_actual_runs,
            inputs=[delete_confirmed],
            outputs=[action_status],
            js="() => confirm('Delete all actual training runs, logs, and checkpoints? Deletion starts after a 2 second pause.')",
        ).then(
            refresh_dashboard,
            inputs=refresh_inputs,
            outputs=outputs,
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the YOLO Lab training control panel.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", dest="inbrowser", action="store_true", default=True)
    parser.add_argument("--no-browser", dest="inbrowser", action="store_false")
    args = parser.parse_args()

    app = build_app()
    app.queue().launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=args.inbrowser)


if __name__ == "__main__":
    main()
