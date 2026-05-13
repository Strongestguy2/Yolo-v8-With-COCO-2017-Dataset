from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from yolo_lab.engine import train
from yolo_lab.smoke import build_smoke_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end training smoke test on synthetic data.")
    parser.add_argument("--base-config", default="configs/smoke.yaml")
    parser.add_argument("--data-root", default="data/smoke_local_yolo")
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()

    cfg = build_smoke_config(args.base_config, args.data_root)
    run_dir = train(cfg, resume=False, max_steps=args.max_steps)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()