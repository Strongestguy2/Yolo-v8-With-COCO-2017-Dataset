from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from yolo_lab.runs import promote_branch, select_branch_checkpoint_via_dialog


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a branch checkpoint back into the main run.")
    parser.add_argument("--checkpoint", default=None, help="Branch checkpoint to promote.")
    parser.add_argument("--pick", action="store_true", help="Open a file dialog to choose the branch checkpoint.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if args.pick or checkpoint is None:
        checkpoint = select_branch_checkpoint_via_dialog(Path.cwd() / "outputs" / "runs")
    if checkpoint is None:
        raise RuntimeError("no checkpoint selected")

    main_run_dir, branch_run_dir = promote_branch(checkpoint)
    print(f"Promoted branch: {branch_run_dir}")
    print(f"Main run updated: {main_run_dir}")


if __name__ == "__main__":
    main()