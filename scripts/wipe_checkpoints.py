"""
Wipe checkpoints based on time criteria.

Usage:
  python scripts/wipe_checkpoints.py --full                           # Wipe everything
  python scripts/wipe_checkpoints.py --minutes 10                     # Wipe last 10 minutes
  python scripts/wipe_checkpoints.py --hours 1                        # Wipe last 1 hour
  python scripts/wipe_checkpoints.py --days 1                         # Wipe last 1 day
  python scripts/wipe_checkpoints.py --since "2026-05-13 12:00:00"   # Wipe since specific time
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def get_checkpoint_dirs(root: str | Path = "outputs/runs") -> list[Path]:
    """Find all directories containing .pt checkpoint files."""
    root = Path(root)
    if not root.exists():
        return []
    
    # Find all checkpoints
    pt_files = list(root.rglob("*.pt"))
    
    # Get unique parent directories
    dirs_to_check = set(f.parent for f in pt_files)
    return sorted(dirs_to_check)


def get_dir_mtime(path: Path) -> datetime:
    """Get the modification time of a directory (most recent file inside)."""
    if not path.exists():
        return datetime.fromtimestamp(0)
    
    times = []
    for item in path.rglob("*"):
        if item.is_file():
            times.append(datetime.fromtimestamp(item.stat().st_mtime))
    
    return max(times) if times else datetime.fromtimestamp(path.stat().st_mtime)


def wipe_full(root: str | Path = "outputs/runs") -> int:
    """Delete all checkpoints."""
    root = Path(root)
    if not root.exists():
        print(f"Directory does not exist: {root}")
        return 0
    
    pt_files = list(root.rglob("*.pt"))
    count = 0
    
    for pt_file in pt_files:
        try:
            pt_file.unlink()
            print(f"Deleted: {pt_file}")
            count += 1
        except Exception as e:
            print(f"Failed to delete {pt_file}: {e}")
    
    return count


def wipe_since(since: datetime, root: str | Path = "outputs/runs") -> int:
    """Delete checkpoints modified after the given datetime."""
    root = Path(root)
    if not root.exists():
        print(f"Directory does not exist: {root}")
        return 0
    
    pt_files = list(root.rglob("*.pt"))
    count = 0
    
    for pt_file in pt_files:
        file_mtime = datetime.fromtimestamp(pt_file.stat().st_mtime)
        if file_mtime >= since:
            try:
                pt_file.unlink()
                print(f"Deleted: {pt_file} (modified: {file_mtime})")
                count += 1
            except Exception as e:
                print(f"Failed to delete {pt_file}: {e}")
    
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wipe checkpoints based on time criteria.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/wipe_checkpoints.py --full
  python scripts/wipe_checkpoints.py --minutes 10
  python scripts/wipe_checkpoints.py --hours 1
  python scripts/wipe_checkpoints.py --days 1
  python scripts/wipe_checkpoints.py --since "2026-05-13 12:00:00"
        """,
    )
    
    parser.add_argument("--full", action="store_true", help="Delete all checkpoints")
    parser.add_argument("--minutes", type=int, help="Delete checkpoints from the last N minutes")
    parser.add_argument("--hours", type=int, help="Delete checkpoints from the last N hours")
    parser.add_argument("--days", type=int, help="Delete checkpoints from the last N days")
    parser.add_argument("--since", type=str, help='Delete checkpoints since this time (format: "YYYY-MM-DD HH:MM:SS")')
    parser.add_argument("--root", type=str, default="outputs/runs", help="Root directory to search for checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    # Determine the cutoff time
    cutoff_time = None
    
    if args.full:
        print("🗑️  Wiping ALL checkpoints...")
        if args.dry_run:
            pt_files = list(Path(args.root).rglob("*.pt"))
            print(f"[DRY RUN] Would delete {len(pt_files)} checkpoint files:")
            for f in pt_files:
                print(f"  {f}")
        else:
            count = wipe_full(args.root)
            print(f"✅ Deleted {count} checkpoint files.")
    
    elif args.minutes is not None:
        cutoff_time = datetime.now() - timedelta(minutes=args.minutes)
        print(f"🗑️  Wiping checkpoints from the last {args.minutes} minute(s) (since {cutoff_time})...")
    
    elif args.hours is not None:
        cutoff_time = datetime.now() - timedelta(hours=args.hours)
        print(f"🗑️  Wiping checkpoints from the last {args.hours} hour(s) (since {cutoff_time})...")
    
    elif args.days is not None:
        cutoff_time = datetime.now() - timedelta(days=args.days)
        print(f"🗑️  Wiping checkpoints from the last {args.days} day(s) (since {cutoff_time})...")
    
    elif args.since is not None:
        try:
            cutoff_time = datetime.strptime(args.since, "%Y-%m-%d %H:%M:%S")
            print(f"🗑️  Wiping checkpoints since {cutoff_time}...")
        except ValueError:
            print(f"❌ Invalid time format: {args.since}")
            print("   Use format: YYYY-MM-DD HH:MM:SS")
            return
    
    else:
        parser.print_help()
        print("\n❌ Please specify one of: --full, --minutes, --hours, --days, or --since")
        return
    
    if cutoff_time is not None:
        if args.dry_run:
            root = Path(args.root)
            pt_files = [f for f in root.rglob("*.pt") if datetime.fromtimestamp(f.stat().st_mtime) >= cutoff_time]
            print(f"[DRY RUN] Would delete {len(pt_files)} checkpoint files:")
            for f in pt_files:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                print(f"  {f} (modified: {mtime})")
        else:
            count = wipe_since(cutoff_time, args.root)
            print(f"✅ Deleted {count} checkpoint files.")


if __name__ == "__main__":
    main()
