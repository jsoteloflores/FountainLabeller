#!/usr/bin/env python3
"""CLI wrapper around :func:`lava_labeler.core.video_relink.relink_workspace_videos`.

Relink an existing FountainLabeller workspace's ``metadata/frames.csv`` rows to
their source videos, filling missing ``video_id`` / ``video_path`` / ``fps``.

Default mode is a dry run. Pass ``--apply`` to write changes.

Examples
--------
    python scripts/relink_workspace_videos.py \
        --dataset "/path/to/workspace" \
        --video-root "/Volumes/Joel HDD/Kilauea_2024-2026_videos_renamed"

    python scripts/relink_workspace_videos.py \
        --dataset "/path/to/workspace" \
        --video-root "/Volumes/Joel HDD/Kilauea_2024-2026_videos_renamed" \
        --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a standalone script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lava_labeler.core.video_relink import relink_workspace_videos  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Relink FountainLabeller workspace videos.")
    p.add_argument("--dataset", required=True, type=Path,
                   help="Workspace root (contains metadata/frames.csv).")
    p.add_argument("--video-root", required=True, type=Path,
                   help="Folder to scan for source videos.")
    p.add_argument("--apply", action="store_true",
                   help="Actually write changes. Without this, runs as a dry run.")
    p.add_argument("--no-recursive", action="store_true",
                   help="Do not recurse into subdirectories when scanning videos.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()

    frames_csv = dataset / "metadata" / "frames.csv"
    if not frames_csv.is_file():
        print(f"ERROR: frames.csv not found at {frames_csv}", file=sys.stderr)
        return 2
    if not video_root.is_dir():
        print(f"ERROR: video root not found: {video_root}", file=sys.stderr)
        return 2

    result = relink_workspace_videos(
        dataset_root=dataset,
        source_video_root=video_root,
        dry_run=not args.apply,
        recursive=not args.no_recursive,
    )

    print("FountainLabeller workspace video relink")
    print(f"  dataset:        {dataset}")
    print(f"  video root:     {video_root}")
    print(f"  videos scanned: {result['videos_scanned']}")
    print(f"  mode:           {'APPLY' if args.apply else 'dry run'}")
    print()
    print(f"  rows checked:        {result['rows_checked']}")
    print(f"  rows already linked: {result['rows_already_linked']}")
    print(f"  rows relinkable:     {result['rows_relinkable']}")
    print(f"  linked existing path:{result['linked_existing_path']}")
    print(f"  linked by filename:  {result['linked_by_filename']}")
    print(f"  ambiguous:           {result['ambiguous']}")
    print(f"  missing:             {result['missing']}")
    print(f"  failed video reads:  {result['failed_to_read_video']}")
    print(f"  unchanged:           {result['unchanged']}")
    print()
    print(f"  report: {result['report_csv']}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write changes.")
    else:
        print("\nApplied. frames.csv and video_registry updated (backups written).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
