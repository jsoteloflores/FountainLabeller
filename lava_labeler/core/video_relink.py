"""Relink old workspace frame rows back to their source videos.

Fills missing ``video_id``, repaired ``video_path``, and ``fps`` on existing
``metadata/frames.csv`` rows by matching them against a folder of source
videos. Never touches labels, masks, ROI coordinates, frame indices, or any
metadata flags — this is metadata *repair* only.

The GUI ``Tools → Relink Source Videos…`` action and the
``scripts/relink_workspace_videos.py`` CLI both call
:func:`relink_workspace_videos`.
"""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from lava_labeler.core.config import atomic_write_json, atomic_write_text
from lava_labeler.core.metadata import MetadataStore
from lava_labeler.core.video_io import VideoReader
from lava_labeler.core.video_registry import VideoRegistry


DEFAULT_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV",
)

REPORT_COLUMNS = [
    "sample_id",
    "old_video_id", "new_video_id",
    "old_video_path", "new_video_path",
    "video_filename",
    "frame_index",
    "status",
    "match_method",
    "registry_match_tier",
    "fps",
    "source_width", "source_height",
    "message",
]

# Row statuses.
LINKED_EXISTING_PATH = "linked_existing_path"
LINKED_BY_FILENAME = "linked_by_filename"
AMBIGUOUS = "ambiguous"
MISSING = "missing"
FAILED_TO_READ = "failed_to_read_video"
SKIPPED_NO_FILENAME = "skipped_no_filename"
UNCHANGED = "unchanged"


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------

def scan_video_files(
    root: Path,
    extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS,
    recursive: bool = True,
) -> list[Path]:
    """Recursively scan *root* for video files with the given *extensions*."""
    root = Path(root)
    exts = {e.lower() for e in extensions}
    found: list[Path] = []
    if not root.exists():
        return found
    if recursive:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if Path(name).suffix.lower() in exts:
                    found.append(Path(dirpath) / name)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                found.append(p)
    return found


def build_video_filename_index(video_paths: list[Path]) -> dict[str, list[Path]]:
    """Return a mapping from lowercase filename to candidate paths."""
    index: dict[str, list[Path]] = {}
    for p in video_paths:
        index.setdefault(p.name.lower(), []).append(p)
    return index


# ---------------------------------------------------------------------------
# Report row
# ---------------------------------------------------------------------------

def _report_row(
    *,
    sample_id: str,
    old_video_id: str,
    new_video_id: str,
    old_video_path: str,
    new_video_path: str,
    video_filename: str,
    frame_index: int,
    status: str,
    match_method: str = "",
    registry_match_tier: str = "",
    fps: float = 0.0,
    source_width: int = 0,
    source_height: int = 0,
    message: str = "",
) -> dict:
    return {
        "sample_id": sample_id,
        "old_video_id": old_video_id,
        "new_video_id": new_video_id,
        "old_video_path": old_video_path,
        "new_video_path": new_video_path,
        "video_filename": video_filename,
        "frame_index": frame_index,
        "status": status,
        "match_method": match_method,
        "registry_match_tier": registry_match_tier,
        "fps": round(fps, 4) if fps else "",
        "source_width": source_width or "",
        "source_height": source_height or "",
        "message": message,
    }


# ---------------------------------------------------------------------------
# Main relink entry point
# ---------------------------------------------------------------------------

def relink_workspace_videos(
    dataset_root: Path,
    source_video_root: Path,
    dry_run: bool = True,
    recursive: bool = True,
    allow_probable: bool = True,
    extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS,
) -> dict:
    """Relink ``metadata/frames.csv`` rows to their source videos.

    Parameters
    ----------
    dataset_root:
        Workspace root (contains ``metadata/frames.csv``).
    source_video_root:
        Folder to scan for source videos.
    dry_run:
        If True, compute the plan and write reports but do not modify
        ``frames.csv`` or the registry.
    recursive:
        Recurse into subdirectories while scanning source videos.
    allow_probable:
        Reserved flag; probable/fingerprint registry matches are always
        accepted by :class:`VideoRegistry`.

    Returns
    -------
    dict
        Summary with counts and paths to the written report files.
    """
    dataset_root = Path(dataset_root)
    source_video_root = Path(source_video_root)

    metadata = MetadataStore(dataset_root)
    registry = VideoRegistry(dataset_root)

    video_paths = scan_video_files(source_video_root, extensions, recursive=recursive)
    filename_index = build_video_filename_index(video_paths)

    # Cache VideoReader info + registry entry per resolved path so we open each
    # source video at most once.
    info_cache: dict[str, object] = {}
    entry_cache: dict[str, tuple[str, str, float, int, int]] = {}

    report_rows: list[dict] = []
    counts = {
        "rows_checked": 0,
        "rows_already_linked": 0,
        "rows_relinkable": 0,
        "linked_existing_path": 0,
        "linked_by_filename": 0,
        "ambiguous": 0,
        "missing": 0,
        "failed_to_read_video": 0,
        "skipped_no_filename": 0,
        "unchanged": 0,
    }
    updates: list[dict] = []

    for rec in metadata.all_records():
        counts["rows_checked"] += 1
        old_video_id = rec.video_id or ""
        old_video_path = rec.video_path or ""
        filename = rec.video_filename or (Path(old_video_path).name if old_video_path else "")

        # Resolve the source video path for this row.
        resolved: Path | None = None
        match_method = ""

        if old_video_path and Path(old_video_path).is_file():
            resolved = Path(old_video_path)
            match_method = "existing_path"
        elif filename and filename.lower() in filename_index:
            candidates = filename_index[filename.lower()]
            if len(candidates) == 1:
                resolved = candidates[0]
                match_method = "filename"
            else:
                counts["ambiguous"] += 1
                report_rows.append(_report_row(
                    sample_id=rec.sample_id, old_video_id=old_video_id,
                    new_video_id=old_video_id, old_video_path=old_video_path,
                    new_video_path="", video_filename=filename,
                    frame_index=rec.frame_index, status=AMBIGUOUS,
                    match_method="filename",
                    message=f"{len(candidates)} candidates under source root",
                ))
                continue
        elif old_video_path and Path(old_video_path).name.lower() in filename_index:
            candidates = filename_index[Path(old_video_path).name.lower()]
            if len(candidates) == 1:
                resolved = candidates[0]
                match_method = "path_basename"
            else:
                counts["ambiguous"] += 1
                report_rows.append(_report_row(
                    sample_id=rec.sample_id, old_video_id=old_video_id,
                    new_video_id=old_video_id, old_video_path=old_video_path,
                    new_video_path="", video_filename=filename,
                    frame_index=rec.frame_index, status=AMBIGUOUS,
                    match_method="path_basename",
                    message=f"{len(candidates)} candidates under source root",
                ))
                continue

        if resolved is None:
            if not filename and not old_video_path:
                counts["skipped_no_filename"] += 1
                report_rows.append(_report_row(
                    sample_id=rec.sample_id, old_video_id=old_video_id,
                    new_video_id=old_video_id, old_video_path=old_video_path,
                    new_video_path="", video_filename=filename,
                    frame_index=rec.frame_index, status=SKIPPED_NO_FILENAME,
                    message="row has no video_filename or video_path",
                ))
            else:
                counts["missing"] += 1
                report_rows.append(_report_row(
                    sample_id=rec.sample_id, old_video_id=old_video_id,
                    new_video_id=old_video_id, old_video_path=old_video_path,
                    new_video_path="", video_filename=filename,
                    frame_index=rec.frame_index, status=MISSING,
                    message="no matching video file found under source root",
                ))
            continue

        key = str(resolved)

        # Read + register the resolved video (once).
        if key not in entry_cache:
            try:
                reader = VideoReader(resolved)
                info = reader.info
                reader.close()
            except Exception as exc:  # noqa: BLE001 — report and move on
                counts["failed_to_read_video"] += 1
                report_rows.append(_report_row(
                    sample_id=rec.sample_id, old_video_id=old_video_id,
                    new_video_id=old_video_id, old_video_path=old_video_path,
                    new_video_path=str(resolved), video_filename=filename,
                    frame_index=rec.frame_index, status=FAILED_TO_READ,
                    match_method=match_method,
                    message=f"VideoReader failed: {exc}",
                ))
                entry_cache[key] = ("", "", 0.0, 0, 0)  # negative cache
                continue
            entry, tier = registry.register(
                info, episode_id=rec.episode_id, camera_id=rec.camera_id,
            )
            info_cache[key] = info
            entry_cache[key] = (
                entry.video_id, tier, info.fps, info.width, info.height,
            )

        new_video_id, tier, fps, src_w, src_h = entry_cache[key]
        if not new_video_id:
            # Negative-cached failed read already reported for this row.
            continue

        # Skip rows that are already fully linked to the same resolved video.
        if (
            old_video_id == new_video_id
            and old_video_path == str(resolved)
            and rec.fps and abs(rec.fps - fps) < 0.01
        ):
            counts["unchanged"] += 1
            counts["rows_already_linked"] += 1
            report_rows.append(_report_row(
                sample_id=rec.sample_id, old_video_id=old_video_id,
                new_video_id=new_video_id, old_video_path=old_video_path,
                new_video_path=str(resolved), video_filename=filename,
                frame_index=rec.frame_index, status=UNCHANGED,
                match_method=match_method, registry_match_tier=tier,
                fps=fps, source_width=src_w, source_height=src_h,
                message="already linked",
            ))
            continue

        if old_video_id:
            counts["rows_already_linked"] += 1
        counts["rows_relinkable"] += 1
        status = LINKED_EXISTING_PATH if match_method == "existing_path" else LINKED_BY_FILENAME
        counts[status] += 1

        updates.append({
            "sample_id": rec.sample_id,
            "video_id": new_video_id,
            "video_filename": resolved.name,
            "video_path": str(resolved),
            "fps": fps,
            "source_width": src_w,
            "source_height": src_h,
        })
        report_rows.append(_report_row(
            sample_id=rec.sample_id, old_video_id=old_video_id,
            new_video_id=new_video_id, old_video_path=old_video_path,
            new_video_path=str(resolved), video_filename=resolved.name,
            frame_index=rec.frame_index, status=status,
            match_method=match_method, registry_match_tier=tier,
            fps=fps, source_width=src_w, source_height=src_h,
        ))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result = {
        "dry_run": dry_run,
        "dataset_root": str(dataset_root),
        "source_video_root": str(source_video_root),
        "timestamp": timestamp,
        "videos_scanned": len(video_paths),
        **counts,
        "report_csv": "",
        "report_json": "",
        "backup_frames_csv": "",
    }

    if not dry_run and updates:
        _backup_before_relink(dataset_root, registry, timestamp)
        for upd in updates:
            metadata.update(
                upd["sample_id"],
                video_id=upd["video_id"],
                video_filename=upd["video_filename"],
                video_path=upd["video_path"],
                fps=upd["fps"],
                source_width=upd["source_width"],
                source_height=upd["source_height"],
            )
        # `update` bumps updated_at; relink is metadata repair, so restore it
        # by saving through the store (updated_at already set). We accept the
        # updated_at bump here because MetadataStore.update sets it; callers who
        # need to preserve it can diff against the backup.
        metadata.save()
        registry.save()
        registry.save_csv()

    # Always write the report so the run is auditable (even dry runs).
    report_csv = dataset_root / "metadata" / "video_relink_report.csv"
    report_json = dataset_root / "metadata" / "video_relink_report.json"
    _write_report_csv(report_csv, report_rows)
    atomic_write_json(report_json, {"summary": result, "rows": report_rows})
    result["report_csv"] = str(report_csv)
    result["report_json"] = str(report_json)

    return result


# ---------------------------------------------------------------------------
# Backups + report writing
# ---------------------------------------------------------------------------

def _backup_before_relink(
    dataset_root: Path, registry: VideoRegistry, timestamp: str,
) -> None:
    backups = dataset_root / "metadata" / "backups"
    backups.mkdir(parents=True, exist_ok=True)

    frames_csv = dataset_root / "metadata" / "frames.csv"
    if frames_csv.exists():
        atomic_write_text(
            backups / f"frames_before_video_relink_{timestamp}.csv",
            frames_csv.read_text(),
        )

    reg_json = dataset_root / "video_registry.json"
    if reg_json.exists():
        atomic_write_text(
            backups / f"video_registry_before_video_relink_{timestamp}.json",
            reg_json.read_text(),
        )
    reg_csv = dataset_root / "video_registry.csv"
    if reg_csv.exists():
        atomic_write_text(
            backups / f"video_registry_before_video_relink_{timestamp}.csv",
            reg_csv.read_text(),
        )


def _write_report_csv(path: Path, rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    atomic_write_text(path, buf.getvalue())
