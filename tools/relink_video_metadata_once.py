#!/usr/bin/env python3
"""One-time video metadata relinking utility for FountainLabeller.

Backfills missing ``video_id`` values in an existing FountainLabeller workspace
by matching each frame row in ``metadata/frames.csv`` back to its source video.

This script is intentionally *standalone*: it does not import the GUI or any
``tkinter`` code, and it copies the small ``VideoRegistry`` fingerprint schema
so it can run safely from the command line.

It NEVER modifies labels, masks, image files, ROI coordinates, frame indices,
or label statuses. It only fills in ``video_id`` / ``video_path`` /
``video_filename`` (and optionally ``fps``) on frame rows, and builds/updates
``video_registry.json`` + ``video_registry.csv``.

Default mode is a dry run. Pass ``--apply`` to write changes.

Usage
-----
    python tools/relink_video_metadata_once.py \
        --workspace "/path/to/workspace"

    python tools/relink_video_metadata_once.py \
        --workspace "/path/to/workspace" \
        --video-root "/Volumes/Joel HDD/Kilauea_2024-2026_videos_renamed" \
        --apply
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    print("ERROR: OpenCV (cv2) is required. Install with: pip install opencv-python", file=sys.stderr)
    raise

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    print("ERROR: pandas is required. Install with: pip install pandas", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Constants (kept in sync with lava_labeler.core.video_registry)
# ---------------------------------------------------------------------------

REGISTRY_JSON = "video_registry.json"
REGISTRY_CSV = "video_registry.csv"

REGISTRY_COLUMNS = [
    "video_id", "video_filename", "video_stem", "video_path",
    "episode_id", "camera_id", "eruption_id", "source_date",
    "total_frames", "fps", "width", "height", "duration_seconds",
    "file_size_bytes", "file_fingerprint",
    "created_at", "last_opened_at", "notes",
]

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv",
    ".MP4", ".MOV", ".AVI", ".MKV",
}

REPORT_COLUMNS = [
    "sample_id",
    "old_video_id", "new_video_id",
    "old_video_path", "new_video_path",
    "video_filename",
    "episode_id", "camera_id",
    "frame_index", "source_width", "source_height",
    "roi_x", "roi_y", "roi_width", "roi_height",
    "match_status", "match_tier",
    "validation_status", "warnings", "errors",
]


# ---------------------------------------------------------------------------
# Value coercion helpers
# ---------------------------------------------------------------------------

def _is_blank(v: object) -> bool:
    return v is None or str(v).strip() in ("", "nan", "None", "NaN")


def _s(row: dict, key: str, default: str = "") -> str:
    v = row.get(key, default)
    return default if _is_blank(v) else str(v).strip()


def _opt_int(row: dict, key: str) -> Optional[int]:
    v = row.get(key, "")
    if _is_blank(v):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _as_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------

def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, obj: object) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2))


# ---------------------------------------------------------------------------
# Fingerprint (identical logic to lava_labeler.core.video_registry)
# ---------------------------------------------------------------------------

def compute_fingerprint(
    filename: str,
    file_size: int,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
) -> str:
    """Return a 16-char hex fingerprint (fast, not a full hash)."""
    duration = total_frames / fps if fps > 0 else 0.0
    raw = f"{filename}|{file_size}|{total_frames}|{fps:.4f}|{width}|{height}|{duration:.2f}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# VideoInfo
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    path: Path
    filename: str
    stem: str
    total_frames: int
    fps: float
    width: int
    height: int
    file_size_bytes: int

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps if self.fps > 0 else 0.0

    @property
    def fingerprint(self) -> str:
        return compute_fingerprint(
            self.filename, self.file_size_bytes,
            self.total_frames, self.fps, self.width, self.height,
        )

    @property
    def is_valid(self) -> bool:
        return (
            self.fps > 0
            and self.total_frames > 0
            and self.width > 0
            and self.height > 0
        )


def read_video_info(path: Path) -> Optional[VideoInfo]:
    """Open *path* with OpenCV and return VideoInfo, or None if it cannot open."""
    path = Path(path)
    if not path.is_file():
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    try:
        file_size = path.stat().st_size
    except OSError:
        file_size = 0
    return VideoInfo(
        path=path,
        filename=path.name,
        stem=path.stem,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        file_size_bytes=file_size,
    )


# ---------------------------------------------------------------------------
# VideoEntry / registry
# ---------------------------------------------------------------------------

@dataclass
class VideoEntry:
    video_id: str
    video_filename: str
    video_stem: str
    video_path: str
    episode_id: str = ""
    camera_id: str = ""
    eruption_id: str = ""
    source_date: str = ""
    total_frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration_seconds: float = 0.0
    file_size_bytes: int = 0
    file_fingerprint: str = ""
    created_at: str = field(default_factory=_now_iso)
    last_opened_at: str = field(default_factory=_now_iso)
    notes: str = ""

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_csv_row(self) -> dict:
        return {col: getattr(self, col, "") for col in REGISTRY_COLUMNS}


def _dict_to_entry(d: dict) -> VideoEntry:
    def s(k: str, df: str = "") -> str:
        v = d.get(k, df)
        return df if _is_blank(v) else str(v)

    def i(k: str, df: int = 0) -> int:
        try:
            return int(float(d.get(k, df)))
        except (TypeError, ValueError):
            return df

    def f(k: str, df: float = 0.0) -> float:
        try:
            return float(d.get(k, df))
        except (TypeError, ValueError):
            return df

    return VideoEntry(
        video_id=s("video_id"),
        video_filename=s("video_filename"),
        video_stem=s("video_stem"),
        video_path=s("video_path"),
        episode_id=s("episode_id"),
        camera_id=s("camera_id"),
        eruption_id=s("eruption_id"),
        source_date=s("source_date"),
        total_frames=i("total_frames"),
        fps=f("fps"),
        width=i("width"),
        height=i("height"),
        duration_seconds=f("duration_seconds"),
        file_size_bytes=i("file_size_bytes"),
        file_fingerprint=s("file_fingerprint"),
        created_at=s("created_at", _now_iso()),
        last_opened_at=s("last_opened_at", _now_iso()),
        notes=s("notes"),
    )


class Registry:
    """Standalone port of lava_labeler.core.video_registry.VideoRegistry.

    Continues ``vid_NNNNNN`` numbering from any existing entries.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)
        self.json_path = self.workspace / REGISTRY_JSON
        self.csv_path = self.workspace / REGISTRY_CSV
        self.entries: dict[str, VideoEntry] = {}
        self._counter = 0
        self._preexisting_ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.json_path.exists():
            return
        try:
            data = json.loads(self.json_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for item in data.get("videos", []):
            entry = _dict_to_entry(item)
            if not entry.video_id:
                continue
            self.entries[entry.video_id] = entry
        self._preexisting_ids = set(self.entries.keys())
        for vid in self.entries:
            try:
                n = int(vid.split("_", 1)[-1])
                self._counter = max(self._counter, n)
            except ValueError:
                pass

    def _new_video_id(self) -> str:
        self._counter += 1
        return f"vid_{self._counter:06d}"

    def was_preexisting(self, video_id: str) -> bool:
        return video_id in self._preexisting_ids

    def match(self, info: VideoInfo) -> tuple[Optional[VideoEntry], str]:
        fp = info.fingerprint
        # Tier 1 — fingerprint
        for entry in self.entries.values():
            if entry.file_fingerprint and entry.file_fingerprint == fp:
                return entry, "exact"
        # Tier 2 — filename + all video properties
        for entry in self.entries.values():
            if (
                entry.video_filename == info.filename
                and entry.total_frames == info.total_frames
                and abs(entry.fps - info.fps) < 0.1
                and entry.width == info.width
                and entry.height == info.height
            ):
                return entry, "probable"
        # Tier 3 — filename only but different properties
        for entry in self.entries.values():
            if entry.video_filename == info.filename:
                return entry, "filename_mismatch"
        return None, "new"

    def match_or_register(
        self,
        info: VideoInfo,
        episode_id: str = "",
        camera_id: str = "",
    ) -> tuple[VideoEntry, str]:
        existing, tier = self.match(info)
        now = _now_iso()
        fp = info.fingerprint

        if existing is not None and tier in ("exact", "probable"):
            existing.video_path = str(info.path)
            existing.last_opened_at = now
            existing.file_fingerprint = fp
            existing.file_size_bytes = info.file_size_bytes
            if episode_id and not existing.episode_id:
                existing.episode_id = episode_id
            if camera_id and not existing.camera_id:
                existing.camera_id = camera_id
            return existing, tier

        # filename_mismatch or new — create a new entry.
        video_id = self._new_video_id()
        entry = VideoEntry(
            video_id=video_id,
            video_filename=info.filename,
            video_stem=info.stem,
            video_path=str(info.path),
            episode_id=episode_id,
            camera_id=camera_id,
            total_frames=info.total_frames,
            fps=info.fps,
            width=info.width,
            height=info.height,
            duration_seconds=round(info.duration_seconds, 3),
            file_size_bytes=info.file_size_bytes,
            file_fingerprint=fp,
            created_at=now,
            last_opened_at=now,
        )
        self.entries[video_id] = entry
        return entry, tier

    def to_json_obj(self) -> dict:
        return {
            "schema_version": "1.0",
            "videos": [e.to_dict() for e in self.entries.values()],
        }

    def to_csv_text(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=REGISTRY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for entry in self.entries.values():
            w.writerow(entry.to_csv_row())
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Source-video discovery
# ---------------------------------------------------------------------------

_EPISODE_TOKEN = re.compile(r"(?<![A-Za-z0-9])[Ee](\d{1,3})(?![0-9])")


def scan_video_root(video_root: Path) -> dict[str, list[Path]]:
    """Recursively index all video files under *video_root* by lowercase name."""
    index: dict[str, list[Path]] = {}
    for dirpath, _dirs, files in os.walk(video_root):
        for name in files:
            if Path(name).suffix in VIDEO_EXTENSIONS:
                index.setdefault(name.lower(), []).append(Path(dirpath) / name)
    return index


def _episode_numbers(text: str) -> set[str]:
    return {m.group(1).lstrip("0") or "0" for m in _EPISODE_TOKEN.finditer(text)}


def _score_candidate(
    candidate: Path,
    info: Optional[VideoInfo],
    *,
    source_width: Optional[int],
    source_height: Optional[int],
    episode_id: str,
    camera_id: str,
    max_frame_index: int,
) -> int:
    """Heuristic disambiguation score for a duplicate-filename candidate."""
    score = 0
    path_str = str(candidate)
    if info is not None:
        if source_width and source_height:
            if info.width == source_width and info.height == source_height:
                score += 4
            else:
                score -= 2
        if info.total_frames > 0 and 0 <= max_frame_index < info.total_frames:
            score += 2
        elif info.total_frames > 0:
            score -= 3
    # Episode token match in folder path.
    if episode_id:
        ep_nums = _episode_numbers(episode_id)
        path_nums = _episode_numbers(path_str)
        if ep_nums and (ep_nums & path_nums):
            score += 2
    # Camera token match in folder path (case-insensitive substring).
    if camera_id and camera_id.lower() in path_str.lower():
        score += 2
    return score


@dataclass
class Resolution:
    path: Optional[Path]
    status: str          # existing_path | video_root_scan | missing | ambiguous
    candidates: list[Path] = field(default_factory=list)


def resolve_video(
    video_path_str: str,
    video_filename: str,
    scan_index: Optional[dict[str, list[Path]]],
    *,
    source_width: Optional[int],
    source_height: Optional[int],
    episode_id: str,
    camera_id: str,
    max_frame_index: int,
    info_cache: dict[str, Optional[VideoInfo]],
) -> Resolution:
    # 1. Direct path.
    if video_path_str:
        direct = Path(video_path_str)
        if direct.is_file():
            return Resolution(direct, "existing_path")

    # 2. Scan under --video-root.
    if scan_index is not None and video_filename:
        candidates = scan_index.get(video_filename.lower(), [])
        if len(candidates) == 1:
            return Resolution(candidates[0], "video_root_scan")
        if len(candidates) > 1:
            scored: list[tuple[int, Path]] = []
            for cand in candidates:
                key = str(cand)
                if key not in info_cache:
                    info_cache[key] = read_video_info(cand)
                score = _score_candidate(
                    cand, info_cache[key],
                    source_width=source_width, source_height=source_height,
                    episode_id=episode_id, camera_id=camera_id,
                    max_frame_index=max_frame_index,
                )
                scored.append((score, cand))
            scored.sort(key=lambda t: t[0], reverse=True)
            best_score = scored[0][0]
            winners = [c for s, c in scored if s == best_score]
            if len(winners) == 1 and best_score > 0:
                return Resolution(winners[0], "video_root_scan", candidates)
            return Resolution(None, "ambiguous", candidates)

    return Resolution(None, "missing")


# ---------------------------------------------------------------------------
# Relink planning + validation
# ---------------------------------------------------------------------------

@dataclass
class RowPlan:
    sample_id: str
    old_video_id: str
    new_video_id: str
    old_video_path: str
    new_video_path: str
    video_filename: str
    episode_id: str
    camera_id: str
    frame_index: Optional[int]
    source_width: Optional[int]
    source_height: Optional[int]
    roi_x: Optional[int]
    roi_y: Optional[int]
    roi_width: Optional[int]
    roi_height: Optional[int]
    is_roi_crop: bool
    fps: float = 0.0
    match_status: str = ""
    match_tier: str = ""
    validation_status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    will_update: bool = False

    def report_row(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "old_video_id": self.old_video_id,
            "new_video_id": self.new_video_id,
            "old_video_path": self.old_video_path,
            "new_video_path": self.new_video_path,
            "video_filename": self.video_filename,
            "episode_id": self.episode_id,
            "camera_id": self.camera_id,
            "frame_index": "" if self.frame_index is None else self.frame_index,
            "source_width": "" if self.source_width is None else self.source_width,
            "source_height": "" if self.source_height is None else self.source_height,
            "roi_x": "" if self.roi_x is None else self.roi_x,
            "roi_y": "" if self.roi_y is None else self.roi_y,
            "roi_width": "" if self.roi_width is None else self.roi_width,
            "roi_height": "" if self.roi_height is None else self.roi_height,
            "match_status": self.match_status,
            "match_tier": self.match_tier,
            "validation_status": self.validation_status,
            "warnings": "; ".join(self.warnings),
            "errors": "; ".join(self.errors),
        }


def _group_key(video_path: str, video_filename: str) -> str:
    return video_path or f"__name__::{video_filename}"


def validate_row(plan: RowPlan, info: Optional[VideoInfo]) -> None:
    """Populate warnings/errors and validation_status on *plan* in place."""
    # Per-row required fields.
    if not plan.video_filename:
        plan.errors.append("missing video_filename")
    if plan.frame_index is None:
        plan.errors.append("missing/invalid frame_index")
    if plan.source_width is None or plan.source_height is None:
        plan.errors.append("missing source_width/source_height")
    if plan.is_roi_crop and (
        plan.roi_x is None or plan.roi_y is None
        or plan.roi_width is None or plan.roi_height is None
    ):
        plan.errors.append("is_roi_crop set but ROI fields missing")

    if info is not None:
        # Frame bounds check.
        if plan.frame_index is not None:
            if not (0 <= plan.frame_index < info.total_frames):
                plan.errors.append(
                    f"frame_out_of_bounds (frame_index={plan.frame_index}, "
                    f"total_frames={info.total_frames})"
                )
        # Resolution consistency check (warning by default).
        if plan.source_width is not None and plan.source_height is not None:
            if info.width and info.height and (
                plan.source_width != info.width or plan.source_height != info.height
            ):
                plan.warnings.append(
                    f"resolution_mismatch (row={plan.source_width}x{plan.source_height}, "
                    f"video={info.width}x{info.height})"
                )
        # ROI bounds check.
        if plan.is_roi_crop and plan.source_width and plan.source_height and all(
            v is not None for v in (plan.roi_x, plan.roi_y, plan.roi_width, plan.roi_height)
        ):
            in_bounds = (
                0 <= plan.roi_x < plan.source_width
                and 0 <= plan.roi_y < plan.source_height
                and plan.roi_x + plan.roi_width <= plan.source_width
                and plan.roi_y + plan.roi_height <= plan.source_height
            )
            if not in_bounds:
                plan.errors.append("roi_out_of_bounds")

    if plan.errors:
        plan.validation_status = "error"
    elif plan.warnings:
        plan.validation_status = "warning"
    else:
        plan.validation_status = "ok"


def build_relink_plan(
    df: pd.DataFrame,
    registry: Registry,
    scan_index: Optional[dict[str, list[Path]]],
    *,
    strict: bool,
) -> tuple[list[RowPlan], dict]:
    rows = df.to_dict("records")

    # Group rows by their source-video reference.
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = _group_key(_s(row, "video_path"), _s(row, "video_filename"))
        groups.setdefault(key, []).append(row)

    # Resolve each group's source video once, then register/match it.
    resolved_video: dict[str, tuple[Resolution, Optional[VideoInfo], Optional[VideoEntry], str, str]] = {}
    info_cache: dict[str, Optional[VideoInfo]] = {}

    for key, group_rows in groups.items():
        sample = group_rows[0]
        video_path_str = _s(sample, "video_path")
        video_filename = _s(sample, "video_filename")
        episode_id = _s(sample, "episode_id")
        camera_id = _s(sample, "camera_id")
        src_w = _opt_int(sample, "source_width")
        src_h = _opt_int(sample, "source_height")
        max_frame_index = max(
            (_opt_int(r, "frame_index") or 0) for r in group_rows
        )

        resolution = resolve_video(
            video_path_str, video_filename, scan_index,
            source_width=src_w, source_height=src_h,
            episode_id=episode_id, camera_id=camera_id,
            max_frame_index=max_frame_index,
            info_cache=info_cache,
        )

        info: Optional[VideoInfo] = None
        entry: Optional[VideoEntry] = None
        match_status = "missing_video_file"
        match_tier = ""

        if resolution.status == "ambiguous":
            match_status = "ambiguous_match"
        elif resolution.path is not None:
            key_path = str(resolution.path)
            if key_path not in info_cache:
                info_cache[key_path] = read_video_info(resolution.path)
            info = info_cache[key_path]
            if info is None or not info.is_valid:
                match_status = "missing_video_file"
            else:
                entry, tier = registry.match_or_register(info, episode_id, camera_id)
                match_tier = tier
                if tier in ("exact", "probable") and registry.was_preexisting(entry.video_id):
                    match_status = "matched_existing_registry"
                else:
                    match_status = "matched_new_registry"

        resolved_video[key] = (resolution, info, entry, match_status, match_tier)

    # Build per-row plans.
    plans: list[RowPlan] = []
    for row in rows:
        key = _group_key(_s(row, "video_path"), _s(row, "video_filename"))
        resolution, info, entry, match_status, match_tier = resolved_video[key]

        old_video_id = _s(row, "video_id")
        frame_index = _opt_int(row, "frame_index")

        plan = RowPlan(
            sample_id=_s(row, "sample_id"),
            old_video_id=old_video_id,
            new_video_id=old_video_id,
            old_video_path=_s(row, "video_path"),
            new_video_path=_s(row, "video_path"),
            video_filename=_s(row, "video_filename"),
            episode_id=_s(row, "episode_id"),
            camera_id=_s(row, "camera_id"),
            frame_index=frame_index,
            source_width=_opt_int(row, "source_width"),
            source_height=_opt_int(row, "source_height"),
            roi_x=_opt_int(row, "roi_x"),
            roi_y=_opt_int(row, "roi_y"),
            roi_width=_opt_int(row, "roi_width"),
            roi_height=_opt_int(row, "roi_height"),
            is_roi_crop=_as_bool(row.get("is_roi_crop", "")),
            fps=info.fps if info is not None else 0.0,
            match_tier=match_tier,
        )

        # Row already has a video_id — do not overwrite it.
        if old_video_id:
            plan.match_status = "skipped_existing_video_id"
            validate_row(plan, info)
            plans.append(plan)
            continue

        plan.match_status = match_status

        if entry is not None and info is not None:
            plan.new_video_id = entry.video_id
            plan.new_video_path = str(info.path)
            plan.video_filename = info.filename
            plan.will_update = True

        validate_row(plan, info)

        # In strict mode, do not assign a video_id to rows that fail validation.
        if strict and plan.errors and plan.will_update:
            plan.new_video_id = old_video_id
            plan.new_video_path = _s(row, "video_path")
            plan.will_update = False

        plans.append(plan)

    summary = _summarize(plans, groups, resolved_video)
    return plans, summary


def _summarize(
    plans: list[RowPlan],
    groups: dict[str, list[dict]],
    resolved_video: dict,
) -> dict:
    unique_filenames = {
        _s(rows[0], "video_filename") for rows in groups.values()
    }
    registered = {
        entry.video_id
        for (_res, _info, entry, _st, _t) in resolved_video.values()
        if entry is not None
    }
    matched_existing = sum(1 for r in resolved_video.values() if r[3] == "matched_existing_registry")
    matched_new = sum(1 for r in resolved_video.values() if r[3] == "matched_new_registry")
    ambiguous = sum(1 for r in resolved_video.values() if r[3] == "ambiguous_match")
    missing = sum(1 for r in resolved_video.values() if r[3] == "missing_video_file")

    return {
        "rows_total": len(plans),
        "rows_with_existing_video_id": sum(1 for p in plans if p.old_video_id),
        "rows_relinked": sum(1 for p in plans if p.will_update),
        "rows_unmatched": sum(
            1 for p in plans
            if not p.old_video_id and not p.will_update
        ),
        "unique_video_filenames": len(unique_filenames),
        "unique_videos_registered": len(registered),
        "ambiguous_matches": ambiguous,
        "missing_videos": missing,
        "videos_matched_existing_registry": matched_existing,
        "videos_matched_new_registry": matched_new,
        "frame_out_of_bounds": sum(
            1 for p in plans if any("frame_out_of_bounds" in e for e in p.errors)
        ),
        "resolution_mismatches": sum(
            1 for p in plans if any("resolution_mismatch" in w for w in p.warnings)
        ),
        "roi_out_of_bounds": sum(
            1 for p in plans if any("roi_out_of_bounds" in e for e in p.errors)
        ),
    }


# ---------------------------------------------------------------------------
# Applying the plan to frames.csv
# ---------------------------------------------------------------------------

def apply_relink_plan(
    df: pd.DataFrame,
    plans: list[RowPlan],
    *,
    write_fps_column: bool,
) -> pd.DataFrame:
    """Return a copy of *df* with video_id/video_path/video_filename updated.

    Preserves original column order and any unknown/extra columns. Only rows
    that are flagged ``will_update`` are changed.
    """
    out = df.copy()
    by_sample = {p.sample_id: p for p in plans}

    # Ensure the columns we may write exist.
    for col in ("video_id", "video_path", "video_filename"):
        if col not in out.columns:
            out[col] = ""

    if write_fps_column and "fps" not in out.columns:
        out["fps"] = ""

    sample_col = out["sample_id"].astype(str)
    for idx in out.index:
        sample_id = str(sample_col.loc[idx]).strip()
        plan = by_sample.get(sample_id)
        if plan is None or not plan.will_update:
            continue
        out.at[idx, "video_id"] = plan.new_video_id
        out.at[idx, "video_path"] = plan.new_video_path
        out.at[idx, "video_filename"] = plan.video_filename
        if write_fps_column and plan.fps > 0:
            out.at[idx, "fps"] = plan.fps

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report_csv(path: Path, plans: list[RowPlan]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for plan in plans:
        w.writerow(plan.report_row())
    atomic_write_text(path, buf.getvalue())


def build_summary_json(
    args: argparse.Namespace,
    workspace: Path,
    frames_path: Path,
    summary: dict,
) -> dict:
    return {
        "created_at": _now_iso(),
        "workspace": str(workspace),
        "frames_csv": str(frames_path),
        "dry_run": not args.apply,
        "strict": args.strict,
        "video_root": str(args.video_root) if args.video_root else None,
        **summary,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_dry_run(summary: dict, resolved_counts: dict) -> None:
    print("FountainLabeller video relink dry run")
    print(f"frames.csv rows: {summary['rows_total']}")
    print(f"unique video filenames: {summary['unique_video_filenames']}")
    print(f"existing video_id values: {summary['rows_with_existing_video_id']}")
    print()
    print("Resolved videos:")
    print(f"  {resolved_counts['existing_path']} matched by existing video_path")
    print(f"  {resolved_counts['video_root_scan']} matched by video-root scan")
    print(f"  {summary['missing_videos']} missing")
    print(f"  {summary['ambiguous_matches']} ambiguous")
    print()
    print("Rows:")
    print(f"  {summary['rows_relinked']} would receive video_id")
    print(f"  {summary['rows_unmatched']} would remain unmatched")
    print(f"  {summary['frame_out_of_bounds']} frame bounds errors")
    print(f"  {summary['roi_out_of_bounds']} ROI bounds errors")
    print(f"  {summary['resolution_mismatches']} resolution mismatches (warnings)")
    print()
    print("Dry run only. Re-run with --apply to write changes.")


# ---------------------------------------------------------------------------
# Backups
# ---------------------------------------------------------------------------

def backup_file(path: Path, backup_path: Path) -> None:
    data = path.read_bytes()
    atomic_write_bytes(backup_path, data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-time FountainLabeller video metadata relinking utility.",
    )
    p.add_argument("--workspace", required=True, type=Path,
                   help="Path to the FountainLabeller workspace (contains metadata/frames.csv).")
    p.add_argument("--video-root", type=Path, default=None,
                   help="Optional root to recursively scan for source videos by filename.")
    p.add_argument("--apply", action="store_true",
                   help="Actually write changes. Without this, runs as a dry run.")
    p.add_argument("--strict", action="store_true",
                   help="Do not assign video_id to rows that fail validation; fail on unmatched rows.")
    p.add_argument("--write-fps-column", action="store_true",
                   help="Also write an fps column into frames.csv (schema must support it).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    frames_path = workspace / "metadata" / "frames.csv"

    if not frames_path.is_file():
        print(f"ERROR: frames.csv not found at {frames_path}", file=sys.stderr)
        return 2

    if args.video_root is not None and not args.video_root.is_dir():
        print(f"ERROR: --video-root does not exist: {args.video_root}", file=sys.stderr)
        return 2

    df = pd.read_csv(frames_path, dtype=str, keep_default_na=False)

    registry = Registry(workspace)

    scan_index: Optional[dict[str, list[Path]]] = None
    if args.video_root is not None:
        print(f"Scanning video root: {args.video_root} …")
        scan_index = scan_video_root(args.video_root)
        print(f"  indexed {sum(len(v) for v in scan_index.values())} video files "
              f"({len(scan_index)} distinct filenames)")

    plans, summary = build_relink_plan(df, registry, scan_index, strict=args.strict)

    # Count resolution sources for console output.
    resolved_counts = {"existing_path": 0, "video_root_scan": 0}
    seen_keys: set[str] = set()
    for plan in plans:
        key = _group_key(plan.old_video_path, plan.video_filename)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if plan.match_status in ("matched_existing_registry", "matched_new_registry"):
            src = plan.new_video_path
            if src and Path(plan.old_video_path).is_file() and str(Path(plan.old_video_path)) == src:
                resolved_counts["existing_path"] += 1
            else:
                resolved_counts["video_root_scan"] += 1

    timestamp = _timestamp()
    metadata_dir = workspace / "metadata"
    report_path = metadata_dir / f"video_relink_report_{timestamp}.csv"
    summary_path = metadata_dir / f"video_relink_summary_{timestamp}.json"
    summary_obj = build_summary_json(args, workspace, frames_path, summary)

    # Strict-mode failure conditions.
    strict_fail = False
    if args.strict:
        if summary["missing_videos"] or summary["ambiguous_matches"]:
            strict_fail = True
        if any(p.errors for p in plans if not p.old_video_id):
            strict_fail = True

    print()
    print_dry_run(summary, resolved_counts)
    print()

    # Always write the report + summary so the run is auditable.
    write_report_csv(report_path, plans)
    atomic_write_json(summary_path, summary_obj)
    print(f"Wrote relink report:  {report_path.relative_to(workspace)}")
    print(f"Wrote relink summary: {summary_path.relative_to(workspace)}")

    # Per-row warnings for rows that are not 2.5D-ready after relinking.
    not_ready = [
        p for p in plans
        if p.will_update and (p.errors or p.warnings)
    ]
    if not_ready:
        print()
        print(f"WARNING: {len(not_ready)} relinked row(s) are not fully 2.5D-ready:")
        for p in not_ready[:20]:
            issues = "; ".join(p.errors + p.warnings)
            print(f"  {p.sample_id}: {issues}")
        if len(not_ready) > 20:
            print(f"  … and {len(not_ready) - 20} more (see report).")

    if not args.apply:
        if strict_fail:
            print()
            print("STRICT: unresolved rows detected — re-run without --strict or fix inputs.",
                  file=sys.stderr)
            return 1
        return 0

    # ------------------------------------------------------------------
    # Apply mode.
    # ------------------------------------------------------------------
    if strict_fail:
        print()
        print("STRICT: refusing to apply — unresolved/invalid rows detected.", file=sys.stderr)
        return 1

    # Backups.
    frames_backup = metadata_dir / f"frames_backup_before_video_relink_{timestamp}.csv"
    backup_file(frames_path, frames_backup)
    print(f"Backed up frames.csv to {frames_backup.relative_to(workspace)}")

    if registry.json_path.exists():
        reg_backup = workspace / f"video_registry_backup_before_video_relink_{timestamp}.json"
        backup_file(registry.json_path, reg_backup)
        print(f"Backed up video_registry.json to {reg_backup.relative_to(workspace)}")

    # Write updated frames.csv (preserve column order + unknown columns).
    updated_df = apply_relink_plan(df, plans, write_fps_column=args.write_fps_column)
    atomic_write_text(frames_path, updated_df.to_csv(index=False))
    print("Wrote updated frames.csv")

    # Write registry.
    atomic_write_json(registry.json_path, registry.to_json_obj())
    print("Wrote video_registry.json")
    atomic_write_text(registry.csv_path, registry.to_csv_text())
    print("Wrote video_registry.csv")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
