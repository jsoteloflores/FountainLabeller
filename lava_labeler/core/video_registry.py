"""Video registry: tracks known videos by fingerprint and filename.

Provides stable ``video_id`` values so that frame metadata rows always link
back to the same video even when the file is moved.

Matching tiers
--------------
Tier 1  — same ``file_fingerprint``             → exact match
Tier 2  — same filename + frame_count + fps + resolution → probable match
Tier 3  — same filename, different properties   → warning required
Tier 4  — no match                              → new entry

The fingerprint is a short MD5 hex string derived from:
  filename | file_size | total_frames | fps | width | height | duration
It is intentionally *not* a full file hash so it remains fast for large
4 K video files.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TYPE_CHECKING

from lava_labeler.core.config import atomic_write_json, atomic_write_text

if TYPE_CHECKING:
    from lava_labeler.core.video_io import VideoInfo


MatchTier = Literal["exact", "probable", "filename_mismatch", "new"]

REGISTRY_JSON = "video_registry.json"
REGISTRY_CSV  = "video_registry.csv"

REGISTRY_COLUMNS = [
    "video_id", "video_filename", "video_stem", "video_path",
    "episode_id", "camera_id", "eruption_id", "source_date",
    "total_frames", "fps", "width", "height", "duration_seconds",
    "file_size_bytes", "file_fingerprint",
    "created_at", "last_opened_at", "notes",
]


# ---------------------------------------------------------------------------
# Fingerprint
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
# VideoEntry dataclass
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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_csv_row(self) -> dict:
        return {col: getattr(self, col, "") for col in REGISTRY_COLUMNS}


# ---------------------------------------------------------------------------
# VideoRegistry
# ---------------------------------------------------------------------------

class VideoRegistry:
    """Persistent registry of all videos opened in a project.

    Parameters
    ----------
    root:
        Project root directory (same as dataset root).
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._json_path = self.root / REGISTRY_JSON
        self._csv_path  = self.root / REGISTRY_CSV
        self._entries: dict[str, VideoEntry] = {}   # video_id → entry
        self._counter: int = 0
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._json_path.exists():
            return
        try:
            data = json.loads(self._json_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for item in data.get("videos", []):
            entry = _dict_to_entry(item)
            self._entries[entry.video_id] = entry
        # Derive counter from existing ids (vid_NNNNNN)
        for vid in self._entries:
            try:
                n = int(vid.split("_", 1)[-1])
                if n > self._counter:
                    self._counter = n
            except ValueError:
                pass

    def save(self) -> None:
        data = {
            "schema_version": "1.0",
            "videos": [e.to_dict() for e in self._entries.values()],
        }
        atomic_write_json(self._json_path, data)

    def save_csv(self) -> None:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=REGISTRY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for entry in self._entries.values():
            w.writerow(entry.to_csv_row())
        atomic_write_text(self._csv_path, buf.getvalue())

    # ------------------------------------------------------------------
    # Video ID generation
    # ------------------------------------------------------------------

    def _new_video_id(self) -> str:
        self._counter += 1
        return f"vid_{self._counter:06d}"

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, info: "VideoInfo") -> tuple["VideoEntry | None", MatchTier]:
        """Find the best existing entry for *info*.

        Returns ``(entry, tier)`` where *tier* indicates the match quality:
        ``"exact"`` / ``"probable"`` / ``"filename_mismatch"`` / ``"new"``.
        """
        filename = Path(info.path).name
        file_size = Path(info.path).stat().st_size if Path(info.path).exists() else 0
        fp = compute_fingerprint(filename, file_size, info.frame_count, info.fps, info.width, info.height)

        # Tier 1 — fingerprint
        for entry in self._entries.values():
            if entry.file_fingerprint and entry.file_fingerprint == fp:
                return entry, "exact"

        # Tier 2 — filename + all video properties
        for entry in self._entries.values():
            if (
                entry.video_filename == filename
                and entry.total_frames == info.frame_count
                and abs(entry.fps - info.fps) < 0.1
                and entry.width == info.width
                and entry.height == info.height
            ):
                return entry, "probable"

        # Tier 3 — filename only but different properties
        for entry in self._entries.values():
            if entry.video_filename == filename:
                return entry, "filename_mismatch"

        return None, "new"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        info: "VideoInfo",
        episode_id: str = "",
        camera_id: str = "",
        eruption_id: str = "",
        source_date: str = "",
        notes: str = "",
    ) -> tuple["VideoEntry", MatchTier]:
        """Register or update a video.  Returns ``(entry, tier)``."""
        existing, tier = self.match(info)
        now = datetime.now(timezone.utc).isoformat()
        filename = Path(info.path).name
        stem = Path(info.path).stem
        file_size = Path(info.path).stat().st_size if Path(info.path).exists() else 0
        fp = compute_fingerprint(filename, file_size, info.frame_count, info.fps, info.width, info.height)

        if existing is not None and tier in ("exact", "probable"):
            # Update mutable fields
            existing.video_path = str(info.path)
            existing.last_opened_at = now
            existing.file_fingerprint = fp
            existing.file_size_bytes = file_size
            # Don't overwrite user-edited episode/camera unless they're blank
            if episode_id and not existing.episode_id:
                existing.episode_id = episode_id
            if camera_id and not existing.camera_id:
                existing.camera_id = camera_id
            self.save()
            return existing, tier

        # filename_mismatch or new — always create a new entry
        video_id = self._new_video_id()
        entry = VideoEntry(
            video_id=video_id,
            video_filename=filename,
            video_stem=stem,
            video_path=str(info.path),
            episode_id=episode_id,
            camera_id=camera_id,
            eruption_id=eruption_id,
            source_date=source_date,
            total_frames=info.frame_count,
            fps=info.fps,
            width=info.width,
            height=info.height,
            duration_seconds=round(info.duration_seconds, 3),
            file_size_bytes=file_size,
            file_fingerprint=fp,
            created_at=now,
            last_opened_at=now,
            notes=notes,
        )
        self._entries[video_id] = entry
        self.save()
        # Return the actual tier so callers can distinguish filename_mismatch from new
        return entry, tier

    def update_entry(self, video_id: str, **kwargs) -> None:
        """Edit mutable fields on an existing entry."""
        entry = self._entries.get(video_id)
        if entry is None:
            return
        allowed = {"episode_id", "camera_id", "eruption_id", "source_date", "notes", "video_path"}
        for k, v in kwargs.items():
            if k in allowed:
                setattr(entry, k, v)
        self.save()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, video_id: str) -> "VideoEntry | None":
        return self._entries.get(video_id)

    def all_entries(self) -> list["VideoEntry"]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_entry(d: dict) -> VideoEntry:
    def _s(k: str, df: str = "") -> str:
        v = d.get(k, df)
        return df if v is None or str(v) in ("nan", "None", "") else str(v)

    def _i(k: str, df: int = 0) -> int:
        try:
            return int(float(d.get(k, df)))
        except (TypeError, ValueError):
            return df

    def _f(k: str, df: float = 0.0) -> float:
        try:
            return float(d.get(k, df))
        except (TypeError, ValueError):
            return df

    return VideoEntry(
        video_id=_s("video_id"),
        video_filename=_s("video_filename"),
        video_stem=_s("video_stem"),
        video_path=_s("video_path"),
        episode_id=_s("episode_id"),
        camera_id=_s("camera_id"),
        eruption_id=_s("eruption_id"),
        source_date=_s("source_date"),
        total_frames=_i("total_frames"),
        fps=_f("fps"),
        width=_i("width"),
        height=_i("height"),
        duration_seconds=_f("duration_seconds"),
        file_size_bytes=_i("file_size_bytes"),
        file_fingerprint=_s("file_fingerprint"),
        created_at=_s("created_at"),
        last_opened_at=_s("last_opened_at"),
        notes=_s("notes"),
    )
