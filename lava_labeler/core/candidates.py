"""Candidate-frame worklist loaded from candidate_frames.csv / .json.

A candidate is a (video, frame_index) the user is asked to label.  The queue
supports filtered navigation (next unlabeled, next needs_review, next
hard_negative, next in same video/camera) and persists status back to disk
with atomic writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from lava_labeler.core.config import atomic_write_text


CANDIDATE_STATUSES = [
    "unlabeled", "in_progress", "labeled", "skipped",
    "needs_review", "bad_frame", "hard_negative", "empty_confirmed",
]

REQUIRED_COLUMNS = [
    "candidate_id", "video_id", "video_path", "frame_index",
    "time_seconds", "camera_id", "reason", "priority", "status", "notes",
]

OPTIONAL_COLUMNS = [
    "eruption_episode", "lighting_condition", "source",
    "roi_x", "roi_y", "roi_width", "roi_height",
    "suggested_by", "uncertainty_score", "review_score",
]

ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


@dataclass
class Candidate:
    candidate_id: str
    video_id: str = ""
    video_path: str = ""
    frame_index: int = 0
    time_seconds: float = 0.0
    camera_id: str = ""
    reason: str = ""
    priority: int = 0
    status: str = "unlabeled"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {
            "candidate_id": self.candidate_id,
            "video_id": self.video_id,
            "video_path": self.video_path,
            "frame_index": self.frame_index,
            "time_seconds": round(self.time_seconds, 4),
            "camera_id": self.camera_id,
            "reason": self.reason,
            "priority": self.priority,
            "status": self.status,
            "notes": self.notes,
        }
        for k in OPTIONAL_COLUMNS:
            if k in self.extra:
                row[k] = self.extra[k]
        return row


class CandidateQueue:
    """Ordered list of candidates, loaded from CSV or JSON."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else None
        self._candidates: list[Candidate] = []
        self._by_id: dict[str, Candidate] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "CandidateQueue":
        path = Path(path)
        q = cls(path)
        if not path.exists():
            return q
        if path.suffix.lower() == ".json":
            q._load_json(path)
        else:
            q._load_csv(path)
        return q

    def _load_csv(self, path: Path) -> None:
        df = pd.read_csv(path, dtype=str).fillna("")
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Candidate CSV {path.name!r} is missing required columns: {missing}. "
                f"Required: {REQUIRED_COLUMNS}"
            )
        for _, row in df.iterrows():
            self._add_from_dict(row.to_dict())

    def _load_json(self, path: Path) -> None:
        data = json.loads(path.read_text())
        rows = data.get("candidates", data) if isinstance(data, dict) else data
        for row in rows:
            self._add_from_dict(row)

    def _add_from_dict(self, row: dict) -> None:
        def _s(k: str, d: str = "") -> str:
            v = row.get(k, d)
            return d if v is None or str(v) in ("nan", "None", "") else str(v)

        def _i(k: str, d: int = 0) -> int:
            try:
                return int(float(row.get(k, d)))
            except (TypeError, ValueError):
                return d

        def _f(k: str, d: float = 0.0) -> float:
            try:
                return float(row.get(k, d))
            except (TypeError, ValueError):
                return d

        cid = _s("candidate_id")
        if not cid:
            # Derive a stable id from video + frame if absent.
            cid = f"{_s('video_id', 'video')}_{_i('frame_index'):08d}"
        extra = {k: row[k] for k in OPTIONAL_COLUMNS if k in row and str(row[k]) not in ("", "nan", "None")}
        cand = Candidate(
            candidate_id=cid,
            video_id=_s("video_id"),
            video_path=_s("video_path"),
            frame_index=_i("frame_index"),
            time_seconds=_f("time_seconds"),
            camera_id=_s("camera_id"),
            reason=_s("reason"),
            priority=_i("priority"),
            status=_s("status", "unlabeled") or "unlabeled",
            notes=_s("notes"),
            extra=extra,
        )
        if cand.candidate_id in self._by_id:
            return
        self._candidates.append(cand)
        self._by_id[cand.candidate_id] = cand

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._candidates)

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def all(self) -> list[Candidate]:
        return list(self._candidates)

    def get(self, candidate_id: str) -> Optional[Candidate]:
        return self._by_id.get(candidate_id)

    def index_of(self, candidate_id: str) -> int:
        for i, c in enumerate(self._candidates):
            if c.candidate_id == candidate_id:
                return i
        return -1

    def set_status(self, candidate_id: str, status: str) -> None:
        c = self._by_id.get(candidate_id)
        if c is not None and status:
            c.status = status

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _find(self, start: int, step: int, predicate) -> Optional[Candidate]:
        n = len(self._candidates)
        if n == 0:
            return None
        i = start + step
        while 0 <= i < n:
            c = self._candidates[i]
            if predicate(c):
                return c
            i += step
        return None

    def next(self, current_id: Optional[str]) -> Optional[Candidate]:
        idx = self.index_of(current_id) if current_id else -1
        return self._find(idx, 1, lambda c: True)

    def previous(self, current_id: Optional[str]) -> Optional[Candidate]:
        idx = self.index_of(current_id) if current_id else len(self._candidates)
        return self._find(idx, -1, lambda c: True)

    def next_unlabeled(self, current_id: Optional[str]) -> Optional[Candidate]:
        idx = self.index_of(current_id) if current_id else -1
        return self._find(idx, 1, lambda c: c.status in ("unlabeled", "in_progress"))

    def next_with_status(self, current_id: Optional[str], status: str) -> Optional[Candidate]:
        idx = self.index_of(current_id) if current_id else -1
        return self._find(idx, 1, lambda c: c.status == status)

    def next_same_video(self, current_id: str) -> Optional[Candidate]:
        cur = self._by_id.get(current_id)
        if cur is None:
            return None
        idx = self.index_of(current_id)
        return self._find(idx, 1, lambda c: c.video_id == cur.video_id)

    def next_same_camera(self, current_id: str) -> Optional[Candidate]:
        cur = self._by_id.get(current_id)
        if cur is None or not cur.camera_id:
            return None
        idx = self.index_of(current_id)
        return self._find(idx, 1, lambda c: c.camera_id == cur.camera_id)

    def first_unlabeled(self) -> Optional[Candidate]:
        for c in self._candidates:
            if c.status in ("unlabeled", "in_progress"):
                return c
        return self._candidates[0] if self._candidates else None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        target = Path(path) if path else self._path
        if target is None:
            return
        rows = [c.to_row() for c in self._candidates]
        # Preserve any optional columns that appeared in the source file.
        cols = list(REQUIRED_COLUMNS)
        for k in OPTIONAL_COLUMNS:
            if any(k in r for r in rows):
                cols.append(k)
        df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
        atomic_write_text(target, df.to_csv(index=False))
