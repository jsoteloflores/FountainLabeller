"""Lightweight CSV session logging for labeling diagnostics."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_HEADER = ["timestamp", "event_type", "candidate_id", "video_id", "frame_index", "details"]


class SessionLogger:
    """Appends one row per event to logs/session_log.csv."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._path = self.root / "logs" / "session_log.csv"
        self._ensure_header()

    def _ensure_header(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if not self._path.exists():
                with self._path.open("w", newline="") as f:
                    csv.writer(f).writerow(_HEADER)
        except OSError:
            pass

    def log(
        self,
        event_type: str,
        candidate_id: str = "",
        video_id: str = "",
        frame_index: Optional[int] = None,
        details: str = "",
    ) -> None:
        try:
            with self._path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now(timezone.utc).isoformat(),
                    event_type,
                    candidate_id,
                    video_id,
                    "" if frame_index is None else frame_index,
                    details,
                ])
        except OSError:
            pass
