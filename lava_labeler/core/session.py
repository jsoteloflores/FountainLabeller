"""Session recovery file (session_recovery.json) and crash-resume support."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from lava_labeler.core.config import atomic_write_json


RECOVERY_NAME = "session_recovery.json"


class SessionRecovery:
    """Tracks enough state to resume a labeling session after a crash."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._path = self.root / RECOVERY_NAME
        self._state: dict[str, Any] = {
            "schema_version": "1.0",
            "video_path": "",
            "frame_index": 0,
            "candidate_id": "",
            "active_sample_id": "",
            "dirty": False,
            "last_saved": "",
            "recent_candidates": [],
            "candidate_queue_path": "",
        }
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._state.update(json.loads(self._path.read_text()))
            except (json.JSONDecodeError, OSError):
                pass

    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def update(self, **kwargs: Any) -> None:
        self._state.update(kwargs)

    def push_recent(self, candidate_id: str, limit: int = 20) -> None:
        if not candidate_id:
            return
        recent = [c for c in self._state.get("recent_candidates", []) if c != candidate_id]
        recent.insert(0, candidate_id)
        self._state["recent_candidates"] = recent[:limit]

    def mark_saved(self) -> None:
        self._state["dirty"] = False
        self._state["last_saved"] = datetime.now(timezone.utc).isoformat()

    def save(self) -> None:
        try:
            atomic_write_json(self._path, self._state)
        except OSError:
            pass

    def clear(self) -> None:
        try:
            if self._path.exists():
                self._path.unlink()
        except OSError:
            pass

    def has_resumable_state(self) -> bool:
        return bool(self._state.get("active_sample_id") or self._state.get("candidate_id"))
