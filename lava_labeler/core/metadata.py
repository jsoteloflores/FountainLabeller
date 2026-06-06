"""FrameRecord dataclass and MetadataStore (frames.csv)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


LABEL_STATUSES = ["queued", "in_progress", "complete", "uncertain", "needs_review"]
DIFFICULTIES = ["easy", "medium", "hard", "unknown"]
LIGHTING_CONDITIONS = ["night", "dawn", "day", "dusk", "unknown"]

_FRAMES_CSV = "metadata/frames.csv"

COLUMNS = [
    "sample_id", "image_path", "mask_path",
    "video_path", "video_id", "episode_id", "camera_id",
    "frame_index", "time_seconds",
    "source_width", "source_height", "export_width", "export_height",
    "roi_x", "roi_y", "roi_width", "roi_height", "is_roi_crop",
    "label_status", "labeler", "notes", "difficulty", "lighting_condition",
    "contains_tephra", "contains_smoke", "contains_base_glow",
    "mask_positive_pixels", "mask_positive_fraction",
    "created_at", "updated_at",
]


@dataclass
class FrameRecord:
    sample_id: str
    video_path: str
    frame_index: int
    source_width: int
    source_height: int
    fps: float = 25.0
    video_id: str = ""
    episode_id: str = "unknownEpisode"
    camera_id: str = "unknownCamera"
    label_status: str = "queued"
    labeler: str = ""
    notes: str = ""
    difficulty: str = "unknown"
    lighting_condition: str = "unknown"
    contains_tephra: bool = False
    contains_smoke: bool = False
    contains_base_glow: bool = False
    mask_positive_pixels: int = 0
    mask_positive_fraction: float = 0.0
    roi_x: Optional[int] = None
    roi_y: Optional[int] = None
    roi_width: Optional[int] = None
    roi_height: Optional[int] = None
    is_roi_crop: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def time_seconds(self) -> float:
        return self.frame_index / self.fps if self.fps > 0 else 0.0

    @property
    def export_width(self) -> int:
        return self.roi_width if (self.is_roi_crop and self.roi_width) else self.source_width

    @property
    def export_height(self) -> int:
        return self.roi_height if (self.is_roi_crop and self.roi_height) else self.source_height

    def to_row(self, dataset_root: Path) -> dict:
        img_path = str(dataset_root / "images" / "all" / f"{self.sample_id}.png")
        msk_path = str(dataset_root / "masks" / "all" / f"{self.sample_id}_mask.png")
        return {
            "sample_id": self.sample_id,
            "image_path": img_path,
            "mask_path": msk_path,
            "video_path": self.video_path,
            "video_id": self.video_id,
            "episode_id": self.episode_id,
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "time_seconds": round(self.time_seconds, 4),
            "source_width": self.source_width,
            "source_height": self.source_height,
            "export_width": self.export_width,
            "export_height": self.export_height,
            "roi_x": self.roi_x,
            "roi_y": self.roi_y,
            "roi_width": self.roi_width,
            "roi_height": self.roi_height,
            "is_roi_crop": self.is_roi_crop,
            "label_status": self.label_status,
            "labeler": self.labeler,
            "notes": self.notes,
            "difficulty": self.difficulty,
            "lighting_condition": self.lighting_condition,
            "contains_tephra": self.contains_tephra,
            "contains_smoke": self.contains_smoke,
            "contains_base_glow": self.contains_base_glow,
            "mask_positive_pixels": self.mask_positive_pixels,
            "mask_positive_fraction": round(self.mask_positive_fraction, 6),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class MetadataStore:
    """Load and save frames.csv for a dataset folder."""

    def __init__(self, dataset_root: Path) -> None:
        self.root = dataset_root
        self._csv_path = dataset_root / _FRAMES_CSV
        self._records: dict[str, FrameRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._csv_path.exists():
            return
        df = pd.read_csv(self._csv_path, dtype=str)
        for _, row in df.iterrows():
            rec = self._row_to_record(row.to_dict())
            self._records[rec.sample_id] = rec

    @staticmethod
    def _row_to_record(row: dict) -> FrameRecord:
        def _s(k: str, default: str = "") -> str:
            v = row.get(k, "")
            return default if (v is None or str(v) in ("nan", "None", "")) else str(v)

        def _i(k: str, default: int = 0) -> int:
            try:
                return int(float(row.get(k, default)))
            except (TypeError, ValueError):
                return default

        def _f(k: str, default: float = 0.0) -> float:
            try:
                return float(row.get(k, default))
            except (TypeError, ValueError):
                return default

        def _b(k: str, default: bool = False) -> bool:
            v = row.get(k, "")
            if isinstance(v, bool):
                return v
            return str(v).lower() in ("true", "1", "yes")

        def _opt_i(k: str) -> Optional[int]:
            v = row.get(k, "")
            if v is None or str(v) in ("nan", "None", ""):
                return None
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None

        return FrameRecord(
            sample_id=_s("sample_id"),
            video_path=_s("video_path"),
            frame_index=_i("frame_index"),
            source_width=_i("source_width"),
            source_height=_i("source_height"),
            video_id=_s("video_id"),
            episode_id=_s("episode_id", "unknownEpisode"),
            camera_id=_s("camera_id", "unknownCamera"),
            label_status=_s("label_status", "queued"),
            labeler=_s("labeler"),
            notes=_s("notes"),
            difficulty=_s("difficulty", "unknown"),
            lighting_condition=_s("lighting_condition", "unknown"),
            contains_tephra=_b("contains_tephra"),
            contains_smoke=_b("contains_smoke"),
            contains_base_glow=_b("contains_base_glow"),
            mask_positive_pixels=_i("mask_positive_pixels"),
            mask_positive_fraction=_f("mask_positive_fraction"),
            roi_x=_opt_i("roi_x"),
            roi_y=_opt_i("roi_y"),
            roi_width=_opt_i("roi_width"),
            roi_height=_opt_i("roi_height"),
            is_roi_crop=_b("is_roi_crop"),
            created_at=_s("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=_s("updated_at", datetime.now(timezone.utc).isoformat()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: FrameRecord) -> None:
        self._records[record.sample_id] = record

    def get(self, sample_id: str) -> FrameRecord | None:
        return self._records.get(sample_id)

    def update(self, sample_id: str, **kwargs) -> None:
        rec = self._records.get(sample_id)
        if rec is None:
            return
        for k, v in kwargs.items():
            if hasattr(rec, k):
                setattr(rec, k, v)
        rec.updated_at = datetime.now(timezone.utc).isoformat()

    def remove(self, sample_id: str) -> None:
        self._records.pop(sample_id, None)

    def all_records(self) -> list[FrameRecord]:
        return list(self._records.values())

    def save(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_row(self.root) for r in self._records.values()]
        df = pd.DataFrame(rows, columns=COLUMNS) if rows else pd.DataFrame(columns=COLUMNS)
        df.to_csv(self._csv_path, index=False)
