"""Dataset summary: compute per-dataset/episode/camera/video statistics.

All counts are derived from the ``MetadataStore`` (frame metadata) and
optionally a ``CandidateQueue``.  The summary is the authoritative source
for the Dataset Context panel and the ``dataset_summary.csv`` mirror.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lava_labeler.core.metadata import MetadataStore
    from lava_labeler.core.candidates import CandidateQueue
    from lava_labeler.core.video_registry import VideoRegistry


SUMMARY_COLUMNS = [
    "scope", "scope_id",
    "total_candidates", "total_labeled", "total_unlabeled",
    "total_skipped", "total_needs_review",
    "positive_masks", "empty_confirmed", "hard_negative",
    "wind_affected", "falling_tephra_visible", "cooling_tephra_visible",
    "smoke_obscured", "ground_glow_visible", "exposure_bloom",
    "ambiguous_boundary",
    "video_count", "episode_id", "camera_id", "last_updated",
]

_TERMINAL_STATUSES = {"complete", "hard_negative", "empty_confirmed"}
_UNLABELED_STATUSES = {"queued", "in_progress", "unlabeled"}
_SKIPPED_STATUSES   = {"skipped", "bad_frame"}


@dataclass
class ScopeStats:
    scope: str      # "dataset" | "video" | "episode" | "camera"
    scope_id: str   # "all" | video_id | episode_id | camera_id

    total_candidates: int = 0
    total_labeled: int = 0
    total_unlabeled: int = 0
    total_skipped: int = 0
    total_needs_review: int = 0

    positive_masks: int = 0
    empty_confirmed: int = 0
    hard_negative: int = 0

    wind_affected: int = 0
    falling_tephra_visible: int = 0
    cooling_tephra_visible: int = 0
    smoke_obscured: int = 0
    ground_glow_visible: int = 0
    exposure_bloom: int = 0
    ambiguous_boundary: int = 0

    video_count: int = 0
    episode_id: str = ""
    camera_id: str = ""
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_csv_row(self) -> dict:
        return {
            "scope": self.scope,
            "scope_id": self.scope_id,
            "total_candidates": self.total_candidates,
            "total_labeled": self.total_labeled,
            "total_unlabeled": self.total_unlabeled,
            "total_skipped": self.total_skipped,
            "total_needs_review": self.total_needs_review,
            "positive_masks": self.positive_masks,
            "empty_confirmed": self.empty_confirmed,
            "hard_negative": self.hard_negative,
            "wind_affected": self.wind_affected,
            "falling_tephra_visible": self.falling_tephra_visible,
            "cooling_tephra_visible": self.cooling_tephra_visible,
            "smoke_obscured": self.smoke_obscured,
            "ground_glow_visible": self.ground_glow_visible,
            "exposure_bloom": self.exposure_bloom,
            "ambiguous_boundary": self.ambiguous_boundary,
            "video_count": self.video_count,
            "episode_id": self.episode_id,
            "camera_id": self.camera_id,
            "last_updated": self.last_updated,
        }


def _tally(stats: ScopeStats, rec: "FrameRecord") -> None:  # type: ignore[name-defined]
    """Add one FrameRecord's counts into *stats*."""
    stats.total_candidates += 1
    ls = rec.label_status or "queued"
    if ls in _TERMINAL_STATUSES:
        stats.total_labeled += 1
    elif ls in _SKIPPED_STATUSES:
        stats.total_skipped += 1
    elif ls == "needs_review":
        stats.total_needs_review += 1
    else:
        stats.total_unlabeled += 1

    if rec.mask_positive_pixels and rec.mask_positive_pixels > 0:
        stats.positive_masks += 1
    if rec.empty_mask_confirmed:
        stats.empty_confirmed += 1
    if rec.hard_negative:
        stats.hard_negative += 1
    if rec.wind_affected:
        stats.wind_affected += 1
    if rec.falling_tephra_visible:
        stats.falling_tephra_visible += 1
    if rec.cooling_tephra_visible:
        stats.cooling_tephra_visible += 1
    if rec.smoke_obscured:
        stats.smoke_obscured += 1
    if rec.ground_glow_visible:
        stats.ground_glow_visible += 1
    if rec.exposure_bloom:
        stats.exposure_bloom += 1
    if rec.ambiguous_boundary:
        stats.ambiguous_boundary += 1


class DatasetSummary:
    """Compute and expose dataset statistics for GUI and CSV export.

    Call :meth:`refresh` to rebuild all counts from the current
    ``MetadataStore``.

    Attributes
    ----------
    dataset : ScopeStats
        Whole-dataset counts.
    by_video : dict[str, ScopeStats]
        Keyed by ``video_id``.
    by_episode : dict[str, ScopeStats]
        Keyed by ``episode_id``.
    by_camera : dict[str, ScopeStats]
        Keyed by ``camera_id``.
    """

    def __init__(
        self,
        metadata: "MetadataStore",
        registry: "VideoRegistry | None" = None,
        candidates: "CandidateQueue | None" = None,
    ) -> None:
        self.metadata = metadata
        self.registry = registry
        self.candidates = candidates

        self.dataset: ScopeStats = ScopeStats(scope="dataset", scope_id="all")
        self.by_video: dict[str, ScopeStats] = {}
        self.by_episode: dict[str, ScopeStats] = {}
        self.by_camera: dict[str, ScopeStats] = {}
        self.refresh()

    def refresh(self) -> None:
        """Recompute all counts from current MetadataStore state."""
        now = datetime.now(timezone.utc).isoformat()
        self.dataset = ScopeStats(scope="dataset", scope_id="all", last_updated=now)
        self.by_video = {}
        self.by_episode = {}
        self.by_camera = {}

        for rec in self.metadata.all_records():
            vid   = rec.video_id   or "unknown_video"
            ep    = rec.episode_id or "unknownEpisode"
            cam   = rec.camera_id  or "unknownCamera"

            # Dataset-level
            _tally(self.dataset, rec)

            # Per-video
            if vid not in self.by_video:
                self.by_video[vid] = ScopeStats(
                    scope="video", scope_id=vid,
                    episode_id=ep, camera_id=cam, last_updated=now,
                )
            _tally(self.by_video[vid], rec)

            # Per-episode
            if ep not in self.by_episode:
                self.by_episode[ep] = ScopeStats(
                    scope="episode", scope_id=ep, episode_id=ep, last_updated=now,
                )
            _tally(self.by_episode[ep], rec)
            # Count distinct videos per episode
            self.by_episode[ep].video_count = len({
                r.video_id for r in self.metadata.all_records()
                if (r.episode_id or "unknownEpisode") == ep and r.video_id
            })

            # Per-camera
            if cam not in self.by_camera:
                self.by_camera[cam] = ScopeStats(
                    scope="camera", scope_id=cam, camera_id=cam, last_updated=now,
                )
            _tally(self.by_camera[cam], rec)

        # Dataset-level video count
        self.dataset.video_count = len({
            r.video_id for r in self.metadata.all_records() if r.video_id
        })

    def stats_for_video(self, video_id: str) -> ScopeStats | None:
        return self.by_video.get(video_id)

    def stats_for_episode(self, episode_id: str) -> ScopeStats | None:
        return self.by_episode.get(episode_id)

    def stats_for_camera(self, camera_id: str) -> ScopeStats | None:
        return self.by_camera.get(camera_id)

    def all_rows(self) -> list[ScopeStats]:
        """All scope rows in a stable order for CSV output."""
        rows: list[ScopeStats] = [self.dataset]
        rows += sorted(self.by_episode.values(), key=lambda s: s.scope_id)
        rows += sorted(self.by_camera.values(),  key=lambda s: s.scope_id)
        rows += sorted(self.by_video.values(),   key=lambda s: s.scope_id)
        return rows
