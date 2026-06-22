"""CSV mirror: regenerate Excel-friendly CSV snapshots from canonical JSON/store.

Three output files are managed:

``video_registry.csv``   — one row per registered video
``frame_metadata.csv``   — one row per labeled/candidate frame (richer than
                           the internal frames.csv)
``dataset_summary.csv``  — one row per scope (dataset / episode / camera / video)

All writes are atomic (temp-then-replace).  A debounce timer in the app
prevents regeneration on every keystroke.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from lava_labeler.core.config import atomic_write_text
from lava_labeler.core.metadata import METADATA_FLAGS, TARGET_DEFINITION
from lava_labeler.core.video_registry import REGISTRY_COLUMNS
from lava_labeler.core.dataset_summary import SUMMARY_COLUMNS

if TYPE_CHECKING:
    from lava_labeler.core.metadata import MetadataStore
    from lava_labeler.core.video_registry import VideoRegistry
    from lava_labeler.core.dataset_summary import DatasetSummary


# ---------------------------------------------------------------------------
# frame_metadata.csv columns
# ---------------------------------------------------------------------------

FRAME_METADATA_COLUMNS = [
    "label_id", "video_id", "video_filename", "episode_id", "camera_id",
    "eruption_id",
    "frame_index", "time_seconds",
    "image_path", "mask_path",
    "target_definition",
    "candidate_id", "candidate_status",
    "label_status", "mask_status", "mask_provenance",
    "has_positive_mask", "mask_pixel_count", "mask_area_fraction",
    "exportable",
    "hard_negative", "empty_mask_confirmed",
    "wind_affected", "falling_tephra_visible", "cooling_tephra_visible",
    "smoke_obscured", "ground_glow_visible", "exposure_bloom",
    "ambiguous_boundary", "needs_review",
    "labeler_notes",
    "created_at", "updated_at",
]

_EXPORTABLE_STATUSES = {"complete", "hard_negative", "empty_confirmed"}


# ---------------------------------------------------------------------------
# Individual writers
# ---------------------------------------------------------------------------

def write_video_registry_csv(registry: "VideoRegistry", path: Path) -> None:
    """Write video_registry.csv from the registry."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REGISTRY_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for entry in registry.all_entries():
        w.writerow(entry.to_csv_row())
    atomic_write_text(path, buf.getvalue())


def write_frame_metadata_csv(
    metadata: "MetadataStore",
    registry: "VideoRegistry | None",
    path: Path,
) -> None:
    """Write frame_metadata.csv with richer columns than internal frames.csv."""
    # Build a lookup for video entries
    vid_lookup: dict[str, object] = {}
    if registry:
        for entry in registry.all_entries():
            vid_lookup[entry.video_id] = entry

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FRAME_METADATA_COLUMNS, extrasaction="ignore")
    w.writeheader()

    for rec in metadata.all_records():
        ve = vid_lookup.get(rec.video_id)
        video_filename = (ve.video_filename if ve else "") or Path(rec.video_path).name
        eruption_id = (ve.eruption_id if ve else "") or ""

        exportable = (
            rec.label_status in _EXPORTABLE_STATUSES
            and not (rec.empty_mask_confirmed and (rec.mask_positive_pixels or 0) > 0)
            and not (rec.mask_positive_pixels == 0 and not rec.empty_mask_confirmed
                     and not rec.hard_negative and rec.label_status == "complete")
        )

        row = {
            "label_id": rec.sample_id,
            "video_id": rec.video_id,
            "video_filename": video_filename,
            "episode_id": rec.episode_id,
            "camera_id": rec.camera_id,
            "eruption_id": eruption_id,
            "frame_index": rec.frame_index,
            "time_seconds": round(rec.time_seconds, 4),
            "image_path": f"images/all/{rec.sample_id}.png",
            "mask_path": f"masks/all/{rec.sample_id}_mask.png",
            "target_definition": rec.target_definition or TARGET_DEFINITION,
            "candidate_id": rec.candidate_id,
            "candidate_status": "",  # resolved below if queue available
            "label_status": rec.label_status,
            "mask_status": rec.label_status,
            "mask_provenance": rec.mask_provenance,
            "has_positive_mask": (rec.mask_positive_pixels or 0) > 0,
            "mask_pixel_count": rec.mask_positive_pixels or 0,
            "mask_area_fraction": round(rec.mask_positive_fraction or 0.0, 6),
            "exportable": exportable,
            "hard_negative": rec.hard_negative,
            "empty_mask_confirmed": rec.empty_mask_confirmed,
            "wind_affected": rec.wind_affected,
            "falling_tephra_visible": rec.falling_tephra_visible,
            "cooling_tephra_visible": rec.cooling_tephra_visible,
            "smoke_obscured": rec.smoke_obscured,
            "ground_glow_visible": rec.ground_glow_visible,
            "exposure_bloom": rec.exposure_bloom,
            "ambiguous_boundary": rec.ambiguous_boundary,
            "needs_review": rec.needs_review,
            "labeler_notes": rec.notes,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
        }
        w.writerow(row)

    atomic_write_text(path, buf.getvalue())


def write_dataset_summary_csv(summary: "DatasetSummary", path: Path) -> None:
    """Write dataset_summary.csv from a DatasetSummary."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for stats in summary.all_rows():
        w.writerow(stats.to_csv_row())
    atomic_write_text(path, buf.getvalue())


# ---------------------------------------------------------------------------
# All-at-once regeneration
# ---------------------------------------------------------------------------

def regenerate_all(
    root: Path,
    metadata: "MetadataStore",
    registry: "VideoRegistry | None" = None,
    summary: "DatasetSummary | None" = None,
) -> dict[str, Path]:
    """Regenerate all three CSV mirrors.  Returns ``{name: path}``."""
    outputs: dict[str, Path] = {}

    if registry is not None:
        p = root / "video_registry.csv"
        write_video_registry_csv(registry, p)
        outputs["video_registry"] = p

    p = root / "frame_metadata.csv"
    write_frame_metadata_csv(metadata, registry, p)
    outputs["frame_metadata"] = p

    if summary is not None:
        summary.refresh()
        p = root / "dataset_summary.csv"
        write_dataset_summary_csv(summary, p)
        outputs["dataset_summary"] = p

    return outputs
