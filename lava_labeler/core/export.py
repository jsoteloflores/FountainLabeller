"""Training-ready export: per-frame metadata JSON + labels_manifest.csv.

Produces, for every labeled/empty-confirmed frame:

* ``metadata/<sample_id>.json`` — schema-versioned per-frame metadata.
* ``labels_manifest.csv`` — one row per exported sample, flat columns the
  training repo can consume directly.

Frames/masks already live under images/all and masks/all; we reference them.
The distinction between "empty because intentionally labeled background" and
"empty because not labeled yet" is preserved via ``empty_mask_confirmed`` and
``mask_status``.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import TYPE_CHECKING

from lava_labeler.core.config import atomic_write_json, atomic_write_text
from lava_labeler.core.metadata import METADATA_FLAGS, TARGET_DEFINITION

if TYPE_CHECKING:
    from lava_labeler.core.dataset import DatasetFolder
    from lava_labeler.core.metadata import FrameRecord, MetadataStore


SCHEMA_VERSION = "1.0"

# Statuses considered "exportable" training samples.
EXPORTABLE_STATUSES = {"complete", "hard_negative", "empty_confirmed"}

MANIFEST_COLUMNS = [
    "sample_id", "video_id", "video_path", "frame_index", "time_seconds",
    "camera_id", "image_path", "mask_path", "mask_status", "mask_provenance",
    "target_definition",
] + METADATA_FLAGS + ["label_notes", "created_at", "updated_at"]


def _frame_metadata_dict(rec: "FrameRecord") -> dict:
    img_path = f"images/all/{rec.sample_id}.png"
    msk_path = f"masks/all/{rec.sample_id}_mask.png"
    flags = {flag: bool(getattr(rec, flag)) for flag in METADATA_FLAGS}
    return {
        "schema_version": SCHEMA_VERSION,
        "target_definition": rec.target_definition or TARGET_DEFINITION,
        "candidate_id": rec.candidate_id,
        "video_id": rec.video_id,
        "video_path": rec.video_path,
        "frame_index": rec.frame_index,
        "time_seconds": round(rec.time_seconds, 4),
        "camera_id": rec.camera_id,
        "image_path": img_path,
        "mask_path": msk_path,
        "mask_status": rec.label_status,
        "mask_provenance": rec.mask_provenance,
        "metadata_flags": flags,
        "label_notes": rec.notes,
        "created_at": rec.created_at,
        "updated_at": rec.updated_at,
    }


def write_frame_metadata_json(dataset: "DatasetFolder", rec: "FrameRecord") -> Path:
    """Write the per-frame metadata JSON for a single record."""
    path = dataset.root / "metadata" / f"{rec.sample_id}.json"
    atomic_write_json(path, _frame_metadata_dict(rec))
    return path


def export_dataset(
    dataset: "DatasetFolder",
    metadata: "MetadataStore",
    statuses: set[str] | None = None,
) -> dict:
    """Write per-frame JSON files and labels_manifest.csv.

    Returns a summary dict: {"exported": n, "manifest": path, "skipped": n}.
    """
    statuses = statuses or EXPORTABLE_STATUSES
    records = metadata.all_records()

    exportable: list["FrameRecord"] = [r for r in records if r.label_status in statuses]

    # Per-frame metadata JSON
    for rec in exportable:
        write_frame_metadata_json(dataset, rec)

    # labels_manifest.csv (flat)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for rec in exportable:
        row = {
            "sample_id": rec.sample_id,
            "video_id": rec.video_id,
            "video_path": rec.video_path,
            "frame_index": rec.frame_index,
            "time_seconds": round(rec.time_seconds, 4),
            "camera_id": rec.camera_id,
            "image_path": f"images/all/{rec.sample_id}.png",
            "mask_path": f"masks/all/{rec.sample_id}_mask.png",
            "mask_status": rec.label_status,
            "mask_provenance": rec.mask_provenance,
            "target_definition": rec.target_definition or TARGET_DEFINITION,
            "label_notes": rec.notes,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
        }
        for flag in METADATA_FLAGS:
            row[flag] = bool(getattr(rec, flag))
        writer.writerow(row)

    manifest_path = dataset.root / "labels_manifest.csv"
    atomic_write_text(manifest_path, buf.getvalue())

    return {
        "exported": len(exportable),
        "manifest": manifest_path,
        "skipped": len(records) - len(exportable),
    }
