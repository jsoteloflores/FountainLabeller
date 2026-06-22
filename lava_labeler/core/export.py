"""Training-ready export: per-frame metadata JSON + labels_manifest.csv.

Produces, for every labeled/empty-confirmed frame:

* ``metadata/<sample_id>.json`` — schema-versioned per-frame metadata.
* ``labels_manifest.csv`` — one row per exported sample, flat columns the
  training repo can consume directly.
* ``export_validation_report.csv`` / ``.json`` — consistency checks flagging
  ambiguous empty masks, mismatched flags, and candidate-status discrepancies.

Frames/masks already live under images/all and masks/all; we reference them.
The distinction between "empty because intentionally labeled background" and
"empty because not labeled yet" is preserved via ``empty_mask_confirmed`` and
``mask_status``.
"""

from __future__ import annotations

import csv
import io
import json
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
    # Stage 2 required columns
    "image_path", "mask_path",
    "video_id", "video_filename", "episode_id", "camera_id",
    "frame_index", "time_seconds",
    "target_definition",
    "label_status", "mask_status", "mask_provenance",
    "hard_negative", "empty_mask_confirmed",
    "wind_affected", "falling_tephra_visible", "cooling_tephra_visible",
    "smoke_obscured", "ground_glow_visible", "exposure_bloom",
    "ambiguous_boundary", "needs_review",
    # Additional recommended columns
    "sample_id", "candidate_id",
    "mask_pixel_count", "has_positive_mask", "mask_area_fraction",
    "exportable", "validation_errors", "validation_warnings",
    "label_notes", "created_at", "updated_at",
]

VALIDATION_COLUMNS = [
    "frame_id", "video_id", "frame_index", "label_status",
    "mask_positive_pixels", "empty_mask_confirmed", "hard_negative",
    "validation_status", "warnings", "included_in_manifest",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_records(
    records: "list[FrameRecord]",
    exportable_ids: set[str] | None = None,
) -> list[dict]:
    """Check every record for consistency.  Returns a list of result dicts.

    Rules checked
    -------------
    1. Empty mask (0 px) without ``empty_mask_confirmed`` → warning.
    2. ``empty_mask_confirmed=True`` but mask has positive pixels → error.
    3. ``hard_negative=True`` but mask has positive pixels → warning.
    4. Candidate-status / label_status mismatch (via label_status alone; we
       don't have the candidate queue here so we check internal consistency):
       label_status in terminal set but no mask provenance set → warning.
    """
    results: list[dict] = []
    for rec in records:
        warnings: list[str] = []
        vstatus = "ok"
        pos_px = rec.mask_positive_pixels or 0
        included = rec.sample_id in (exportable_ids or set())

        # Rule 1: empty mask, unconfirmed
        if pos_px == 0 and not rec.empty_mask_confirmed and not rec.hard_negative:
            if rec.label_status in ("complete", "in_progress", "queued"):
                warnings.append(
                    "Mask is empty but empty_mask_confirmed=False — may be unlabeled."
                )
                vstatus = "warning"
                included = False  # exclude regardless of status

        # Rule 2: confirmed empty but positive pixels
        if rec.empty_mask_confirmed and pos_px > 0:
            warnings.append(
                "empty_mask_confirmed=True but mask has positive pixels. "
                "Clear the mask or unset empty_mask_confirmed."
            )
            vstatus = "error"
            included = False

        # Rule 3: hard_negative with positive pixels
        if rec.hard_negative and pos_px > 0:
            warnings.append(
                "hard_negative=True but mask has positive pixels. "
                "Confirm this is intentional or unset hard_negative."
            )
            if vstatus == "ok":
                vstatus = "warning"

        # Rule 4: terminal status with no provenance
        if rec.label_status in EXPORTABLE_STATUSES and not rec.mask_provenance:
            warnings.append(
                f"label_status={rec.label_status} but mask_provenance is not set."
            )
            if vstatus == "ok":
                vstatus = "warning"

        # Rule 5: wrong target_definition
        if rec.target_definition and rec.target_definition != TARGET_DEFINITION:
            warnings.append(
                f"target_definition='{rec.target_definition}' does not match "
                f"expected '{TARGET_DEFINITION}'."
            )
            if vstatus == "ok":
                vstatus = "warning"

        # Rule 6: missing video_id on exportable rows
        if rec.label_status in EXPORTABLE_STATUSES and not rec.video_id:
            warnings.append("video_id is missing on an exportable row.")
            if vstatus == "ok":
                vstatus = "warning"

        results.append({
            "frame_id": rec.sample_id,
            "video_id": rec.video_id,
            "frame_index": rec.frame_index,
            "label_status": rec.label_status,
            "mask_positive_pixels": pos_px,
            "empty_mask_confirmed": bool(rec.empty_mask_confirmed),
            "hard_negative": bool(rec.hard_negative),
            "validation_status": vstatus,
            "warnings": "; ".join(warnings),
            "included_in_manifest": included,
        })
    return results


def write_validation_report(dataset: "DatasetFolder", results: list[dict]) -> tuple[Path, Path]:
    """Write export_validation_report.json and .csv; return (json_path, csv_path)."""
    json_path = dataset.root / "export_validation_report.json"
    csv_path  = dataset.root / "export_validation_report.csv"

    atomic_write_json(json_path, {"schema_version": SCHEMA_VERSION, "rows": results})

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=VALIDATION_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    atomic_write_text(csv_path, buf.getvalue())

    return json_path, csv_path


# ---------------------------------------------------------------------------
# Per-frame metadata + manifest
# ---------------------------------------------------------------------------

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
    """Write per-frame JSON files, labels_manifest.csv, and validation report.

    Returns a summary dict:
    ``{"exported": n, "manifest": path, "skipped": n,
       "validation_warnings": n, "validation_report": path}``.
    """
    statuses = statuses or EXPORTABLE_STATUSES
    records = metadata.all_records()

    # --- Validation pass (all records) ---
    # Build the set of IDs that would be exported under normal rules.
    candidate_ids = {r.sample_id for r in records if r.label_status in statuses}
    val_results = validate_records(records, exportable_ids=candidate_ids)

    # Remove records that validation marked as excluded (Rule 1, 2).
    excluded_by_validation = {
        row["frame_id"] for row in val_results
        if not row["included_in_manifest"] and row["frame_id"] in candidate_ids
    }
    exportable: list["FrameRecord"] = [
        r for r in records
        if r.label_status in statuses and r.sample_id not in excluded_by_validation
    ]

    # Per-frame metadata JSON
    for rec in exportable:
        write_frame_metadata_json(dataset, rec)

    # labels_manifest.csv (flat)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for rec in exportable:
        pos_px = rec.mask_positive_pixels or 0
        total_px = (rec.source_width or 1) * (rec.source_height or 1)
        row = {
            "image_path": f"images/all/{rec.sample_id}.png",
            "mask_path": f"masks/all/{rec.sample_id}_mask.png",
            "video_id": rec.video_id,
            "video_filename": rec.video_filename or Path(rec.video_path).name,
            "episode_id": rec.episode_id,
            "camera_id": rec.camera_id,
            "frame_index": rec.frame_index,
            "time_seconds": round(rec.time_seconds, 4),
            "target_definition": rec.target_definition or TARGET_DEFINITION,
            "label_status": rec.label_status,
            "mask_status": rec.label_status,
            "mask_provenance": rec.mask_provenance,
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
            "sample_id": rec.sample_id,
            "candidate_id": rec.candidate_id,
            "mask_pixel_count": pos_px,
            "has_positive_mask": pos_px > 0,
            "mask_area_fraction": round(pos_px / total_px, 6) if total_px else 0.0,
            "exportable": True,
            "validation_errors": "",
            "validation_warnings": "",
            "label_notes": rec.notes,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
        }
        writer.writerow(row)

    manifest_path = dataset.root / "labels_manifest.csv"
    atomic_write_text(manifest_path, buf.getvalue())

    # Validation report
    json_report, csv_report = write_validation_report(dataset, val_results)
    warn_count = sum(1 for r in val_results if r["validation_status"] != "ok")

    return {
        "exported": len(exportable),
        "manifest": manifest_path,
        "skipped": len(records) - len(exportable),
        "validation_warnings": warn_count,
        "validation_report": csv_report,
    }

