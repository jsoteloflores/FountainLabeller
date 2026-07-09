"""2.5D temporal dataset export.

Packages labeled center frames, masks, metadata, and surrounding temporal
frames (using the center frame's ROI) so a downstream trainer (KFS) can build
2.5D tensors without guessing.

The center frame metadata is always the authority for:

    video_id / video_path / video_filename / frame_index / fps / ROI / dims

Nothing in the canonical workspace (``metadata/frames.csv``, masks, images) is
modified by export.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from lava_labeler.core.config import atomic_write_json, atomic_write_text
from lava_labeler.core.metadata import METADATA_FLAGS
from lava_labeler.core.video_io import VideoReader

if TYPE_CHECKING:
    from lava_labeler.core.dataset import DatasetFolder
    from lava_labeler.core.metadata import FrameRecord, MetadataStore
    from lava_labeler.core.video_registry import VideoRegistry


EXPORTABLE_STATUSES: tuple[str, ...] = ("complete", "hard_negative", "empty_confirmed")

WINDOW_MODES = ("centered", "causal")
EDGE_POLICIES = ("skip", "replicate")


# ---------------------------------------------------------------------------
# Config + neighbor dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TemporalExportConfig:
    output_root: Path
    temporal_radius: int = 2
    temporal_stride: int = 1
    window_mode: str = "centered"       # centered | causal
    edge_policy: str = "skip"           # skip | replicate
    statuses: tuple[str, ...] = EXPORTABLE_STATUSES
    include_qc_contact_sheets: bool = True
    overwrite_existing: bool = False


@dataclass
class TemporalNeighbor:
    offset_label: str
    offset_index: int
    frame_index: int
    valid: bool
    path: str
    message: str = ""


# ---------------------------------------------------------------------------
# Offset math
# ---------------------------------------------------------------------------

def temporal_offsets(radius: int, stride: int, mode: str) -> list[int]:
    """Return frame offsets for a centered or causal temporal window.

    >>> temporal_offsets(2, 1, "centered")
    [-2, -1, 0, 1, 2]
    >>> temporal_offsets(4, 1, "causal")
    [-4, -3, -2, -1, 0]
    """
    if radius < 0:
        raise ValueError("temporal_radius must be >= 0")
    if stride < 1:
        raise ValueError("temporal_stride must be >= 1")
    if mode == "centered":
        return [i * stride for i in range(-radius, radius + 1)]
    if mode == "causal":
        return [i * stride for i in range(-radius, 1)]
    raise ValueError(f"Unknown window_mode: {mode!r} (expected 'centered' or 'causal')")


def offset_label(offset: int) -> str:
    """Return a path-safe offset label.

    >>> offset_label(-2), offset_label(0), offset_label(2)
    ('offset_m002', 'offset_0000', 'offset_p002')
    """
    if offset == 0:
        return "offset_0000"
    sign = "m" if offset < 0 else "p"
    return f"offset_{sign}{abs(offset):03d}"


# ---------------------------------------------------------------------------
# ROI cropping
# ---------------------------------------------------------------------------

def crop_frame_for_record(frame_bgr: np.ndarray, rec: "FrameRecord") -> np.ndarray:
    """Apply the record's ROI if ``is_roi_crop`` is set; otherwise return full frame."""
    if not rec.is_roi_crop:
        return frame_bgr
    if None in (rec.roi_x, rec.roi_y, rec.roi_width, rec.roi_height):
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    x = max(0, int(rec.roi_x))
    y = max(0, int(rec.roi_y))
    x2 = min(w, x + int(rec.roi_width))
    y2 = min(h, y + int(rec.roi_height))
    return frame_bgr[y:y2, x:x2]


# ---------------------------------------------------------------------------
# Readiness checking
# ---------------------------------------------------------------------------

READINESS_COLUMNS = [
    "sample_id", "ready",
    "video_link_ok", "video_id_present", "video_path_exists",
    "fps_present", "frame_index_valid", "roi_valid",
    "center_image_exists", "center_mask_exists", "neighbors_available",
    "reason",
]


def _resolve_video_path(rec: "FrameRecord") -> Optional[Path]:
    if rec.video_path and Path(rec.video_path).is_file():
        return Path(rec.video_path)
    return None


def check_temporal_readiness(
    dataset: "DatasetFolder",
    metadata: "MetadataStore",
    config: TemporalExportConfig,
) -> tuple[list[dict], dict]:
    """Return (per-row readiness dicts, summary) for exportable label rows.

    Opens each resolvable source video once to check frame bounds.
    """
    offsets = temporal_offsets(config.temporal_radius, config.temporal_stride, config.window_mode)
    frame_count_cache: dict[str, int] = {}

    rows: list[dict] = []
    summary = {
        "total_rows_checked": 0,
        "exportable_label_rows": 0,
        "ready": 0,
        "missing_video_id": 0,
        "missing_video_file": 0,
        "missing_fps": 0,
        "invalid_roi": 0,
        "edge_window_skipped": 0,
        "missing_center_image": 0,
        "missing_center_mask": 0,
    }

    for rec in metadata.all_records():
        summary["total_rows_checked"] += 1
        if rec.label_status not in config.statuses:
            continue
        summary["exportable_label_rows"] += 1

        reasons: list[str] = []
        video_id_present = bool(rec.video_id)
        vpath = _resolve_video_path(rec)
        video_path_exists = vpath is not None
        fps_present = bool(rec.fps and rec.fps > 0)
        video_link_ok = video_id_present and video_path_exists

        if not video_id_present:
            reasons.append("missing video_id")
            summary["missing_video_id"] += 1
        if not video_path_exists:
            reasons.append("video file not found")
            summary["missing_video_file"] += 1
        if not fps_present:
            reasons.append("missing fps")
            summary["missing_fps"] += 1

        frame_index_valid = False
        total_frames = 0
        if vpath is not None:
            key = str(vpath)
            if key not in frame_count_cache:
                try:
                    reader = VideoReader(vpath)
                    frame_count_cache[key] = reader.info.frame_count
                    reader.close()
                except Exception:  # noqa: BLE001
                    frame_count_cache[key] = 0
            total_frames = frame_count_cache[key]
            frame_index_valid = 0 <= rec.frame_index < total_frames
            if not frame_index_valid:
                reasons.append("frame_index out of bounds")

        roi_valid = _roi_is_valid(rec)
        if not roi_valid:
            reasons.append("invalid ROI")
            summary["invalid_roi"] += 1

        center_image_exists = dataset.image_path(rec.sample_id).exists()
        center_mask_exists = dataset.mask_path(rec.sample_id).exists()
        if not center_image_exists:
            reasons.append("center image missing")
            summary["missing_center_image"] += 1
        if not center_mask_exists:
            reasons.append("center mask missing")
            summary["missing_center_mask"] += 1

        # Neighbor availability under edge policy.
        neighbors_available = True
        if video_path_exists and frame_index_valid:
            for off in offsets:
                ni = rec.frame_index + off
                if not (0 <= ni < total_frames):
                    if config.edge_policy == "skip":
                        neighbors_available = False
                        break
            if not neighbors_available:
                reasons.append("edge neighbor out of bounds")
                summary["edge_window_skipped"] += 1

        ready = (
            video_link_ok and fps_present and frame_index_valid and roi_valid
            and center_image_exists and center_mask_exists and neighbors_available
        )
        if ready:
            summary["ready"] += 1

        rows.append({
            "sample_id": rec.sample_id,
            "ready": ready,
            "video_link_ok": video_link_ok,
            "video_id_present": video_id_present,
            "video_path_exists": video_path_exists,
            "fps_present": fps_present,
            "frame_index_valid": frame_index_valid,
            "roi_valid": roi_valid,
            "center_image_exists": center_image_exists,
            "center_mask_exists": center_mask_exists,
            "neighbors_available": neighbors_available,
            "reason": "; ".join(reasons),
        })

    return rows, summary


def _roi_is_valid(rec: "FrameRecord") -> bool:
    if not rec.is_roi_crop:
        return True
    if None in (rec.roi_x, rec.roi_y, rec.roi_width, rec.roi_height):
        return False
    if rec.roi_width <= 0 or rec.roi_height <= 0:
        return False
    if rec.roi_x < 0 or rec.roi_y < 0:
        return False
    if rec.source_width and rec.source_height:
        if rec.roi_x + rec.roi_width > rec.source_width:
            return False
        if rec.roi_y + rec.roi_height > rec.source_height:
            return False
    return True


# ---------------------------------------------------------------------------
# Manifest columns
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS = [
    "sample_id",
    "center_image_path", "center_mask_path",
    "center_frame_index", "fps", "time_seconds",
    "video_id", "video_filename", "video_path",
    "video_total_frames", "video_duration_seconds",
    "episode_id", "camera_id", "candidate_id", "target_definition",
    "label_status", "mask_provenance",
    "mask_positive_pixels", "mask_positive_fraction",
    "source_width", "source_height", "export_width", "export_height",
    "is_roi_crop", "roi_x", "roi_y", "roi_width", "roi_height",
    "roi_mode", "roi_size_policy",
    "temporal_radius", "temporal_stride", "window_mode", "edge_policy",
    "temporal_offsets_json", "temporal_frame_indices_json",
    "temporal_frame_paths_json", "temporal_valid_json",
    "all_temporal_frames_valid",
    "split_group_id",
    "metadata_flags_json",
    "notes", "created_at", "updated_at",
]

EXPORT_REPORT_COLUMNS = [
    "sample_id", "status", "exported", "reason",
    "center_frame_index", "video_id", "video_filename",
    "n_temporal_frames", "n_valid_temporal_frames",
]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_temporal_dataset(
    dataset: "DatasetFolder",
    metadata: "MetadataStore",
    video_registry: "Optional[VideoRegistry]",
    config: TemporalExportConfig,
) -> dict:
    """Export a 2.5D temporal dataset package. Returns a summary dict.

    Does NOT modify the canonical workspace.
    """
    if config.window_mode not in WINDOW_MODES:
        raise ValueError(f"Invalid window_mode: {config.window_mode}")
    if config.edge_policy not in EDGE_POLICIES:
        raise ValueError(f"Invalid edge_policy: {config.edge_policy}")

    out = Path(config.output_root)
    if out.exists() and any(out.iterdir()) and not config.overwrite_existing:
        raise FileExistsError(
            f"Output folder is not empty: {out}. "
            f"Enable overwrite to write into it."
        )

    # Create output layout.
    dirs = {
        "metadata": out / "metadata",
        "images": out / "images" / "all",
        "masks": out / "masks" / "all",
        "temporal": out / "temporal_frames",
        "qc": out / "qc" / "contact_sheets",
        "reports": out / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    offsets = temporal_offsets(config.temporal_radius, config.temporal_stride, config.window_mode)

    manifest_rows: list[dict] = []
    report_rows: list[dict] = []
    reader_cache: dict[str, VideoReader] = {}
    frame_count_cache: dict[str, int] = {}

    exported = 0
    skipped = 0

    try:
        for rec in metadata.all_records():
            if rec.label_status not in config.statuses:
                continue

            ok, reason, manifest_row = _export_sample(
                rec, dataset, config, offsets, dirs,
                reader_cache, frame_count_cache, video_registry,
            )
            if ok:
                exported += 1
                manifest_rows.append(manifest_row)
                report_rows.append({
                    "sample_id": rec.sample_id, "status": "exported", "exported": True,
                    "reason": "", "center_frame_index": rec.frame_index,
                    "video_id": rec.video_id, "video_filename": rec.video_filename,
                    "n_temporal_frames": len(offsets),
                    "n_valid_temporal_frames": manifest_row["_n_valid"],
                })
            else:
                skipped += 1
                report_rows.append({
                    "sample_id": rec.sample_id, "status": "skipped", "exported": False,
                    "reason": reason, "center_frame_index": rec.frame_index,
                    "video_id": rec.video_id, "video_filename": rec.video_filename,
                    "n_temporal_frames": len(offsets), "n_valid_temporal_frames": 0,
                })
    finally:
        for reader in reader_cache.values():
            reader.close()

    # Strip private helper keys before writing the manifest.
    for row in manifest_rows:
        row.pop("_n_valid", None)

    # Write manifests + copies + reports.
    _write_manifest(dirs["metadata"].parent, manifest_rows)
    _copy_workspace_metadata(dataset, video_registry, dirs["metadata"])

    readiness_rows, readiness_summary = check_temporal_readiness(dataset, metadata, config)
    _write_csv(dirs["reports"] / "temporal_readiness_report.csv", READINESS_COLUMNS, readiness_rows)
    _write_csv(dirs["reports"] / "temporal_export_report.csv", EXPORT_REPORT_COLUMNS, report_rows)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(out),
        "temporal_radius": config.temporal_radius,
        "temporal_stride": config.temporal_stride,
        "window_mode": config.window_mode,
        "edge_policy": config.edge_policy,
        "statuses": list(config.statuses),
        "samples_exported": exported,
        "samples_skipped": skipped,
        "temporal_frames_per_sample": len(offsets),
        "readiness": readiness_summary,
    }
    atomic_write_json(out / "export_summary.json", summary)
    _write_export_config(out / "export_config.json", config)

    return summary


def _export_sample(
    rec: "FrameRecord",
    dataset: "DatasetFolder",
    config: TemporalExportConfig,
    offsets: list[int],
    dirs: dict[str, Path],
    reader_cache: dict[str, VideoReader],
    frame_count_cache: dict[str, int],
    video_registry: "Optional[VideoRegistry]",
) -> tuple[bool, str, dict]:
    # Validate mask emptiness policy.
    empty_ok = (
        rec.empty_mask_confirmed or rec.hard_negative
        or rec.label_status in ("hard_negative", "empty_confirmed")
    )

    center_img_path = dataset.image_path(rec.sample_id)
    center_mask_path = dataset.mask_path(rec.sample_id)
    if not center_img_path.exists():
        return False, "center image missing", {}
    if not center_mask_path.exists():
        return False, "center mask missing", {}

    center_img = cv2.imread(str(center_img_path))
    center_mask = cv2.imread(str(center_mask_path), cv2.IMREAD_GRAYSCALE)
    if center_img is None or center_mask is None:
        return False, "failed to read center image/mask", {}
    if center_img.shape[:2] != center_mask.shape[:2]:
        return False, "mask/image dimension mismatch", {}

    positive = int(np.count_nonzero(center_mask))
    if positive == 0 and not empty_ok:
        return False, "empty mask not confirmed", {}

    if not _roi_is_valid(rec):
        return False, "invalid ROI", {}

    vpath = _resolve_video_path(rec)
    if vpath is None:
        return False, "video file not found", {}
    if not (rec.fps and rec.fps > 0):
        return False, "missing fps", {}

    key = str(vpath)
    if key not in reader_cache:
        try:
            reader_cache[key] = VideoReader(vpath)
            frame_count_cache[key] = reader_cache[key].info.frame_count
        except Exception as exc:  # noqa: BLE001
            return False, f"failed to open video: {exc}", {}
    reader = reader_cache[key]
    total_frames = frame_count_cache[key]

    if not (0 <= rec.frame_index < total_frames):
        return False, "center frame_index out of bounds", {}

    # Resolve neighbor frame indices under the edge policy.
    neighbors: list[TemporalNeighbor] = []
    for off in offsets:
        target = rec.frame_index + off
        valid = 0 <= target < total_frames
        resolved_index = target
        message = ""
        if not valid:
            if config.edge_policy == "skip":
                return False, f"neighbor index out of bounds (offset {off})", {}
            # replicate — clamp to nearest valid frame
            resolved_index = max(0, min(total_frames - 1, target))
            message = "replicated"
        neighbors.append(TemporalNeighbor(
            offset_label=offset_label(off),
            offset_index=off,
            frame_index=resolved_index,
            valid=valid,
            path="",
            message=message,
        ))

    # Extract + crop + write neighbor frames.
    sample_dir = dirs["temporal"] / rec.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    center_h, center_w = center_img.shape[:2]

    for nb in neighbors:
        frame = reader.read_frame(nb.frame_index)
        if frame is None:
            return False, f"failed to read frame {nb.frame_index}", {}
        crop = crop_frame_for_record(frame, rec)
        if crop.shape[:2] != (center_h, center_w):
            # Resize defensively so all temporal frames match the center exactly.
            crop = cv2.resize(crop, (center_w, center_h), interpolation=cv2.INTER_AREA)
        rel = Path("temporal_frames") / rec.sample_id / f"{nb.offset_label}.png"
        cv2.imwrite(str(dirs["temporal"].parent / rel), crop)
        nb.path = str(rel)

    # Copy center image + mask into the export package.
    shutil.copy2(center_img_path, dirs["images"] / f"{rec.sample_id}.png")
    shutil.copy2(center_mask_path, dirs["masks"] / f"{rec.sample_id}_mask.png")

    # QC contact sheet.
    if config.include_qc_contact_sheets:
        _write_contact_sheet(
            dirs["qc"] / f"{rec.sample_id}_temporal_contact.png",
            neighbors, dirs["temporal"].parent, center_img, center_mask,
        )

    n_valid = sum(1 for nb in neighbors if nb.valid)
    all_valid = all(nb.valid for nb in neighbors)

    manifest_row = _build_manifest_row(
        rec, config, offsets, neighbors, positive,
        center_w, center_h, total_frames, all_valid, video_registry,
    )
    manifest_row["_n_valid"] = n_valid
    return True, "", manifest_row


def _build_manifest_row(
    rec: "FrameRecord",
    config: TemporalExportConfig,
    offsets: list[int],
    neighbors: list[TemporalNeighbor],
    positive: int,
    export_w: int,
    export_h: int,
    total_frames: int,
    all_valid: bool,
    video_registry: "Optional[VideoRegistry]",
) -> dict:
    entry = video_registry.get(rec.video_id) if (video_registry and rec.video_id) else None
    duration = entry.duration_seconds if entry else (
        total_frames / rec.fps if rec.fps > 0 else 0.0
    )
    split_video = rec.video_id or rec.video_filename
    split_group = f"{split_video}__{rec.episode_id}__{rec.camera_id}"
    flags = {flag: bool(getattr(rec, flag, False)) for flag in METADATA_FLAGS}

    mask_fraction = (
        positive / float(export_w * export_h) if export_w and export_h else 0.0
    )

    return {
        "sample_id": rec.sample_id,
        "center_image_path": f"images/all/{rec.sample_id}.png",
        "center_mask_path": f"masks/all/{rec.sample_id}_mask.png",
        "center_frame_index": rec.frame_index,
        "fps": rec.fps,
        "time_seconds": round(rec.time_seconds, 4),
        "video_id": rec.video_id,
        "video_filename": rec.video_filename,
        "video_path": rec.video_path,
        "video_total_frames": total_frames,
        "video_duration_seconds": round(duration, 3),
        "episode_id": rec.episode_id,
        "camera_id": rec.camera_id,
        "candidate_id": rec.candidate_id,
        "target_definition": rec.target_definition,
        "label_status": rec.label_status,
        "mask_provenance": rec.mask_provenance,
        "mask_positive_pixels": positive,
        "mask_positive_fraction": round(mask_fraction, 6),
        "source_width": rec.source_width,
        "source_height": rec.source_height,
        "export_width": export_w,
        "export_height": export_h,
        "is_roi_crop": rec.is_roi_crop,
        "roi_x": rec.roi_x if rec.roi_x is not None else "",
        "roi_y": rec.roi_y if rec.roi_y is not None else "",
        "roi_width": rec.roi_width if rec.roi_width is not None else "",
        "roi_height": rec.roi_height if rec.roi_height is not None else "",
        "roi_mode": rec.roi_mode,
        "roi_size_policy": rec.roi_size_policy,
        "temporal_radius": config.temporal_radius,
        "temporal_stride": config.temporal_stride,
        "window_mode": config.window_mode,
        "edge_policy": config.edge_policy,
        "temporal_offsets_json": json.dumps(offsets),
        "temporal_frame_indices_json": json.dumps([nb.frame_index for nb in neighbors]),
        "temporal_frame_paths_json": json.dumps([nb.path for nb in neighbors]),
        "temporal_valid_json": json.dumps([nb.valid for nb in neighbors]),
        "all_temporal_frames_valid": all_valid,
        "split_group_id": split_group,
        "metadata_flags_json": json.dumps(flags),
        "notes": rec.notes,
        "created_at": rec.created_at,
        "updated_at": rec.updated_at,
    }


# ---------------------------------------------------------------------------
# Contact sheets
# ---------------------------------------------------------------------------

def _write_contact_sheet(
    out_path: Path,
    neighbors: list[TemporalNeighbor],
    export_root: Path,
    center_img: np.ndarray,
    center_mask: np.ndarray,
    tile_max: int = 240,
) -> None:
    tiles: list[np.ndarray] = []
    for nb in neighbors:
        img = cv2.imread(str(export_root / nb.path))
        if img is None:
            img = np.zeros((tile_max, tile_max, 3), np.uint8)
        tiles.append(_label_tile(_fit(img, tile_max), nb.offset_label, nb.valid))

    # Mask overlay tile (center).
    overlay = center_img.copy()
    colored = np.zeros_like(overlay)
    colored[center_mask > 0] = (0, 255, 0)
    cv2.addWeighted(colored, 0.45, overlay, 1.0, 0, overlay)
    tiles.append(_label_tile(_fit(overlay, tile_max), "mask", True))

    h = max(t.shape[0] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[0] < h:
            pad = np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)
            t = np.vstack([t, pad])
        padded.append(t)
    sheet = np.hstack(padded)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def _fit(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w, 1))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img


def _label_tile(img: np.ndarray, text: str, valid: bool) -> np.ndarray:
    color = (255, 255, 255) if valid else (0, 0, 255)
    out = img.copy()
    cv2.putText(out, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_manifest(out_root: Path, rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    atomic_write_text(out_root / "temporal_manifest.csv", buf.getvalue())
    atomic_write_json(out_root / "temporal_manifest.json", rows)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    atomic_write_text(path, buf.getvalue())


def _write_export_config(path: Path, config: TemporalExportConfig) -> None:
    atomic_write_json(path, {
        "output_root": str(config.output_root),
        "temporal_radius": config.temporal_radius,
        "temporal_stride": config.temporal_stride,
        "window_mode": config.window_mode,
        "edge_policy": config.edge_policy,
        "statuses": list(config.statuses),
        "include_qc_contact_sheets": config.include_qc_contact_sheets,
        "overwrite_existing": config.overwrite_existing,
    })


def _copy_workspace_metadata(
    dataset: "DatasetFolder",
    video_registry: "Optional[VideoRegistry]",
    out_metadata: Path,
) -> None:
    """Copy frames.csv, registry files, dataset_config, and class_definition."""
    out_metadata.mkdir(parents=True, exist_ok=True)
    mapping = [
        (dataset.root / "metadata" / "frames.csv", out_metadata / "frames.csv"),
        (dataset.root / "video_registry.json", out_metadata / "video_registry.json"),
        (dataset.root / "video_registry.csv", out_metadata / "video_registry.csv"),
        (dataset.root / "dataset_config.json", out_metadata / "dataset_config.json"),
        (dataset.root / "metadata" / "class_definition.md", out_metadata / "class_definition.md"),
    ]
    for src, dst in mapping:
        if src.exists():
            shutil.copy2(src, dst)
