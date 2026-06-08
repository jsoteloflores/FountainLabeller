"""Dataset validation: image/mask pairing, dimensions, dtype, binary values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class ValidationIssue:
    sample_id: str
    message: str
    severity: str   # "error" or "warning"

    def __str__(self) -> str:
        tag = "ERROR" if self.severity == "error" else "WARN "
        sid = f"[{self.sample_id}] " if self.sample_id else ""
        return f"{tag}  {sid}{self.message}"


def _resolve_path(dataset_root: Path, p_str: str) -> Path:
    """Resolve a path that may be relative (Colab-style) or absolute (legacy)."""
    p = Path(p_str)
    if not p.is_absolute():
        p = dataset_root / p
    return p


def validate_dataset(dataset_root: Path) -> list[ValidationIssue]:
    """Run all validation checks and return a list of issues."""
    issues: list[ValidationIssue] = []

    csv_path = dataset_root / "metadata" / "frames.csv"
    if not csv_path.exists():
        issues.append(ValidationIssue("", "metadata/frames.csv not found", "error"))
        return issues

    df = pd.read_csv(csv_path, dtype=str)
    if df.empty:
        issues.append(ValidationIssue("", "frames.csv is empty", "warning"))
        return issues

    seen_ids: set[str] = set()

    for _, row in df.iterrows():
        sid = str(row.get("sample_id", "")).strip()
        label_status = str(row.get("label_status", "queued")).strip()

        if sid in seen_ids:
            issues.append(ValidationIssue(sid, "Duplicate sample_id", "error"))
        seen_ids.add(sid)

        # Resolve image/mask paths: support both relative (Colab) and absolute (legacy)
        img_p = _resolve_path(dataset_root, str(row.get("image_path", "")))
        msk_p = _resolve_path(dataset_root, str(row.get("mask_path", "")))

        if not img_p.exists():
            issues.append(ValidationIssue(sid, f"Image missing: {img_p.name}", "error"))
            continue

        mask_exists = msk_p.exists()
        if not mask_exists:
            # Frames that are complete/uncertain/needs_review must have a mask
            if label_status in ("complete", "uncertain", "needs_review"):
                issues.append(ValidationIssue(sid, f"Mask missing for '{label_status}' frame", "error"))
            else:
                issues.append(ValidationIssue(sid, f"Mask missing (status: {label_status})", "warning"))
            continue

        img = cv2.imread(str(img_p))
        msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)

        if img is None:
            issues.append(ValidationIssue(sid, "Cannot read image file", "error"))
            continue
        if msk is None:
            issues.append(ValidationIssue(sid, "Cannot read mask file", "error"))
            continue

        ih, iw = img.shape[:2]
        mh, mw = msk.shape[:2]
        if ih != mh or iw != mw:
            issues.append(
                ValidationIssue(
                    sid,
                    f"Size mismatch — image {iw}×{ih} vs mask {mw}×{mh}",
                    "error",
                )
            )

        if msk.dtype != np.uint8:
            issues.append(ValidationIssue(sid, f"Mask dtype {msk.dtype}, expected uint8", "error"))

        unique_vals = set(int(v) for v in np.unique(msk))
        invalid = unique_vals - {0, 255}
        if invalid:
            issues.append(
                ValidationIssue(
                    sid,
                    f"Mask contains non-binary pixel values: {sorted(invalid)}",
                    "error",
                )
            )

        # Check mask_positive_pixels accuracy in metadata
        try:
            recorded = int(float(row.get("mask_positive_pixels", -1)))
            actual = int(np.sum(msk > 0))
            if recorded >= 0 and recorded != actual:
                issues.append(
                    ValidationIssue(
                        sid,
                        f"mask_positive_pixels mismatch: recorded {recorded}, actual {actual}",
                        "warning",
                    )
                )
        except (TypeError, ValueError):
            pass

        # ROI geometry checks
        try:
            def _safe_int(key: str, default: int = 0) -> int:
                v = row.get(key, "")
                if v is None or str(v).strip() in ("nan", "None", ""):
                    return default
                return int(float(v))

            is_roi = str(row.get("is_roi_crop", "false")).lower() in ("true", "1", "yes")
            roi_mode_val = str(row.get("roi_mode", "full_frame")).strip()
            src_w = _safe_int("source_width")
            src_h = _safe_int("source_height")
            rx = _safe_int("roi_x")
            ry = _safe_int("roi_y")
            rw = _safe_int("roi_width")
            rh = _safe_int("roi_height")
            ew = _safe_int("export_width")
            eh = _safe_int("export_height")

            if rw <= 0:
                issues.append(ValidationIssue(sid, "roi_width <= 0", "error"))
            if rh <= 0:
                issues.append(ValidationIssue(sid, "roi_height <= 0", "error"))
            if rx < 0:
                issues.append(ValidationIssue(sid, "roi_x < 0", "error"))
            if ry < 0:
                issues.append(ValidationIssue(sid, "roi_y < 0", "error"))
            if src_w > 0 and rw > 0 and rx + rw > src_w:
                issues.append(ValidationIssue(
                    sid, f"ROI x+w ({rx + rw}) exceeds source_width ({src_w})", "error"
                ))
            if src_h > 0 and rh > 0 and ry + rh > src_h:
                issues.append(ValidationIssue(
                    sid, f"ROI y+h ({ry + rh}) exceeds source_height ({src_h})", "error"
                ))
            if ew <= 0:
                issues.append(ValidationIssue(sid, "export_width <= 0", "error"))
            if eh <= 0:
                issues.append(ValidationIssue(sid, "export_height <= 0", "error"))

            if is_roi and roi_mode_val == "fixed_roi_crop":
                if rw > 0 and ew != rw:
                    issues.append(ValidationIssue(
                        sid, f"export_width ({ew}) != roi_width ({rw})", "error"
                    ))
                if rh > 0 and eh != rh:
                    issues.append(ValidationIssue(
                        sid, f"export_height ({eh}) != roi_height ({rh})", "error"
                    ))
                # Image on disk should match roi dimensions
                if rw > 0 and rh > 0 and (iw != rw or ih != rh):
                    issues.append(ValidationIssue(
                        sid,
                        f"ROI crop image {iw}\u00d7{ih} does not match roi_width\u00d7roi_height ({rw}\u00d7{rh})",
                        "error",
                    ))
            elif not is_roi:
                # Full frame: roi_x/roi_y should be 0
                if rx != 0 or ry != 0:
                    issues.append(ValidationIssue(
                        sid, f"full_frame sample has non-zero roi offset ({rx}, {ry})", "warning"
                    ))
                if src_w > 0 and rw > 0 and (rw != src_w or rh != src_h):
                    issues.append(ValidationIssue(
                        sid,
                        f"full_frame roi size ({rw}\u00d7{rh}) != source size ({src_w}\u00d7{src_h})",
                        "warning",
                    ))
        except (TypeError, ValueError):
            pass

    return issues
