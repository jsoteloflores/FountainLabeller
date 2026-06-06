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

        if sid in seen_ids:
            issues.append(ValidationIssue(sid, "Duplicate sample_id", "error"))
        seen_ids.add(sid)

        img_p = Path(str(row.get("image_path", "")))
        msk_p = Path(str(row.get("mask_path", "")))

        if not img_p.exists():
            issues.append(ValidationIssue(sid, f"Image missing: {img_p.name}", "error"))
            continue

        if not msk_p.exists():
            issues.append(ValidationIssue(sid, f"Mask missing: {msk_p.name}", "error"))
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

    return issues
