"""QC overlay and thumbnail generation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def generate_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    overlay_color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
    max_dim: int = 1280,
) -> None:
    """Write a semi-transparent mask overlay image to *out_path*."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w, 1))

    if scale < 1.0:
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        display = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_disp = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        display = frame_bgr.copy()
        mask_disp = mask.copy()

    # Blend positive pixels with overlay colour
    overlay = display.copy()
    colored = np.zeros_like(overlay)
    colored[mask_disp > 0] = overlay_color
    cv2.addWeighted(colored, alpha, overlay, 1.0, 0, overlay)

    # Draw contour
    contours, _ = cv2.findContours(mask_disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, overlay_color, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def generate_thumbnail(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    max_dim: int = 320,
) -> None:
    generate_overlay(frame_bgr, mask, out_path, max_dim=max_dim)
