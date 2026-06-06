"""Full-resolution mask operations: create, paint, erase."""

from __future__ import annotations

import math

import cv2
import numpy as np


def create_mask(h: int, w: int) -> np.ndarray:
    """Return a zeroed uint8 mask of shape (h, w)."""
    return np.zeros((h, w), dtype=np.uint8)


def apply_stroke(
    mask: np.ndarray,
    points: list[tuple[float, float]],
    radius: int,
    value: int = 255,
) -> tuple[int, int, int, int] | None:
    """Paint circles along a stroke path into *mask* (in-place).

    Parameters
    ----------
    mask   : uint8 array, modified in-place.
    points : list of (image_x, image_y) float coords, in order.
    radius : brush radius in source-image pixels.
    value  : 255 for paint, 0 for erase.

    Returns the bounding box (x0, y0, x1, y1) of all painted pixels,
    or None if nothing was drawn.
    """
    if not points:
        return None

    h, w = mask.shape
    touched_x: list[int] = []
    touched_y: list[int] = []

    def _circle(x: float, y: float) -> None:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(mask, (cx, cy), radius, int(value), -1)
        touched_x.append(cx)
        touched_y.append(cy)

    _circle(*points[0])

    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(dist))
        for j in range(1, steps + 1):
            t = j / steps
            _circle(x0 + t * (x1 - x0), y0 + t * (y1 - y0))

    if not touched_x:
        return None

    return (
        max(0, min(touched_x) - radius),
        max(0, min(touched_y) - radius),
        min(w - 1, max(touched_x) + radius),
        min(h - 1, max(touched_y) + radius),
    )


def apply_otsu_stroke(
    mask: np.ndarray,
    frame_bgr: np.ndarray,
    points: list[tuple[float, float]],
    radius: int,
    value: int = 255,
) -> tuple[int, int, int, int] | None:
    """Paint using local Otsu thresholding within the brush radius.

    For each point along the stroke the grayscale Otsu threshold is computed
    from the circular ROI in *frame_bgr*.  Pixels brighter than the threshold
    (within the brush circle) are set to *value*.  This automatically snaps
    to incandescent lava edges without manual edge-chasing.

    Parameters
    ----------
    mask      : uint8 mask to modify in-place.
    frame_bgr : source BGR image at full resolution.
    points    : list of (image_x, image_y) float coords.
    radius    : brush radius in source-image pixels.
    value     : 255 to paint positive, 0 to erase the full circle.
    """
    if not points or frame_bgr is None:
        return None

    h, w = mask.shape
    touched_rects: list[tuple[int, int, int, int]] = []

    def _otsu_circle(x: float, y: float) -> None:
        cx, cy = int(round(x)), int(round(y))
        bx0 = max(0, cx - radius)
        by0 = max(0, cy - radius)
        bx1 = min(w, cx + radius + 1)
        by1 = min(h, cy + radius + 1)
        if bx1 <= bx0 or by1 <= by0:
            return

        roi = frame_bgr[by0:by1, bx0:bx1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Otsu finds the optimal binary split — works well for bright lava vs
        # dark background.  If the ROI is uniform the threshold degenerates; we
        # fall back to painting the whole circle so the brush stays responsive.
        _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Circle footprint within the bounding box
        circ = np.zeros((by1 - by0, bx1 - bx0), dtype=np.uint8)
        cv2.circle(circ, (cx - bx0, cy - by0), radius, 255, -1)

        if value == 255:
            combined = (threshed > 0) & (circ > 0)
        else:
            # Eraser: clear the full circle regardless of threshold
            combined = circ > 0

        mask[by0:by1, bx0:bx1][combined] = value
        touched_rects.append((bx0, by0, bx1, by1))

    _otsu_circle(*points[0])
    for i in range(1, len(points)):
        x0f, y0f = points[i - 1]
        x1f, y1f = points[i]
        dist = math.hypot(x1f - x0f, y1f - y0f)
        # Step at half-radius intervals: enough coverage, avoids redundant Otsu calls
        step_size = max(1.0, radius / 2.0)
        steps = max(1, int(dist / step_size))
        for j in range(1, steps + 1):
            t = j / steps
            _otsu_circle(x0f + t * (x1f - x0f), y0f + t * (y1f - y0f))

    if not touched_rects:
        return None

    return (
        min(r[0] for r in touched_rects),
        min(r[1] for r in touched_rects),
        max(r[2] for r in touched_rects),
        max(r[3] for r in touched_rects),
    )
