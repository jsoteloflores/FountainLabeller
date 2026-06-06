"""Viewport: zoom/pan state and coordinate transforms for the labeling canvas."""

from __future__ import annotations


class Viewport:
    """Manages zoom and pan, and converts between screen and image coordinates.

    Invariants
    ----------
    screen_x = image_x * zoom + pan_x
    screen_y = image_y * zoom + pan_y

    image_x = (screen_x - pan_x) / zoom
    image_y = (screen_y - pan_y) / zoom
    """

    ZOOM_MIN: float = 0.02
    ZOOM_MAX: float = 64.0

    def __init__(self) -> None:
        self.zoom: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def screen_to_image(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx - self.pan_x) / self.zoom, (sy - self.pan_y) / self.zoom

    def image_to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        return ix * self.zoom + self.pan_x, iy * self.zoom + self.pan_y

    # ------------------------------------------------------------------
    # View operations
    # ------------------------------------------------------------------

    def zoom_at(self, screen_x: float, screen_y: float, factor: float) -> None:
        """Cursor-centred zoom: the image pixel under the cursor stays fixed."""
        ix, iy = self.screen_to_image(screen_x, screen_y)
        self.zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self.zoom * factor))
        self.pan_x = screen_x - ix * self.zoom
        self.pan_y = screen_y - iy * self.zoom

    def fit_to_view(self, img_w: int, img_h: int, canvas_w: int, canvas_h: int) -> None:
        if img_w <= 0 or img_h <= 0 or canvas_w <= 0 or canvas_h <= 0:
            return
        scale = min(canvas_w / img_w, canvas_h / img_h)
        self.zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, scale))
        self.pan_x = (canvas_w - img_w * self.zoom) / 2.0
        self.pan_y = (canvas_h - img_h * self.zoom) / 2.0

    def reset_100(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def get_visible_image_rect(
        self, canvas_w: int, canvas_h: int
    ) -> tuple[float, float, float, float]:
        """Return (x0, y0, x1, y1) in image coordinates for the visible canvas."""
        x0, y0 = self.screen_to_image(0, 0)
        x1, y1 = self.screen_to_image(canvas_w, canvas_h)
        return x0, y0, x1, y1

    @property
    def zoom_percent(self) -> float:
        return self.zoom * 100.0
