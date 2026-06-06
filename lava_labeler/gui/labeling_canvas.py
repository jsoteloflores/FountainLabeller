"""Labeling canvas: viewport rendering, brush/eraser, zoom/pan, undo/redo.

All annotation coordinates live in full-resolution image space.
The canvas is only a viewport into the source image.

Coordinate invariants
---------------------
screen_x = image_x * zoom + pan_x
screen_y = image_y * zoom + pan_y
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageTk

from lava_labeler.core.frame_cache import FrameCache
from lava_labeler.core.mask_ops import apply_otsu_stroke, apply_stroke, create_mask
from lava_labeler.core.undo_stack import MaskPatch, UndoStack
from lava_labeler.core.viewport import Viewport

if TYPE_CHECKING:
    from lava_labeler.gui.app import App


class LabelingCanvas(ttk.Frame):
    """Central canvas widget used for both video browsing and mask labeling."""

    _CURSOR_PAINT = "crosshair"
    _CURSOR_PAN = "fleur"

    def __init__(self, parent: tk.Widget, app: "App") -> None:
        super().__init__(parent)
        self.app = app
        self.viewport = Viewport()
        self.undo_stack = UndoStack()

        # Full-resolution image state
        self._frame: np.ndarray | None = None      # BGR uint8
        self._mask: np.ndarray | None = None        # uint8
        self._mode: str = "browse"                  # "browse" | "label"

        # Tool state
        self._tool: str = "brush"
        self._brush_radius: int = 10
        self._mask_visible: bool = True
        self._mask_alpha: float = 0.5
        self._unsaved: bool = False

        # Stroke tracking
        self._stroking: bool = False
        self._stroke_points: list[tuple[float, float]] = []
        self._pre_stroke_mask: np.ndarray | None = None   # full-mask copy before stroke

        # Pan tracking
        self._panning: bool = False
        self._space_held: bool = False
        self._pan_start_screen: tuple[int, int] = (0, 0)
        self._pan_origin: tuple[float, float] = (0.0, 0.0)

        # Tkinter canvas
        self._canvas = tk.Canvas(self, bg="#1a1a1a", cursor=self._CURSOR_PAINT, highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._photo: ImageTk.PhotoImage | None = None
        self._image_id: int | None = None

        # Debounce
        self._redraw_pending: bool = False

        self._bind_events()

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_browse_frame(self, frame_bgr: np.ndarray) -> None:
        self._mode = "browse"
        self._frame = frame_bgr
        self._mask = None
        self._schedule_redraw()

    def set_labeling_frame(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray | None,
        sample_id: str,
    ) -> None:
        self._mode = "label"
        self._frame = frame_bgr
        h, w = frame_bgr.shape[:2]
        if mask is not None and mask.shape == (h, w) and mask.dtype == np.uint8:
            self._mask = mask.copy()
        else:
            self._mask = create_mask(h, w)
        self.undo_stack.clear()
        self._unsaved = False

        cw = self._canvas.winfo_width() or 900
        ch = self._canvas.winfo_height() or 600
        self.viewport.fit_to_view(w, h, cw, ch)
        self._schedule_redraw()

    def get_current_mask(self) -> np.ndarray | None:
        return self._mask.copy() if self._mask is not None else None

    def get_current_frame(self) -> np.ndarray | None:
        return self._frame

    def mark_saved(self) -> None:
        self._unsaved = False

    def set_tool(self, tool: str) -> None:
        self._tool = tool

    def set_brush_radius(self, r: int) -> None:
        self._brush_radius = max(1, min(500, r))
        if hasattr(self, "app") and hasattr(self.app, "toolbar"):
            self.app.toolbar.update_brush_size(self._brush_radius)

    def adjust_brush(self, delta: int) -> None:
        self.set_brush_radius(self._brush_radius + delta)

    def set_mask_visible(self, visible: bool) -> None:
        self._mask_visible = visible
        self._schedule_redraw()

    def set_mask_alpha(self, alpha: float) -> None:
        self._mask_alpha = alpha
        self._schedule_redraw()

    def undo(self) -> None:
        if self._mask is not None and self.undo_stack.undo(self._mask):
            self._unsaved = True
            self._schedule_redraw()

    def redo(self) -> None:
        if self._mask is not None and self.undo_stack.redo(self._mask):
            self._unsaved = True
            self._schedule_redraw()

    def zoom_fit(self) -> None:
        if self._frame is not None:
            h, w = self._frame.shape[:2]
            cw = self._canvas.winfo_width() or 900
            ch = self._canvas.winfo_height() or 600
            self.viewport.fit_to_view(w, h, cw, ch)
            self._schedule_redraw()

    def zoom_100(self) -> None:
        self.viewport.reset_100()
        self._schedule_redraw()

    def zoom_step(self, factor: float) -> None:
        cw = self._canvas.winfo_width() or 900
        ch = self._canvas.winfo_height() or 600
        self.viewport.zoom_at(cw / 2, ch / 2, factor)
        self._schedule_redraw()

    # ------------------------------------------------------------------
    # Event bindings
    # ------------------------------------------------------------------

    def _bind_events(self) -> None:
        c = self._canvas

        c.bind("<Configure>", self._on_resize)

        # Drawing
        c.bind("<Button-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Motion>", self._on_motion)

        # Trackpad / mouse scroll
        c.bind("<MouseWheel>", self._on_scroll)
        # On macOS, horizontal trackpad scroll fires as Shift-MouseWheel in some Tk builds
        c.bind("<Shift-MouseWheel>", self._on_scroll_h)

        # Middle mouse pan
        c.bind("<Button-2>", self._on_middle_press)
        c.bind("<B2-Motion>", self._on_middle_drag)
        c.bind("<ButtonRelease-2>", self._on_middle_release)

        # Spacebar pan
        self.bind_all("<KeyPress-space>", self._on_space_press, add=True)
        self.bind_all("<KeyRelease-space>", self._on_space_release, add=True)

        # Brush size keys
        self.bind_all("<bracketleft>", lambda _: self.adjust_brush(-2))
        self.bind_all("<bracketright>", lambda _: self.adjust_brush(2))
        self.bind_all("<Shift-bracketleft>", lambda _: self.adjust_brush(-10))
        self.bind_all("<Shift-bracketright>", lambda _: self.adjust_brush(10))

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def _on_resize(self, _event: tk.Event) -> None:
        self._schedule_redraw()

    def _on_press(self, event: tk.Event) -> None:
        if self._space_held:
            self._start_pan(event.x, event.y)
            return
        if self._mode == "label" and self._mask is not None:
            self._start_stroke(event.x, event.y)

    def _on_drag(self, event: tk.Event) -> None:
        if self._panning:
            self._update_pan(event.x, event.y)
            return
        if self._stroking:
            self._continue_stroke(event.x, event.y)
        if self._mode == "label":
            self._draw_cursor(event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self._panning:
            self._end_pan()
            return
        if self._stroking:
            self._end_stroke()

    def _on_motion(self, event: tk.Event) -> None:
        if self._mode == "label":
            self._draw_cursor(event.x, event.y)
        if self._frame is not None:
            ix, iy = self.viewport.screen_to_image(event.x, event.y)
            self.app.set_status(
                f"img ({int(ix)}, {int(iy)})  |  "
                f"zoom {self.viewport.zoom_percent:.0f}%  |  "
                f"brush r={self._brush_radius}px"
                + ("  [UNSAVED]" if self._unsaved else "")
            )

    def _on_scroll(self, event: tk.Event) -> None:
        # Detect Ctrl / Command modifier in event.state bitmask
        # macOS: Ctrl=0x4, Command=0x8
        ctrl = bool(event.state & 0x4)
        cmd = bool(event.state & 0x8)

        if ctrl or cmd:
            # Zoom
            factor = 1.1 if event.delta > 0 else 1 / 1.1
            self.viewport.zoom_at(event.x, event.y, factor)
        else:
            # Pan vertically
            self.viewport.pan_y += event.delta * 0.4
        self._schedule_redraw()

    def _on_scroll_h(self, event: tk.Event) -> None:
        self.viewport.pan_x += event.delta * 0.4
        self._schedule_redraw()

    def _on_middle_press(self, event: tk.Event) -> None:
        self._start_pan(event.x, event.y)

    def _on_middle_drag(self, event: tk.Event) -> None:
        self._update_pan(event.x, event.y)

    def _on_middle_release(self, _event: tk.Event) -> None:
        self._end_pan()

    def _on_space_press(self, _event: tk.Event) -> None:
        self._space_held = True
        self._canvas.config(cursor=self._CURSOR_PAN)

    def _on_space_release(self, _event: tk.Event) -> None:
        self._space_held = False
        self._panning = False
        self._canvas.config(cursor=self._CURSOR_PAINT)

    # ------------------------------------------------------------------
    # Pan
    # ------------------------------------------------------------------

    def _start_pan(self, sx: int, sy: int) -> None:
        self._panning = True
        self._pan_start_screen = (sx, sy)
        self._pan_origin = (self.viewport.pan_x, self.viewport.pan_y)
        self._canvas.config(cursor=self._CURSOR_PAN)

    def _update_pan(self, sx: int, sy: int) -> None:
        dx = sx - self._pan_start_screen[0]
        dy = sy - self._pan_start_screen[1]
        self.viewport.pan_x = self._pan_origin[0] + dx
        self.viewport.pan_y = self._pan_origin[1] + dy
        self._schedule_redraw()

    def _end_pan(self) -> None:
        self._panning = False
        if not self._space_held:
            self._canvas.config(cursor=self._CURSOR_PAINT)

    # ------------------------------------------------------------------
    # Brush / eraser stroke
    # ------------------------------------------------------------------

    def _apply_stroke_segment(
        self, points: list[tuple[float, float]]
    ) -> None:
        """Apply the active tool to *points* on the full-resolution mask."""
        if self._mask is None or self._frame is None:
            return
        h, w = self._mask.shape
        # Filter points to those inside the image
        valid = [(x, y) for (x, y) in points if 0 <= x < w and 0 <= y < h]
        if not valid:
            return
        if self._tool == "brightness_assist":
            apply_otsu_stroke(self._mask, self._frame, valid, self._brush_radius, 255)
        else:
            value = 255 if self._tool == "brush" else 0
            apply_stroke(self._mask, valid, self._brush_radius, value)

    def _start_stroke(self, sx: int, sy: int) -> None:
        if self._mask is None:
            return
        ix, iy = self.viewport.screen_to_image(sx, sy)
        # Reject strokes that start entirely outside the image
        if self._frame is not None:
            h, w = self._frame.shape[:2]
            if not (0 <= ix < w and 0 <= iy < h):
                return
        # Snapshot full mask before stroke for undo patch extraction at end
        self._pre_stroke_mask = self._mask.copy()
        self._stroking = True
        self._stroke_points = [(ix, iy)]
        self._apply_stroke_segment([(ix, iy)])
        self._schedule_redraw()

    def _continue_stroke(self, sx: int, sy: int) -> None:
        if not self._stroking or self._mask is None:
            return
        ix, iy = self.viewport.screen_to_image(sx, sy)
        prev = self._stroke_points[-1]
        self._stroke_points.append((ix, iy))
        self._apply_stroke_segment([prev, (ix, iy)])
        self._schedule_redraw()

    def _end_stroke(self) -> None:
        if not self._stroking:
            return
        self._stroking = False

        if self._pre_stroke_mask is not None and self._mask is not None and self._stroke_points:
            r = self._brush_radius
            h, w = self._mask.shape
            xs = [p[0] for p in self._stroke_points]
            ys = [p[1] for p in self._stroke_points]
            x0 = max(0, int(min(xs)) - r)
            y0 = max(0, int(min(ys)) - r)
            x1 = min(w, int(max(xs)) + r + 1)
            y1 = min(h, int(max(ys)) + r + 1)
            before = self._pre_stroke_mask[y0:y1, x0:x1].copy()
            after = self._mask[y0:y1, x0:x1].copy()
            self.undo_stack.push(MaskPatch(x0=x0, y0=y0, before=before, after=after))

        self._pre_stroke_mask = None
        self._stroke_points = []
        self._unsaved = True
        self._schedule_redraw()

    # ------------------------------------------------------------------
    # Cursor overlay
    # ------------------------------------------------------------------

    def _draw_cursor(self, sx: int, sy: int) -> None:
        self._canvas.delete("cursor_ring")
        if self._mode != "label":
            return
        r_screen = self._brush_radius * self.viewport.zoom
        color = (
            "#00ff88" if self._tool == "brush"
            else "#ff9900" if self._tool == "brightness_assist"
            else "#ff4444"
        )
        self._canvas.create_oval(
            sx - r_screen, sy - r_screen,
            sx + r_screen, sy + r_screen,
            outline=color, width=1,
            tags="cursor_ring",
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _schedule_redraw(self) -> None:
        if not self._redraw_pending:
            self._redraw_pending = True
            self.after(16, self._do_redraw)

    def _do_redraw(self) -> None:
        self._redraw_pending = False
        if self._frame is None:
            return

        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        h, w = self._frame.shape[:2]

        # Determine visible image region (float coords)
        vp = self.viewport
        x0f, y0f, x1f, y1f = vp.get_visible_image_rect(cw, ch)

        # Clamp to image bounds
        x0 = max(0, int(x0f))
        y0 = max(0, int(y0f))
        x1 = min(w, int(x1f) + 1)
        y1 = min(h, int(y1f) + 1)

        if x1 <= x0 or y1 <= y0:
            # Image is entirely off-screen; just show background
            canvas_img = Image.new("RGB", (cw, ch), (26, 26, 26))
            self._blit(canvas_img)
            return

        # Crop visible region (BGR → RGB)
        crop_bgr = self._frame[y0:y1, x0:x1]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        # Compute display size for this crop
        screen_x0 = x0 * vp.zoom + vp.pan_x
        screen_y0 = y0 * vp.zoom + vp.pan_y
        screen_x1 = x1 * vp.zoom + vp.pan_x
        screen_y1 = y1 * vp.zoom + vp.pan_y

        disp_w = max(1, int(round(screen_x1 - screen_x0)))
        disp_h = max(1, int(round(screen_y1 - screen_y0)))

        pil_frame = Image.fromarray(crop_rgb).resize((disp_w, disp_h), Image.BILINEAR)

        # Composite mask overlay (label mode only)
        if self._mode == "label" and self._mask is not None and self._mask_visible:
            crop_mask = self._mask[y0:y1, x0:x1]
            pil_mask = Image.fromarray(crop_mask, mode="L").resize(
                (disp_w, disp_h), Image.NEAREST
            )
            mask_arr = np.array(pil_mask, dtype=np.float32)
            frame_arr = np.array(pil_frame, dtype=np.float32)

            alpha = self._mask_alpha
            pos = mask_arr > 0
            composite = frame_arr.copy()
            # Green channel boost for positive pixels
            composite[pos, 0] = frame_arr[pos, 0] * (1.0 - alpha)
            composite[pos, 1] = np.clip(frame_arr[pos, 1] * (1.0 - alpha) + 255 * alpha, 0, 255)
            composite[pos, 2] = frame_arr[pos, 2] * (1.0 - alpha)

            display_img = Image.fromarray(composite.astype(np.uint8))
        else:
            display_img = pil_frame

        # Paste crop onto a full-canvas background image.
        # paste_x/paste_y can legitimately be negative when the image is panned
        # so that its left/top edge is off the left/top of the canvas.
        # PIL.Image.paste() handles negative offsets correctly (clips to dest).
        # DO NOT clamp to max(0, ...) — that would shift the displayed image
        # relative to the coordinate system and break brush accuracy.
        canvas_img = Image.new("RGB", (cw, ch), (26, 26, 26))
        paste_x = int(round(screen_x0))
        paste_y = int(round(screen_y0))
        canvas_img.paste(display_img, (paste_x, paste_y))

        self._blit(canvas_img)

    def _blit(self, img: Image.Image) -> None:
        photo = ImageTk.PhotoImage(img)
        self._photo = photo  # keep reference — GC would blank the canvas otherwise
        if self._image_id is None:
            self._image_id = self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        else:
            self._canvas.itemconfig(self._image_id, image=photo)
        # Raise cursor ring above the image
        self._canvas.tag_raise("cursor_ring")
