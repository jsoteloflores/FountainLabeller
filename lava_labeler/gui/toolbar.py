"""Top toolbar: tools, zoom controls, mask toggle, brush size."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lava_labeler.gui.app import App


class Toolbar(ttk.Frame):
    def __init__(self, parent: tk.Widget, app: "App") -> None:
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self) -> None:
        PAD = {"padx": 2, "pady": 2}

        # ---- File group ----
        ttk.Button(self, text="Open Video", command=self.app.open_video).pack(side=tk.LEFT, **PAD)
        ttk.Button(self, text="New Dataset", command=self.app.new_dataset).pack(side=tk.LEFT, **PAD)
        ttk.Button(self, text="Open Dataset", command=self.app.open_dataset).pack(side=tk.LEFT, **PAD)
        ttk.Button(self, text="Save", command=self.app.save_current_mask).pack(side=tk.LEFT, **PAD)
        ttk.Button(self, text="Export…", command=self.app.open_export_dialog).pack(side=tk.LEFT, **PAD)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # ---- Tool group ----
        self._tool_var = tk.StringVar(value="brush")
        ttk.Radiobutton(
            self, text="Brush", variable=self._tool_var, value="brush",
            command=self._on_tool_change,
        ).pack(side=tk.LEFT, **PAD)
        ttk.Radiobutton(
            self, text="Eraser", variable=self._tool_var, value="eraser",
            command=self._on_tool_change,
        ).pack(side=tk.LEFT, **PAD)
        ttk.Radiobutton(
            self, text="Otsu", variable=self._tool_var, value="otsu_brush",
            command=self._on_tool_change,
        ).pack(side=tk.LEFT, **PAD)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # ---- Edit group ----
        ttk.Button(self, text="Undo", command=self.app.undo).pack(side=tk.LEFT, **PAD)
        ttk.Button(self, text="Redo", command=self.app.redo).pack(side=tk.LEFT, **PAD)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # ---- Mask visibility ----
        self._mask_visible_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self, text="Mask On", variable=self._mask_visible_var,
            command=self._on_mask_toggle,
        ).pack(side=tk.LEFT, **PAD)

        ttk.Label(self, text="Opacity:").pack(side=tk.LEFT, padx=(4, 0))
        self._opacity_var = tk.DoubleVar(value=0.5)
        ttk.Scale(
            self, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self._opacity_var, length=80,
            command=self._on_opacity_change,
        ).pack(side=tk.LEFT, **PAD)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # ---- Brush size ----
        ttk.Label(self, text="Brush:").pack(side=tk.LEFT, padx=(4, 0))
        self._brush_var = tk.IntVar(value=10)
        self._brush_spin = ttk.Spinbox(
            self, from_=1, to=500, textvariable=self._brush_var, width=5,
            command=self._on_brush_change,
        )
        self._brush_spin.pack(side=tk.LEFT, **PAD)
        self._brush_spin.bind("<Return>", lambda _: self._on_brush_change())

        ttk.Label(self, text="px").pack(side=tk.LEFT)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # ---- Zoom group ----
        ttk.Button(self, text="Fit", command=self.app.zoom_fit).pack(side=tk.LEFT, **PAD)
        ttk.Button(self, text="100%", command=self.app.zoom_100).pack(side=tk.LEFT, **PAD)
        ttk.Button(self, text="＋", width=2, command=self.app.zoom_in).pack(side=tk.LEFT, **PAD)
        ttk.Button(self, text="－", width=2, command=self.app.zoom_out).pack(side=tk.LEFT, **PAD)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_tool_change(self) -> None:
        self.app.canvas.set_tool(self._tool_var.get())

    def _on_mask_toggle(self) -> None:
        self.app.canvas.set_mask_visible(self._mask_visible_var.get())

    def _on_opacity_change(self, _=None) -> None:
        self.app.canvas.set_mask_alpha(self._opacity_var.get())

    def _on_brush_change(self) -> None:
        try:
            r = int(self._brush_var.get())
            self.app.canvas.set_brush_radius(r)
        except (ValueError, tk.TclError):
            pass

    # ------------------------------------------------------------------
    # External update (called by canvas when [ / ] keys change radius)
    # ------------------------------------------------------------------

    def update_brush_size(self, r: int) -> None:
        self._brush_var.set(r)
