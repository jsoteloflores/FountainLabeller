"""ROI panel: fixed-size draggable ROI controls."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lava_labeler.gui.app import App


class ROIPanel(ttk.Frame):
    """Horizontal toolbar row for the fixed-size ROI workflow.

    The panel sits below the main toolbar.  It controls:
      - ROI mode (Full Frame vs Fixed ROI Crop)
      - Fixed ROI dimensions (width × height)
      - ROI visibility toggle
      - Size policy (global_fixed / camera_fixed)
      - Clear / Save-placement actions
      - Read-only position display (updated as the ROI is dragged)

    Arrow-key nudge (Ctrl+Arrow = 1 px, Ctrl+Shift+Arrow = 10 px) is bound
    globally in app.py.
    """

    def __init__(self, parent: tk.Widget, app: "App") -> None:
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self) -> None:
        PAD = {"padx": 3, "pady": 2}

        # ---- Mode ----
        ttk.Label(self, text="ROI:").pack(side=tk.LEFT, **PAD)
        self._mode_var = tk.StringVar(value="full_frame")
        ttk.Radiobutton(
            self, text="Full Frame", variable=self._mode_var, value="full_frame",
            command=self._on_mode_change,
        ).pack(side=tk.LEFT, **PAD)
        ttk.Radiobutton(
            self, text="Fixed ROI Crop", variable=self._mode_var, value="fixed_roi_crop",
            command=self._on_mode_change,
        ).pack(side=tk.LEFT, **PAD)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # ---- Dimensions ----
        ttk.Label(self, text="W:").pack(side=tk.LEFT, padx=(4, 0))
        self._w_var = tk.IntVar(value=1280)
        self._w_spin = ttk.Spinbox(
            self, from_=64, to=9999, textvariable=self._w_var, width=6,
            command=self._on_size_change,
        )
        self._w_spin.pack(side=tk.LEFT, **PAD)
        self._w_spin.bind("<Return>", lambda _: self._on_size_change())
        self._w_spin.bind("<FocusOut>", lambda _: self._on_size_change())

        ttk.Label(self, text="H:").pack(side=tk.LEFT, padx=(2, 0))
        self._h_var = tk.IntVar(value=960)
        self._h_spin = ttk.Spinbox(
            self, from_=64, to=9999, textvariable=self._h_var, width=6,
            command=self._on_size_change,
        )
        self._h_spin.pack(side=tk.LEFT, **PAD)
        self._h_spin.bind("<Return>", lambda _: self._on_size_change())
        self._h_spin.bind("<FocusOut>", lambda _: self._on_size_change())

        # Start with dim spinboxes disabled (mode = Full Frame)
        self._w_spin.config(state="disabled")
        self._h_spin.config(state="disabled")

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # ---- Show ROI ----
        self._show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self, text="Show ROI", variable=self._show_var,
            command=self._on_show_toggle,
        ).pack(side=tk.LEFT, **PAD)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # ---- Size policy ----
        ttk.Label(self, text="Policy:").pack(side=tk.LEFT, padx=(4, 0))
        self._policy_var = tk.StringVar(value="global_fixed")
        policy_combo = ttk.Combobox(
            self, textvariable=self._policy_var,
            values=["global_fixed", "camera_fixed"],
            state="readonly", width=14,
        )
        policy_combo.pack(side=tk.LEFT, **PAD)
        policy_combo.bind("<<ComboboxSelected>>", self._on_policy_change)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # ---- Action buttons ----
        ttk.Button(self, text="Clear ROI", command=self._on_clear).pack(side=tk.LEFT, **PAD)
        ttk.Button(self, text="Save Placement", command=self._on_save_placement).pack(
            side=tk.LEFT, **PAD
        )

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # ---- Position readout ----
        ttk.Label(self, text="Pos:").pack(side=tk.LEFT, padx=(4, 0))
        self._pos_var = tk.StringVar(value="x=0  y=0")
        ttk.Label(self, textvariable=self._pos_var, width=14, anchor=tk.W).pack(
            side=tk.LEFT, **PAD
        )

        # ---- Hint ----
        ttk.Label(
            self,
            text="Ctrl+↑↓←→ nudge 1px  |  Ctrl+Shift+↑↓←→ nudge 10px",
            foreground="#808080",
        ).pack(side=tk.LEFT, padx=(10, 0))

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_mode_change(self) -> None:
        mode = self._mode_var.get()
        state = "normal" if mode == "fixed_roi_crop" else "disabled"
        self._w_spin.config(state=state)
        self._h_spin.config(state=state)
        self.app.set_roi_mode(mode)

    def _on_size_change(self) -> None:
        try:
            w = int(self._w_var.get())
            h = int(self._h_var.get())
        except (ValueError, tk.TclError):
            return
        self.app.set_roi_size(w, h)

    def _on_show_toggle(self) -> None:
        self.app.canvas.set_roi_visible(self._show_var.get())

    def _on_policy_change(self, _event=None) -> None:
        self.app._roi_size_policy = self._policy_var.get()

    def _on_clear(self) -> None:
        self.app.clear_roi()

    def _on_save_placement(self) -> None:
        self.app.save_roi_placement()

    # ------------------------------------------------------------------
    # External update helpers
    # ------------------------------------------------------------------

    def update_position_display(self, x: int, y: int) -> None:
        self._pos_var.set(f"x={x}  y={y}")

    def set_mode(self, mode: str) -> None:
        """Sync the radio button (called externally if mode changes in code)."""
        self._mode_var.set(mode)
        state = "normal" if mode == "fixed_roi_crop" else "disabled"
        self._w_spin.config(state=state)
        self._h_spin.config(state=state)

    def set_size(self, w: int, h: int) -> None:
        """Sync the W/H spinboxes (called externally when ROI size is restored)."""
        self._w_var.set(int(w))
        self._h_var.set(int(h))

    def set_policy(self, policy: str) -> None:
        """Sync the policy combobox (called externally when ROI policy is restored)."""
        if policy in ("global_fixed", "camera_fixed"):
            self._policy_var.set(policy)
