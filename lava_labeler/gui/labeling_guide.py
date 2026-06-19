"""Collapsible labeling-guidance panel and a shortcut cheat-sheet dialog."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from lava_labeler.core.config import SHORTCUT_GROUPS

if TYPE_CHECKING:
    from lava_labeler.core.config import ShortcutConfig
    from lava_labeler.gui.app import App


_GUIDE_TEXT = """Label: active rising lava fountain material.

Include:
• coherent incandescent material rising from the vent/source
• the active fountain core/envelope
• material still dynamically part of the upward jet/plume

Exclude:
• wind-drifted tephra
• falling particles
• detached glowing fragments
• ground glow
• smoke/steam/ash without active lava
• cooled diffuse plume

When unsure:
• be conservative
• mark ambiguous_boundary (U)"""


class LabelingGuidePanel(ttk.Frame):
    """A compact, collapsible labeling-guidance card."""

    def __init__(self, parent: tk.Widget, app: "App", start_open: bool = True) -> None:
        super().__init__(parent)
        self.app = app
        self._open = tk.BooleanVar(value=start_open)
        self._build()

    def _build(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill=tk.X)
        self._toggle_btn = ttk.Button(
            header, text="▼ Labeling Guide", command=self._toggle, width=22
        )
        self._toggle_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._body = tk.Label(
            self,
            text=_GUIDE_TEXT,
            justify=tk.LEFT,
            anchor=tk.W,
            wraplength=240,
            font=("TkDefaultFont", 9),
            bg="#23272e",
            fg="#d7dae0",
            padx=6,
            pady=6,
        )
        if self._open.get():
            self._body.pack(fill=tk.X, pady=(2, 0))

    def _toggle(self) -> None:
        if self._open.get():
            self._body.pack_forget()
            self._toggle_btn.config(text="▶ Labeling Guide")
            self._open.set(False)
        else:
            self._body.pack(fill=tk.X, pady=(2, 0))
            self._toggle_btn.config(text="▼ Labeling Guide")
            self._open.set(True)


# Human-friendly action labels for the cheat sheet.
_ACTION_LABELS: dict[str, str] = {
    "previous_frame": "Previous frame",
    "next_frame": "Next frame",
    "jump_back_small": "Jump back (small)",
    "jump_forward_small": "Jump forward (small)",
    "jump_back_large": "Jump back (large)",
    "jump_forward_large": "Jump forward (large)",
    "previous_candidate": "Previous candidate",
    "next_candidate": "Next candidate",
    "save_and_next": "Save & next candidate",
    "save": "Save mask",
    "force_save": "Force save session",
    "play_pause": "Play / pause preview",
    "toggle_playback_panel": "Toggle playback panel",
    "fit_view": "Fit image to view",
    "zoom_100": "Zoom 100%",
    "reset_view": "Reset view",
    "toggle_mask": "Toggle mask overlay",
    "toggle_metadata_panel": "Toggle metadata panel",
    "cheat_sheet": "Show this cheat sheet",
    "brush": "Brush tool",
    "eraser": "Eraser tool",
    "brush_smaller": "Decrease brush size",
    "brush_larger": "Increase brush size",
    "undo": "Undo",
    "redo": "Redo",
    "otsu_brush": "Otsu assist brush",
    "clear_mask": "Clear current mask",
    "mark_empty": "Mark intentionally empty",
    "toggle_wind_affected": "Toggle wind_affected",
    "toggle_falling_tephra_visible": "Toggle falling_tephra_visible",
    "toggle_cooling_tephra_visible": "Toggle cooling_tephra_visible",
    "toggle_smoke_obscured": "Toggle smoke_obscured",
    "toggle_ground_glow_visible": "Toggle ground_glow_visible",
    "toggle_exposure_bloom": "Toggle exposure_bloom",
    "toggle_ambiguous_boundary": "Toggle ambiguous_boundary",
    "toggle_hard_negative": "Toggle hard_negative",
    "approve_human_clean": "Approve (human_clean)",
    "mark_needs_review": "Mark needs_review",
    "toggle_model_draft_corrected": "Toggle model_draft_corrected",
}


class CheatSheetDialog(tk.Toplevel):
    """Scrollable, grouped cheat sheet reflecting the active shortcut config."""

    def __init__(self, parent: tk.Widget, shortcuts: "ShortcutConfig") -> None:
        super().__init__(parent)
        self.title("Keyboard Shortcuts")
        self.geometry("520x640")
        self._shortcuts = shortcuts
        self._build()
        self.transient(parent)

    def _build(self) -> None:
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, bg="#1e1e1e", highlightthickness=0)
        sb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        actions = self._shortcuts.all_actions()
        for group, action_names in SHORTCUT_GROUPS.items():
            ttk.Label(
                inner, text=group, font=("TkDefaultFont", 11, "bold")
            ).pack(anchor=tk.W, padx=10, pady=(10, 2))
            for name in action_names:
                key = actions.get(name, "")
                if not key:
                    continue
                row = ttk.Frame(inner)
                row.pack(fill=tk.X, padx=16, pady=1)
                ttk.Label(row, text=key, width=12, font=("Courier", 10, "bold"),
                          foreground="#4fc3f7").pack(side=tk.LEFT)
                ttk.Label(row, text=_ACTION_LABELS.get(name, name)).pack(side=tk.LEFT, padx=6)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=8)
