"""Session progress / statistics panel for the right sidebar.

Shows a compact summary of candidate-queue status, per-status counts,
and session throughput.  Refreshed on save, metadata toggle, and queue
navigation — never on every video frame during playback.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lava_labeler.gui.app import App


class ProgressPanel(ttk.LabelFrame):
    """Compact progress stats panel."""

    def __init__(self, parent: tk.Widget, app: "App") -> None:
        super().__init__(parent, text="Progress", padding=(4, 2))
        self.app = app
        self._build()
        self.refresh()

    def _build(self) -> None:
        lbl = ttk.Label
        s = ttk.Style()

        def row(key: str, label_text: str) -> tk.StringVar:
            f = ttk.Frame(self)
            f.pack(fill=tk.X, pady=0)
            ttk.Label(f, text=label_text, width=16, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            ttk.Label(f, textvariable=var, anchor=tk.E).pack(side=tk.RIGHT)
            return var

        self._candidate_var = row("candidate", "Candidate:")
        self._remaining_var = row("remaining", "Remaining:")
        self._labeled_var   = row("labeled",   "Labeled:")
        self._nr_var        = row("needs_rev",  "Needs review:")
        self._hn_var        = row("hardneg",    "Hard negatives:")
        self._empty_var     = row("empty",      "Empty confirmed:")
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        self._session_var   = row("session",   "This session:")

    def refresh(self) -> None:
        """Recompute stats from app state and update labels."""
        app = self.app

        if app.candidates is None:
            for v in (self._candidate_var, self._remaining_var, self._labeled_var,
                      self._nr_var, self._hn_var, self._empty_var):
                v.set("—")
            self._session_var.set(str(app._session_labeled_count))
            return

        total = len(app.candidates)
        counts: dict[str, int] = {}
        for c in app.candidates.all():
            counts[c.status] = counts.get(c.status, 0) + 1

        # Current position
        cur = app._active_candidate_id
        idx: int | None = None
        if cur is not None:
            for i, c in enumerate(app.candidates.all()):
                if c.candidate_id == cur:
                    idx = i + 1
                    break
        if idx is not None:
            self._candidate_var.set(f"{idx} / {total}")
        else:
            self._candidate_var.set(f"— / {total}")

        unlabeled = counts.get("unlabeled", 0) + counts.get("in_progress", 0)
        labeled = counts.get("labeled", 0)
        nr = counts.get("needs_review", 0)
        hn = counts.get("hard_negative", 0)
        empty = counts.get("empty_confirmed", 0)

        self._remaining_var.set(str(unlabeled))
        self._labeled_var.set(str(labeled))
        self._nr_var.set(str(nr))
        self._hn_var.set(str(hn))
        self._empty_var.set(str(empty))
        self._session_var.set(str(app._session_labeled_count))
