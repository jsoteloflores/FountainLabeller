"""Right-panel frame queue: list of queued frames with status badges."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lava_labeler.gui.app import App

# Colour by label status
_STATUS_COLORS: dict[str, str] = {
    "queued":       "#aaaaaa",
    "in_progress":  "#4fc3f7",
    "complete":     "#81c784",
    "uncertain":    "#ffb74d",
    "needs_review": "#e57373",
}


class FrameQueuePanel(ttk.Frame):
    def __init__(self, parent: tk.Widget, app: "App") -> None:
        super().__init__(parent)
        self.app = app
        self._sample_ids: list[str] = []
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Frame Queue", font=("TkDefaultFont", 10, "bold")).pack(
            anchor=tk.W, padx=4, pady=(4, 0)
        )

        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            activestyle="dotbox",
            font=("Courier", 9),
        )
        scrollbar.config(command=self._listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._listbox.bind("<Double-Button-1>", self._on_open)
        self._listbox.bind("<Return>", self._on_open)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(btn_frame, text="Label Selected", command=self._on_open).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(btn_frame, text="Remove", command=self._on_remove).pack(side=tk.LEFT, padx=(2, 0))

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the list from the current metadata store."""
        self._listbox.delete(0, tk.END)
        self._sample_ids.clear()

        if self.app.metadata is None:
            return

        for rec in self.app.metadata.all_records():
            self._sample_ids.append(rec.sample_id)
            label = f"{rec.sample_id[-30:]:30s}  {rec.label_status}"
            self._listbox.insert(tk.END, label)

            # Colour-code by status
            color = _STATUS_COLORS.get(rec.label_status, "#aaaaaa")
            self._listbox.itemconfig(tk.END, fg=color)

    def _selected_id(self) -> str | None:
        sel = self._listbox.curselection()
        if not sel:
            return None
        idx = sel[0]
        return self._sample_ids[idx] if idx < len(self._sample_ids) else None

    def _on_open(self, _event=None) -> None:
        sid = self._selected_id()
        if sid:
            self.app.open_frame_for_labeling(sid)

    def _on_remove(self) -> None:
        sid = self._selected_id()
        if sid and self.app.metadata:
            self.app.metadata.remove(sid)
            self.app.metadata.save()
            self.refresh()
