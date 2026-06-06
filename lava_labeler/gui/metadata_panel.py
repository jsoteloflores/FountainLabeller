"""Right-panel metadata editor for the currently open frame."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from lava_labeler.core.metadata import (
    FrameRecord, DIFFICULTIES, LIGHTING_CONDITIONS, LABEL_STATUSES,
)

if TYPE_CHECKING:
    from lava_labeler.gui.app import App


class MetadataPanel(ttk.LabelFrame):
    def __init__(self, parent: tk.Widget, app: "App") -> None:
        super().__init__(parent, text="Frame Metadata")
        self.app = app
        self._sample_id: str | None = None
        self._build()

    def _build(self) -> None:
        PAD = {"padx": 4, "pady": 2}
        sticky = {"sticky": tk.EW}

        self.columnconfigure(1, weight=1)

        # Episode / camera IDs (also used when adding frames to queue)
        ttk.Label(self, text="Episode:").grid(row=0, column=0, **PAD, **sticky)
        self.episode_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.episode_var, width=14).grid(row=0, column=1, **PAD, **sticky)

        ttk.Label(self, text="Camera:").grid(row=1, column=0, **PAD, **sticky)
        self.camera_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.camera_var, width=14).grid(row=1, column=1, **PAD, **sticky)

        # Status
        ttk.Label(self, text="Status:").grid(row=2, column=0, **PAD, **sticky)
        self._status_var = tk.StringVar(value="queued")
        ttk.Combobox(
            self, textvariable=self._status_var, values=LABEL_STATUSES,
            state="readonly", width=13,
        ).grid(row=2, column=1, **PAD, **sticky)

        # Difficulty
        ttk.Label(self, text="Difficulty:").grid(row=3, column=0, **PAD, **sticky)
        self._difficulty_var = tk.StringVar(value="unknown")
        ttk.Combobox(
            self, textvariable=self._difficulty_var, values=DIFFICULTIES,
            state="readonly", width=13,
        ).grid(row=3, column=1, **PAD, **sticky)

        # Lighting
        ttk.Label(self, text="Lighting:").grid(row=4, column=0, **PAD, **sticky)
        self._lighting_var = tk.StringVar(value="unknown")
        ttk.Combobox(
            self, textvariable=self._lighting_var, values=LIGHTING_CONDITIONS,
            state="readonly", width=13,
        ).grid(row=4, column=1, **PAD, **sticky)

        # Notes
        ttk.Label(self, text="Notes:").grid(row=5, column=0, **PAD, **sticky)
        self._notes_var = tk.StringVar()
        ttk.Entry(self, textvariable=self._notes_var).grid(
            row=5, column=1, columnspan=1, **PAD, **sticky
        )

        # Checkboxes
        self._tephra_var = tk.BooleanVar()
        self._smoke_var = tk.BooleanVar()
        self._base_glow_var = tk.BooleanVar()
        ttk.Checkbutton(self, text="Tephra", variable=self._tephra_var).grid(
            row=6, column=0, **PAD, sticky=tk.W
        )
        ttk.Checkbutton(self, text="Smoke", variable=self._smoke_var).grid(
            row=6, column=1, **PAD, sticky=tk.W
        )
        ttk.Checkbutton(self, text="Base glow", variable=self._base_glow_var).grid(
            row=7, column=0, columnspan=2, **PAD, sticky=tk.W
        )

        # Action buttons
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=8, column=0, columnspan=2, pady=4)
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Complete", command=lambda: self._set_status("complete")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Uncertain", command=lambda: self._set_status("uncertain")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Needs Review", command=lambda: self._set_status("needs_review")).pack(
            side=tk.LEFT, padx=2
        )

    # ------------------------------------------------------------------

    def load_record(self, rec: FrameRecord) -> None:
        self._sample_id = rec.sample_id
        self.episode_var.set(rec.episode_id)
        self.camera_var.set(rec.camera_id)
        self._status_var.set(rec.label_status)
        self._difficulty_var.set(rec.difficulty)
        self._lighting_var.set(rec.lighting_condition)
        self._notes_var.set(rec.notes)
        self._tephra_var.set(rec.contains_tephra)
        self._smoke_var.set(rec.contains_smoke)
        self._base_glow_var.set(rec.contains_base_glow)

    def _apply(self) -> None:
        sid = self._sample_id
        if sid and self.app.metadata:
            self.app.metadata.update(
                sid,
                episode_id=self.episode_var.get(),
                camera_id=self.camera_var.get(),
                label_status=self._status_var.get(),
                difficulty=self._difficulty_var.get(),
                lighting_condition=self._lighting_var.get(),
                notes=self._notes_var.get(),
                contains_tephra=self._tephra_var.get(),
                contains_smoke=self._smoke_var.get(),
                contains_base_glow=self._base_glow_var.get(),
            )
            self.app.metadata.save()
            self.app.frame_queue.refresh()
            self.app.set_status(f"Metadata saved for {sid}")

    def _set_status(self, status: str) -> None:
        self._status_var.set(status)
        self._apply()
