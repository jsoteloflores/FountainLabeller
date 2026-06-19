"""Right-panel metadata editor for the currently open frame."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

from lava_labeler.core.metadata import (
    FrameRecord, DIFFICULTIES, LIGHTING_CONDITIONS, LABEL_STATUSES,
    METADATA_FLAGS, MASK_PROVENANCES,
)

if TYPE_CHECKING:
    from lava_labeler.gui.app import App


class MetadataPanel(ttk.Frame):
    """Two-section panel.

    Section 1 — "New Frame Defaults" (always editable):
        Episode / Camera used as the stamp when clicking "Add Frame".
        Never overwritten by loading a queued frame for labeling.
        Changing these here does NOT rename any existing frame.

    Section 2 — "Loaded Frame" (shown/active only when a frame is open):
        Displays the stored sample_id of the frame being labeled.
        Has a "Rename ID…" button for the rare case of needing to change
        the stored identity after queueing.
        Editable fields: status, difficulty, lighting, notes, flags.
    """

    def __init__(self, parent: tk.Widget, app: "App") -> None:
        super().__init__(parent)
        self.app = app
        self._sample_id: str | None = None
        self._build()

    def _build(self) -> None:
        PAD = {"padx": 4, "pady": 2}
        sticky_ew = {"sticky": tk.EW}

        # ── Section 1: New Frame Defaults ─────────────────────────────
        defaults_frame = ttk.LabelFrame(self, text="New Frame Defaults")
        defaults_frame.pack(fill=tk.X, padx=4, pady=(4, 2))
        defaults_frame.columnconfigure(1, weight=1)

        ttk.Label(defaults_frame, text="Episode:").grid(row=0, column=0, **PAD, **sticky_ew)
        self.episode_var = tk.StringVar()
        ttk.Entry(defaults_frame, textvariable=self.episode_var, width=16).grid(
            row=0, column=1, **PAD, **sticky_ew
        )

        ttk.Label(defaults_frame, text="Camera:").grid(row=1, column=0, **PAD, **sticky_ew)
        self.camera_var = tk.StringVar()
        ttk.Entry(defaults_frame, textvariable=self.camera_var, width=16).grid(
            row=1, column=1, **PAD, **sticky_ew
        )

        ttk.Label(
            defaults_frame,
            text="Used when adding new frames. Change freely between videos.",
            foreground="#888888",
            wraplength=220,
            justify=tk.LEFT,
            font=("TkDefaultFont", 8),
        ).grid(row=2, column=0, columnspan=2, **PAD, sticky=tk.W)

        # ── Section 2: Loaded Frame ────────────────────────────────────
        self._frame_section = ttk.LabelFrame(self, text="Loaded Frame")
        self._frame_section.pack(fill=tk.X, padx=4, pady=(2, 4))
        self._frame_section.columnconfigure(1, weight=1)

        # Sample ID display + rename
        ttk.Label(self._frame_section, text="ID:").grid(row=0, column=0, **PAD, sticky=tk.W)
        self._sid_var = tk.StringVar(value="—")
        ttk.Label(
            self._frame_section, textvariable=self._sid_var,
            foreground="#aaaaff", font=("Courier", 8), wraplength=160, justify=tk.LEFT,
        ).grid(row=0, column=1, **PAD, sticky=tk.W)

        self._rename_btn = ttk.Button(
            self._frame_section, text="Rename ID…", command=self._on_rename_click
        )
        self._rename_btn.grid(row=1, column=0, columnspan=2, **PAD, sticky=tk.EW)

        # Rename edit fields (hidden until rename is confirmed)
        self._rename_frame = ttk.Frame(self._frame_section)
        # Not packed initially

        ttk.Label(self._rename_frame, text="New Ep:").pack(side=tk.LEFT)
        self._new_ep_var = tk.StringVar()
        ttk.Entry(self._rename_frame, textvariable=self._new_ep_var, width=10).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Label(self._rename_frame, text="Cam:").pack(side=tk.LEFT)
        self._new_cam_var = tk.StringVar()
        ttk.Entry(self._rename_frame, textvariable=self._new_cam_var, width=8).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(self._rename_frame, text="✓", width=2, command=self._execute_rename).pack(
            side=tk.LEFT
        )
        ttk.Button(self._rename_frame, text="✕", width=2, command=self._cancel_rename).pack(
            side=tk.LEFT, padx=(2, 0)
        )

        ttk.Separator(self._frame_section, orient=tk.HORIZONTAL).grid(
            row=3, column=0, columnspan=2, sticky=tk.EW, pady=2
        )

        # Status
        ttk.Label(self._frame_section, text="Status:").grid(row=4, column=0, **PAD, **sticky_ew)
        self._status_var = tk.StringVar(value="queued")
        ttk.Combobox(
            self._frame_section, textvariable=self._status_var, values=LABEL_STATUSES,
            state="readonly", width=13,
        ).grid(row=4, column=1, **PAD, **sticky_ew)

        # Difficulty
        ttk.Label(self._frame_section, text="Difficulty:").grid(row=5, column=0, **PAD, **sticky_ew)
        self._difficulty_var = tk.StringVar(value="unknown")
        ttk.Combobox(
            self._frame_section, textvariable=self._difficulty_var, values=DIFFICULTIES,
            state="readonly", width=13,
        ).grid(row=5, column=1, **PAD, **sticky_ew)

        # Lighting
        ttk.Label(self._frame_section, text="Lighting:").grid(row=6, column=0, **PAD, **sticky_ew)
        self._lighting_var = tk.StringVar(value="unknown")
        ttk.Combobox(
            self._frame_section, textvariable=self._lighting_var, values=LIGHTING_CONDITIONS,
            state="readonly", width=13,
        ).grid(row=6, column=1, **PAD, **sticky_ew)

        # Notes
        ttk.Label(self._frame_section, text="Notes:").grid(row=7, column=0, **PAD, **sticky_ew)
        self._notes_var = tk.StringVar()
        ttk.Entry(self._frame_section, textvariable=self._notes_var).grid(
            row=7, column=1, **PAD, **sticky_ew
        )

        # Flags
        self._tephra_var = tk.BooleanVar()
        self._smoke_var = tk.BooleanVar()
        self._base_glow_var = tk.BooleanVar()
        ttk.Checkbutton(self._frame_section, text="Tephra", variable=self._tephra_var).grid(
            row=8, column=0, **PAD, sticky=tk.W
        )
        ttk.Checkbutton(self._frame_section, text="Smoke", variable=self._smoke_var).grid(
            row=8, column=1, **PAD, sticky=tk.W
        )
        ttk.Checkbutton(self._frame_section, text="Base glow", variable=self._base_glow_var).grid(
            row=9, column=0, columnspan=2, **PAD, sticky=tk.W
        )

        # Mask provenance
        ttk.Label(self._frame_section, text="Provenance:").grid(
            row=10, column=0, **PAD, **sticky_ew
        )
        self._provenance_var = tk.StringVar(value="human_rough")
        ttk.Combobox(
            self._frame_section, textvariable=self._provenance_var, values=MASK_PROVENANCES,
            state="readonly", width=13,
        ).grid(row=10, column=1, **PAD, **sticky_ew)

        # ── Per-frame metadata flags (hotkey-toggleable) ──────────────
        flags_box = ttk.LabelFrame(self._frame_section, text="Flags (hotkeys)")
        flags_box.grid(row=11, column=0, columnspan=2, padx=2, pady=4, sticky=tk.EW)
        flags_box.columnconfigure(0, weight=1)
        flags_box.columnconfigure(1, weight=1)

        self._flag_vars: dict[str, tk.BooleanVar] = {}
        # Short labels for compactness; full names are the metadata field names.
        flag_labels = {
            "wind_affected": "Wind (W)",
            "falling_tephra_visible": "Falling (T)",
            "cooling_tephra_visible": "Cooling (⇧C)",
            "smoke_obscured": "Smoke (K)",
            "ground_glow_visible": "Grnd glow (G)",
            "exposure_bloom": "Bloom (L)",
            "ambiguous_boundary": "Ambig (U)",
            "hard_negative": "Hard-neg (N)",
            "empty_mask_confirmed": "Empty (0)",
            "needs_review": "Review (J)",
            "bad_frame": "Bad frame",
            "model_draft_present": "Draft present",
            "model_draft_corrected": "Draft corr (P)",
            "human_clean": "Clean (Y)",
        }
        for i, flag in enumerate(METADATA_FLAGS):
            var = tk.BooleanVar(value=False)
            self._flag_vars[flag] = var
            ttk.Checkbutton(
                flags_box, text=flag_labels.get(flag, flag), variable=var,
                command=lambda f=flag: self._on_flag_checkbox(f),
            ).grid(row=i // 2, column=i % 2, padx=2, pady=1, sticky=tk.W)

        # Action buttons
        btn_frame = ttk.Frame(self._frame_section)
        btn_frame.grid(row=20, column=0, columnspan=2, pady=4)
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Complete",
                   command=lambda: self._set_status("complete")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Uncertain",
                   command=lambda: self._set_status("uncertain")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Needs Review",
                   command=lambda: self._set_status("needs_review")).pack(side=tk.LEFT, padx=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_record(self, rec: FrameRecord) -> None:
        """Populate the Loaded Frame section.

        Deliberately does NOT touch episode_var / camera_var — those are
        session-level defaults for adding new frames and must remain free
        to change between videos.
        """
        self._sample_id = rec.sample_id
        self._sid_var.set(rec.sample_id)
        self._status_var.set(rec.label_status)
        self._difficulty_var.set(rec.difficulty)
        self._lighting_var.set(rec.lighting_condition)
        self._notes_var.set(rec.notes)
        self._tephra_var.set(rec.contains_tephra)
        self._smoke_var.set(rec.contains_smoke)
        self._base_glow_var.set(rec.contains_base_glow)
        self._provenance_var.set(rec.mask_provenance or "human_rough")
        for flag, var in self._flag_vars.items():
            var.set(bool(getattr(rec, flag, False)))
        self._cancel_rename()   # make sure rename row is hidden

    # ------------------------------------------------------------------
    # Rename flow (loaded frame only)
    # ------------------------------------------------------------------

    def _on_rename_click(self) -> None:
        old_sid = self._sample_id
        if not old_sid:
            return
        confirmed = messagebox.askyesno(
            "Rename frame identity",
            f"Renaming will move all files for:\n\n"
            f"  {old_sid}\n\n"
            "The image, mask, QC files, and metadata row will all be updated.\n\n"
            "Proceed?",
            icon="warning",
        )
        if not confirmed:
            return
        # Pre-fill with the record's current values
        rec = self.app.metadata.get(old_sid) if self.app.metadata else None
        self._new_ep_var.set(rec.episode_id if rec else "")
        self._new_cam_var.set(rec.camera_id if rec else "")
        self._rename_frame.grid(row=2, column=0, columnspan=2, padx=4, pady=2, sticky=tk.EW)
        self._rename_btn.config(state="disabled")

    def _cancel_rename(self) -> None:
        self._rename_frame.grid_remove()
        self._rename_btn.config(state="normal", text="Rename ID…")

    def _execute_rename(self) -> None:
        old_sid = self._sample_id
        if not old_sid or not self.app.metadata or not self.app.dataset:
            self._cancel_rename()
            return

        new_ep = self._new_ep_var.get().strip().replace(" ", "_") or "unknownEpisode"
        new_cam = self._new_cam_var.get().strip().replace(" ", "_") or "unknownCamera"

        from lava_labeler.core.dataset import make_sample_id
        rec = self.app.metadata.get(old_sid)
        if rec is None:
            self._cancel_rename()
            return
        new_sid = make_sample_id(new_ep, new_cam, rec.frame_index)

        if new_sid != old_sid:
            success = self.app.rename_sample(old_sid, new_sid, new_ep, new_cam)
            if success:
                self._sample_id = new_sid
                self._sid_var.set(new_sid)

        self._cancel_rename()

    # ------------------------------------------------------------------
    # Per-frame metadata apply
    # ------------------------------------------------------------------

    def _apply(self) -> None:
        sid = self._sample_id
        if sid and self.app.metadata:
            # Save the mask to disk first so pixels are never lost on navigation
            # and so mask_positive_pixels is accurate before writing metadata.
            self.app.save_current_mask()
            updates = dict(
                label_status=self._status_var.get(),
                mask_provenance=self._provenance_var.get(),
                difficulty=self._difficulty_var.get(),
                lighting_condition=self._lighting_var.get(),
                notes=self._notes_var.get(),
                contains_tephra=self._tephra_var.get(),
                contains_smoke=self._smoke_var.get(),
                contains_base_glow=self._base_glow_var.get(),
            )
            for flag, var in self._flag_vars.items():
                updates[flag] = var.get()
            self.app.metadata.update(sid, **updates)
            self.app.metadata.save()
            self.app.frame_queue.refresh()
            self.app.mark_clean()
            self.app.set_status(f"Metadata saved for {sid}")

    # ------------------------------------------------------------------
    # Flag hotkey support
    # ------------------------------------------------------------------

    def _on_flag_checkbox(self, flag: str) -> None:
        """User clicked a flag checkbox: persist immediately + autosave."""
        if not self._sample_id or not self.app.metadata:
            return
        value = self._flag_vars[flag].get()
        self.app.apply_flag(flag, value)

    def toggle_flag(self, flag: str) -> bool | None:
        """Toggle *flag* in the UI and return the new value (None if no frame)."""
        var = self._flag_vars.get(flag)
        if var is None or not self._sample_id:
            return None
        new_val = not var.get()
        var.set(new_val)
        return new_val

    def set_flag(self, flag: str, value: bool) -> None:
        var = self._flag_vars.get(flag)
        if var is not None:
            var.set(value)

    def set_provenance(self, provenance: str) -> None:
        self._provenance_var.set(provenance)

    def set_status_value(self, status: str) -> None:
        self._status_var.set(status)

    def _set_status(self, status: str) -> None:
        if status == "complete":
            if not self.app.warn_if_empty_mask_complete():
                return
        self._status_var.set(status)
        self._apply()

