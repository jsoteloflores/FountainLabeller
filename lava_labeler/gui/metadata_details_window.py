"""Metadata Details window — full dataset accounting summary.

Opens as a Toplevel (non-modal) showing:
  1. Current video metadata (registry fields)
  2. Per-video / per-episode / per-camera counts
  3. Whole-dataset flag counts
  4. Validation warnings
  5. CSV file paths and last regeneration time
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lava_labeler.gui.app import App


class MetadataDetailsWindow(tk.Toplevel):
    """Non-modal dataset accounting window."""

    def __init__(self, app: "App") -> None:
        super().__init__(app)
        self.app = app
        self.title("Dataset Details")
        self.geometry("780x680")
        self.minsize(600, 500)
        self.resizable(True, True)
        self._build()
        self.refresh()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # Notebook for tabbed sections
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._tab_video   = ttk.Frame(nb)
        self._tab_counts  = ttk.Frame(nb)
        self._tab_flags   = ttk.Frame(nb)
        self._tab_files   = ttk.Frame(nb)
        nb.add(self._tab_video,  text="Current Video")
        nb.add(self._tab_counts, text="Counts by Scope")
        nb.add(self._tab_flags,  text="Flag Counts")
        nb.add(self._tab_files,  text="Files")

        self._build_video_tab()
        self._build_counts_tab()
        self._build_flags_tab()
        self._build_files_tab()

        # Bottom bar
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side=tk.LEFT)
        ttk.Button(bar, text="Regenerate CSVs", command=self._regen_csvs).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(bar, text="Close", command=self.destroy).pack(side=tk.RIGHT)

    # --- Video tab ---

    def _build_video_tab(self) -> None:
        f = self._tab_video
        self._video_rows: dict[str, tk.StringVar] = {}
        fields = [
            ("video_id",          "Video ID:"),
            ("video_filename",    "Filename:"),
            ("video_path",        "Path:"),
            ("episode_id",        "Episode:"),
            ("camera_id",         "Camera:"),
            ("eruption_id",       "Eruption ID:"),
            ("source_date",       "Source date:"),
            ("total_frames",      "Total frames:"),
            ("fps",               "FPS:"),
            ("resolution",        "Resolution:"),
            ("duration_seconds",  "Duration (s):"),
            ("file_size_bytes",   "File size:"),
            ("file_fingerprint",  "Fingerprint:"),
            ("created_at",        "First opened:"),
            ("last_opened_at",    "Last opened:"),
            ("notes",             "Notes:"),
        ]
        for key, label in fields:
            row = ttk.Frame(f)
            row.pack(fill=tk.X, padx=6, pady=1)
            ttk.Label(row, text=label, width=18, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            ttk.Label(row, textvariable=var, anchor=tk.W, wraplength=500).pack(
                side=tk.LEFT, fill=tk.X, expand=True
            )
            self._video_rows[key] = var

        # Edit button
        edit_bar = ttk.Frame(f)
        edit_bar.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(edit_bar, text="Edit Video Metadata…",
                   command=self._edit_video_metadata).pack(side=tk.LEFT)

    # --- Counts tab ---

    def _build_counts_tab(self) -> None:
        f = self._tab_counts
        scroll = _ScrolledFrame(f)
        scroll.pack(fill=tk.BOTH, expand=True)
        self._counts_frame = scroll.inner

    def _populate_counts(self) -> None:
        for w in self._counts_frame.winfo_children():
            w.destroy()

        ds = self.app.dataset_summary
        if ds is None:
            ttk.Label(self._counts_frame, text="No dataset open.").pack(padx=8, pady=8)
            return

        # Columns
        col_labels = [
            "Scope", "ID",
            "Cand.", "Labeled", "Unlabeled", "Skipped", "NR",
            "Pos.Masks", "Empty", "H.Neg.", "Videos",
        ]
        _header_row(self._counts_frame, col_labels, row=0)

        r = 1
        for stats in ds.all_rows():
            vals = [
                stats.scope, stats.scope_id,
                stats.total_candidates, stats.total_labeled,
                stats.total_unlabeled, stats.total_skipped,
                stats.total_needs_review,
                stats.positive_masks, stats.empty_confirmed,
                stats.hard_negative, stats.video_count,
            ]
            bg = "#1a2a1a" if stats.scope == "dataset" else None
            _data_row(self._counts_frame, vals, row=r, bg=bg)
            r += 1

    # --- Flags tab ---

    def _build_flags_tab(self) -> None:
        f = self._tab_flags
        scroll = _ScrolledFrame(f)
        scroll.pack(fill=tk.BOTH, expand=True)
        self._flags_frame = scroll.inner

    def _populate_flags(self) -> None:
        for w in self._flags_frame.winfo_children():
            w.destroy()

        ds = self.app.dataset_summary
        if ds is None:
            ttk.Label(self._flags_frame, text="No dataset open.").pack(padx=8, pady=8)
            return

        flags = [
            ("wind_affected",           "Wind affected"),
            ("falling_tephra_visible",  "Falling tephra visible"),
            ("cooling_tephra_visible",  "Cooling tephra visible"),
            ("smoke_obscured",          "Smoke obscured"),
            ("ground_glow_visible",     "Ground glow visible"),
            ("exposure_bloom",          "Exposure bloom"),
            ("ambiguous_boundary",      "Ambiguous boundary"),
            ("hard_negative",           "Hard negative"),
            ("empty_confirmed",         "Empty confirmed"),
        ]

        col_labels = ["Flag", "Dataset", "By episode →"]
        _header_row(self._flags_frame, col_labels, row=0, col_widths=[24, 8, 0])

        for r, (flag, label) in enumerate(flags, start=1):
            ds_val = getattr(ds.dataset, flag, "—")
            ep_parts = ", ".join(
                f"{eid}:{getattr(s, flag, 0)}"
                for eid, s in sorted(ds.by_episode.items())
                if getattr(s, flag, 0) > 0
            ) or "—"
            _data_row(self._flags_frame, [label, ds_val, ep_parts], row=r, col_widths=[24, 8, 0])

    # --- Files tab ---

    def _build_files_tab(self) -> None:
        f = self._tab_files
        scroll = _ScrolledFrame(f)
        scroll.pack(fill=tk.BOTH, expand=True)
        self._files_frame = scroll.inner

    def _populate_files(self) -> None:
        for w in self._files_frame.winfo_children():
            w.destroy()

        app = self.app
        if app.dataset is None:
            ttk.Label(self._files_frame, text="No dataset open.").pack(padx=8, pady=8)
            return

        root = app.dataset.root
        files = [
            ("video_registry.json",               "Video registry (JSON)"),
            ("video_registry.csv",                "Video registry (CSV)"),
            ("frame_metadata.csv",                "Frame metadata (CSV)"),
            ("metadata/frames.csv",               "Internal frames.csv"),
            ("dataset_summary.csv",               "Dataset summary (CSV)"),
            ("labels_manifest.csv",               "Training manifest (CSV)"),
            ("export_validation_report.csv",      "Validation report (CSV)"),
            ("session_recovery.json",             "Session recovery"),
        ]

        ttk.Label(self._files_frame, text="Dataset root: " + str(root),
                  anchor=tk.W, wraplength=700).grid(
            row=0, column=0, columnspan=3, sticky=tk.W, padx=6, pady=4)

        for r, (name, label) in enumerate(files, start=1):
            p = root / name
            exists = p.exists()
            mtime = ""
            if exists:
                try:
                    mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                except OSError:
                    mtime = "?"

            ttk.Label(self._files_frame, text=label, width=28, anchor=tk.W).grid(
                row=r, column=0, sticky=tk.W, padx=6, pady=1)
            status_lbl = ttk.Label(
                self._files_frame,
                text="✓" if exists else "missing",
                foreground="#81c784" if exists else "#e57373",
                width=8,
            )
            status_lbl.grid(row=r, column=1, padx=2)
            ttk.Label(self._files_frame, text=mtime, anchor=tk.W, foreground="#888").grid(
                row=r, column=2, sticky=tk.W, padx=4)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        app = self.app

        # Video tab
        ve = None
        if app.video_registry is not None and app._active_video_id:
            ve = app.video_registry.get(app._active_video_id)

        if ve is not None:
            self._video_rows["video_id"].set(ve.video_id)
            self._video_rows["video_filename"].set(ve.video_filename)
            self._video_rows["video_path"].set(ve.video_path)
            self._video_rows["episode_id"].set(ve.episode_id or "—")
            self._video_rows["camera_id"].set(ve.camera_id or "—")
            self._video_rows["eruption_id"].set(ve.eruption_id or "—")
            self._video_rows["source_date"].set(ve.source_date or "—")
            self._video_rows["total_frames"].set(str(ve.total_frames))
            self._video_rows["fps"].set(f"{ve.fps:.3f}")
            self._video_rows["resolution"].set(f"{ve.width}×{ve.height}")
            self._video_rows["duration_seconds"].set(f"{ve.duration_seconds:.2f}")
            self._video_rows["file_size_bytes"].set(_fmt_bytes(ve.file_size_bytes))
            self._video_rows["file_fingerprint"].set(ve.file_fingerprint or "—")
            self._video_rows["created_at"].set(ve.created_at or "—")
            self._video_rows["last_opened_at"].set(ve.last_opened_at or "—")
            self._video_rows["notes"].set(ve.notes or "—")
        else:
            if app.video_reader is not None:
                info = app.video_reader.info
                self._video_rows["video_filename"].set(info.path.name)
                self._video_rows["video_path"].set(str(info.path))
                self._video_rows["total_frames"].set(str(info.frame_count))
                self._video_rows["fps"].set(f"{info.fps:.3f}")
                self._video_rows["resolution"].set(f"{info.width}×{info.height}")
                self._video_rows["duration_seconds"].set(f"{info.duration_seconds:.2f}")
            for k in ("video_id", "episode_id", "camera_id", "eruption_id",
                      "source_date", "file_size_bytes", "file_fingerprint",
                      "created_at", "last_opened_at", "notes"):
                self._video_rows[k].set("—")

        self._populate_counts()
        self._populate_flags()
        self._populate_files()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _regen_csvs(self) -> None:
        self.app._regenerate_csv_mirrors(force=True)
        self.refresh()
        self.app.set_status("CSV mirrors regenerated.")

    def _edit_video_metadata(self) -> None:
        """Open a small dialog to edit episode/camera/notes for current video."""
        app = self.app
        if app.video_registry is None or not app._active_video_id:
            return
        ve = app.video_registry.get(app._active_video_id)
        if ve is None:
            return
        _VideoMetadataEditor(self, app, ve)


# ---------------------------------------------------------------------------
# Video metadata editor dialog
# ---------------------------------------------------------------------------

class _VideoMetadataEditor(tk.Toplevel):
    def __init__(self, parent: tk.Widget, app: "App", entry) -> None:
        super().__init__(parent)
        self.app = app
        self.entry = entry
        self.title("Edit Video Metadata")
        self.resizable(False, False)
        self._build()
        self.grab_set()

    def _build(self) -> None:
        f = ttk.Frame(self, padding=10)
        f.pack(fill=tk.BOTH, expand=True)

        self._vars: dict[str, tk.StringVar] = {}
        fields = [
            ("episode_id",  "Episode ID:"),
            ("camera_id",   "Camera ID:"),
            ("eruption_id", "Eruption ID:"),
            ("source_date", "Source date:"),
            ("notes",       "Notes:"),
        ]
        for r, (key, lbl) in enumerate(fields):
            ttk.Label(f, text=lbl, width=14, anchor=tk.W).grid(
                row=r, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar(value=getattr(self.entry, key, ""))
            ttk.Entry(f, textvariable=var, width=36).grid(
                row=r, column=1, sticky=tk.EW, padx=(4, 0))
            self._vars[key] = var

        btn_row = ttk.Frame(f)
        btn_row.grid(row=len(fields), column=0, columnspan=2, pady=(8, 0))
        ttk.Button(btn_row, text="Save", command=self._save).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=(4, 0))

    def _save(self) -> None:
        kwargs = {k: v.get().strip() for k, v in self._vars.items()}
        self.app.video_registry.update_entry(self.entry.video_id, **kwargs)
        self.app.video_registry.save_csv()
        # Propagate episode/camera to active metadata records for this video
        if self.app.metadata and kwargs.get("episode_id"):
            for rec in self.app.metadata.all_records():
                if rec.video_id == self.entry.video_id:
                    self.app.metadata.update(
                        rec.sample_id,
                        episode_id=kwargs["episode_id"] or rec.episode_id,
                        camera_id=kwargs.get("camera_id") or rec.camera_id,
                    )
            self.app.metadata.save()
        if self.app.dataset_summary:
            self.app.dataset_summary.refresh()
        if hasattr(self.app, "dataset_context_panel"):
            self.app.dataset_context_panel.refresh()
        # Refresh parent details window
        self.master.refresh()
        self.app.set_status(f"Updated video metadata for {self.entry.video_id}")
        self.destroy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ScrolledFrame(ttk.Frame):
    """A frame with a vertical scrollbar."""

    def __init__(self, parent: tk.Widget, **kw) -> None:
        super().__init__(parent, **kw)
        vsb = ttk.Scrollbar(self, orient=tk.VERTICAL)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas = tk.Canvas(self, yscrollcommand=vsb.set, bd=0, highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self._win = canvas.create_window((0, 0), window=self.inner, anchor=tk.NW)
        self.inner.bind("<Configure>", lambda _: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(self._win, width=e.width))


def _header_row(parent: tk.Widget, labels: list[str], row: int,
                col_widths: list[int] | None = None) -> None:
    for c, lbl in enumerate(labels):
        w = (col_widths[c] if col_widths and c < len(col_widths) else 10) or 0
        kw: dict = {"anchor": tk.W, "font": ("", 9, "bold")}
        if w:
            kw["width"] = w
        ttk.Label(parent, text=lbl, **kw).grid(
            row=row, column=c, sticky=tk.W, padx=3, pady=1)


def _data_row(parent: tk.Widget, vals: list, row: int,
              bg: str | None = None,
              col_widths: list[int] | None = None) -> None:
    for c, val in enumerate(vals):
        w = (col_widths[c] if col_widths and c < len(col_widths) else 10) or 0
        kw: dict = {"anchor": tk.W}
        if w:
            kw["width"] = w
        if bg:
            tk.Label(parent, text=str(val), bg=bg, **kw).grid(
                row=row, column=c, sticky=tk.W, padx=3, pady=0)
        else:
            ttk.Label(parent, text=str(val), **kw).grid(
                row=row, column=c, sticky=tk.W, padx=3, pady=0)


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1e9:.2f} GB"
    if n >= 1_000_000:
        return f"{n/1e6:.1f} MB"
    if n >= 1_000:
        return f"{n/1e3:.0f} KB"
    return f"{n} B"
