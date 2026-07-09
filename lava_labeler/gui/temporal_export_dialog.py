"""2.5D temporal dataset export dialog.

Lets the user configure and run a temporal export package: center images,
masks, metadata copies, surrounding temporal frames (using the center ROI),
manifests, and optional QC contact sheets.

Runs a readiness check when opened so the user sees how many rows are
2.5D-ready before exporting.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

from lava_labeler.core.temporal_export import (
    EDGE_POLICIES,
    EXPORTABLE_STATUSES,
    WINDOW_MODES,
    TemporalExportConfig,
    check_temporal_readiness,
    export_temporal_dataset,
)

if TYPE_CHECKING:
    from lava_labeler.core.dataset import DatasetFolder
    from lava_labeler.core.metadata import MetadataStore
    from lava_labeler.core.video_registry import VideoRegistry


class TemporalExportDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Widget,
        dataset: "DatasetFolder",
        metadata: "MetadataStore",
        video_registry: "VideoRegistry | None",
    ) -> None:
        super().__init__(parent)
        self.title("Export 2.5D Temporal Dataset")
        self.resizable(True, True)
        self.geometry("640x640")
        self._dataset = dataset
        self._metadata = metadata
        self._video_registry = video_registry

        # Config vars
        self._output_var = tk.StringVar(value="")
        self._radius_var = tk.IntVar(value=2)
        self._stride_var = tk.IntVar(value=1)
        self._window_var = tk.StringVar(value="centered")
        self._edge_var = tk.StringVar(value="skip")
        self._qc_var = tk.BooleanVar(value=True)
        self._overwrite_var = tk.BooleanVar(value=False)
        self._status_vars = {
            s: tk.BooleanVar(value=s in EXPORTABLE_STATUSES)
            for s in ("complete", "hard_negative", "empty_confirmed",
                      "uncertain", "needs_review")
        }

        self._build()
        self.transient(parent)
        self.grab_set()
        self._refresh_readiness()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # Output folder
        out_frame = ttk.Frame(self)
        out_frame.pack(fill=tk.X, **pad)
        ttk.Label(out_frame, text="Output folder:").pack(side=tk.LEFT)
        ttk.Entry(out_frame, textvariable=self._output_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(out_frame, text="Browse…", command=self._browse_output).pack(side=tk.LEFT)

        # Temporal settings
        grid = ttk.Frame(self)
        grid.pack(fill=tk.X, **pad)
        ttk.Label(grid, text="Temporal radius:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(grid, from_=0, to=30, textvariable=self._radius_var, width=6).grid(
            row=0, column=1, sticky=tk.W, padx=6)
        ttk.Label(grid, text="Temporal stride:").grid(row=0, column=2, sticky=tk.W)
        ttk.Spinbox(grid, from_=1, to=30, textvariable=self._stride_var, width=6).grid(
            row=0, column=3, sticky=tk.W, padx=6)

        ttk.Label(grid, text="Window mode:").grid(row=1, column=0, sticky=tk.W)
        ttk.Combobox(grid, textvariable=self._window_var, values=list(WINDOW_MODES),
                     state="readonly", width=10).grid(row=1, column=1, sticky=tk.W, padx=6)
        ttk.Label(grid, text="Edge policy:").grid(row=1, column=2, sticky=tk.W)
        ttk.Combobox(grid, textvariable=self._edge_var, values=list(EDGE_POLICIES),
                     state="readonly", width=10).grid(row=1, column=3, sticky=tk.W, padx=6)

        # Statuses
        st_frame = ttk.LabelFrame(self, text="Statuses to include")
        st_frame.pack(fill=tk.X, **pad)
        for i, (status, var) in enumerate(self._status_vars.items()):
            ttk.Checkbutton(st_frame, text=status, variable=var,
                            command=self._refresh_readiness).grid(
                row=i // 3, column=i % 3, sticky=tk.W, padx=8, pady=2)

        # Options
        opt = ttk.Frame(self)
        opt.pack(fill=tk.X, **pad)
        ttk.Checkbutton(opt, text="Include QC contact sheets",
                        variable=self._qc_var).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(opt, text="Overwrite existing",
                        variable=self._overwrite_var).pack(side=tk.LEFT, padx=6)

        # Readiness / log
        ttk.Label(self, text="Readiness & log:", font=("TkDefaultFont", 10, "bold")).pack(
            anchor=tk.W, padx=10, pady=(8, 2))
        log_frame = ttk.Frame(self)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self._log = tk.Text(log_frame, height=12, wrap=tk.WORD, font=("Courier", 10))
        sb = ttk.Scrollbar(log_frame, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        self._log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons
        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(btns, text="Re-check readiness", command=self._refresh_readiness).pack(side=tk.LEFT)
        self._export_btn = ttk.Button(btns, text="Export…", command=self._on_export)
        self._export_btn.pack(side=tk.RIGHT)
        ttk.Button(btns, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=6)

    # ------------------------------------------------------------------
    def _browse_output(self) -> None:
        folder = filedialog.askdirectory(title="Select export output folder")
        if folder:
            self._output_var.set(folder)

    def _log_write(self, text: str) -> None:
        self._log.insert(tk.END, text + "\n")
        self._log.see(tk.END)

    def _selected_statuses(self) -> tuple[str, ...]:
        return tuple(s for s, v in self._status_vars.items() if v.get())

    def _make_config(self) -> TemporalExportConfig:
        return TemporalExportConfig(
            output_root=Path(self._output_var.get()),
            temporal_radius=int(self._radius_var.get()),
            temporal_stride=int(self._stride_var.get()),
            window_mode=self._window_var.get(),
            edge_policy=self._edge_var.get(),
            statuses=self._selected_statuses(),
            include_qc_contact_sheets=bool(self._qc_var.get()),
            overwrite_existing=bool(self._overwrite_var.get()),
        )

    def _refresh_readiness(self) -> None:
        try:
            cfg = self._make_config()
            _rows, summary = check_temporal_readiness(self._dataset, self._metadata, cfg)
        except Exception as exc:  # noqa: BLE001
            self._log_write(f"Readiness check failed: {exc}")
            return
        self._log.delete("1.0", tk.END)
        self._log_write("Temporal readiness:")
        self._log_write(f"  Total rows checked:     {summary['total_rows_checked']}")
        self._log_write(f"  Exportable label rows:  {summary['exportable_label_rows']}")
        self._log_write(f"  2.5D ready:             {summary['ready']}")
        self._log_write(f"  Missing video_id:       {summary['missing_video_id']}")
        self._log_write(f"  Missing video file:     {summary['missing_video_file']}")
        self._log_write(f"  Missing fps:            {summary['missing_fps']}")
        self._log_write(f"  Invalid ROI:            {summary['invalid_roi']}")
        self._log_write(f"  Edge-window skipped:    {summary['edge_window_skipped']}")

    # ------------------------------------------------------------------
    def _on_export(self) -> None:
        if not self._output_var.get():
            messagebox.showwarning("No output folder", "Choose an output folder first.")
            return
        if not self._selected_statuses():
            messagebox.showwarning("No statuses", "Select at least one status to include.")
            return
        if not messagebox.askyesno(
            "Export 2.5D dataset",
            "Run the 2.5D temporal export now?\n\n"
            "This reads the source videos and can take a while.",
        ):
            return

        self._export_btn.configure(state=tk.DISABLED)
        self._log_write("\nExporting… (this reads source videos)")
        cfg = self._make_config()
        threading.Thread(target=self._run_export, args=(cfg,), daemon=True).start()

    def _run_export(self, cfg: TemporalExportConfig) -> None:
        try:
            summary = export_temporal_dataset(
                self._dataset, self._metadata, self._video_registry, cfg)
        except Exception as exc:  # noqa: BLE001
            self.after(0, lambda: self._export_done(None, str(exc)))
            return
        self.after(0, lambda: self._export_done(summary, ""))

    def _export_done(self, summary: dict | None, error: str) -> None:
        self._export_btn.configure(state=tk.NORMAL)
        if error:
            self._log_write(f"Export failed: {error}")
            messagebox.showerror("Export failed", error)
            return
        self._log_write("\nExport complete.")
        self._log_write(f"  Samples exported: {summary['samples_exported']}")
        self._log_write(f"  Samples skipped:  {summary['samples_skipped']}")
        self._log_write(f"  Frames/sample:    {summary['temporal_frames_per_sample']}")
        self._log_write(f"  Output: {summary['output_root']}")
        messagebox.showinfo(
            "Export complete",
            f"Exported {summary['samples_exported']} samples "
            f"({summary['samples_skipped']} skipped).\n\n{summary['output_root']}",
        )
