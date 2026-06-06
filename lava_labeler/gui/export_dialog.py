"""Export dialog: validate dataset and write final output structure."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

from lava_labeler.core.validation import validate_dataset

if TYPE_CHECKING:
    from lava_labeler.core.dataset import DatasetFolder
    from lava_labeler.core.metadata import MetadataStore


class ExportDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Widget,
        dataset: "DatasetFolder",
        metadata: "MetadataStore",
    ) -> None:
        super().__init__(parent)
        self.title("Export Dataset")
        self.resizable(True, True)
        self.geometry("640x480")
        self._dataset = dataset
        self._metadata = metadata
        self._build()
        self.transient(parent)
        self.grab_set()

    def _build(self) -> None:
        # Summary
        records = self._metadata.all_records()
        total = len(records)
        complete = sum(1 for r in records if r.label_status == "complete")
        uncertain = sum(1 for r in records if r.label_status == "uncertain")
        queued = sum(1 for r in records if r.label_status == "queued")

        info = (
            f"Dataset folder:  {self._dataset.root}\n"
            f"Total frames:    {total}\n"
            f"Complete:        {complete}\n"
            f"Uncertain:       {uncertain}\n"
            f"Queued / other:  {queued}\n"
        )
        ttk.Label(self, text=info, justify=tk.LEFT, font=("Courier", 10)).pack(
            anchor=tk.W, padx=12, pady=8
        )

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)

        # Validation log
        ttk.Label(self, text="Validation log:", font=("TkDefaultFont", 10, "bold")).pack(
            anchor=tk.W, padx=12, pady=(8, 2)
        )

        log_frame = ttk.Frame(self)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        sb = ttk.Scrollbar(log_frame, orient=tk.VERTICAL)
        self._log_text = tk.Text(
            log_frame,
            yscrollcommand=sb.set,
            font=("Courier", 9),
            bg="#1e1e1e",
            fg="#cccccc",
            state=tk.DISABLED,
            wrap=tk.NONE,
        )
        sb.config(command=self._log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btn_frame, text="Validate", command=self._run_validation).pack(side=tk.LEFT, padx=4)
        self._export_btn = ttk.Button(
            btn_frame, text="Export (finalize metadata)", command=self._run_export, state=tk.DISABLED
        )
        self._export_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=4)

        # Run validation immediately on open
        self.after(100, self._run_validation)

    # ------------------------------------------------------------------

    def _log(self, text: str) -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, text + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _run_validation(self) -> None:
        self._clear_log()
        self._log("Running validation…\n")
        issues = validate_dataset(self._dataset.root)

        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]

        if not issues:
            self._log("✓  No issues found. Dataset looks clean.")
            self._export_btn.config(state=tk.NORMAL)
        else:
            for issue in issues:
                self._log(str(issue))
            self._log(
                f"\n{len(errors)} error(s)  {len(warnings)} warning(s)"
            )
            if errors:
                self._export_btn.config(state=tk.DISABLED)
                self._log("\nFix errors before exporting.")
            else:
                self._export_btn.config(state=tk.NORMAL)
                self._log("\nWarnings only — export is allowed.")

    def _run_export(self) -> None:
        """Finalise metadata CSV and write validation log to qc/."""
        try:
            self._metadata.save()
            issues = validate_dataset(self._dataset.root)
            log_path = self._dataset.root / "qc" / "validation_log.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as f:
                if issues:
                    for issue in issues:
                        f.write(str(issue) + "\n")
                else:
                    f.write("No issues found.\n")
            self._log(f"\n✓  Export complete. Validation log: {log_path}")
            messagebox.showinfo(
                "Export complete",
                f"Metadata saved.\nValidation log written to:\n{log_path}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)
