"""Main application window."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np

from lava_labeler.core.dataset import DatasetFolder, make_sample_id
from lava_labeler.core.frame_cache import FrameCache
from lava_labeler.core.metadata import FrameRecord, MetadataStore
from lava_labeler.core.qc import generate_overlay, generate_thumbnail
from lava_labeler.core.video_io import VideoReader
from lava_labeler.gui.export_dialog import ExportDialog
from lava_labeler.gui.frame_queue import FrameQueuePanel
from lava_labeler.gui.labeling_canvas import LabelingCanvas
from lava_labeler.gui.metadata_panel import MetadataPanel
from lava_labeler.gui.toolbar import Toolbar


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Lava Labeler — Fountain Dataset Builder")
        self.geometry("1400x900")
        self.minsize(900, 600)

        # Core state
        self.video_reader: VideoReader | None = None
        self.dataset: DatasetFolder | None = None
        self.metadata: MetadataStore | None = None
        self.frame_cache = FrameCache()          # 512 MB byte-budgeted LRU
        self.current_frame_index: int = 0
        self._active_sample_id: str | None = None

        self._build_menu()
        self._build_layout()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self.quit_app)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open Video…", command=self.open_video, accelerator="Cmd+O")
        file_menu.add_command(label="New Dataset…", command=self.new_dataset)
        file_menu.add_command(label="Open Dataset…", command=self.open_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Save Mask", command=self.save_current_mask, accelerator="Cmd+S")
        file_menu.add_separator()
        file_menu.add_command(label="Export Dataset…", command=self.open_export_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit_app, accelerator="Cmd+Q")
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Cmd+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Cmd+Shift+Z")
        menubar.add_cascade(label="Edit", menu=edit_menu)

    def _build_layout(self) -> None:
        # Top toolbar
        self.toolbar = Toolbar(self, app=self)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Main area (canvas + right panel)
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        # Right panel
        right = ttk.Frame(main, width=270)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=4)
        right.pack_propagate(False)

        self.frame_queue = FrameQueuePanel(right, app=self)
        self.frame_queue.pack(fill=tk.BOTH, expand=True)

        self.metadata_panel = MetadataPanel(right, app=self)
        self.metadata_panel.pack(fill=tk.X, pady=(4, 0))

        # Center canvas
        canvas_frame = ttk.Frame(main)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas = LabelingCanvas(canvas_frame, app=self)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom timeline
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=4, pady=2)
        self._build_timeline(bottom)

        # Status bar
        self.status_var = tk.StringVar(value="Ready. Open a video to begin.")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            fill=tk.X, side=tk.BOTTOM, padx=2
        )

    def _build_timeline(self, parent: ttk.Frame) -> None:
        self.scrubber_var = tk.IntVar(value=0)
        self._scrubber = ttk.Scale(
            parent, from_=0, to=1, orient=tk.HORIZONTAL,
            variable=self.scrubber_var, command=self._on_scrubber,
        )
        self._scrubber.pack(fill=tk.X)

        row = ttk.Frame(parent)
        row.pack(fill=tk.X)

        ttk.Button(row, text="◀◀", width=3, command=lambda: self._jump(-10)).pack(side=tk.LEFT)
        ttk.Button(row, text="◀", width=3, command=lambda: self._jump(-1)).pack(side=tk.LEFT)

        ttk.Label(row, text="Frame:").pack(side=tk.LEFT, padx=(8, 2))
        self._frame_entry_var = tk.StringVar(value="0")
        frame_entry = ttk.Entry(row, textvariable=self._frame_entry_var, width=8)
        frame_entry.pack(side=tk.LEFT)
        frame_entry.bind("<Return>", self._on_frame_entry)

        ttk.Button(row, text="▶", width=3, command=lambda: self._jump(1)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(row, text="▶▶", width=3, command=lambda: self._jump(10)).pack(side=tk.LEFT)

        ttk.Label(row, text="Step:").pack(side=tk.LEFT, padx=(12, 2))
        self.step_var = tk.IntVar(value=30)
        ttk.Spinbox(row, from_=1, to=9999, textvariable=self.step_var, width=6).pack(side=tk.LEFT)
        ttk.Button(row, text="−Step", command=lambda: self._jump(-self.step_var.get())).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(row, text="+Step", command=lambda: self._jump(self.step_var.get())).pack(side=tk.LEFT)
        ttk.Button(row, text="Add Frame", command=self.add_current_frame).pack(side=tk.LEFT, padx=(16, 0))

        self._video_info_var = tk.StringVar(value="No video loaded")
        ttk.Label(row, textvariable=self._video_info_var).pack(side=tk.RIGHT, padx=8)

    def _bind_keys(self) -> None:
        self.bind_all("<Command-o>", lambda _: self.open_video())
        self.bind_all("<Command-s>", lambda _: self.save_current_mask())
        self.bind_all("<Control-s>", lambda _: self.save_current_mask())
        self.bind_all("<Command-z>", lambda _: self.undo())
        self.bind_all("<Command-Z>", lambda _: self.redo())
        self.bind_all("<Command-q>", lambda _: self.quit_app())
        self.bind_all("<Left>",  lambda _: self._jump(-1))
        self.bind_all("<Right>", lambda _: self._jump(1))
        self.bind_all("<Shift-Left>",  lambda _: self._jump(-self.step_var.get()))
        self.bind_all("<Shift-Right>", lambda _: self._jump(self.step_var.get()))

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.MOV *.AVI *.MKV"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            # Guard unsaved changes before switching to a new video
            if not self._check_unsaved(context="before opening a new video"):
                return
            if self.video_reader:
                self.video_reader.close()
            self.video_reader = VideoReader(path)
            self.frame_cache.clear()
            info = self.video_reader.info
            self._scrubber.configure(to=max(1, info.frame_count - 1))
            self.scrubber_var.set(0)
            self.current_frame_index = 0
            self._video_info_var.set(info.summary)
            self.title(f"Lava Labeler — {Path(path).name}")
            self._load_frame(0)
        except Exception as exc:
            messagebox.showerror("Cannot open video", str(exc))

    def _get_frame(self, index: int) -> np.ndarray | None:
        if self.video_reader is None:
            return None
        path = str(self.video_reader.info.path)
        cached = self.frame_cache.get(path, index)
        if cached is not None:
            return cached
        frame = self.video_reader.read_frame(index)
        if frame is not None:
            self.frame_cache.put(path, index, frame)
        return frame

    def _load_frame(self, index: int) -> None:
        if self.video_reader is None:
            return
        info = self.video_reader.info
        index = max(0, min(info.frame_count - 1, index))
        self.current_frame_index = index
        self.scrubber_var.set(index)
        self._frame_entry_var.set(str(index))
        frame = self._get_frame(index)
        if frame is not None:
            self.canvas.set_browse_frame(frame)
        t = index / info.fps if info.fps > 0 else 0.0
        self.set_status(f"Frame {index}/{info.frame_count-1}  t={t:.2f}s")

    def _jump(self, delta: int) -> None:
        self._load_frame(self.current_frame_index + delta)

    def _on_scrubber(self, val: str) -> None:
        try:
            idx = int(float(val))
        except ValueError:
            return
        if idx != self.current_frame_index:
            self._load_frame(idx)

    def _on_frame_entry(self, _event=None) -> None:
        try:
            self._load_frame(int(self._frame_entry_var.get()))
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def new_dataset(self) -> None:
        if not self._check_unsaved(context="before creating a new dataset"):
            return
        path = filedialog.askdirectory(title="Select or Create Dataset Folder")
        if not path:
            return
        ds = DatasetFolder(path)
        ds.create(name=Path(path).name)
        self.dataset = ds
        self.metadata = MetadataStore(ds.root)
        self.frame_queue.refresh()
        self.set_status(f"Dataset created: {path}")

    def open_dataset(self) -> None:
        if not self._check_unsaved(context="before opening another dataset"):
            return
        path = filedialog.askdirectory(title="Open Existing Dataset Folder")
        if not path:
            return
        ds = DatasetFolder(path)
        if not ds.exists:
            messagebox.showerror("Error", "Folder not found.")
            return
        self.dataset = ds
        self.metadata = MetadataStore(ds.root)
        self.frame_queue.refresh()
        self.set_status(f"Dataset opened: {path}")

    # ------------------------------------------------------------------
    # Frame queue
    # ------------------------------------------------------------------

    def add_current_frame(self) -> None:
        if self.video_reader is None:
            messagebox.showwarning("No video", "Open a video file first.")
            return
        if self.dataset is None:
            messagebox.showwarning("No dataset", "Create or open a dataset first.")
            return

        info = self.video_reader.info
        idx = self.current_frame_index
        ep = self.metadata_panel.episode_var.get() or "unknownEpisode"
        cam = self.metadata_panel.camera_var.get() or "unknownCamera"
        sid = make_sample_id(ep, cam, idx)

        if self.metadata.get(sid) is not None:
            messagebox.showinfo("Already queued", f"{sid} is already in the queue.")
            return

        rec = FrameRecord(
            sample_id=sid,
            video_path=str(info.path),
            frame_index=idx,
            source_width=info.width,
            source_height=info.height,
            fps=info.fps,
            episode_id=ep,
            camera_id=cam,
        )
        self.metadata.add(rec)

        # Save source-resolution image immediately
        frame = self._get_frame(idx)
        if frame is not None:
            self.dataset.save_image(sid, frame)

        self.metadata.save()
        self.frame_queue.refresh()
        self.set_status(f"Added frame {idx} → {sid}")

    # ------------------------------------------------------------------
    # Labeling
    # ------------------------------------------------------------------

    def open_frame_for_labeling(self, sample_id: str) -> None:
        if self.dataset is None or self.metadata is None:
            return
        rec = self.metadata.get(sample_id)
        if rec is None:
            return

        # Guard unsaved changes when switching to a different frame
        if (
            self._active_sample_id is not None
            and self._active_sample_id != sample_id
        ):
            if not self._check_unsaved(context=f"before switching away from {self._active_sample_id}"):
                return

        self._active_sample_id = sample_id

        # Load frame from the record's specific video — never use the browse cache
        # directly, because the user may have opened a different video since this
        # frame was queued, which would cause the wrong frame to be displayed.
        frame: np.ndarray | None = None
        vp_path = str(self.video_reader.info.path) if self.video_reader else None
        if vp_path == rec.video_path:
            # Current video matches: use the normal (path-keyed) cache.
            frame = self._get_frame(rec.frame_index)
        else:
            # Different video: check its own cache slot, then open it directly.
            frame = self.frame_cache.get(rec.video_path, rec.frame_index)
            if frame is None:
                try:
                    r = VideoReader(rec.video_path)
                    frame = r.read_frame(rec.frame_index)
                    r.close()
                    if frame is not None:
                        self.frame_cache.put(rec.video_path, rec.frame_index, frame)
                except Exception:
                    pass

        if frame is None:
            frame = self.dataset.load_image(sample_id)

        if frame is None:
            messagebox.showerror("Frame unavailable", f"Cannot load frame for {sample_id}.")
            return

        mask = self.dataset.load_mask(sample_id)
        self.canvas.set_labeling_frame(frame, mask, sample_id)
        self.metadata_panel.load_record(rec)
        # Only advance status if the frame hasn't been worked on yet
        if rec.label_status == "queued":
            self.metadata.update(sample_id, label_status="in_progress")
        self.metadata.save()
        self.frame_queue.refresh()
        self.set_status(f"Labeling: {sample_id}  [{rec.label_status}]")

    def save_current_mask(self) -> None:
        sid = self._active_sample_id
        if sid is None or self.dataset is None or self.metadata is None:
            return
        mask = self.canvas.get_current_mask()
        if mask is None:
            return

        pos_pixels = int(np.sum(mask > 0))
        total = int(mask.size)
        self.dataset.save_mask(sid, mask)
        self.metadata.update(
            sid,
            mask_positive_pixels=pos_pixels,
            mask_positive_fraction=pos_pixels / total if total > 0 else 0.0,
        )
        self.metadata.save()
        self.canvas.mark_saved()

        # QC outputs
        frame = self.canvas.get_current_frame()
        if frame is not None:
            generate_overlay(frame, mask, self.dataset.qc_overlay_path(sid))
            generate_thumbnail(frame, mask, self.dataset.qc_thumb_path(sid))

        self.set_status(f"Saved {sid}  ({pos_pixels} positive px, {pos_pixels/total*100:.1f}%)")

    def warn_if_empty_mask_complete(self) -> bool:
        """Called before marking a frame 'complete'.  Returns True if safe to proceed."""
        mask = self.canvas.get_current_mask()
        if mask is not None and int(np.sum(mask > 0)) == 0:
            answer = messagebox.askyesnocancel(
                "Empty mask",
                "This mask contains no positive pixels.\n\n"
                "Is this a true-negative frame (no visible lava)?\n\n"
                "  Yes  → Mark complete (true negative)\n"
                "  No   → Cancel, keep editing\n"
                "  Cancel → Mark as Needs Review instead",
            )
            if answer is None:      # Cancel → needs_review
                if self._active_sample_id and self.metadata:
                    self.metadata.update(self._active_sample_id, label_status="needs_review")
                    self.metadata.save()
                    self.frame_queue.refresh()
                return False
            if not answer:          # No → keep editing
                return False
            # Yes → allow through as true negative
        return True

    # ------------------------------------------------------------------
    # Edit
    # ------------------------------------------------------------------

    def undo(self) -> None:
        self.canvas.undo()

    def redo(self) -> None:
        self.canvas.redo()

    # ------------------------------------------------------------------
    # Zoom convenience methods (called by toolbar)
    # ------------------------------------------------------------------

    def zoom_fit(self) -> None:
        self.canvas.zoom_fit()

    def zoom_100(self) -> None:
        self.canvas.zoom_100()

    def zoom_in(self) -> None:
        self.canvas.zoom_step(1.25)

    def zoom_out(self) -> None:
        self.canvas.zoom_step(1 / 1.25)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def open_export_dialog(self) -> None:
        if self.dataset is None or self.metadata is None:
            messagebox.showwarning("No dataset", "Open a dataset first.")
            return
        ExportDialog(self, dataset=self.dataset, metadata=self.metadata)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    # ------------------------------------------------------------------
    # Unsaved-changes guard
    # ------------------------------------------------------------------

    def _check_unsaved(self, context: str = "") -> bool:
        """If there are unsaved mask edits, prompt Save / Discard / Cancel.

        Returns True if the caller may proceed, False if the user cancelled.
        """
        if not (self.canvas._mode == "label" and self.canvas._unsaved):
            return True  # nothing to guard

        sid = self._active_sample_id or "current frame"
        prompt = f"Unsaved mask changes on\n{sid}"
        if context:
            prompt += f"\n({context})"
        answer = messagebox.askyesnocancel(
            "Unsaved changes",
            prompt + "\n\nSave changes before continuing?",
        )
        if answer is None:   # Cancel → stay put
            return False
        if answer:           # Yes → save, then proceed
            self.save_current_mask()
        # False (Discard) or saved: proceed
        return True

    # ------------------------------------------------------------------
    # Quit
    # ------------------------------------------------------------------

    def quit_app(self) -> None:
        if not self._check_unsaved(context="before quitting"):
            return
        if self.video_reader:
            self.video_reader.close()
        self.destroy()


def main() -> None:
    app = App()
    app.mainloop()
