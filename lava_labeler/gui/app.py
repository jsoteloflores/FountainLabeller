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
from lava_labeler.gui.roi_panel import ROIPanel
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

        # ROI state
        self._roi_mode: str = "full_frame"           # "full_frame" | "fixed_roi_crop"
        self._roi_size_policy: str = "global_fixed"  # "none" | "global_fixed" | "camera_fixed"
        self._roi_x: int = 0
        self._roi_y: int = 0
        self._roi_w: int = 1280
        self._roi_h: int = 960

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

        # ROI panel (below main toolbar)
        self.roi_panel = ROIPanel(self, app=self)
        self.roi_panel.pack(side=tk.TOP, fill=tk.X, padx=4, pady=1)
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
        # ROI nudge — Ctrl+Arrow (1 px), Ctrl+Shift+Arrow (10 px)
        self.bind_all("<Control-Left>",        lambda _: self.nudge_roi(-1, 0))
        self.bind_all("<Control-Right>",       lambda _: self.nudge_roi(1, 0))
        self.bind_all("<Control-Up>",          lambda _: self.nudge_roi(0, -1))
        self.bind_all("<Control-Down>",        lambda _: self.nudge_roi(0, 1))
        self.bind_all("<Control-Shift-Left>",  lambda _: self.nudge_roi(-10, 0))
        self.bind_all("<Control-Shift-Right>", lambda _: self.nudge_roi(10, 0))
        self.bind_all("<Control-Shift-Up>",    lambda _: self.nudge_roi(0, -10))
        self.bind_all("<Control-Shift-Down>",  lambda _: self.nudge_roi(0, 10))

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
            self._load_frame(0, fit=True)
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

    def _load_frame(self, index: int, fit: bool = False) -> None:
        if self.video_reader is None:
            return
        info = self.video_reader.info
        index = max(0, min(info.frame_count - 1, index))
        self.current_frame_index = index
        self.scrubber_var.set(index)
        self._frame_entry_var.set(str(index))
        frame = self._get_frame(index)
        if frame is not None:
            self.canvas.set_browse_frame(frame, fit=fit)
        # Sync ROI overlay with current mode
        if self._roi_mode == "fixed_roi_crop":
            self.canvas.set_roi(self._roi_x, self._roi_y, self._roi_w, self._roi_h)
        else:
            self.canvas.clear_roi()
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

        frame = self._get_frame(idx)
        if frame is None:
            messagebox.showerror("Frame unavailable", f"Cannot load frame {idx}.")
            return

        # Determine ROI export geometry
        is_roi = self._roi_mode == "fixed_roi_crop"
        if is_roi:
            rx, ry, rw, rh = self._clamped_roi(info.width, info.height)
            save_frame = frame[ry:ry + rh, rx:rx + rw]
            roi_mode_val = "fixed_roi_crop"
            roi_policy = self._roi_size_policy
        else:
            rx, ry = 0, 0
            rw, rh = info.width, info.height
            save_frame = frame
            roi_mode_val = "full_frame"
            roi_policy = "none"

        rec = FrameRecord(
            sample_id=sid,
            video_path=str(info.path),
            frame_index=idx,
            source_width=info.width,
            source_height=info.height,
            fps=info.fps,
            episode_id=ep,
            camera_id=cam,
            is_roi_crop=is_roi,
            roi_mode=roi_mode_val,
            roi_size_policy=roi_policy,
            roi_x=rx,
            roi_y=ry,
            roi_width=rw,
            roi_height=rh,
        )
        self.metadata.add(rec)
        self.dataset.save_image(sid, save_frame)
        self.metadata.save()
        self.frame_queue.refresh()
        mode_str = f"ROI {rw}\u00d7{rh} @ ({rx},{ry})" if is_roi else "full frame"
        self.set_status(f"Added frame {idx} \u2192 {sid}  [{mode_str}]")

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

        # Load the frame image.
        # For ROI crops the saved image is already the crop — always load from disk
        # so the canvas receives the right-sized array and mask alignment is exact.
        frame: np.ndarray | None = None
        if rec.is_roi_crop:
            frame = self.dataset.load_image(sample_id)
        else:
            # Full-frame: try video cache first, fall back to saved image
            vp_path = str(self.video_reader.info.path) if self.video_reader else None
            if vp_path == rec.video_path:
                frame = self._get_frame(rec.frame_index)
            else:
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

    # ------------------------------------------------------------------
    # ROI management
    # ------------------------------------------------------------------

    def _clamped_roi(self, src_w: int, src_h: int) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) clamped so the ROI fits inside the source frame."""
        rw = min(self._roi_w, src_w)
        rh = min(self._roi_h, src_h)
        rx = max(0, min(self._roi_x, src_w - rw))
        ry = max(0, min(self._roi_y, src_h - rh))
        return rx, ry, rw, rh

    def set_roi_mode(self, mode: str) -> None:
        self._roi_mode = mode
        if mode == "fixed_roi_crop":
            self.canvas.set_roi(self._roi_x, self._roi_y, self._roi_w, self._roi_h)
        else:
            self.canvas.clear_roi()

    def set_roi_size(self, w: int, h: int) -> None:
        """Validate and apply new ROI dimensions. Shows error if too large for current video."""
        if self.video_reader:
            info = self.video_reader.info
            if w > info.width or h > info.height:
                messagebox.showerror(
                    "ROI too large",
                    f"ROI {w}\u00d7{h} exceeds source frame {info.width}\u00d7{info.height}.",
                )
                return
        self._roi_w = max(1, w)
        self._roi_h = max(1, h)
        # Re-clamp position
        if self.video_reader:
            info = self.video_reader.info
            self._roi_x = min(self._roi_x, max(0, info.width - self._roi_w))
            self._roi_y = min(self._roi_y, max(0, info.height - self._roi_h))
        if self._roi_mode == "fixed_roi_crop":
            self.canvas.set_roi(self._roi_x, self._roi_y, self._roi_w, self._roi_h)

    def clear_roi(self) -> None:
        self._roi_x = 0
        self._roi_y = 0
        self.canvas.clear_roi()
        if hasattr(self, "roi_panel"):
            self.roi_panel.update_position_display(0, 0)

    def save_roi_placement(self) -> None:
        """Display current ROI placement in the status bar."""
        self.set_status(
            f"ROI placement: x={self._roi_x}, y={self._roi_y}, "
            f"w={self._roi_w}, h={self._roi_h}  (used for next Add Frame)"
        )

    def on_roi_dragged(self, x: int, y: int, w: int, h: int) -> None:
        """Called by LabelingCanvas when the ROI is drag-repositioned."""
        self._roi_x = x
        self._roi_y = y
        self._roi_w = w
        self._roi_h = h
        if hasattr(self, "roi_panel"):
            self.roi_panel.update_position_display(x, y)

    def nudge_roi(self, dx: int, dy: int) -> None:
        """Nudge the ROI by (dx, dy) pixels (keyboard shortcut)."""
        if self._roi_mode != "fixed_roi_crop":
            return
        new_x = self._roi_x + dx
        new_y = self._roi_y + dy
        if self.video_reader:
            info = self.video_reader.info
            new_x = max(0, min(new_x, info.width - self._roi_w))
            new_y = max(0, min(new_y, info.height - self._roi_h))
        else:
            new_x = max(0, new_x)
            new_y = max(0, new_y)
        self._roi_x = new_x
        self._roi_y = new_y
        self.canvas.set_roi(self._roi_x, self._roi_y, self._roi_w, self._roi_h)
        if hasattr(self, "roi_panel"):
            self.roi_panel.update_position_display(self._roi_x, self._roi_y)

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

    def rename_sample(self, old_sid: str, new_sid: str, new_ep: str, new_cam: str) -> bool:
        """Rename all files and metadata for *old_sid* → *new_sid*.

        Moves image, mask, and QC files; updates the metadata row; updates
        _active_sample_id if it matches. Returns True on success.
        """
        if self.dataset is None or self.metadata is None:
            return False

        if self.metadata.get(new_sid) is not None:
            from tkinter import messagebox
            messagebox.showerror(
                "Rename conflict",
                f"A frame with ID\n{new_sid}\nalready exists in the dataset.",
            )
            return False

        import shutil
        from pathlib import Path

        ds = self.dataset

        def _move(src: Path, dst: Path) -> None:
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

        try:
            _move(ds.image_path(old_sid),      ds.image_path(new_sid))
            _move(ds.mask_path(old_sid),        ds.mask_path(new_sid))
            _move(ds.qc_overlay_path(old_sid),  ds.qc_overlay_path(new_sid))
            _move(ds.qc_thumb_path(old_sid),    ds.qc_thumb_path(new_sid))
        except Exception as exc:
            from tkinter import messagebox
            messagebox.showerror("Rename failed", f"File move error:\n{exc}")
            return False

        # Update metadata record in-place (keep everything except identity fields)
        rec = self.metadata.get(old_sid)
        if rec is not None:
            rec.sample_id = new_sid
            rec.episode_id = new_ep
            rec.camera_id = new_cam
            self.metadata.remove(old_sid)
            self.metadata.add(rec)
            self.metadata.save()

        if self._active_sample_id == old_sid:
            self._active_sample_id = new_sid

        self.frame_queue.refresh()
        self.set_status(f"Renamed {old_sid} → {new_sid}")
        return True

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
