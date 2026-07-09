"""Main application window."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np

from lava_labeler.core.candidates import Candidate, CandidateQueue
from lava_labeler.core.config import ProjectConfig, ShortcutConfig
from lava_labeler.core.csv_mirror import regenerate_all as _regen_csv_all
from lava_labeler.core.dataset import DatasetFolder, make_sample_id
from lava_labeler.core.dataset_summary import DatasetSummary
from lava_labeler.core.frame_cache import FrameCache
from lava_labeler.core.logging_utils import SessionLogger
from lava_labeler.core.metadata import FrameRecord, MetadataStore
from lava_labeler.core.playback import PlaybackController
from lava_labeler.core.qc import generate_overlay, generate_thumbnail
from lava_labeler.core.session import SessionRecovery
from lava_labeler.core.video_io import VideoReader
from lava_labeler.core.video_registry import VideoRegistry, MatchTier
from lava_labeler.gui.dataset_context_panel import DatasetContextPanel
from lava_labeler.gui.export_dialog import ExportDialog
from lava_labeler.gui.frame_queue import FrameQueuePanel
from lava_labeler.gui.labeling_canvas import LabelingCanvas
from lava_labeler.gui.labeling_guide import CheatSheetDialog, LabelingGuidePanel
from lava_labeler.gui.metadata_panel import MetadataPanel
from lava_labeler.gui.playback_panel import PlaybackPanel
from lava_labeler.gui.roi_panel import ROIPanel
from lava_labeler.gui.toolbar import Toolbar


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Lava Labeler — Fountain Dataset Builder")
        self.geometry("1500x950")
        self.minsize(900, 600)

        # Core state
        self.video_reader: VideoReader | None = None
        self.dataset: DatasetFolder | None = None
        self.metadata: MetadataStore | None = None
        self.frame_cache = FrameCache()          # 512 MB byte-budgeted LRU
        self.current_frame_index: int = 0
        self._active_sample_id: str | None = None

        # Stage-1 high-throughput state
        self.project_config: ProjectConfig | None = None
        self.shortcuts: ShortcutConfig = ShortcutConfig()
        self.candidates: CandidateQueue | None = None
        self.recovery: SessionRecovery | None = None
        self.logger: SessionLogger | None = None
        self.playback = PlaybackController()
        self._active_candidate_id: str | None = None
        self._last_saved_candidate_id: str | None = None
        self._last_saved_sample_id: str | None = None
        self._session_labeled_count: int = 0
        self._review_mode: bool = False
        self._dirty: bool = False
        self._empty_confirm_done: bool = False
        self._playback_after_id: str | None = None
        self._autosave_after_id: str | None = None
        self._toast_after_id: str | None = None
        self._candidate_filter: str = "all"
        # Stage-1c metadata registry & accounting
        self.video_registry: VideoRegistry | None = None
        self.dataset_summary: DatasetSummary | None = None
        self._active_video_id: str | None = None
        self._video_registry_tier: MatchTier = "new"
        self._csv_mirror_after_id: str | None = None

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
        file_menu.add_command(label="Load Candidate Queue…", command=self.open_candidate_queue)
        file_menu.add_separator()
        file_menu.add_command(label="Save Mask", command=self.save_current_mask, accelerator="Cmd+S")
        file_menu.add_command(label="Save & Next Candidate", command=self.save_and_next)
        file_menu.add_separator()
        file_menu.add_command(label="Export Dataset…", command=self.open_export_dialog)
        file_menu.add_command(label="Export Training Manifest", command=self.export_training_manifest)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit_app, accelerator="Cmd+Q")
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Cmd+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Cmd+Shift+Z")
        menubar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        self._review_mode_var = tk.BooleanVar(value=False)
        view_menu.add_checkbutton(
            label="Review Mode", variable=self._review_mode_var, command=self.toggle_review_mode
        )
        view_menu.add_separator()
        view_menu.add_command(label="Keyboard Shortcuts…", command=self.show_cheat_sheet)
        view_menu.add_command(label="Dataset Details…", command=self.show_dataset_details)
        menubar.add_cascade(label="View", menu=view_menu)

        tools_menu = tk.Menu(menubar, tearoff=False)
        tools_menu.add_command(label="Relink Source Videos…", command=self.relink_source_videos)
        tools_menu.add_separator()
        tools_menu.add_command(
            label="Export 2.5D Temporal Dataset…",
            command=self.open_temporal_export_dialog,
        )
        menubar.add_cascade(label="Tools", menu=tools_menu)

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

        # Right panel — scrollable
        right_outer = ttk.Frame(main, width=290)
        right_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=4)
        right_outer.pack_propagate(False)

        _right_canvas = tk.Canvas(right_outer, highlightthickness=0)
        _right_sb = ttk.Scrollbar(right_outer, orient=tk.VERTICAL,
                                   command=_right_canvas.yview)
        _right_canvas.configure(yscrollcommand=_right_sb.set)
        _right_sb.pack(side=tk.RIGHT, fill=tk.Y)
        _right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(_right_canvas)
        _right_win = _right_canvas.create_window((0, 0), window=right, anchor="nw")

        def _on_right_inner_resize(event):
            _right_canvas.configure(scrollregion=_right_canvas.bbox("all"))
        right.bind("<Configure>", _on_right_inner_resize)

        def _on_right_canvas_resize(event):
            _right_canvas.itemconfig(_right_win, width=event.width)
        _right_canvas.bind("<Configure>", _on_right_canvas_resize)

        def _scroll_right_panel(event):
            _right_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        _right_canvas.bind("<Enter>",
                           lambda e: _right_canvas.bind_all("<MouseWheel>", _scroll_right_panel))
        _right_canvas.bind("<Leave>",
                           lambda e: _right_canvas.unbind_all("<MouseWheel>"))

        # Candidate filter
        filter_row = ttk.Frame(right)
        filter_row.pack(fill=tk.X)
        ttk.Label(filter_row, text="Filter:").pack(side=tk.LEFT)
        self._filter_var = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(
            filter_row, textvariable=self._filter_var, state="readonly", width=14,
            values=["all", "unlabeled", "needs_review", "hard_negative", "complete"],
        )
        filter_combo.pack(side=tk.LEFT, padx=4)
        filter_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        self.frame_queue = FrameQueuePanel(right, app=self)
        self.frame_queue.pack(fill=tk.BOTH, expand=True)

        self.metadata_panel = MetadataPanel(right, app=self)
        self.metadata_panel.pack(fill=tk.X, pady=(4, 0))

        # Dataset Context panel (compact video/episode/dataset stats)
        self.dataset_context_panel = DatasetContextPanel(right, app=self)
        self.dataset_context_panel.pack(fill=tk.X, pady=(4, 0))

        # Progress / session stats panel
        from lava_labeler.gui.progress_panel import ProgressPanel
        self.progress_panel = ProgressPanel(right, app=self)
        self.progress_panel.pack(fill=tk.X, pady=(4, 0))

        self.labeling_guide = LabelingGuidePanel(right, app=self, start_open=True)
        self.labeling_guide.pack(fill=tk.X, pady=(4, 0))

        # Center canvas
        canvas_frame = ttk.Frame(main)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas = LabelingCanvas(canvas_frame, app=self)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Playback panel
        self.playback_panel = PlaybackPanel(self, app=self)
        self.playback_panel.pack(fill=tk.X, padx=4, pady=2)
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Bottom timeline
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=4, pady=2)
        self._build_timeline(bottom)

        # Status bar with save-state indicator
        status_bar = ttk.Frame(self)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=2)
        self.save_state_var = tk.StringVar(value="Saved")
        self._save_state_lbl = ttk.Label(
            status_bar, textvariable=self.save_state_var, relief=tk.SUNKEN,
            anchor=tk.W, width=18, foreground="#81c784",
        )
        self._save_state_lbl.pack(side=tk.LEFT, padx=(0, 2))
        self.status_var = tk.StringVar(value="Ready. Open a dataset to begin.")
        ttk.Label(status_bar, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            fill=tk.X, side=tk.LEFT, expand=True
        )
        # Toast label — temporarily shown for hotkey feedback
        self._toast_var = tk.StringVar(value="")
        self._toast_lbl = tk.Label(
            status_bar, textvariable=self._toast_var, relief=tk.FLAT,
            anchor=tk.W, bg="#1e2d3d", fg="#80cbc4", padx=8, pady=0,
        )
        # not packed until a toast fires

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
        self._bind_shortcuts()

    # ------------------------------------------------------------------
    # Configurable keyboard shortcuts
    # ------------------------------------------------------------------

    def _focus_is_text_entry(self) -> bool:
        """True when keyboard focus is in a text-entry widget or listbox.

        Used to suppress single-key navigation/metadata shortcuts while the
        user is typing in episode/camera/notes fields or selecting in a list.
        """
        try:
            w = self.focus_get()
        except (KeyError, tk.TclError):
            return False
        if w is None:
            return False
        cls = w.winfo_class()
        return cls in ("TEntry", "Entry", "TCombobox", "TSpinbox", "Spinbox", "Text", "Listbox")

    def _bind_shortcuts(self) -> None:
        """Bind every action in the shortcut config to its handler."""
        handlers = self._shortcut_handlers()
        for action, handler in handlers.items():
            seq = self.shortcuts.sequence_for(action)
            if not seq:
                continue
            self.bind_all(seq, self._make_shortcut_callback(handler))

    def _make_shortcut_callback(self, handler):
        def _cb(event):
            if self._focus_is_text_entry():
                return  # let the widget receive the key
            handler()
            return "break"
        return _cb

    def _shortcut_handlers(self) -> dict:
        return {
            # Navigation
            "previous_frame": lambda: self._jump(-1),
            "next_frame": lambda: self._jump(1),
            "jump_back_small": lambda: self._jump(-10),
            "jump_forward_small": lambda: self._jump(10),
            "jump_back_large": lambda: self._jump(-100),
            "jump_forward_large": lambda: self._jump(100),
            "previous_candidate": self.previous_candidate,
            "next_candidate": self.next_candidate,
            "save_and_next": self.save_and_next,
            "save": self.save_current_mask,
            "force_save": self.force_save,
            # View
            "fit_view": self.zoom_fit,
            "zoom_100": self.zoom_100,
            "reset_view": self.reset_view,
            "toggle_mask": self.toggle_mask_overlay,
            "toggle_playback_panel": self.toggle_playback_panel,
            "toggle_metadata_panel": self.toggle_metadata_panel,
            "cheat_sheet": self.show_cheat_sheet,
            # Playback
            "play_pause": self.toggle_playback,
            # Drawing
            "brush": lambda: self.set_tool("brush"),
            "eraser": lambda: self.set_tool("eraser"),
            "otsu_brush": lambda: self.set_tool("otsu_brush"),
            "undo": self.undo,
            "redo": self.redo,
            "clear_mask": self.clear_mask,
            "mark_empty": self.mark_empty,
            # Metadata toggles
            "toggle_wind_affected": lambda: self.toggle_metadata_flag("wind_affected"),
            "toggle_falling_tephra_visible": lambda: self.toggle_metadata_flag("falling_tephra_visible"),
            "toggle_cooling_tephra_visible": lambda: self.toggle_metadata_flag("cooling_tephra_visible"),
            "toggle_smoke_obscured": lambda: self.toggle_metadata_flag("smoke_obscured"),
            "toggle_ground_glow_visible": lambda: self.toggle_metadata_flag("ground_glow_visible"),
            "toggle_exposure_bloom": lambda: self.toggle_metadata_flag("exposure_bloom"),
            "toggle_ambiguous_boundary": lambda: self.toggle_metadata_flag("ambiguous_boundary"),
            "toggle_hard_negative": self.mark_hard_negative,
            "approve_human_clean": self.approve_human_clean,
            "mark_needs_review": self.mark_needs_review,
            "toggle_model_draft_corrected": lambda: self.toggle_metadata_flag("model_draft_corrected"),
            "jump_to_last_saved": self.jump_to_last_saved,
        }

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
            self._register_video(info)
        except Exception as exc:
            messagebox.showerror("Cannot open video", str(exc))

    def _register_video(self, info) -> None:
        """Register the current VideoInfo in the VideoRegistry (if enabled)."""
        if self.video_registry is None:
            return
        ep = self.metadata_panel.episode_var.get() if hasattr(self, "metadata_panel") else ""
        cam = self.metadata_panel.camera_var.get() if hasattr(self, "metadata_panel") else ""
        entry, tier = self.video_registry.register(info, episode_id=ep, camera_id=cam)
        self._active_video_id = entry.video_id
        self._video_registry_tier = tier
        self.video_registry.save_csv()
        if tier == "filename_mismatch":
            messagebox.showwarning(
                "Video registry",
                f"A video named '{info.path.name}' was seen before, but the file properties "
                f"differ (frame count, fps, or resolution changed).\n\n"
                f"A new registry entry has been created: {entry.video_id}.",
            )
        elif tier == "new":
            self.set_status(f"Registered new video: {entry.video_id} ({info.path.name})")
        else:
            self.set_status(f"Recognized video: {entry.video_id} ({tier})")
        if self.dataset_summary is not None:
            self.dataset_summary.refresh()
        if hasattr(self, "dataset_context_panel"):
            self.dataset_context_panel.refresh()

    def _ensure_active_video_registered(self):
        """Ensure the current video has a registry entry and return it.

        Unlike :meth:`_register_video`, this is safe to call even when the
        registry did not exist when the video was first opened (e.g. the user
        opened a video *before* opening a dataset). Returns the active
        ``VideoEntry`` or ``None`` if no video/dataset/registry is available.
        """
        if self.video_reader is None:
            return None

        if self.video_registry is None and self.dataset is not None:
            self.video_registry = VideoRegistry(self.dataset.root)

        if self.video_registry is None:
            return None

        info = self.video_reader.info
        ep = self.metadata_panel.episode_var.get() if hasattr(self, "metadata_panel") else ""
        cam = self.metadata_panel.camera_var.get() if hasattr(self, "metadata_panel") else ""

        entry, tier = self.video_registry.register(info, episode_id=ep, camera_id=cam)
        self._active_video_id = entry.video_id
        self._video_registry_tier = tier
        self.video_registry.save()
        self.video_registry.save_csv()
        return entry

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
        self._init_session(ds.root)
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
        self._init_session(ds.root)
        self.frame_queue.refresh()
        self._restore_roi_settings()
        self.set_status(f"Dataset opened: {path}")
        self._maybe_resume_session()

    def _init_session(self, root) -> None:
        """Load project config, shortcuts, recovery, logger, registry for a dataset."""
        self.project_config = ProjectConfig.load(root)
        shortcut_file = str(self.project_config.get("shortcut_config_path", "shortcuts.json"))
        self.shortcuts = ShortcutConfig.load(root, shortcut_file)
        self._bind_shortcuts()  # rebind in case the user customised shortcuts.json
        self.recovery = SessionRecovery(root)
        self.logger = SessionLogger(root)
        self._empty_confirm_done = False
        # Apply config defaults to playback + cache + overlay.
        self.playback.loop_radius = int(self.project_config.get("default_loop_radius_frames", 15))
        self.playback.speed = float(self.project_config.get("default_playback_speed", 0.5))
        if hasattr(self, "playback_panel"):
            self.playback_panel.loop_radius_var.set(self.playback.loop_radius)
            self.playback_panel.speed_var.set(self.playback.speed)
        opacity = float(self.project_config.get("default_mask_opacity", 0.4))
        self.canvas.set_mask_alpha(opacity)
        if hasattr(self, "toolbar") and hasattr(self.toolbar, "_opacity_var"):
            self.toolbar._opacity_var.set(opacity)
        # Video registry
        use_reg = True
        meta_cfg = self.project_config.get("metadata", {})
        if isinstance(meta_cfg, dict):
            use_reg = meta_cfg.get("use_video_registry", True)
        if use_reg:
            self.video_registry = VideoRegistry(root)
        # Dataset summary (needs metadata store to be set first)
        if self.metadata is not None:
            self.dataset_summary = DatasetSummary(self.metadata, self.video_registry)
        # Auto-load a candidate queue if one lives in the dataset folder.
        for name in ("candidate_frames.csv", "candidate_frames.json"):
            cand_path = Path(root) / name
            if cand_path.exists():
                self.candidates = CandidateQueue.load(cand_path)
                self.set_status(f"Loaded {len(self.candidates)} candidates from {name}")
                break

    def _maybe_resume_session(self) -> None:
        """Offer to resume the previous candidate/frame after a crash."""
        if self.recovery is None or not self.recovery.has_resumable_state():
            return
        cand_id = self.recovery.get("candidate_id", "")
        sid = self.recovery.get("active_sample_id", "")
        label = cand_id or sid
        if not label:
            return
        if not messagebox.askyesno(
            "Resume session",
            f"A previous session was interrupted.\n\nResume from:\n  {label}\n\n"
            "Yes → reopen it.   No → start fresh.",
        ):
            return
        # Restore candidate queue if it was loaded.
        q_path = self.recovery.get("candidate_queue_path", "")
        if q_path and Path(q_path).exists() and self.candidates is None:
            self.candidates = CandidateQueue.load(q_path)
        if cand_id and self.candidates is not None:
            cand = self.candidates.get(cand_id)
            if cand is not None:
                self.open_candidate(cand)
                return
        if sid and self.metadata and self.metadata.get(sid) is not None:
            self.open_frame_for_labeling(sid)

    def _restore_roi_settings(self) -> None:
        """Restore ROI size/mode/policy from dataset config or infer from frames.

        Priority:
          1. standard_roi_size in dataset_config.json (explicitly saved)
          2. The most common roi_width/roi_height among existing ROI-crop frames
        """
        if self.dataset is None:
            return

        applied_w: int | None = None
        applied_h: int | None = None
        policy = self._roi_size_policy
        source = ""

        # 1. Saved config
        cfg = self.dataset.read_config()
        saved = cfg.get("standard_roi_size")
        if isinstance(saved, dict) and saved.get("width") and saved.get("height"):
            applied_w = int(saved["width"])
            applied_h = int(saved["height"])
            policy = str(saved.get("scope", policy)) if saved.get("scope") in (
                "global_fixed", "camera_fixed",
            ) else policy
            source = "config"

        # 2. Infer from existing frames (most common ROI size)
        if applied_w is None and self.metadata is not None:
            from collections import Counter
            sizes: Counter = Counter()
            for rec in self.metadata.all_records():
                if rec.is_roi_crop and rec.roi_width and rec.roi_height:
                    sizes[(int(rec.roi_width), int(rec.roi_height))] += 1
            if sizes:
                (cw, ch), _ = sizes.most_common(1)[0]
                applied_w, applied_h = cw, ch
                source = "frames"

        if applied_w and applied_h:
            self._roi_w = applied_w
            self._roi_h = applied_h
            self._roi_size_policy = policy
            self._roi_mode = "fixed_roi_crop"
            # Sync UI
            if hasattr(self, "roi_panel"):
                self.roi_panel.set_size(applied_w, applied_h)
                self.roi_panel.set_mode("fixed_roi_crop")
                self.roi_panel.set_policy(policy)
            self.set_status(
                f"ROI size restored from {source}: {applied_w}\u00d7{applied_h}"
            )

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

        entry = self._ensure_active_video_registered()

        rec = FrameRecord(
            sample_id=sid,
            video_path=str(info.path),
            video_id=entry.video_id if entry is not None else (self._active_video_id or ""),
            video_filename=entry.video_filename if entry is not None else Path(info.path).name,
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
        self.mark_clean()
        # Anchor playback on this frame's index and stop any active preview.
        self._stop_playback()
        if self.video_reader is not None:
            self.playback.fps = self.video_reader.info.fps
            self.playback.set_anchor(rec.frame_index, self.video_reader.info.frame_count - 1)
        self.current_frame_index = rec.frame_index
        self._update_playback_info()
        if self.recovery is not None:
            self.recovery.update(
                active_sample_id=sample_id,
                video_path=rec.video_path,
                frame_index=rec.frame_index,
                candidate_id=self._active_candidate_id or "",
            )
            self.recovery.save()
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
        self._persist_roi_size()

    def _persist_roi_size(self) -> None:
        """Save the current ROI size to dataset_config.json so it's restored later."""
        if self.dataset is None:
            return
        self.dataset.update_config(standard_roi_size={
            "width": self._roi_w,
            "height": self._roi_h,
            "scope": self._roi_size_policy,
        })

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
        frame = self.canvas.get_label_frame()
        if frame is not None:
            generate_overlay(frame, mask, self.dataset.qc_overlay_path(sid))
            generate_thumbnail(frame, mask, self.dataset.qc_thumb_path(sid))

        self.mark_clean()
        self._log_event("save_mask", details=f"{pos_pixels}px")
        self.set_status(f"Saved {sid}  ({pos_pixels} positive px, {pos_pixels/total*100:.1f}%)")
        self._show_toast(f"[save] frame saved  ({pos_pixels}px)")
        # Track last saved for jump_to_last_saved
        self._last_saved_sample_id = sid
        if self._active_candidate_id:
            self._last_saved_candidate_id = self._active_candidate_id
            if self.recovery is not None:
                self.recovery.update(last_saved_candidate_id=self._active_candidate_id,
                                     last_saved_sample_id=sid)
        if hasattr(self, "progress_panel"):
            self.progress_panel.refresh()
        # Refresh dataset accounting panels and schedule CSV mirror
        if self.dataset_summary is not None:
            self.dataset_summary.refresh()
        if hasattr(self, "dataset_context_panel"):
            self.dataset_context_panel.refresh()
        self._schedule_csv_mirror()

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

    def open_temporal_export_dialog(self) -> None:
        if self.dataset is None or self.metadata is None:
            messagebox.showwarning("No dataset", "Open a dataset first.")
            return
        from lava_labeler.gui.temporal_export_dialog import TemporalExportDialog
        if self.video_registry is None:
            self.video_registry = VideoRegistry(self.dataset.root)
        TemporalExportDialog(
            self, dataset=self.dataset, metadata=self.metadata,
            video_registry=self.video_registry,
        )

    def relink_source_videos(self) -> None:
        if self.dataset is None or self.metadata is None:
            messagebox.showwarning("No dataset", "Open a dataset first.")
            return
        from lava_labeler.core.video_relink import relink_workspace_videos

        folder = filedialog.askdirectory(title="Select source-video root folder")
        if not folder:
            return

        # Dry run first.
        try:
            plan = relink_workspace_videos(
                dataset_root=self.dataset.root,
                source_video_root=Path(folder),
                dry_run=True,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Relink failed", str(exc))
            return

        summary = (
            f"Rows checked:        {plan['rows_checked']}\n"
            f"Rows already linked: {plan['rows_already_linked']}\n"
            f"Rows relinkable:     {plan['rows_relinkable']}\n"
            f"Missing:             {plan['missing']}\n"
            f"Ambiguous:           {plan['ambiguous']}\n"
            f"Failed video reads:  {plan['failed_to_read_video']}\n\n"
            f"Report: {plan['report_csv']}"
        )
        if plan["rows_relinkable"] == 0:
            messagebox.showinfo("Relink source videos", "Nothing to relink.\n\n" + summary)
            return
        if not messagebox.askyesno(
            "Relink source videos",
            summary + "\n\nApply these changes? A backup of frames.csv will be written.",
        ):
            return

        try:
            result = relink_workspace_videos(
                dataset_root=self.dataset.root,
                source_video_root=Path(folder),
                dry_run=False,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Relink failed", str(exc))
            return

        # Reload metadata and refresh UI.
        self.metadata = MetadataStore(self.dataset.root)
        if hasattr(self, "frame_queue"):
            self.frame_queue.refresh()
        if self.dataset_summary is not None:
            self.dataset_summary.refresh()
        if hasattr(self, "dataset_context_panel"):
            self.dataset_context_panel.refresh()
        self.set_status(
            f"Relinked {result['rows_relinkable']} rows "
            f"({result['linked_existing_path']} by path, "
            f"{result['linked_by_filename']} by filename)"
        )
        messagebox.showinfo(
            "Relink complete",
            f"Relinked {result['rows_relinkable']} rows.\n\n"
            f"Backup + report written under metadata/.",
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    # ------------------------------------------------------------------
    # Toast notifications
    # ------------------------------------------------------------------

    def _show_toast(self, msg: str) -> None:
        """Show a temporary hotkey-feedback message in the status bar."""
        enabled = True
        if self.project_config is not None:
            enabled = bool(self.project_config.get("show_hotkey_toasts", True))
        if not enabled:
            return
        duration = 1500
        if self.project_config is not None:
            duration = int(self.project_config.get("toast_duration_ms", 1500))
        self._toast_var.set(msg)
        self._toast_lbl.pack(side=tk.RIGHT, padx=(2, 2))
        if self._toast_after_id is not None:
            try:
                self.after_cancel(self._toast_after_id)
            except Exception:
                pass
        self._toast_after_id = self.after(duration, self._hide_toast)

    def _hide_toast(self) -> None:
        self._toast_after_id = None
        self._toast_lbl.pack_forget()
        self._toast_var.set("")

    # ------------------------------------------------------------------
    # Dirty state / save indicator
    # ------------------------------------------------------------------

    def mark_dirty(self) -> None:
        self._dirty = True
        self.save_state_var.set("Unsaved changes")
        self._save_state_lbl.config(foreground="#e57373")
        if self.recovery is not None:
            self.recovery.update(dirty=True)
        self._schedule_autosave()

    def mark_clean(self) -> None:
        self._dirty = False
        self.save_state_var.set("Saved")
        self._save_state_lbl.config(foreground="#81c784")
        if self.recovery is not None:
            self.recovery.mark_saved()
            self.recovery.save()

    def _schedule_autosave(self) -> None:
        if self._autosave_after_id is not None:
            return
        interval = 10
        if self.project_config is not None:
            interval = int(self.project_config.get("autosave_interval_seconds", 10))
        self._autosave_after_id = self.after(max(1000, interval * 1000), self._autosave_tick)

    def _autosave_tick(self) -> None:
        self._autosave_after_id = None
        if not self._dirty:
            return
        try:
            self.save_state_var.set("Autosaving…")
            self.update_idletasks()
            self._autosave_now()
            self.mark_clean()
            self._log_event("autosave_mask")
        except Exception:
            self.save_state_var.set("Autosave failed")
            self._save_state_lbl.config(foreground="#e57373")

    def _autosave_now(self) -> None:
        """Persist the current mask + metadata without UI side effects."""
        if self._active_sample_id and self.dataset and self.metadata:
            mask = self.canvas.get_current_mask()
            if mask is not None:
                pos = int(np.sum(mask > 0))
                total = int(mask.size)
                self.dataset.save_mask(self._active_sample_id, mask)
                self.metadata.update(
                    self._active_sample_id,
                    mask_positive_pixels=pos,
                    mask_positive_fraction=pos / total if total else 0.0,
                )
                self.canvas.mark_saved()
            self.metadata.save()

    # ------------------------------------------------------------------
    # CSV mirror
    # ------------------------------------------------------------------

    def _schedule_csv_mirror(self) -> None:
        """Debounce CSV regeneration so rapid saves don't thrash disk."""
        if self._csv_mirror_after_id is not None:
            try:
                self.after_cancel(self._csv_mirror_after_id)
            except Exception:
                pass
        debounce = 1500
        if self.project_config:
            meta = self.project_config.get("metadata", {})
            if isinstance(meta, dict):
                debounce = int(meta.get("csv_mirror_debounce_ms", 1500))
        self._csv_mirror_after_id = self.after(debounce, self._csv_mirror_tick)

    def _csv_mirror_tick(self) -> None:
        self._csv_mirror_after_id = None
        self._regenerate_csv_mirrors()

    def _regenerate_csv_mirrors(self, force: bool = False) -> None:
        """Regenerate video_registry.csv, frame_metadata.csv, dataset_summary.csv."""
        if self.metadata is None or self.dataset is None:
            return
        enabled = True
        if not force and self.project_config:
            meta = self.project_config.get("metadata", {})
            if isinstance(meta, dict):
                enabled = meta.get("regenerate_csv_on_save", True)
        if not enabled:
            return
        if self.dataset_summary is not None:
            self.dataset_summary.refresh()
        try:
            _regen_csv_all(
                self.dataset.root,
                self.metadata,
                registry=self.video_registry,
                summary=self.dataset_summary,
            )
        except Exception:
            pass  # CSV mirrors are non-critical; don't interrupt the workflow

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, details: str = "") -> None:
        if self.logger is None:
            return
        rec = self.metadata.get(self._active_sample_id) if (self.metadata and self._active_sample_id) else None
        self.logger.log(
            event_type,
            candidate_id=self._active_candidate_id or "",
            video_id=rec.video_id if rec else "",
            frame_index=rec.frame_index if rec else self.current_frame_index,
            details=details,
        )

    # ------------------------------------------------------------------
    # Tool / view shortcuts
    # ------------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        if hasattr(self, "toolbar") and hasattr(self.toolbar, "_tool_var"):
            self.toolbar._tool_var.set(tool)
        self.set_status(f"Tool: {tool}")

    def reset_view(self) -> None:
        self.canvas.zoom_fit()

    def toggle_mask_overlay(self) -> None:
        new_state = not self.canvas._mask_visible
        self.canvas.set_mask_visible(new_state)
        self.set_status(f"Mask overlay: {'ON' if new_state else 'OFF'}")

    def toggle_playback_panel(self) -> None:
        if self.playback_panel.winfo_ismapped():
            self.playback_panel.pack_forget()
        else:
            self.playback_panel.pack(fill=tk.X, padx=4, pady=2)

    def toggle_metadata_panel(self) -> None:
        if self.metadata_panel.winfo_ismapped():
            self.metadata_panel.pack_forget()
        else:
            self.metadata_panel.pack(fill=tk.X, pady=(4, 0))

    def show_cheat_sheet(self) -> None:
        CheatSheetDialog(self, self.shortcuts)

    def show_dataset_details(self) -> None:
        from lava_labeler.gui.metadata_details_window import MetadataDetailsWindow
        MetadataDetailsWindow(self)

    def toggle_review_mode(self) -> None:
        self._review_mode = self._review_mode_var.get()
        self.set_status(f"Review mode: {'ON' if self._review_mode else 'OFF'}")

    def clear_mask(self) -> None:
        if self._active_sample_id is None:
            return
        # Confirmation for non-empty masks (per project config)
        if self.canvas.has_positive_pixels():
            confirm = True
            if self.project_config is not None:
                confirm = bool(self.project_config.get("confirm_clear_nonempty_mask", True))
            if confirm:
                if not messagebox.askyesno(
                    "Clear mask?",
                    "The current mask has positive pixels.\n\n"
                    "Clear it? This can be undone with Ctrl+Z.",
                ):
                    return
        self.canvas.clear_mask()
        self.mark_dirty()
        self._show_toast("[mask] cleared")
        self.set_status("Mask cleared (undo with Ctrl+Z)")

    # ------------------------------------------------------------------
    # Metadata flag hotkeys
    # ------------------------------------------------------------------

    def toggle_metadata_flag(self, flag: str) -> None:
        if self._active_sample_id is None or self.metadata is None:
            self.set_status("Open a frame before tagging metadata.")
            return
        new_val = self.metadata_panel.toggle_flag(flag)
        if new_val is None:
            return
        self.apply_flag(flag, new_val)

    def apply_flag(self, flag: str, value: bool) -> None:
        """Persist a single metadata flag immediately and autosave."""
        if self._active_sample_id is None or self.metadata is None:
            return
        self.metadata.update(self._active_sample_id, **{flag: value})
        self.metadata.save()
        self.frame_queue.refresh()
        self.mark_clean()
        self._log_event("toggle_metadata", details=f"{flag}={'ON' if value else 'OFF'}")
        self._show_toast(f"[metadata] {flag}: {'ON' if value else 'OFF'}")
        self.set_status(f"{flag}: {'ON' if value else 'OFF'}")
        if hasattr(self, "progress_panel"):
            self.progress_panel.refresh()

    def mark_hard_negative(self) -> None:
        if self._active_sample_id is None or self.metadata is None:
            return
        new_val = self.metadata_panel.toggle_flag("hard_negative")
        if new_val is None:
            return
        self.apply_flag("hard_negative", new_val)
        if new_val:
            self.metadata.update(self._active_sample_id, label_status="hard_negative")
            self.metadata.save()
            self.metadata_panel.set_status_value("hard_negative")
            self._update_candidate_status_from_record()
            self.frame_queue.refresh()
        self._log_event("mark_hard_negative")
        self._show_toast(f"[metadata] hard_negative: {'ON' if new_val else 'OFF'}")
        if hasattr(self, "progress_panel"):
            self.progress_panel.refresh()

    def approve_human_clean(self) -> None:
        if self._active_sample_id is None or self.metadata is None:
            return
        self.save_current_mask()
        self.metadata_panel.set_flag("human_clean", True)
        self.metadata.update(
            self._active_sample_id, human_clean=True,
            label_status="complete", mask_provenance="human_clean",
        )
        self.metadata.save()
        self.metadata_panel.set_status_value("complete")
        self.metadata_panel.set_provenance("human_clean")
        self._update_candidate_status_from_record()
        self.frame_queue.refresh()
        self.mark_clean()
        self._log_event("approve_mask")
        self.set_status("Approved: human_clean")
        self._show_toast("[metadata] human_clean: approved")
        if hasattr(self, "progress_panel"):
            self.progress_panel.refresh()

    def mark_needs_review(self) -> None:
        if self._active_sample_id is None or self.metadata is None:
            return
        self.metadata_panel.set_flag("needs_review", True)
        self.metadata.update(
            self._active_sample_id, needs_review=True, label_status="needs_review",
        )
        self.metadata.save()
        self.metadata_panel.set_status_value("needs_review")
        self._update_candidate_status_from_record()
        self.frame_queue.refresh()
        self.mark_clean()
        self._log_event("needs_review")
        self.set_status("Marked needs_review")
        self._show_toast("[metadata] needs_review: ON")
        if hasattr(self, "progress_panel"):
            self.progress_panel.refresh()

    def mark_empty(self) -> None:
        """Mark the current frame as an intentionally-empty (true-negative) label."""
        if self._active_sample_id is None or self.metadata is None:
            self.set_status("Open a frame first.")
            return
        # One-time-per-session confirmation, then fast marking with undo support.
        require_confirm = True
        if self.project_config is not None:
            require_confirm = bool(
                self.project_config.get("require_empty_mask_confirmation_once_per_session", True)
            )
        if require_confirm and not self._empty_confirm_done:
            ans = messagebox.askyesno(
                "Mark intentionally empty",
                "Mark this frame as an intentionally-empty (true-negative) label?\n\n"
                "This clears the mask. You can undo with Ctrl+Z.\n\n"
                "Marking empty will not ask again this session.",
            )
            if not ans:
                return
            self._empty_confirm_done = True

        self.canvas.clear_mask()
        self.dataset.save_mask(self._active_sample_id, self.canvas.get_current_mask())
        self.metadata_panel.set_flag("empty_mask_confirmed", True)
        # Preserve hard_negative status if already set; otherwise empty_confirmed.
        rec = self.metadata.get(self._active_sample_id)
        new_status = "hard_negative" if (rec and rec.hard_negative) else "empty_confirmed"
        self.metadata.update(
            self._active_sample_id,
            empty_mask_confirmed=True,
            mask_provenance="empty_confirmed",
            label_status=new_status,
            mask_positive_pixels=0,
            mask_positive_fraction=0.0,
        )
        self.metadata.save()
        self.metadata_panel.set_status_value(new_status)
        self.metadata_panel.set_provenance("empty_confirmed")
        self.canvas.mark_saved()
        self._update_candidate_status_from_record()
        self.frame_queue.refresh()
        self.mark_clean()
        self._log_event("mark_empty")
        self.set_status(f"Marked intentionally empty ({new_status})")
        self._show_toast(f"[mask] empty mask confirmed ({new_status})")
        if hasattr(self, "progress_panel"):
            self.progress_panel.refresh()

    def force_save(self) -> None:
        self.save_current_mask()
        if self.metadata is not None:
            self.metadata.save()
        if self.recovery is not None:
            self.recovery.save()
        self.set_status("Project saved.")

    # ------------------------------------------------------------------
    # Candidate queue
    # ------------------------------------------------------------------

    def open_candidate_queue(self) -> None:
        if self.dataset is None or self.metadata is None:
            messagebox.showwarning("No dataset", "Create or open a dataset first.")
            return
        path = filedialog.askopenfilename(
            title="Open Candidate Queue",
            filetypes=[("Candidate files", "*.csv *.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self.candidates = CandidateQueue.load(path)
        if self.recovery is not None:
            self.recovery.update(candidate_queue_path=path)
            self.recovery.save()
        self.set_status(f"Loaded {len(self.candidates)} candidates from {Path(path).name}")
        first = self.candidates.first_unlabeled()
        if first is not None:
            self.open_candidate(first)

    def open_candidate(self, candidate: "Candidate") -> None:
        """Ensure a dataset record/image exists for *candidate*, then label it."""
        if self.dataset is None or self.metadata is None:
            return
        if not self._check_unsaved(context="before opening another candidate"):
            return
        # Ensure the candidate's video is open.
        if not self._ensure_video(candidate.video_path):
            messagebox.showerror("Video unavailable", f"Cannot open:\n{candidate.video_path}")
            return
        sid = self._ensure_record_for_candidate(candidate)
        if sid is None:
            return
        self._active_candidate_id = candidate.candidate_id
        if candidate.status == "unlabeled":
            self.candidates.set_status(candidate.candidate_id, "in_progress")
            self._save_candidates()
        if self.recovery is not None:
            self.recovery.update(candidate_id=candidate.candidate_id)
            self.recovery.push_recent(candidate.candidate_id)
        self.open_frame_for_labeling(sid)
        self._log_event("open_candidate", details=candidate.reason)

    def _ensure_video(self, video_path: str) -> bool:
        if not video_path:
            return self.video_reader is not None
        cur = str(self.video_reader.info.path) if self.video_reader else None
        if cur == video_path:
            return True
        if not Path(video_path).exists():
            return self.video_reader is not None  # fall back to whatever is open
        try:
            if self.video_reader:
                self.video_reader.close()
            self.video_reader = VideoReader(video_path)
            self.frame_cache.clear()
            info = self.video_reader.info
            self._scrubber.configure(to=max(1, info.frame_count - 1))
            self._video_info_var.set(info.summary)
            self.title(f"Lava Labeler — {Path(video_path).name}")
            return True
        except Exception:
            return False

    def _ensure_record_for_candidate(self, candidate: "Candidate") -> str | None:
        """Create (or fetch) the FrameRecord + image for a candidate frame."""
        if self.video_reader is None or self.dataset is None or self.metadata is None:
            return None
        info = self.video_reader.info
        idx = max(0, min(info.frame_count - 1, candidate.frame_index))
        ep = candidate.extra.get("eruption_episode", "") or self.metadata_panel.episode_var.get() or "unknownEpisode"
        cam = candidate.camera_id or self.metadata_panel.camera_var.get() or "unknownCamera"
        sid = make_sample_id(ep, cam, idx)
        if self.metadata.get(sid) is not None:
            return sid
        frame = self._get_frame(idx)
        if frame is None:
            messagebox.showerror("Frame unavailable", f"Cannot load frame {idx}.")
            return None
        rec = FrameRecord(
            sample_id=sid,
            video_path=str(info.path),
            frame_index=idx,
            source_width=info.width,
            source_height=info.height,
            fps=info.fps,
            video_id=candidate.video_id or self._active_video_id or "",
            video_filename=info.path.name,
            episode_id=ep,
            camera_id=cam,
            candidate_id=candidate.candidate_id,
            roi_x=0, roi_y=0, roi_width=info.width, roi_height=info.height,
            lighting_condition=candidate.extra.get("lighting_condition", "unknown") or "unknown",
            notes=candidate.notes,
        )
        self.metadata.add(rec)
        self.dataset.save_image(sid, frame)
        self.metadata.save()
        self.frame_queue.refresh()
        return sid

    def _current_candidate(self) -> "Candidate | None":
        if self.candidates is None or self._active_candidate_id is None:
            return None
        return self.candidates.get(self._active_candidate_id)

    def next_candidate(self) -> None:
        if self.candidates is None:
            self.set_status("No candidate queue loaded.")
            return
        nxt = self.candidates.next(self._active_candidate_id)
        if nxt is None:
            self.set_status("No more candidates.")
            return
        self.open_candidate(nxt)

    def previous_candidate(self) -> None:
        if self.candidates is None:
            self.set_status("No candidate queue loaded.")
            return
        prv = self.candidates.previous(self._active_candidate_id)
        if prv is None:
            self.set_status("At first candidate.")
            return
        self.open_candidate(prv)

    def save_and_next(self) -> None:
        """Save the current frame, update candidate status, jump to next unlabeled."""
        if self._active_sample_id is not None:
            self.save_current_mask()
            self._finalize_current_label_status()
            self._update_candidate_status_from_record()
            self._save_candidates()
            self.mark_clean()
            self._session_labeled_count += 1
            if hasattr(self, "progress_panel"):
                self.progress_panel.refresh()
        if self.candidates is None:
            self.set_status("Saved. (No candidate queue loaded.)")
            return
        nxt = self.candidates.next_unlabeled(self._active_candidate_id)
        if nxt is None:
            self.set_status("Saved. No more unlabeled candidates.")
            return
        self.open_candidate(nxt)

    def _finalize_current_label_status(self) -> None:
        """Promote in_progress frames to a terminal status based on content."""
        if self._active_sample_id is None or self.metadata is None:
            return
        rec = self.metadata.get(self._active_sample_id)
        if rec is None:
            return
        if rec.label_status in ("queued", "in_progress"):
            if rec.hard_negative and rec.empty_mask_confirmed:
                new = "hard_negative"
            elif rec.needs_review:
                new = "needs_review"
            elif rec.mask_positive_pixels > 0:
                new = "complete"
                if rec.mask_provenance in ("human_rough", "unknown", ""):
                    self.metadata.update(self._active_sample_id, mask_provenance="human_clean")
            elif rec.empty_mask_confirmed:
                new = "empty_confirmed"
            else:
                new = rec.label_status  # leave as-is if nothing decisive
            self.metadata.update(self._active_sample_id, label_status=new)
            self.metadata.save()
            self.metadata_panel.set_status_value(new)

    def _update_candidate_status_from_record(self) -> None:
        """Sync the candidate queue status from the active frame record."""
        if self.candidates is None or self._active_candidate_id is None or self.metadata is None:
            return
        rec = self.metadata.get(self._active_sample_id) if self._active_sample_id else None
        if rec is None:
            return
        mapping = {
            "complete": "labeled",
            "hard_negative": "hard_negative",
            "empty_confirmed": "empty_confirmed",
            "needs_review": "needs_review",
            "bad_frame": "bad_frame",
            "skipped": "skipped",
        }
        status = mapping.get(rec.label_status)
        if status:
            self.candidates.set_status(self._active_candidate_id, status)

    def _save_candidates(self) -> None:
        if self.candidates is not None:
            self.candidates.save()

    def _on_filter_change(self, _event=None) -> None:
        self._candidate_filter = self._filter_var.get()
        self.frame_queue.set_filter(self._candidate_filter)

    # ------------------------------------------------------------------
    # Jump to last saved
    # ------------------------------------------------------------------

    def jump_to_last_saved(self) -> None:
        """Jump back to the last saved candidate / sample."""
        # Check recovery for last_saved_candidate_id in case app just loaded.
        last_cid = self._last_saved_candidate_id
        last_sid = self._last_saved_sample_id
        if last_cid is None and self.recovery is not None:
            last_cid = self.recovery.get("last_saved_candidate_id", "") or None
        if last_sid is None and self.recovery is not None:
            last_sid = self.recovery.get("last_saved_sample_id", "") or None

        if last_cid is None and last_sid is None:
            self._show_toast("No last saved candidate yet.")
            self.set_status("No last saved candidate yet.")
            return

        if not self._check_unsaved(context="before jumping to last saved"):
            return

        if last_cid and self.candidates is not None:
            cand = self.candidates.get(last_cid)
            if cand is not None:
                self.open_candidate(cand)
                self._show_toast(f"[nav] returned to {last_cid}")
                return

        if last_sid and self.metadata is not None and self.metadata.get(last_sid) is not None:
            self.open_frame_for_labeling(last_sid)
            self._show_toast(f"[nav] returned to {last_sid}")
            return

        self._show_toast("Last saved frame no longer available.")
        self.set_status("Last saved frame no longer available.")

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def toggle_playback(self) -> None:
        if self.playback.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self.video_reader is None:
            self.set_status("Open a video to play.")
            return
        info = self.video_reader.info
        # Anchor on the active label frame if labeling, else the browse frame.
        anchor = self.current_frame_index
        if self._active_sample_id and self.metadata:
            rec = self.metadata.get(self._active_sample_id)
            if rec is not None:
                anchor = rec.frame_index
        self.playback.fps = info.fps
        self.playback.set_anchor(anchor, info.frame_count - 1)
        self.playback.is_playing = True
        self.playback_panel.set_playing(True)
        self._log_event("playback_start")
        self._playback_tick()

    def _stop_playback(self) -> None:
        self.playback.is_playing = False
        if self._playback_after_id is not None:
            try:
                self.after_cancel(self._playback_after_id)
            except Exception:
                pass
            self._playback_after_id = None
        self.playback_panel.set_playing(False)
        # Restore the anchored label frame / browse frame.
        if self.canvas.is_previewing:
            self.canvas.end_preview()
        self._update_playback_info()
        self._log_event("playback_stop")

    def _playback_tick(self) -> None:
        if not self.playback.is_playing or self.video_reader is None:
            return
        idx = self.playback.next_frame()
        frame = self._get_frame(idx)
        if frame is not None:
            info = self.video_reader.info
            t = idx / info.fps if info.fps > 0 else 0.0
            if self.canvas._mode == "label" and idx != self.playback.anchor_frame:
                banner = (
                    f"Previewing frame {idx} (t={t:.2f}s)  —  "
                    f"Editing mask for frame {self.playback.anchor_frame}"
                )
                self.canvas.show_preview(frame, banner)
            elif self.canvas._mode == "label":
                # Reached the anchor: show the real label frame + mask.
                self.canvas.end_preview()
            else:
                self.canvas.set_browse_frame(frame)
            self._update_playback_info()
        self._playback_after_id = self.after(self.playback.interval_ms(), self._playback_tick)

    def playback_step(self, delta: int) -> None:
        """Step the preview frame by *delta* without starting continuous play."""
        if self.video_reader is None:
            return
        if not self.canvas._mode == "label":
            self._jump(delta)
            return
        if not self.playback.max_frame:
            info = self.video_reader.info
            anchor = self.current_frame_index
            if self._active_sample_id and self.metadata:
                rec = self.metadata.get(self._active_sample_id)
                if rec is not None:
                    anchor = rec.frame_index
            self.playback.set_anchor(anchor, info.frame_count - 1)
        target = max(0, min(self.playback.max_frame, self.playback.preview_frame + delta))
        self.playback.preview_frame = target
        frame = self._get_frame(target)
        if frame is None:
            return
        if target == self.playback.anchor_frame:
            self.canvas.end_preview()
        else:
            info = self.video_reader.info
            t = target / info.fps if info.fps > 0 else 0.0
            self.canvas.show_preview(
                frame,
                f"Previewing frame {target} (t={t:.2f}s)  —  "
                f"Editing mask for frame {self.playback.anchor_frame}",
            )
        self._update_playback_info()

    def playback_reset(self) -> None:
        """Stop playback and return to the anchored label frame."""
        self._stop_playback()
        self.playback.reset_to_anchor()
        if self.canvas.is_previewing:
            self.canvas.end_preview()
        self._update_playback_info()

    def set_loop_radius(self, n: int) -> None:
        self.playback.loop_radius = max(0, n)

    def set_loop_enabled(self, enabled: bool) -> None:
        self.playback.loop_enabled = enabled

    def set_playback_speed(self, speed: float) -> None:
        self.playback.speed = max(0.05, speed)

    def _update_playback_info(self) -> None:
        if self.canvas._mode == "label" and self._active_sample_id:
            anchor = self.playback.anchor_frame
            preview = self.playback.preview_frame
            if preview != anchor and self.canvas.is_previewing:
                self.playback_panel.set_info(
                    f"Preview {preview}  |  Editing {anchor}", previewing=True
                )
            else:
                self.playback_panel.set_info(f"Editing frame {anchor}", previewing=False)
        else:
            self.playback_panel.set_info(f"Frame {self.current_frame_index}", previewing=False)

    # ------------------------------------------------------------------
    # Training export
    # ------------------------------------------------------------------

    def export_training_manifest(self) -> None:
        if self.dataset is None or self.metadata is None:
            messagebox.showwarning("No dataset", "Open a dataset first.")
            return
        from lava_labeler.core.export import export_dataset
        try:
            summary = export_dataset(self.dataset, self.metadata)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        warnings = summary.get("validation_warnings", 0)
        warn_text = f"\n\nValidation warnings: {warnings}" if warnings else ""
        report_text = f"\nValidation report:\n{summary.get('validation_report', '')}" if warnings else ""
        messagebox.showinfo(
            "Export complete",
            f"Exported {summary['exported']} samples.\n"
            f"Skipped (not finalized): {summary['skipped']}{warn_text}\n\n"
            f"Manifest:\n{summary['manifest']}{report_text}",
        )
        status = f"Exported {summary['exported']} samples → labels_manifest.csv"
        if warnings:
            status += f"  ({warnings} validation warnings)"
        self.set_status(status)

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
        # Best-effort autosave of any pending edits before the guard prompt.
        if self._dirty:
            try:
                self._autosave_now()
                self.mark_clean()
            except Exception:
                pass
        if not self._check_unsaved(context="before quitting"):
            return
        self._stop_playback()
        # Cancel pending timers
        for attr in ("_autosave_after_id", "_csv_mirror_after_id", "_toast_after_id"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.after_cancel(aid)
                except Exception:
                    pass
        # Save canonical JSON state
        if self.video_registry is not None:
            try:
                self.video_registry.save()
            except Exception:
                pass
        if self.metadata is not None:
            try:
                self.metadata.save()
            except Exception:
                pass
        if self.candidates is not None:
            try:
                self.candidates.save()
            except Exception:
                pass
        # Regenerate CSV mirrors on close
        regen_on_close = True
        if self.project_config:
            meta = self.project_config.get("metadata", {})
            if isinstance(meta, dict):
                regen_on_close = meta.get("regenerate_csv_on_close", True)
        if regen_on_close:
            self._regenerate_csv_mirrors(force=True)
        if self.recovery is not None:
            # Clean exit: drop the recovery file so we don't prompt next time.
            self.recovery.clear()
        if self.video_reader:
            self.video_reader.close()
        self.destroy()


def main() -> None:
    app = App()
    app.mainloop()
