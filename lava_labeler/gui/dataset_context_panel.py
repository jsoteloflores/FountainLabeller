"""Compact Dataset Context panel for the main GUI right sidebar.

Shows only the most operationally useful stats so the labeler can stay
oriented without leaving the canvas:

  • current video filename + recognition status
  • current episode / camera
  • labeled / candidates for this video
  • labeled for this episode
  • whole-dataset totals

A ``Dataset Details…`` button opens :class:`MetadataDetailsWindow`.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lava_labeler.gui.app import App


class DatasetContextPanel(ttk.LabelFrame):
    """Compact stats panel wired to app state."""

    def __init__(self, parent: tk.Widget, app: "App") -> None:
        super().__init__(parent, text="Dataset Context", padding=(4, 2))
        self.app = app
        self._build()
        self.refresh()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        def _row(label_text: str, width: int = 18) -> tk.StringVar:
            f = ttk.Frame(self)
            f.pack(fill=tk.X, pady=0)
            ttk.Label(f, text=label_text, width=width, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            ttk.Label(f, textvariable=var, anchor=tk.E).pack(side=tk.RIGHT)
            return var

        # Video info
        self._video_var   = _row("Video:")
        self._recog_var   = _row("Recognized:")
        self._episode_var = _row("Episode:")
        self._camera_var  = _row("Camera:")

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # This-video counts
        ttk.Label(self, text="This video:", anchor=tk.W,
                  font=("", 9, "bold")).pack(fill=tk.X)
        self._vid_labeled_var    = _row("  Labeled:")
        self._vid_cand_var       = _row("  Candidates:")
        self._vid_hardneg_var    = _row("  Hard neg.:")
        self._vid_empty_var      = _row("  Empty conf.:")
        self._vid_review_var     = _row("  Needs review:")

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # This-episode counts
        ttk.Label(self, text="This episode:", anchor=tk.W,
                  font=("", 9, "bold")).pack(fill=tk.X)
        self._ep_labeled_var = _row("  Labeled:")
        self._ep_videos_var  = _row("  Videos:")

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # Dataset totals
        ttk.Label(self, text="Dataset:", anchor=tk.W,
                  font=("", 9, "bold")).pack(fill=tk.X)
        self._ds_labeled_var  = _row("  Labeled:")
        self._ds_hardneg_var  = _row("  Hard neg.:")
        self._ds_empty_var    = _row("  Empty conf.:")
        self._ds_review_var   = _row("  Needs review:")
        self._ds_videos_var   = _row("  Videos:")

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # Details button
        lbl = "Dataset Details…"
        if (self.app.project_config is not None
                and isinstance(self.app.project_config.get("gui"), dict)):
            lbl = self.app.project_config.get("gui", {}).get(
                "metadata_details_button_label", lbl)
        ttk.Button(self, text=lbl, command=self._open_details).pack(
            fill=tk.X, pady=(2, 0))

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        app = self.app

        # Video info
        ve = None
        if app.video_registry is not None and app._active_video_id:
            ve = app.video_registry.get(app._active_video_id)

        if ve is not None:
            self._video_var.set(_trunc(ve.video_filename, 20))
            self._recog_var.set("Yes" if app._video_registry_tier in ("exact", "probable") else "New")
            self._episode_var.set(ve.episode_id or "—")
            self._camera_var.set(ve.camera_id or "—")
        elif app.video_reader is not None:
            fn = app.video_reader.info.path.name
            self._video_var.set(_trunc(fn, 20))
            self._recog_var.set("New")
            ep = app.metadata_panel.episode_var.get() if hasattr(app, "metadata_panel") else "—"
            cam = app.metadata_panel.camera_var.get() if hasattr(app, "metadata_panel") else "—"
            self._episode_var.set(ep or "—")
            self._camera_var.set(cam or "—")
        else:
            for v in (self._video_var, self._recog_var, self._episode_var, self._camera_var):
                v.set("—")

        # Summary stats
        ds = app.dataset_summary
        if ds is None:
            for v in (self._vid_labeled_var, self._vid_cand_var,
                      self._vid_hardneg_var, self._vid_empty_var, self._vid_review_var,
                      self._ep_labeled_var, self._ep_videos_var,
                      self._ds_labeled_var, self._ds_hardneg_var,
                      self._ds_empty_var, self._ds_review_var, self._ds_videos_var):
                v.set("—")
            return

        # Per-video
        vid_stats = ds.stats_for_video(app._active_video_id) if app._active_video_id else None
        if vid_stats:
            self._vid_labeled_var.set(str(vid_stats.total_labeled))
            self._vid_cand_var.set(str(vid_stats.total_candidates))
            self._vid_hardneg_var.set(str(vid_stats.hard_negative))
            self._vid_empty_var.set(str(vid_stats.empty_confirmed))
            self._vid_review_var.set(str(vid_stats.total_needs_review))
        else:
            for v in (self._vid_labeled_var, self._vid_cand_var,
                      self._vid_hardneg_var, self._vid_empty_var, self._vid_review_var):
                v.set("0")

        # Per-episode
        ep_id = (ve.episode_id if ve else "") or (
            app.metadata_panel.episode_var.get() if hasattr(app, "metadata_panel") else "")
        ep_stats = ds.stats_for_episode(ep_id) if ep_id else None
        if ep_stats:
            self._ep_labeled_var.set(str(ep_stats.total_labeled))
            self._ep_videos_var.set(str(ep_stats.video_count))
        else:
            self._ep_labeled_var.set("0")
            self._ep_videos_var.set("0")

        # Dataset
        d = ds.dataset
        tot = d.total_labeled + d.total_unlabeled + d.total_skipped + d.total_needs_review
        self._ds_labeled_var.set(f"{d.total_labeled} / {tot}")
        self._ds_hardneg_var.set(str(d.hard_negative))
        self._ds_empty_var.set(str(d.empty_confirmed))
        self._ds_review_var.set(str(d.total_needs_review))
        self._ds_videos_var.set(str(d.video_count))

    def _open_details(self) -> None:
        from lava_labeler.gui.metadata_details_window import MetadataDetailsWindow
        MetadataDetailsWindow(self.app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"
