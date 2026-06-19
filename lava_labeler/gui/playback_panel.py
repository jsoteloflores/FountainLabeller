"""Bottom playback panel: loop-around-current-frame motion preview controls."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lava_labeler.gui.app import App


class PlaybackPanel(ttk.Frame):
    """Controls for the motion-preview playback loop around the label frame."""

    def __init__(self, parent: tk.Widget, app: "App") -> None:
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self) -> None:
        PAD = {"padx": 2, "pady": 2}

        ttk.Label(self, text="Playback:", font=("TkDefaultFont", 9, "bold")).pack(
            side=tk.LEFT, **PAD
        )

        self._play_btn = ttk.Button(self, text="▶ Play", width=8, command=self.app.toggle_playback)
        self._play_btn.pack(side=tk.LEFT, **PAD)

        ttk.Button(self, text="⏮", width=3,
                   command=lambda: self.app.playback_step(-1)).pack(side=tk.LEFT, **PAD)
        ttk.Button(self, text="⏭", width=3,
                   command=lambda: self.app.playback_step(1)).pack(side=tk.LEFT, **PAD)
        ttk.Button(self, text="⟲ Anchor", width=8,
                   command=self.app.playback_reset).pack(side=tk.LEFT, **PAD)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # Loop radius
        ttk.Label(self, text="Loop ±").pack(side=tk.LEFT, padx=(4, 0))
        self.loop_radius_var = tk.IntVar(value=15)
        spin = ttk.Spinbox(
            self, from_=0, to=300, textvariable=self.loop_radius_var, width=5,
            command=self._on_loop_radius,
        )
        spin.pack(side=tk.LEFT, **PAD)
        spin.bind("<Return>", lambda _: self._on_loop_radius())
        ttk.Label(self, text="frames").pack(side=tk.LEFT)

        self.loop_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Loop", variable=self.loop_enabled_var,
                        command=self._on_loop_toggle).pack(side=tk.LEFT, **PAD)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # Speed
        ttk.Label(self, text="Speed").pack(side=tk.LEFT, padx=(4, 0))
        self.speed_var = tk.DoubleVar(value=0.5)
        speed_spin = ttk.Spinbox(
            self, from_=0.1, to=4.0, increment=0.1, textvariable=self.speed_var,
            width=5, format="%.1f", command=self._on_speed,
        )
        speed_spin.pack(side=tk.LEFT, **PAD)
        speed_spin.bind("<Return>", lambda _: self._on_speed())
        ttk.Label(self, text="×").pack(side=tk.LEFT)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # Preview / anchor indicator
        self._info_var = tk.StringVar(value="No label frame")
        self._info_lbl = ttk.Label(self, textvariable=self._info_var, foreground="#4fc3f7")
        self._info_lbl.pack(side=tk.LEFT, padx=8)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_loop_radius(self) -> None:
        try:
            self.app.set_loop_radius(int(self.loop_radius_var.get()))
        except (ValueError, tk.TclError):
            pass

    def _on_loop_toggle(self) -> None:
        self.app.set_loop_enabled(self.loop_enabled_var.get())

    def _on_speed(self) -> None:
        try:
            self.app.set_playback_speed(float(self.speed_var.get()))
        except (ValueError, tk.TclError):
            pass

    # ------------------------------------------------------------------
    # External updates
    # ------------------------------------------------------------------

    def set_playing(self, playing: bool) -> None:
        self._play_btn.config(text="⏸ Pause" if playing else "▶ Play")

    def set_info(self, text: str, previewing: bool = False) -> None:
        self._info_var.set(text)
        self._info_lbl.config(foreground="#ffb74d" if previewing else "#4fc3f7")
