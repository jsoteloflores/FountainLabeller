"""Playback controller for looping motion preview around a label frame.

This is intentionally Tk-agnostic: it holds playback *state* and computes the
next frame index and timer interval.  The GUI owns the actual ``after`` timer
and the rendering.  The editable label frame is the *anchor*; previewing never
changes it.
"""

from __future__ import annotations


class PlaybackController:
    """Computes preview frame stepping for loop-around-current-frame playback."""

    def __init__(self, fps: float = 25.0, loop_radius: int = 15, speed: float = 0.5) -> None:
        self.is_playing: bool = False
        self.loop_enabled: bool = True
        self.fps: float = fps if fps > 0 else 25.0
        self.loop_radius: int = max(0, loop_radius)
        self.speed: float = max(0.05, speed)

        self.anchor_frame: int = 0     # editable label frame (mask anchor)
        self.preview_frame: int = 0    # currently displayed frame
        self.max_frame: int = 0        # last valid frame index in the video

    # ------------------------------------------------------------------

    @property
    def loop_start(self) -> int:
        return max(0, self.anchor_frame - self.loop_radius)

    @property
    def loop_end(self) -> int:
        end = self.anchor_frame + self.loop_radius
        return min(end, self.max_frame) if self.max_frame > 0 else end

    @property
    def is_previewing(self) -> bool:
        """True when the displayed frame differs from the editable anchor."""
        return self.preview_frame != self.anchor_frame

    def set_anchor(self, frame_index: int, max_frame: int) -> None:
        self.anchor_frame = max(0, frame_index)
        self.max_frame = max(0, max_frame)
        self.preview_frame = self.anchor_frame

    def interval_ms(self) -> int:
        """Milliseconds between preview frames for the current fps and speed."""
        base = 1000.0 / self.fps if self.fps > 0 else 40.0
        return max(10, int(round(base / self.speed)))

    def next_frame(self) -> int:
        """Advance the preview frame one step (wrapping around the loop)."""
        nf = self.preview_frame + 1
        if self.loop_enabled:
            start, end = self.loop_start, self.loop_end
            if nf > end:
                nf = start
        else:
            if self.max_frame > 0 and nf > self.max_frame:
                nf = self.loop_start
        self.preview_frame = nf
        return nf

    def reset_to_anchor(self) -> None:
        self.preview_frame = self.anchor_frame
