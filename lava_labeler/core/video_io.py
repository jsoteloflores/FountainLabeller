"""Video file IO using OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def duration_str(self) -> str:
        t = self.duration_seconds
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        if h:
            return f"{h}:{m:02d}:{s:05.2f}"
        return f"{m}:{s:05.2f}"

    @property
    def summary(self) -> str:
        return (
            f"{self.frame_count} frames  |  {self.fps:.3f} fps  |  "
            f"{self.width}×{self.height}  |  {self.duration_str}"
        )


class VideoReader:
    """Thin wrapper around cv2.VideoCapture with frame-index seeking."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")

        self._info = VideoInfo(
            path=self.path,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def info(self) -> VideoInfo:
        return self._info

    def read_frame(self, index: int) -> np.ndarray | None:
        """Read a frame by 0-based index. Returns BGR uint8 array or None."""
        if not (0 <= index < self._info.frame_count):
            return None
        current = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current != index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()
