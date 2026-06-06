"""LRU frame cache to avoid re-decoding recently visited frames."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class FrameCache:
    """LRU cache mapping (video_path, frame_index) → BGR uint8 array.

    Keying by video_path prevents frames from one video polluting another
    when the user switches between videos in the same session.
    """

    def __init__(self, max_size: int = 60) -> None:
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.max_size = max_size

    def get(self, video_path: str, index: int) -> np.ndarray | None:
        key = (video_path, index)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, video_path: str, index: int, frame: np.ndarray) -> None:
        key = (video_path, index)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        self._cache[key] = frame

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
