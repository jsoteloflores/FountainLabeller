"""LRU frame cache with a byte-budget eviction policy.

At 4K (3840×2160×3 uint8 ≈ 24.9 MB/frame) a count of 60 would consume ~1.5 GB.
A memory budget evicts least-recently-used frames automatically.
Default budget is 512 MB, holding ~20 4K frames with room to spare.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class FrameCache:
    """LRU cache mapping (video_path, frame_index) → BGR uint8 array.

    Evicts least-recently-used frames once the total byte footprint
    exceeds *max_bytes*.  Keying by video_path prevents frames from
    different videos from colliding.
    """

    def __init__(self, max_bytes: int = 512 * 1024 * 1024) -> None:
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.max_bytes = max_bytes
        self._current_bytes: int = 0

    # ------------------------------------------------------------------

    def get(self, video_path: str, index: int) -> np.ndarray | None:
        key = (video_path, index)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, video_path: str, index: int, frame: np.ndarray) -> None:
        key = (video_path, index)
        new_bytes = frame.nbytes

        if key in self._cache:
            self._current_bytes -= self._cache[key].nbytes
            self._cache.move_to_end(key)
            self._cache[key] = frame
            self._current_bytes += new_bytes
            return

        # Evict LRU frames until the budget is satisfied
        while self._cache and self._current_bytes + new_bytes > self.max_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._current_bytes -= evicted.nbytes

        self._cache[key] = frame
        self._current_bytes += new_bytes

    def clear(self) -> None:
        self._cache.clear()
        self._current_bytes = 0

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def used_mb(self) -> float:
        return self._current_bytes / (1024 * 1024)
