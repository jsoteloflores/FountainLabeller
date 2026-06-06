"""Undo/redo stack using bounding-box mask patches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MaskPatch:
    """Records the changed region of a single committed stroke."""
    x0: int
    y0: int
    before: np.ndarray   # uint8 snapshot of region before stroke
    after: np.ndarray    # uint8 snapshot of region after stroke


class UndoStack:
    def __init__(self, max_depth: int = 50) -> None:
        self._undo: list[MaskPatch] = []
        self._redo: list[MaskPatch] = []
        self.max_depth = max_depth

    def push(self, patch: MaskPatch) -> None:
        self._undo.append(patch)
        if len(self._undo) > self.max_depth:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, mask: np.ndarray) -> bool:
        if not self._undo:
            return False
        patch = self._undo.pop()
        h, w = patch.before.shape
        mask[patch.y0 : patch.y0 + h, patch.x0 : patch.x0 + w] = patch.before
        self._redo.append(patch)
        return True

    def redo(self, mask: np.ndarray) -> bool:
        if not self._redo:
            return False
        patch = self._redo.pop()
        h, w = patch.after.shape
        mask[patch.y0 : patch.y0 + h, patch.x0 : patch.x0 + w] = patch.after
        self._undo.append(patch)
        return True

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)
