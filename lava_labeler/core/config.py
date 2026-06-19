"""Project configuration, keyboard shortcut config, and atomic file writes.

This module centralises:

* :func:`atomic_write_bytes` / :func:`atomic_write_text` — crash-safe file
  writes using the write-temp-then-replace pattern.
* :class:`ProjectConfig` — loads/saves ``project_config.json`` with sane
  defaults (loop radius, playback speed, autosave interval, cache settings…).
* :class:`ShortcutConfig` — loads/saves ``shortcuts.json`` and converts the
  human-friendly key names ("Space", "Shift+A", "Ctrl+S") into Tk binding
  sequences ("<space>", "<Shift-A>", "<Control-s>").

All files default to living in the dataset root, but the loaders accept an
explicit path so the same code works for project-level or session files.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------
# Atomic writes
# ----------------------------------------------------------------------

def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically."""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, obj: Any) -> None:
    """Serialise *obj* as pretty JSON and write it atomically."""
    atomic_write_text(path, json.dumps(obj, indent=2))


# ----------------------------------------------------------------------
# Project config
# ----------------------------------------------------------------------

PROJECT_CONFIG_NAME = "project_config.json"

DEFAULT_PROJECT_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "target_definition": "active_rising_lava_fountain",
    "default_loop_radius_frames": 15,
    "default_playback_speed": 0.5,
    "autosave_interval_seconds": 10,
    "frame_cache_radius": 30,
    "max_cached_frames": 200,
    "show_labeling_guide_on_startup": True,
    "require_empty_mask_confirmation_once_per_session": True,
    "default_mask_opacity": 0.4,
    "shortcut_config_path": "shortcuts.json",
}


class ProjectConfig:
    """Loads and persists ``project_config.json`` with defaults applied."""

    def __init__(self, values: dict[str, Any] | None = None, path: Path | None = None) -> None:
        self._values: dict[str, Any] = dict(DEFAULT_PROJECT_CONFIG)
        if values:
            self._values.update(values)
        self._path = path

    @classmethod
    def load(cls, root: str | Path) -> "ProjectConfig":
        root = Path(root)
        path = root / PROJECT_CONFIG_NAME
        values: dict[str, Any] = {}
        if path.exists():
            try:
                values = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                values = {}
        cfg = cls(values, path=path)
        # Persist defaults the first time so the file is discoverable/editable.
        if not path.exists():
            cfg.save()
        return cfg

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, DEFAULT_PROJECT_CONFIG.get(key, default))

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value

    def update(self, **kwargs: Any) -> None:
        self._values.update(kwargs)

    def save(self) -> None:
        if self._path is not None:
            atomic_write_json(self._path, self._values)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)


# ----------------------------------------------------------------------
# Shortcut config
# ----------------------------------------------------------------------

DEFAULT_SHORTCUTS: dict[str, str] = {
    # Navigation
    "previous_frame": "A",
    "next_frame": "D",
    "jump_back_small": "Q",
    "jump_forward_small": "E",
    "jump_back_large": "Z",
    "jump_forward_large": "X",
    "previous_candidate": "Shift+A",
    "next_candidate": "Shift+D",
    "save_and_next": "Return",
    "save": "S",
    "force_save": "Ctrl+S",
    # View
    "fit_view": "F",
    "zoom_100": "1",
    "reset_view": "R",
    "toggle_mask": "H",
    "toggle_playback_panel": "V",
    "toggle_metadata_panel": "M",
    "cheat_sheet": "?",
    # Playback
    "play_pause": "space",
    # Drawing
    "brush": "B",
    "eraser": "Shift+E",
    "brush_smaller": "[",
    "brush_larger": "]",
    "undo": "Ctrl+Z",
    "redo": "Ctrl+Y",
    "otsu_brush": "O",
    "clear_mask": "C",
    "mark_empty": "0",
    # Metadata toggles
    "toggle_wind_affected": "W",
    "toggle_falling_tephra_visible": "T",
    "toggle_cooling_tephra_visible": "Shift+C",
    "toggle_smoke_obscured": "K",
    "toggle_ground_glow_visible": "G",
    "toggle_exposure_bloom": "L",
    "toggle_ambiguous_boundary": "U",
    "toggle_hard_negative": "N",
    "approve_human_clean": "Y",
    "mark_needs_review": "J",
    "toggle_model_draft_corrected": "P",
}

# Human-readable grouping used by the cheat sheet.
SHORTCUT_GROUPS: dict[str, list[str]] = {
    "Navigation": [
        "previous_frame", "next_frame", "jump_back_small", "jump_forward_small",
        "jump_back_large", "jump_forward_large", "previous_candidate",
        "next_candidate", "save_and_next", "save", "force_save",
    ],
    "Playback": ["play_pause", "toggle_playback_panel"],
    "View": [
        "fit_view", "zoom_100", "reset_view", "toggle_mask",
        "toggle_metadata_panel", "cheat_sheet",
    ],
    "Drawing": [
        "brush", "eraser", "brush_smaller", "brush_larger", "undo", "redo",
        "otsu_brush", "clear_mask", "mark_empty",
    ],
    "Metadata": [
        "toggle_wind_affected", "toggle_falling_tephra_visible",
        "toggle_cooling_tephra_visible", "toggle_smoke_obscured",
        "toggle_ground_glow_visible", "toggle_exposure_bloom",
        "toggle_ambiguous_boundary", "toggle_hard_negative",
        "approve_human_clean", "mark_needs_review",
        "toggle_model_draft_corrected",
    ],
}

# Special key names → Tk keysyms
_SPECIAL_KEYS: dict[str, str] = {
    "space": "space",
    "enter": "Return",
    "return": "Return",
    "tab": "Tab",
    "esc": "Escape",
    "escape": "Escape",
    "[": "bracketleft",
    "]": "bracketright",
    "?": "question",
    "/": "slash",
    ".": "period",
    ",": "comma",
    "-": "minus",
    "=": "equal",
}

_MODIFIER_MAP: dict[str, str] = {
    "ctrl": "Control",
    "control": "Control",
    "shift": "Shift",
    "alt": "Alt",
    "option": "Alt",
    "cmd": "Command",
    "command": "Command",
    "meta": "Command",
}


def shortcut_to_tk_sequence(friendly: str) -> str | None:
    """Convert a friendly shortcut string into a Tk binding sequence.

    Examples
    --------
    "A"        -> "<a>"
    "Shift+A"  -> "<Shift-A>"
    "Ctrl+S"   -> "<Control-s>"
    "Space"    -> "<space>"
    "?"        -> "<question>"
    "["        -> "<bracketleft>"
    "0"        -> "<Key-0>"
    """
    if not friendly:
        return None
    parts = [p for p in friendly.replace(" ", "").split("+") if p]
    if not parts:
        return None

    modifiers: list[str] = []
    key = parts[-1]
    for raw in parts[:-1]:
        mod = _MODIFIER_MAP.get(raw.lower())
        if mod:
            modifiers.append(mod)

    has_shift = "Shift" in modifiers
    key_lower = key.lower()

    if key_lower in _SPECIAL_KEYS:
        keysym = _SPECIAL_KEYS[key_lower]
    elif len(key) == 1 and key.isalpha():
        # Letters: Shift uppercases the keysym.
        keysym = key.upper() if has_shift else key.lower()
    elif len(key) == 1 and key.isdigit():
        keysym = f"Key-{key}"
    else:
        keysym = key

    seq = "".join(f"{m}-" for m in modifiers) + keysym
    return f"<{seq}>"


class ShortcutConfig:
    """Loads, persists, and resolves keyboard shortcuts."""

    def __init__(self, mapping: dict[str, str] | None = None, path: Path | None = None) -> None:
        self._map: dict[str, str] = dict(DEFAULT_SHORTCUTS)
        if mapping:
            # Only accept known actions; ignore typos to keep bindings valid.
            for k, v in mapping.items():
                if k in DEFAULT_SHORTCUTS and isinstance(v, str) and v:
                    self._map[k] = v
        self._path = path

    @classmethod
    def load(cls, root: str | Path, filename: str = "shortcuts.json") -> "ShortcutConfig":
        root = Path(root)
        path = root / filename
        mapping: dict[str, str] = {}
        if path.exists():
            try:
                mapping = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                mapping = {}
        cfg = cls(mapping, path=path)
        if not path.exists():
            cfg.save()
        return cfg

    def key_for(self, action: str) -> str:
        """Return the friendly key string bound to *action*."""
        return self._map.get(action, "")

    def sequence_for(self, action: str) -> str | None:
        """Return the Tk binding sequence for *action* (or None)."""
        return shortcut_to_tk_sequence(self._map.get(action, ""))

    def all_actions(self) -> dict[str, str]:
        return dict(self._map)

    def save(self) -> None:
        if self._path is not None:
            atomic_write_json(self._path, self._map)
