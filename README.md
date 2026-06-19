# FountainLabeller

High-throughput, keyboard-first, video-aware annotation tool for building the
lava-fountain segmentation training dataset.

The positive class is **active rising lava fountain material**: coherent
incandescent material still part of the upward rising jet/fountain/plume
emerging from the vent. It is *not* every bright pixel — wind-drifted tephra,
falling particles, detached fragments, ground glow, and cooled plume are
negatives. See the in-app **Labeling Guide** panel for the full definition.

## Run

```bash
pip install -e .
python -m lava_labeler
```

## High-throughput workflow

1. Open a dataset (`File ▸ Open Dataset…`). A `project_config.json` and
   `shortcuts.json` are created on first open.
2. Load a worklist (`File ▸ Load Candidate Queue…`) — see
   `examples/candidate_frames_example.csv` for the schema.
3. Watch motion around the current frame with the **Playback** panel
   (loop-around-current-frame). The editable mask stays anchored to the label
   frame; previewed neighbour frames show a clear banner.
4. Draw/edit the mask, toggle metadata flags with one-key shortcuts, then press
   **Enter** to save and jump to the next unlabeled candidate.
5. Press `Shift+R` at any time to jump back to the last saved candidate.

Work autosaves while dirty and a `session_recovery.json` lets the app resume
after an interrupted session.  A **Progress** panel in the sidebar shows
per-status counts and session throughput.

## High-throughput labeling safety features

- **Clear-mask requires `Ctrl+Backspace`** — plain `C` is not bound to clear,
  preventing accidental mask loss during fast labeling.
- Clearing a non-empty mask asks for confirmation by default
  (`confirm_clear_nonempty_mask` in `project_config.json`).
- **Empty masks must be explicitly confirmed** — pressing `0` marks a frame as
  an intentionally-empty label; unconfirmed empty masks are excluded from the
  training manifest.
- **Hard negatives** can be saved quickly with `N` + `Enter`.
- **Export validation** prevents unlabeled empty masks from entering the
  training manifest and writes `export_validation_report.csv` / `.json` beside
  the manifest.
- Hotkey feedback **toasts** show the flag name and new state after every
  metadata toggle.

## Default keyboard shortcuts

Navigation: `A`/`D` prev/next frame · `Q`/`E` ±10 · `Z`/`X` ±100 ·
`Shift+A`/`Shift+D` prev/next candidate · `Enter` save & next · `S` save ·
`Ctrl+S` force save · `Shift+R` jump to last saved.

Playback: `Space` play/pause · `V` toggle panel.

View: `F` fit · `1` 100% · `R` reset · `H` toggle mask · `M` toggle metadata ·
`?` cheat sheet.

Drawing: `B` brush · `Shift+E` eraser · `[`/`]` brush size · `Ctrl+Z`/`Ctrl+Y`
undo/redo · `O` Otsu brush · **`Ctrl+Backspace` clear mask** · `0` mark
intentionally empty.

Metadata: `W` wind · `T` falling tephra · `Shift+C` cooling tephra · `K` smoke ·
`G` ground glow · `L` exposure bloom · `U` ambiguous · `N` hard-negative ·
`Y` approve (human_clean) · `J` needs_review · `P` model_draft_corrected.

All shortcuts are configurable in `shortcuts.json`. Press `?` for a cheat sheet
that reflects the active config.

## Candidate queue schema

Load a `candidate_frames.csv` via `File ▸ Load Candidate Queue…` or place one
in the dataset folder root for auto-load.

Required columns:

| Column | Description |
|---|---|
| `candidate_id` | Unique ID for this candidate frame |
| `video_id` | Video identifier |
| `video_path` | Absolute path to the source video |
| `frame_index` | 0-based frame index |
| `time_seconds` | Time in seconds |
| `camera_id` | Camera identifier |
| `reason` | Why this frame was selected (free text) |
| `priority` | Integer priority (lower = higher priority) |
| `status` | Starting status — `unlabeled`, `needs_review`, etc. |
| `notes` | Optional notes |

Valid status values: `unlabeled`, `in_progress`, `labeled`, `skipped`,
`needs_review`, `bad_frame`, `hard_negative`, `empty_confirmed`.

See `examples/candidate_frames_example.csv` for concrete examples.

## Export

`File ▸ Export Training Manifest` writes, for every finalized sample:

```
metadata/<sample_id>.json        per-frame metadata (schema_version 1.0)
labels_manifest.csv              one flat row per sample for training
export_validation_report.csv     consistency check results
export_validation_report.json    same, structured
```

The manifest preserves `target_definition = active_rising_lava_fountain`,
`mask_provenance`, and all metadata flags, and distinguishes intentionally-empty
(true-negative / hard-negative) frames from unlabeled ones.

Unconfirmed empty masks and internally-inconsistent records are excluded from
the manifest and flagged in the validation report.

## Running smoke tests

```bash
cd FountainLabeller
pip install -e ".[dev]"          # or: pip install pytest
pytest tests/test_stage1_infrastructure.py -v
```

The tests exercise core modules without opening a GUI window. They check:
shortcut safety (plain C ≠ clear-mask), candidate queue loading/persistence,
metadata flag roundtrips, session recovery, playback loop math, frame-cache
LRU eviction, and export validation rules.
