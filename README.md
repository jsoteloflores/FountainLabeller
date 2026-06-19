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
2. Load a worklist (`File ▸ Load Candidate Queue…`) or drop a
   `candidate_frames.csv` into the dataset folder to auto-load it.
3. Watch motion around the current frame with the **Playback** panel
   (loop-around-current-frame). The editable mask stays anchored to the label
   frame; previewed neighbour frames show a clear banner.
4. Draw/edit the mask, toggle metadata flags with one-key shortcuts, then press
   **Enter** to save and jump to the next unlabeled candidate.

Work autosaves while dirty and a `session_recovery.json` lets the app resume
after an interrupted session.

## Default keyboard shortcuts

Navigation: `A`/`D` prev/next frame · `Q`/`E` ±10 · `Z`/`X` ±100 ·
`Shift+A`/`Shift+D` prev/next candidate · `Enter` save & next · `S` save ·
`Ctrl+S` force save.

Playback: `Space` play/pause · `V` toggle panel.

View: `F` fit · `1` 100% · `R` reset · `H` toggle mask · `M` toggle metadata ·
`?` cheat sheet.

Drawing: `B` brush · `Shift+E` eraser · `[`/`]` brush size · `Ctrl+Z`/`Ctrl+Y`
undo/redo · `O` Otsu brush · `C` clear mask · `0` mark intentionally empty.

Metadata: `W` wind · `T` falling tephra · `Shift+C` cooling tephra · `K` smoke ·
`G` ground glow · `L` exposure bloom · `U` ambiguous · `N` hard-negative ·
`Y` approve (human_clean) · `J` needs_review · `P` model_draft_corrected.

All shortcuts are configurable in `shortcuts.json`. Press `?` for a cheat sheet
that reflects the active config.

## Export

`File ▸ Export Training Manifest` writes, for every finalized sample:

```
metadata/<sample_id>.json     per-frame metadata (schema_version 1.0)
labels_manifest.csv           one flat row per sample for training
```

The manifest preserves `target_definition = active_rising_lava_fountain`,
`mask_provenance`, and all metadata flags, and distinguishes intentionally-empty
(true-negative / hard-negative) frames from unlabeled ones.
