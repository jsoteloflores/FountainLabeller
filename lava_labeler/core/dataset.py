"""On-disk dataset folder layout and file I/O."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


DATASET_CONFIG_NAME = "dataset_config.json"

_DEFAULT_CONFIG: dict = {
    "dataset_name": "lava_fountain_dataset",
    "dataset_version": "0.1.0",
    "class_positive": "active_rising_lava_fountain",
    "target_definition": "active_rising_lava_fountain",
    "mask_format": {
        "dtype": "uint8",
        "background": 0,
        "positive": 255,
    },
    "images_are_source_resolution": True,
    "masks_match_images_exactly": True,
    "roi_policy": {
        "enabled": True,
        "preferred_mode": "fixed_size_draggable_roi",
        "description": (
            "Samples may be source-resolution crops using a fixed-size ROI placed over "
            "the target fountain. Pixels outside the ROI are outside the analysis domain "
            "and are not treated as negative examples."
        ),
    },
    "created_by": "lava-labeler",
    "notes": "",
}

_CLASS_DEFINITION = """\
# Class Definition

## Positive class: active_rising_lava_fountain

Label pixels that are part of coherent incandescent lava material still
dynamically connected to the upward fountain, jet, or rising plume structure
emerging from the vent/source region.

Include:

- coherent rising fountain core/body
- incandescent material actively moving upward as part of the fountain
- connected or visually continuous rising fountain envelope
- turbulent active fountain body before it detaches, cools, falls, or drifts away

## Negative class: background

Do not label:

- wind-drifted tephra
- falling incandescent particles
- detached fragments no longer part of the rising fountain
- cooling/diffuse plume material
- smoke, steam, ash, or obscuration
- ground glow or old incandescent deposits
- vent/crater walls/sky/background terrain
- camera artifacts, exposure bloom, or lens glare
- any incandescent material that is not part of the active rising fountain structure

When boundary behavior is unclear, label conservatively and set
`ambiguous_boundary = true` in frame metadata.

## Outside ROI

Pixels outside the ROI are outside the analysis domain for ROI-crop datasets.
They are not labeled as positive or negative and should not be used to penalize
the model.

## Nearby Vents

Nearby vents outside the target ROI are excluded from training and measurement
so that non-target incandescent activity does not bias target-fountain
segmentation.
"""

_README = """\
# Lava Fountain Segmentation Dataset

Binary segmentation dataset for active rising lava fountain material in
volcanic eruption video.

## Positive class: active_rising_lava_fountain

```
1 = active rising lava fountain material
0 = background / non-rising / detached / falling / drifting / cooling material
```

Label pixels that are part of coherent incandescent lava material still
dynamically connected to the upward fountain, jet, or rising plume structure
emerging from the vent/source region.

## Negative class: background

Explicitly excluded from the positive class:

- wind-drifted tephra
- falling incandescent particles or detached fragments
- cooling/diffuse plume material
- smoke, steam, ash, or obscuration
- ground glow or old incandescent deposits
- exposure bloom, lens glare, or camera artifacts
- vent/crater walls, sky, or background terrain

See `metadata/class_definition.md` for full edge-case guidance.

## Metadata flags

The following per-frame flags are part of the dataset accounting system and
are expected by Stage 2 training:

- `hard_negative` — frame with no fountain visible (explicit negative example)
- `wind_affected` — wind significantly deflects or disperses the fountain
- `falling_tephra_visible` — falling incandescent particles visible in frame
- `cooling_tephra_visible` — cooling/diffuse tephra visible in frame
- `smoke_obscured` — fountain partially or fully obscured by smoke/steam/ash
- `ambiguous_boundary` — boundary between rising fountain and background is unclear

## File Pairing

Each sample has an image and a matching mask:

```
images/all/<sample_id>.png
masks/all/<sample_id>_mask.png
```

## Format

- **Images**: PNG, RGB, full source resolution.
- **Masks**: PNG, uint8 grayscale. Background = `0`, positive = `255`.
  Masks are NOT antialiased. Every pixel is exactly 0 or 255.
- Image and mask have identical (width, height).

## Metadata

`metadata/frames.csv` is the complete index. Load it to pair image/mask paths
and filter by status, episode, camera, etc.

```python
import pandas as pd
df = pd.read_csv("metadata/frames.csv")
complete = df[df.label_status == "complete"]
for _, row in complete.iterrows():
    img_path  = row["image_path"]
    mask_path = row["mask_path"]
```
"""


def make_sample_id(episode_id: str, camera_id: str, frame_index: int) -> str:
    ep = (episode_id.strip() or "unknownEpisode").replace(" ", "_")
    cam = (camera_id.strip() or "unknownCamera").replace(" ", "_")
    return f"{ep}_{cam}_frame{frame_index:08d}"


class DatasetFolder:
    """Manages the on-disk layout of an lava-labeler dataset."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def create(self, name: str = "") -> None:
        """Create all subdirectories, config, class definition, and README."""
        for subdir in (
            self.root / "images" / "all",
            self.root / "masks" / "all",
            self.root / "metadata",
            self.root / "qc" / "overlays",
            self.root / "qc" / "thumbnails",
        ):
            subdir.mkdir(parents=True, exist_ok=True)

        cfg_path = self.root / DATASET_CONFIG_NAME
        if not cfg_path.exists():
            cfg = dict(_DEFAULT_CONFIG)
            if name:
                cfg["dataset_name"] = name
            cfg["created_at"] = datetime.now(timezone.utc).isoformat()
            cfg_path.write_text(json.dumps(cfg, indent=2))

        class_def = self.root / "metadata" / "class_definition.md"
        if not class_def.exists():
            class_def.write_text(_CLASS_DEFINITION)

        readme = self.root / "README.md"
        if not readme.exists():
            readme.write_text(_README)

    @property
    def exists(self) -> bool:
        return self.root.exists()

    # ------------------------------------------------------------------
    # Config read/write
    # ------------------------------------------------------------------

    def config_path(self) -> Path:
        return self.root / DATASET_CONFIG_NAME

    def read_config(self) -> dict:
        """Return the dataset_config.json contents (empty dict if missing/invalid)."""
        p = self.config_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def write_config(self, cfg: dict) -> None:
        self.config_path().write_text(json.dumps(cfg, indent=2))

    def update_config(self, **kwargs) -> None:
        """Merge *kwargs* into the existing config and persist it."""
        cfg = self.read_config()
        cfg.update(kwargs)
        self.write_config(cfg)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def image_path(self, sample_id: str) -> Path:
        return self.root / "images" / "all" / f"{sample_id}.png"

    def mask_path(self, sample_id: str) -> Path:
        return self.root / "masks" / "all" / f"{sample_id}_mask.png"

    def qc_overlay_path(self, sample_id: str) -> Path:
        return self.root / "qc" / "overlays" / f"{sample_id}_overlay.png"

    def qc_thumb_path(self, sample_id: str) -> Path:
        return self.root / "qc" / "thumbnails" / f"{sample_id}_thumb.png"

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save_image(self, sample_id: str, frame_bgr: np.ndarray) -> Path:
        path = self.image_path(sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        from lava_labeler.core.config import atomic_write_bytes
        ok, buf = cv2.imencode(".png", frame_bgr)
        if not ok:
            raise IOError(f"Failed to encode image for {sample_id}")
        atomic_write_bytes(path, buf.tobytes())
        return path

    def save_mask(self, sample_id: str, mask: np.ndarray) -> Path:
        if mask.dtype != np.uint8:
            raise ValueError("Mask must be uint8.")
        # Canonicalize to strict binary before writing
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        path = self.mask_path(sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        from lava_labeler.core.config import atomic_write_bytes
        ok, buf = cv2.imencode(".png", mask)
        if not ok:
            raise IOError(f"Failed to encode mask for {sample_id}")
        atomic_write_bytes(path, buf.tobytes())
        return path

    def load_mask(self, sample_id: str) -> np.ndarray | None:
        path = self.mask_path(sample_id)
        if not path.exists():
            return None
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            return None
        # Canonicalize to strict binary: handles any imported or non-binary mask
        return np.where(raw > 0, 255, 0).astype(np.uint8)

    def load_image(self, sample_id: str) -> np.ndarray | None:
        path = self.image_path(sample_id)
        if not path.exists():
            return None
        return cv2.imread(str(path))
