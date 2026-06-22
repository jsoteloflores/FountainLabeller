"""Stage 1d — dataset definition cleanup tests.

All tests are non-GUI and do not require launching Tkinter.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rec(sample_id: str = "s1", video_id: str = "vid_000001",
              label_status: str = "complete", mask_positive_pixels: int = 100,
              target_definition: str = "active_rising_lava_fountain",
              **kwargs):
    from lava_labeler.core.metadata import FrameRecord
    return FrameRecord(
        sample_id=sample_id,
        video_path="/v.mp4",
        frame_index=0,
        source_width=1920, source_height=1080,
        video_id=video_id,
        label_status=label_status,
        mask_positive_pixels=mask_positive_pixels,
        target_definition=target_definition,
        mask_provenance="human_clean",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. New dataset defaults
# ---------------------------------------------------------------------------

class TestNewDatasetUsesActiveRisingTarget:
    def test_default_config_class_positive(self) -> None:
        from lava_labeler.core.dataset import _DEFAULT_CONFIG
        assert _DEFAULT_CONFIG["class_positive"] == "active_rising_lava_fountain"

    def test_default_config_target_definition(self) -> None:
        from lava_labeler.core.dataset import _DEFAULT_CONFIG
        assert _DEFAULT_CONFIG.get("target_definition") == "active_rising_lava_fountain"

    def test_created_dataset_config_has_active_rising(self) -> None:
        from lava_labeler.core.dataset import DatasetFolder
        with tempfile.TemporaryDirectory() as tmp:
            ds = DatasetFolder(tmp)
            ds.create(name="test_ds")
            cfg = ds.read_config()
        assert cfg["class_positive"] == "active_rising_lava_fountain"
        assert cfg.get("target_definition") == "active_rising_lava_fountain"

    def test_old_broad_string_not_in_defaults(self) -> None:
        from lava_labeler.core.dataset import _DEFAULT_CONFIG
        assert "visible_airborne" not in _DEFAULT_CONFIG["class_positive"]


# ---------------------------------------------------------------------------
# 2. Generated class definition
# ---------------------------------------------------------------------------

class TestClassDefinitionMentionsActiveRising:
    def _load(self) -> str:
        from lava_labeler.core.dataset import DatasetFolder
        with tempfile.TemporaryDirectory() as tmp:
            DatasetFolder(tmp).create()
            return (Path(tmp) / "metadata" / "class_definition.md").read_text()

    def test_mentions_active_rising(self) -> None:
        text = self._load()
        assert "active_rising_lava_fountain" in text

    def test_excludes_drifting_tephra(self) -> None:
        text = self._load()
        assert "wind-drifted tephra" in text or "drifting" in text

    def test_excludes_falling_particles(self) -> None:
        text = self._load()
        assert "falling" in text

    def test_excludes_cooling_plume(self) -> None:
        text = self._load()
        assert "cooling" in text

    def test_excludes_ground_glow(self) -> None:
        text = self._load()
        assert "ground glow" in text

    def test_excludes_exposure_bloom(self) -> None:
        text = self._load()
        assert "exposure bloom" in text

    def test_does_not_contain_old_broad_language(self) -> None:
        text = self._load()
        assert "Visible airborne incandescent" not in text


# ---------------------------------------------------------------------------
# 3. Generated README uses active-rising language
# ---------------------------------------------------------------------------

class TestGeneratedReadmeUsesActiveRisingLanguage:
    def _load(self) -> str:
        from lava_labeler.core.dataset import DatasetFolder
        with tempfile.TemporaryDirectory() as tmp:
            DatasetFolder(tmp).create()
            return (Path(tmp) / "README.md").read_text()

    def test_positive_class_label(self) -> None:
        text = self._load()
        assert "active rising lava fountain material" in text or \
               "active_rising_lava_fountain" in text

    def test_mask_legend_present(self) -> None:
        text = self._load()
        assert "1 = active rising lava fountain material" in text

    def test_lists_metadata_flags(self) -> None:
        text = self._load()
        assert "hard_negative" in text
        assert "wind_affected" in text
        assert "falling_tephra_visible" in text
        assert "smoke_obscured" in text
        assert "ambiguous_boundary" in text

    def test_no_old_broad_positive_class(self) -> None:
        text = self._load()
        assert "Visible airborne incandescent blackbody" not in text


# ---------------------------------------------------------------------------
# 4. Export validation warns on stale target definitions
# ---------------------------------------------------------------------------

class TestExportValidationWarnsOnStaleTargetDefinition:
    def test_stale_visible_airborne_warns(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = _make_rec(target_definition="visible_airborne_incandescent_lava_fountain")
        results = validate_records([rec])
        r = results[0]
        assert r["validation_status"] in ("warning", "error")
        assert "stale" in r["warnings"].lower()

    def test_stale_incandescent_lava_warns(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = _make_rec(target_definition="incandescent_lava")
        results = validate_records([rec])
        assert results[0]["validation_status"] in ("warning", "error")
        assert "stale" in results[0]["warnings"].lower()

    def test_correct_target_no_warning(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = _make_rec(target_definition="active_rising_lava_fountain")
        results = validate_records([rec])
        assert results[0]["validation_status"] == "ok"
        assert not results[0]["warnings"]

    def test_unknown_nonstandard_target_warns(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = _make_rec(target_definition="some_future_class")
        results = validate_records([rec])
        assert results[0]["validation_status"] in ("warning", "error")


# ---------------------------------------------------------------------------
# 5. Strict mode excludes stale-target rows from manifest
# ---------------------------------------------------------------------------

class TestTrainingManifestStrictMode:
    def test_strict_mode_excludes_stale_rows(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = _make_rec(target_definition="visible_airborne_incandescent_lava_fountain")
        results = validate_records([rec], exportable_ids={rec.sample_id},
                                   strict_target=True)
        r = results[0]
        assert r["validation_status"] == "error"
        assert not r["included_in_manifest"]

    def test_strict_mode_keeps_correct_target(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = _make_rec(target_definition="active_rising_lava_fountain")
        results = validate_records([rec], exportable_ids={rec.sample_id},
                                   strict_target=True)
        r = results[0]
        assert r["validation_status"] == "ok"
        assert r["included_in_manifest"]

    def test_non_strict_stale_is_warning_not_excluded(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = _make_rec(target_definition="incandescent_lava")
        results = validate_records([rec], exportable_ids={rec.sample_id},
                                   strict_target=False)
        r = results[0]
        assert r["validation_status"] == "warning"
        # Non-strict: included is determined by exportable_ids, stale does NOT force exclusion
        assert r["included_in_manifest"]

    def test_config_default_strict_true(self) -> None:
        from lava_labeler.core.config import DEFAULT_PROJECT_CONFIG
        assert DEFAULT_PROJECT_CONFIG["metadata"]["strict_target_definition_validation"] is True
