"""Stage 1 infrastructure smoke tests.

Run from the project root:

    pytest tests/test_stage1_infrastructure.py -v

Or without pytest:

    python -m pytest tests/test_stage1_infrastructure.py -v

These tests exercise core modules only — no GUI, no Tk window.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent  # FountainLabeller/


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


# ---------------------------------------------------------------------------
# 1. ShortcutConfig — default shortcuts and persistence
# ---------------------------------------------------------------------------

class TestShortcutConfig:
    def test_load_defaults(self) -> None:
        from lava_labeler.core.config import ShortcutConfig, DEFAULT_SHORTCUTS
        sc = ShortcutConfig()
        assert sc.key_for("save_and_next") == "Return"
        assert sc.key_for("play_pause") == "space"
        assert sc.key_for("jump_to_last_saved") == "Shift+R"

    def test_clear_mask_not_plain_c(self) -> None:
        """Brief requirement: plain C must NOT clear the mask."""
        from lava_labeler.core.config import ShortcutConfig
        sc = ShortcutConfig()
        clear_key = sc.key_for("clear_mask")
        assert clear_key.upper() != "C", (
            f"clear_mask is bound to '{clear_key}' — plain C is forbidden "
            "per the Stage 1 polish brief (accidental mask loss)."
        )

    def test_clear_mask_is_ctrl_backspace(self) -> None:
        from lava_labeler.core.config import ShortcutConfig
        sc = ShortcutConfig()
        assert sc.key_for("clear_mask") == "Ctrl+Backspace"

    def test_metadata_shortcuts_present(self) -> None:
        from lava_labeler.core.config import ShortcutConfig
        sc = ShortcutConfig()
        for action in (
            "toggle_wind_affected", "toggle_falling_tephra_visible",
            "toggle_cooling_tephra_visible", "toggle_smoke_obscured",
            "toggle_hard_negative", "mark_needs_review", "approve_human_clean",
        ):
            assert sc.key_for(action), f"No shortcut for {action}"

    def test_custom_shortcut_roundtrip(self) -> None:
        from lava_labeler.core.config import ShortcutConfig
        root = _tmp()
        sc = ShortcutConfig({"save_and_next": "Shift+Return"}, path=root / "shortcuts.json")
        sc.save()
        sc2 = ShortcutConfig.load(root)
        assert sc2.key_for("save_and_next") == "Shift+Return"

    def test_sequence_conversion(self) -> None:
        from lava_labeler.core.config import shortcut_to_tk_sequence as s
        assert s("A") == "<a>"
        assert s("Shift+A") == "<Shift-A>"
        assert s("Ctrl+S") == "<Control-s>"
        assert s("space") == "<space>"
        assert s("?") == "<question>"
        assert s("[") == "<bracketleft>"
        assert s("0") == "<Key-0>"
        assert s("Return") == "<Return>"
        assert s("Ctrl+Backspace") == "<Control-BackSpace>"
        assert s("Shift+R") == "<Shift-R>"


# ---------------------------------------------------------------------------
# 2. CandidateQueue — load, navigate, persist
# ---------------------------------------------------------------------------

class TestCandidateQueue:
    def _example_csv(self, root: Path) -> Path:
        csv_path = root / "candidate_frames.csv"
        csv_path.write_text(
            "candidate_id,video_id,video_path,frame_index,time_seconds,"
            "camera_id,reason,priority,status,notes\n"
            "c1,v,/v.mp4,100,4.0,camA,test,1,unlabeled,\n"
            "c2,v,/v.mp4,200,8.0,camA,test,1,unlabeled,\n"
            "c3,v,/v.mp4,300,12.0,camA,test,1,needs_review,hard case\n"
        )
        return csv_path

    def test_load_example_csv(self) -> None:
        from lava_labeler.core.candidates import CandidateQueue
        csv_path = ROOT / "examples" / "candidate_frames_example.csv"
        assert csv_path.exists(), "examples/candidate_frames_example.csv not found"
        q = CandidateQueue.load(csv_path)
        assert len(q) >= 3

    def test_navigation_and_save(self) -> None:
        from lava_labeler.core.candidates import CandidateQueue
        root = _tmp()
        csv_path = self._example_csv(root)
        q = CandidateQueue.load(csv_path)
        assert len(q) == 3
        first = q.first_unlabeled()
        assert first is not None
        assert first.candidate_id == "c1"
        nxt = q.next_unlabeled("c1")
        assert nxt is not None
        assert nxt.candidate_id == "c2"
        q.set_status("c1", "labeled")
        q.save()
        q2 = CandidateQueue.load(csv_path)
        assert q2.get("c1").status == "labeled"

    def test_invalid_columns_raises(self) -> None:
        from lava_labeler.core.candidates import CandidateQueue
        root = _tmp()
        bad = root / "bad.csv"
        bad.write_text("foo,bar\n1,2\n")
        with pytest.raises(Exception):
            CandidateQueue.load(bad)


# ---------------------------------------------------------------------------
# 3. MetadataStore — create, flag roundtrip, frames.csv
# ---------------------------------------------------------------------------

class TestMetadataStore:
    def test_create_and_roundtrip(self) -> None:
        from lava_labeler.core.metadata import MetadataStore, FrameRecord
        root = _tmp()
        md = MetadataStore(root)
        rec = FrameRecord(
            sample_id="ep1_camA_frame00000042",
            video_path="/v.mp4",
            frame_index=42,
            source_width=640, source_height=480,
        )
        md.add(rec)
        md.save()
        md2 = MetadataStore(root)
        r2 = md2.get("ep1_camA_frame00000042")
        assert r2 is not None
        assert r2.frame_index == 42

    def test_flag_roundtrip(self) -> None:
        from lava_labeler.core.metadata import MetadataStore, FrameRecord
        root = _tmp()
        md = MetadataStore(root)
        rec = FrameRecord(sample_id="s1", video_path="/v.mp4", frame_index=1,
                         source_width=0, source_height=0)
        md.add(rec)
        md.update("s1", wind_affected=True, human_clean=True, mask_provenance="human_clean")
        md.save()
        md2 = MetadataStore(root)
        r = md2.get("s1")
        assert r.wind_affected is True
        assert r.human_clean is True
        assert r.mask_provenance == "human_clean"

    def test_manifest_row_written(self) -> None:
        from lava_labeler.core.metadata import MetadataStore, FrameRecord
        root = _tmp()
        md = MetadataStore(root)
        md.add(FrameRecord(sample_id="s1", video_path="/v.mp4", frame_index=1,
                           source_width=0, source_height=0))
        md.save()
        frames_csv = root / "metadata" / "frames.csv"
        assert frames_csv.exists()
        content = frames_csv.read_text()
        assert "s1" in content


# ---------------------------------------------------------------------------
# 4. SessionRecovery — write, reload, has_resumable_state
# ---------------------------------------------------------------------------

class TestSessionRecovery:
    def test_write_and_reload(self) -> None:
        from lava_labeler.core.session import SessionRecovery
        root = _tmp()
        rec = SessionRecovery(root)
        rec.update(
            candidate_id="c1",
            active_sample_id="ep1_camA_frame00000010",
            video_path="/v.mp4",
            frame_index=10,
        )
        rec.save()
        rec2 = SessionRecovery(root)
        assert rec2.get("candidate_id") == "c1"
        assert rec2.get("active_sample_id") == "ep1_camA_frame00000010"
        assert rec2.has_resumable_state()

    def test_clear_removes_file(self) -> None:
        from lava_labeler.core.session import SessionRecovery
        root = _tmp()
        rec = SessionRecovery(root)
        rec.update(active_sample_id="s1")
        rec.save()
        assert rec.path.exists()
        rec.clear()
        assert not rec.path.exists()

    def test_last_saved_fields(self) -> None:
        from lava_labeler.core.session import SessionRecovery
        root = _tmp()
        rec = SessionRecovery(root)
        rec.update(last_saved_candidate_id="c5", last_saved_sample_id="ep1_camA_frame00000050")
        rec.save()
        rec2 = SessionRecovery(root)
        assert rec2.get("last_saved_candidate_id") == "c5"


# ---------------------------------------------------------------------------
# 5. PlaybackController — loop math
# ---------------------------------------------------------------------------

class TestPlaybackController:
    def test_loop_range(self) -> None:
        from lava_labeler.core.playback import PlaybackController
        p = PlaybackController(fps=25, loop_radius=3)
        p.set_anchor(10, 100)
        assert p.loop_start == 7
        assert p.loop_end == 13
        assert p.interval_ms() == 80  # 1000/25*2 ≈ 80

    def test_wraps_at_loop_end(self) -> None:
        from lava_labeler.core.playback import PlaybackController
        p = PlaybackController(fps=25, loop_radius=3)
        p.set_anchor(10, 100)
        frames = [p.next_frame() for _ in range(8)]
        # Should go 11,12,13, wrap to 7,8,9,10,11
        assert frames[2] == 13
        assert frames[3] == 7  # wrapped

    def test_reset_to_anchor(self) -> None:
        from lava_labeler.core.playback import PlaybackController
        p = PlaybackController(fps=25, loop_radius=5)
        p.set_anchor(20, 100)
        for _ in range(4):
            p.next_frame()
        p.reset_to_anchor()
        assert p.preview_frame == p.anchor_frame


# ---------------------------------------------------------------------------
# 6. FrameCache — byte-budget eviction
# ---------------------------------------------------------------------------

class TestFrameCache:
    def _make_frame(self, h: int = 100, w: int = 100) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_basic_put_get(self) -> None:
        from lava_labeler.core.frame_cache import FrameCache
        cache = FrameCache(max_bytes=10 * 1024 * 1024)
        f = self._make_frame()
        cache.put("/v.mp4", 0, f)
        result = cache.get("/v.mp4", 0)
        assert result is not None
        assert result.shape == f.shape

    def test_evicts_lru_under_budget(self) -> None:
        from lava_labeler.core.frame_cache import FrameCache
        frame = self._make_frame(100, 100)  # 30 KB each
        budget = 2 * frame.nbytes + 1
        cache = FrameCache(max_bytes=budget)
        cache.put("/v.mp4", 0, frame.copy())
        cache.put("/v.mp4", 1, frame.copy())
        # Access frame 0 to make it recently used
        cache.get("/v.mp4", 0)
        # Adding frame 2 should evict frame 1 (LRU)
        cache.put("/v.mp4", 2, frame.copy())
        assert cache.get("/v.mp4", 0) is not None   # still present
        assert cache.get("/v.mp4", 1) is None        # evicted
        assert cache.get("/v.mp4", 2) is not None   # just added

    def test_clear(self) -> None:
        from lava_labeler.core.frame_cache import FrameCache
        cache = FrameCache()
        cache.put("/v.mp4", 0, self._make_frame())
        cache.clear()
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# 7. Export validation — empty-mask rules
# ---------------------------------------------------------------------------

class TestExportValidation:
    def _rec(self, **kwargs):
        from lava_labeler.core.metadata import FrameRecord
        defaults = dict(
            sample_id="s1", video_path="/v.mp4", frame_index=1,
            source_width=0, source_height=0,
            video_id="vid_000001",
            mask_positive_pixels=0, empty_mask_confirmed=False,
            hard_negative=False, label_status="complete", mask_provenance="human_clean",
        )
        defaults.update(kwargs)
        return FrameRecord(**defaults)

    def test_empty_unconfirmed_is_warning(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = self._rec(
            mask_positive_pixels=0,
            empty_mask_confirmed=False,
            hard_negative=False,
            label_status="complete",
        )
        results = validate_records([rec], exportable_ids={"s1"})
        assert results[0]["validation_status"] == "warning"
        assert results[0]["included_in_manifest"] is False

    def test_empty_confirmed_exports_ok(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = self._rec(
            mask_positive_pixels=0,
            empty_mask_confirmed=True,
            hard_negative=False,
            label_status="empty_confirmed",
            mask_provenance="empty_confirmed",
        )
        results = validate_records([rec], exportable_ids={"s1"})
        assert results[0]["validation_status"] == "ok"
        assert results[0]["included_in_manifest"] is True

    def test_hard_negative_empty_exports_ok(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = self._rec(
            mask_positive_pixels=0,
            empty_mask_confirmed=True,
            hard_negative=True,
            label_status="hard_negative",
            mask_provenance="empty_confirmed",
        )
        results = validate_records([rec], exportable_ids={"s1"})
        assert results[0]["included_in_manifest"] is True

    def test_confirmed_empty_with_positive_pixels_is_error(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = self._rec(
            mask_positive_pixels=500,
            empty_mask_confirmed=True,
            label_status="complete",
        )
        results = validate_records([rec], exportable_ids={"s1"})
        assert results[0]["validation_status"] == "error"
        assert results[0]["included_in_manifest"] is False

    def test_hard_negative_with_positive_pixels_is_warning(self) -> None:
        from lava_labeler.core.export import validate_records
        rec = self._rec(
            mask_positive_pixels=100,
            hard_negative=True,
            empty_mask_confirmed=False,
            label_status="hard_negative",
        )
        results = validate_records([rec], exportable_ids={"s1"})
        assert results[0]["validation_status"] == "warning"

    def test_export_writes_validation_report(self) -> None:
        import numpy as np
        from lava_labeler.core.dataset import DatasetFolder
        from lava_labeler.core.metadata import MetadataStore, FrameRecord
        from lava_labeler.core.export import export_dataset

        root = _tmp()
        ds = DatasetFolder(root)
        ds.create("t")
        md = MetadataStore(root)
        # One good record, one bad (empty unconfirmed but complete)
        good = FrameRecord(
            sample_id="good", video_path="/v.mp4", frame_index=1,
            source_width=10, source_height=10,
            mask_positive_pixels=100, label_status="complete",
            mask_provenance="human_clean",
        )
        bad = FrameRecord(
            sample_id="bad", video_path="/v.mp4", frame_index=2,
            source_width=10, source_height=10,
            mask_positive_pixels=0, empty_mask_confirmed=False,
            label_status="complete", mask_provenance="human_clean",
        )
        # Save placeholder images/masks for the good record
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)
        ds.save_image("good", dummy)
        ds.save_mask("good", dummy[:, :, 0])
        md.add(good)
        md.add(bad)
        md.save()

        summary = export_dataset(ds, md)
        assert summary["validation_warnings"] >= 1
        assert (root / "export_validation_report.csv").exists()
        assert (root / "export_validation_report.json").exists()
        # bad record excluded from manifest
        assert summary["exported"] == 1
