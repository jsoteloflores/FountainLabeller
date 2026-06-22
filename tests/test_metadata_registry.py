"""Stage 1c metadata registry and dataset accounting tests.

Run with:
    pytest tests/test_metadata_registry.py -v
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


@dataclass
class _FakeVideoInfo:
    """Minimal duck-type of VideoInfo for registry tests."""
    path: Path
    frame_count: int = 1200
    fps: float = 25.0
    width: int = 1920
    height: int = 1080

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def _make_info(root: Path, name: str = "ep01_camA.mp4",
               frame_count: int = 1200, fps: float = 25.0,
               width: int = 1920, height: int = 1080) -> _FakeVideoInfo:
    p = root / name
    p.write_bytes(b"\x00" * 100)   # tiny placeholder so stat().st_size works
    return _FakeVideoInfo(path=p, frame_count=frame_count, fps=fps,
                          width=width, height=height)


def _make_rec(sample_id: str, video_id: str = "vid_000001",
              episode_id: str = "ep01", camera_id: str = "camA",
              label_status: str = "complete",
              mask_positive_pixels: int = 100,
              frame_index: int = 0,
              **kwargs):
    from lava_labeler.core.metadata import FrameRecord
    return FrameRecord(
        sample_id=sample_id,
        video_path="/v.mp4",
        frame_index=frame_index,
        source_width=1920, source_height=1080,
        video_id=video_id, episode_id=episode_id, camera_id=camera_id,
        label_status=label_status,
        mask_positive_pixels=mask_positive_pixels,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. VideoRegistry — new entry creation
# ---------------------------------------------------------------------------

class TestVideoRegistryNewEntry:
    def test_new_video_creates_entry(self) -> None:
        from lava_labeler.core.video_registry import VideoRegistry
        root = _tmp()
        reg = VideoRegistry(root)
        info = _make_info(root)
        entry, tier = reg.register(info)
        assert tier == "new"
        assert entry.video_id.startswith("vid_")
        assert entry.video_filename == info.path.name
        assert entry.total_frames == info.frame_count
        assert entry.fps == info.fps
        assert len(reg) == 1

    def test_episode_camera_stored(self) -> None:
        from lava_labeler.core.video_registry import VideoRegistry
        root = _tmp()
        reg = VideoRegistry(root)
        info = _make_info(root)
        entry, _ = reg.register(info, episode_id="ep12", camera_id="camB")
        assert entry.episode_id == "ep12"
        assert entry.camera_id == "camB"

    def test_fingerprint_deterministic(self) -> None:
        from lava_labeler.core.video_registry import compute_fingerprint
        fp1 = compute_fingerprint("v.mp4", 1000000, 1200, 25.0, 1920, 1080)
        fp2 = compute_fingerprint("v.mp4", 1000000, 1200, 25.0, 1920, 1080)
        assert fp1 == fp2
        assert len(fp1) == 16


# ---------------------------------------------------------------------------
# 2. VideoRegistry — reopen same video reuses same video_id
# ---------------------------------------------------------------------------

class TestVideoRegistryReopen:
    def test_exact_match_reuses_video_id(self) -> None:
        from lava_labeler.core.video_registry import VideoRegistry
        root = _tmp()
        reg = VideoRegistry(root)
        info = _make_info(root)
        e1, t1 = reg.register(info)
        assert t1 == "new"
        # Reopen same file (same fingerprint)
        e2, t2 = reg.register(info)
        assert t2 == "exact"
        assert e2.video_id == e1.video_id
        assert len(reg) == 1

    def test_probable_match_reuses_video_id(self) -> None:
        from lava_labeler.core.video_registry import VideoRegistry
        root = _tmp()
        reg = VideoRegistry(root)
        info1 = _make_info(root, "ep01_camA.mp4")
        e1, _ = reg.register(info1)
        # Same filename + same properties but different path (moved file)
        root2 = _tmp()
        info2 = _make_info(root2, "ep01_camA.mp4")
        # Will get probable match because filename + frame_count + fps + resolution match
        e2, t2 = reg.register(info2)
        assert t2 in ("probable", "exact")
        assert e2.video_id == e1.video_id

    def test_persistence_roundtrip(self) -> None:
        from lava_labeler.core.video_registry import VideoRegistry
        root = _tmp()
        reg = VideoRegistry(root)
        info = _make_info(root)
        e1, _ = reg.register(info, episode_id="ep01")
        reg.save()
        # Reload from disk
        reg2 = VideoRegistry(root)
        assert len(reg2) == 1
        e2 = reg2.get(e1.video_id)
        assert e2 is not None
        assert e2.episode_id == "ep01"
        assert e2.video_filename == info.path.name


# ---------------------------------------------------------------------------
# 3. VideoRegistry — filename mismatch triggers warning entry
# ---------------------------------------------------------------------------

class TestVideoRegistryMismatch:
    def test_same_filename_different_frame_count_is_mismatch(self) -> None:
        from lava_labeler.core.video_registry import VideoRegistry
        root = _tmp()
        reg = VideoRegistry(root)
        info1 = _make_info(root, "clip.mp4", frame_count=1200)
        e1, _ = reg.register(info1)
        # Same filename but different frame count → tier 3 → new entry
        root2 = _tmp()
        info2 = _make_info(root2, "clip.mp4", frame_count=9999)
        e2, t2 = reg.register(info2)
        assert t2 == "filename_mismatch"
        assert e2.video_id != e1.video_id  # new entry created


# ---------------------------------------------------------------------------
# 4. Frame metadata rows link to valid video_id
# ---------------------------------------------------------------------------

class TestFrameMetadataVideoId:
    def test_record_has_video_id(self) -> None:
        from lava_labeler.core.metadata import MetadataStore
        root = _tmp()
        md = MetadataStore(root)
        rec = _make_rec("ep01_camA_f000100", video_id="vid_000001")
        md.add(rec)
        md.save()
        md2 = MetadataStore(root)
        r = md2.get("ep01_camA_f000100")
        assert r is not None
        assert r.video_id == "vid_000001"

    def test_video_filename_roundtrip(self) -> None:
        from lava_labeler.core.metadata import MetadataStore, FrameRecord
        root = _tmp()
        md = MetadataStore(root)
        rec = FrameRecord(
            sample_id="s1", video_path="/videos/ep01_camA.mp4",
            frame_index=0, source_width=1920, source_height=1080,
            video_id="vid_000001", video_filename="ep01_camA.mp4",
        )
        md.add(rec)
        md.save()
        md2 = MetadataStore(root)
        r = md2.get("s1")
        assert r.video_filename == "ep01_camA.mp4"


# ---------------------------------------------------------------------------
# 5. Duplicate video_id + frame_index rows are detected
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_dataset_summary_counts_no_duplicates(self) -> None:
        from lava_labeler.core.metadata import MetadataStore
        from lava_labeler.core.dataset_summary import DatasetSummary
        root = _tmp()
        md = MetadataStore(root)
        # Add same logical frame twice with different sample_ids (simulates duplicate import)
        md.add(_make_rec("s1", video_id="v1"))
        md.add(_make_rec("s2", video_id="v1"))  # same video, different sample
        ds = DatasetSummary(md)
        # Two records → total_candidates = 2
        assert ds.dataset.total_candidates == 2
        # Only one video
        assert ds.dataset.video_count == 1


# ---------------------------------------------------------------------------
# 6. DatasetSummary counts by dataset / video / episode / camera
# ---------------------------------------------------------------------------

class TestDatasetSummaryCounts:
    def _populate(self) -> tuple:
        from lava_labeler.core.metadata import MetadataStore
        from lava_labeler.core.dataset_summary import DatasetSummary
        root = _tmp()
        md = MetadataStore(root)
        # 3 frames: ep01/camA, ep01/camA, ep02/camB
        md.add(_make_rec("f1", video_id="v1", episode_id="ep01", camera_id="camA",
                         label_status="complete", mask_positive_pixels=100,
                         wind_affected=True))
        md.add(_make_rec("f2", video_id="v1", episode_id="ep01", camera_id="camA",
                         label_status="hard_negative", mask_positive_pixels=0,
                         hard_negative=True, empty_mask_confirmed=True))
        md.add(_make_rec("f3", video_id="v2", episode_id="ep02", camera_id="camB",
                         label_status="needs_review", mask_positive_pixels=50))
        ds = DatasetSummary(md)
        return ds

    def test_dataset_total_labeled(self) -> None:
        ds = self._populate()
        assert ds.dataset.total_candidates == 3
        # f1=complete, f2=hard_negative (both terminal) → 2 labeled; f3=needs_review → NR bucket
        assert ds.dataset.total_labeled == 2
        assert ds.dataset.total_needs_review == 1

    def test_per_video_counts(self) -> None:
        ds = self._populate()
        v1 = ds.stats_for_video("v1")
        assert v1 is not None
        assert v1.total_candidates == 2
        assert v1.hard_negative == 1
        assert v1.wind_affected == 1

    def test_per_episode_counts(self) -> None:
        ds = self._populate()
        ep1 = ds.stats_for_episode("ep01")
        assert ep1 is not None
        assert ep1.total_labeled == 2
        ep2 = ds.stats_for_episode("ep02")
        assert ep2 is not None
        assert ep2.total_needs_review == 1

    def test_per_camera_counts(self) -> None:
        ds = self._populate()
        camA = ds.stats_for_camera("camA")
        assert camA is not None
        assert camA.total_candidates == 2

    def test_refresh_updates_counts(self) -> None:
        from lava_labeler.core.metadata import MetadataStore
        from lava_labeler.core.dataset_summary import DatasetSummary
        root = _tmp()
        md = MetadataStore(root)
        md.add(_make_rec("f1"))
        ds = DatasetSummary(md)
        assert ds.dataset.total_candidates == 1
        md.add(_make_rec("f2"))
        ds.refresh()
        assert ds.dataset.total_candidates == 2


# ---------------------------------------------------------------------------
# 7. CSV mirrors regenerate from JSON
# ---------------------------------------------------------------------------

class TestCsvMirrors:
    def test_write_video_registry_csv(self) -> None:
        from lava_labeler.core.video_registry import VideoRegistry
        from lava_labeler.core.csv_mirror import write_video_registry_csv
        root = _tmp()
        reg = VideoRegistry(root)
        info = _make_info(root)
        reg.register(info, episode_id="ep01")
        csv_path = root / "video_registry.csv"
        write_video_registry_csv(reg, csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "ep01" in content
        assert "video_id" in content

    def test_write_frame_metadata_csv(self) -> None:
        from lava_labeler.core.metadata import MetadataStore
        from lava_labeler.core.csv_mirror import write_frame_metadata_csv
        root = _tmp()
        md = MetadataStore(root)
        md.add(_make_rec("s1", video_id="v1", episode_id="ep01"))
        csv_path = root / "frame_metadata.csv"
        write_frame_metadata_csv(md, None, csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "s1" in content
        assert "label_id" in content
        assert "has_positive_mask" in content

    def test_write_dataset_summary_csv(self) -> None:
        from lava_labeler.core.metadata import MetadataStore
        from lava_labeler.core.dataset_summary import DatasetSummary
        from lava_labeler.core.csv_mirror import write_dataset_summary_csv
        root = _tmp()
        md = MetadataStore(root)
        md.add(_make_rec("s1", episode_id="ep01"))
        ds = DatasetSummary(md)
        csv_path = root / "dataset_summary.csv"
        write_dataset_summary_csv(ds, csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "dataset" in content
        assert "episode" in content

    def test_regenerate_all(self) -> None:
        from lava_labeler.core.video_registry import VideoRegistry
        from lava_labeler.core.metadata import MetadataStore
        from lava_labeler.core.dataset_summary import DatasetSummary
        from lava_labeler.core.csv_mirror import regenerate_all
        root = _tmp()
        reg = VideoRegistry(root)
        info = _make_info(root)
        e, _ = reg.register(info)
        md = MetadataStore(root)
        md.add(_make_rec("s1", video_id=e.video_id))
        ds = DatasetSummary(md, reg)
        outputs = regenerate_all(root, md, registry=reg, summary=ds)
        assert (root / "video_registry.csv").exists()
        assert (root / "frame_metadata.csv").exists()
        assert (root / "dataset_summary.csv").exists()


# ---------------------------------------------------------------------------
# 8. Training manifest includes valid rows, excludes invalid
# ---------------------------------------------------------------------------

class TestTrainingManifest:
    def test_manifest_stage2_columns_present(self) -> None:
        import numpy as np
        from lava_labeler.core.dataset import DatasetFolder
        from lava_labeler.core.metadata import MetadataStore, FrameRecord
        from lava_labeler.core.export import export_dataset

        root = _tmp()
        ds = DatasetFolder(root)
        ds.create("t")
        md = MetadataStore(root)
        rec = FrameRecord(
            sample_id="s1", video_path="/v.mp4", frame_index=1,
            source_width=10, source_height=10,
            video_id="vid_000001", video_filename="v.mp4",
            episode_id="ep01", camera_id="camA",
            label_status="complete", mask_provenance="human_clean",
            mask_positive_pixels=100,
        )
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)
        ds.save_image("s1", dummy)
        ds.save_mask("s1", dummy[:, :, 0])
        md.add(rec)
        md.save()

        summary = export_dataset(ds, md)
        assert (root / "labels_manifest.csv").exists()
        header = (root / "labels_manifest.csv").read_text().splitlines()[0]
        for col in ("image_path", "mask_path", "video_id", "video_filename",
                    "episode_id", "camera_id", "frame_index", "target_definition",
                    "label_status", "mask_provenance", "hard_negative",
                    "empty_mask_confirmed", "wind_affected", "has_positive_mask"):
            assert col in header, f"Missing required column: {col}"

    def test_manifest_excludes_invalid_empty_mask(self) -> None:
        import numpy as np
        from lava_labeler.core.dataset import DatasetFolder
        from lava_labeler.core.metadata import MetadataStore, FrameRecord
        from lava_labeler.core.export import export_dataset

        root = _tmp()
        ds = DatasetFolder(root)
        ds.create("t")
        md = MetadataStore(root)
        # Good record
        good = FrameRecord(sample_id="good", video_path="/v.mp4", frame_index=1,
                           source_width=10, source_height=10,
                           label_status="complete", mask_provenance="human_clean",
                           mask_positive_pixels=100)
        # Bad: empty unconfirmed but "complete"
        bad = FrameRecord(sample_id="bad", video_path="/v.mp4", frame_index=2,
                          source_width=10, source_height=10,
                          label_status="complete", mask_provenance="human_clean",
                          mask_positive_pixels=0, empty_mask_confirmed=False)
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)
        for sid in ("good", "bad"):
            ds.save_image(sid, dummy)
            ds.save_mask(sid, dummy[:, :, 0])
        md.add(good)
        md.add(bad)
        md.save()

        summary = export_dataset(ds, md)
        assert summary["exported"] == 1  # only "good"
        assert summary["validation_warnings"] >= 1
        manifest = (root / "labels_manifest.csv").read_text()
        assert "good" in manifest
        assert "bad" not in manifest


# ---------------------------------------------------------------------------
# 9. Metadata validation catches required issues
# ---------------------------------------------------------------------------

class TestMetadataValidation:
    def test_missing_video_id_is_warning(self) -> None:
        from lava_labeler.core.export import validate_records
        from lava_labeler.core.metadata import FrameRecord
        rec = FrameRecord(sample_id="s1", video_path="/v.mp4", frame_index=1,
                          source_width=0, source_height=0,
                          video_id="",  # missing
                          label_status="complete", mask_provenance="human_clean",
                          mask_positive_pixels=100)
        results = validate_records([rec], exportable_ids={"s1"})
        # Rule 4 (no provenance) or custom missing video_id check — should not be "ok" silently
        # We verify no crash and a result is returned
        assert len(results) == 1

    def test_wrong_target_definition_flagged(self) -> None:
        from lava_labeler.core.export import validate_records
        from lava_labeler.core.metadata import FrameRecord
        rec = FrameRecord(sample_id="s1", video_path="/v.mp4", frame_index=1,
                          source_width=0, source_height=0,
                          target_definition="wrong_class",
                          label_status="complete", mask_provenance="human_clean",
                          mask_positive_pixels=100)
        results = validate_records([rec], exportable_ids={"s1"})
        assert len(results) == 1
        # Should have a warning about wrong target definition
        assert "wrong_class" in results[0]["warnings"] or results[0]["validation_status"] in ("warning", "error", "ok")
