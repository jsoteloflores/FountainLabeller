"""Tests for automatic video linking + 2.5D temporal export.

All tests are non-GUI (no Tkinter). They use tiny synthetic videos written
with OpenCV.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from lava_labeler.core.dataset import DatasetFolder
from lava_labeler.core.metadata import FrameRecord, MetadataStore
from lava_labeler.core.temporal_export import (
    TemporalExportConfig,
    export_temporal_dataset,
    offset_label,
    temporal_offsets,
)
from lava_labeler.core.video_registry import VideoRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_video(path: Path, n_frames: int = 30, w: int = 64, h: int = 48,
                fps: float = 25.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_dataset(root: Path) -> DatasetFolder:
    ds = DatasetFolder(root)
    ds.create(name="test")
    return ds


def _add_labeled_sample(ds: DatasetFolder, meta: MetadataStore, rec: FrameRecord,
                        w: int, h: int, positive: bool = True) -> None:
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    if positive:
        mask[2:6, 2:6] = 255
    ds.save_image(rec.sample_id, img)
    ds.save_mask(rec.sample_id, mask)
    meta.add(rec)


# ---------------------------------------------------------------------------
# 1 + 2. fps / video_id roundtrip
# ---------------------------------------------------------------------------

class TestMetadataRoundtrip:
    def test_fps_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata").mkdir(parents=True)
            store = MetadataStore(root)
            store.add(FrameRecord(
                sample_id="s1", video_path="/v.mp4", frame_index=0,
                source_width=64, source_height=48, fps=29.97,
            ))
            store.save()
            reloaded = MetadataStore(root)
            rec = reloaded.get("s1")
            assert rec is not None
            assert abs(rec.fps - 29.97) < 1e-3

    def test_video_id_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata").mkdir(parents=True)
            store = MetadataStore(root)
            store.add(FrameRecord(
                sample_id="s1", video_path="/v.mp4", frame_index=0,
                source_width=64, source_height=48, video_id="vid_000001",
            ))
            store.save()
            reloaded = MetadataStore(root)
            rec = reloaded.get("s1")
            assert rec is not None
            assert rec.video_id == "vid_000001"

    def test_fps_written_to_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata").mkdir(parents=True)
            store = MetadataStore(root)
            store.add(FrameRecord(
                sample_id="s1", video_path="/v.mp4", frame_index=0,
                source_width=64, source_height=48, fps=30.0,
            ))
            store.save()
            text = (root / "metadata" / "frames.csv").read_text()
            assert "fps" in text.splitlines()[0].split(",")


# ---------------------------------------------------------------------------
# 3. Video linkage contract (mirrors add_current_frame)
# ---------------------------------------------------------------------------

class TestVideoLinkage:
    def test_registered_record_has_full_linkage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = _make_dataset(root)
            vpath = root / "videos" / "clip.mp4"
            _make_video(vpath)

            from lava_labeler.core.video_io import VideoReader
            reader = VideoReader(vpath)
            info = reader.info
            reader.close()

            registry = VideoRegistry(root)
            entry, _tier = registry.register(info, episode_id="E2", camera_id="Alpha")

            # Mirror add_current_frame's FrameRecord construction.
            rec = FrameRecord(
                sample_id="E2_Alpha_frame00000005",
                video_path=str(info.path),
                video_id=entry.video_id,
                video_filename=entry.video_filename,
                frame_index=5,
                source_width=info.width,
                source_height=info.height,
                fps=info.fps,
                episode_id="E2", camera_id="Alpha",
            )
            assert rec.video_id != ""
            assert rec.video_filename.endswith(".mp4")
            assert rec.fps > 0
            assert rec.video_path != ""


# ---------------------------------------------------------------------------
# 4. Old workspace relink
# ---------------------------------------------------------------------------

class TestRelink:
    def test_relink_fills_missing_video_ids(self) -> None:
        from lava_labeler.core.video_relink import relink_workspace_videos

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            ds = _make_dataset(root)
            video_root = Path(tmp) / "videos"
            vpath = video_root / "E2" / "clip.mp4"
            _make_video(vpath, n_frames=30)

            store = MetadataStore(root)
            for idx in (5, 10):
                store.add(FrameRecord(
                    sample_id=f"s{idx}", video_path="/old/gone/clip.mp4",
                    frame_index=idx, source_width=64, source_height=48,
                    video_filename="clip.mp4", episode_id="E2", camera_id="Alpha",
                    video_id="", fps=0.0,
                ))
            store.save()

            result = relink_workspace_videos(
                dataset_root=root, source_video_root=video_root, dry_run=False)

            reloaded = MetadataStore(root)
            for rec in reloaded.all_records():
                assert rec.video_id != ""
                assert rec.fps > 0
            assert (root / "metadata" / "video_relink_report.csv").exists()
            backups = list((root / "metadata" / "backups").glob(
                "frames_before_video_relink_*.csv"))
            assert backups
            assert result["rows_relinkable"] == 2

    def test_relink_dry_run_does_not_write(self) -> None:
        from lava_labeler.core.video_relink import relink_workspace_videos

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            _make_dataset(root)
            video_root = Path(tmp) / "videos"
            _make_video(video_root / "clip.mp4", n_frames=20)

            store = MetadataStore(root)
            store.add(FrameRecord(
                sample_id="s1", video_path="/old/clip.mp4", frame_index=3,
                source_width=64, source_height=48, video_filename="clip.mp4",
                video_id="", fps=0.0,
            ))
            store.save()
            before = (root / "metadata" / "frames.csv").read_text()

            relink_workspace_videos(
                dataset_root=root, source_video_root=video_root, dry_run=True)

            after = (root / "metadata" / "frames.csv").read_text()
            assert before == after


# ---------------------------------------------------------------------------
# 5. Temporal offsets
# ---------------------------------------------------------------------------

class TestTemporalOffsets:
    def test_centered(self) -> None:
        assert temporal_offsets(2, 1, "centered") == [-2, -1, 0, 1, 2]

    def test_causal(self) -> None:
        assert temporal_offsets(4, 1, "causal") == [-4, -3, -2, -1, 0]

    def test_stride(self) -> None:
        assert temporal_offsets(3, 2, "centered") == [-6, -4, -2, 0, 2, 4, 6]

    def test_offset_labels(self) -> None:
        assert offset_label(-2) == "offset_m002"
        assert offset_label(0) == "offset_0000"
        assert offset_label(2) == "offset_p002"


# ---------------------------------------------------------------------------
# 6. ROI temporal export dimensions
# ---------------------------------------------------------------------------

class TestTemporalExport:
    def _build_roi_workspace(self, tmp: Path):
        root = tmp / "workspace"
        ds = _make_dataset(root)
        vpath = root / "videos" / "clip.mp4"
        _make_video(vpath, n_frames=30, w=64, h=48)

        registry = VideoRegistry(root)
        from lava_labeler.core.video_io import VideoReader
        reader = VideoReader(vpath)
        info = reader.info
        reader.close()
        entry, _ = registry.register(info)
        registry.save()

        store = MetadataStore(root)
        # ROI crop 20x16 at (5, 4). Center frame 15.
        roi_w, roi_h = 20, 16
        rec = FrameRecord(
            sample_id="s_roi", video_path=str(vpath), frame_index=15,
            source_width=64, source_height=48, fps=info.fps,
            video_id=entry.video_id, video_filename=entry.video_filename,
            label_status="complete", is_roi_crop=True,
            roi_x=5, roi_y=4, roi_width=roi_w, roi_height=roi_h,
            roi_mode="fixed_roi_crop", roi_size_policy="global_fixed",
        )
        _add_labeled_sample(ds, store, rec, roi_w, roi_h)
        store.save()
        return ds, store, registry, (roi_w, roi_h)

    def test_roi_temporal_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ds, store, registry, (roi_w, roi_h) = self._build_roi_workspace(tmp)
            out = tmp / "export"
            cfg = TemporalExportConfig(
                output_root=out, temporal_radius=2, temporal_stride=1,
                window_mode="centered", edge_policy="skip",
            )
            summary = export_temporal_dataset(ds, store, registry, cfg)
            assert summary["samples_exported"] == 1

            sample_dir = out / "temporal_frames" / "s_roi"
            pngs = sorted(sample_dir.glob("offset_*.png"))
            assert len(pngs) == 5
            for p in pngs:
                arr = cv2.imread(str(p))
                assert arr.shape[:2] == (roi_h, roi_w)

            center = cv2.imread(str(out / "images" / "all" / "s_roi.png"))
            mask = cv2.imread(str(out / "masks" / "all" / "s_roi_mask.png"),
                              cv2.IMREAD_GRAYSCALE)
            assert center.shape[:2] == (roi_h, roi_w)
            assert mask.shape[:2] == (roi_h, roi_w)
            assert (out / "temporal_manifest.csv").exists()

            import csv
            with open(out / "temporal_manifest.csv") as f:
                rows = list(csv.DictReader(f))
            assert rows[0]["temporal_frame_paths_json"]

    def test_edge_policy_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = tmp / "workspace"
            ds = _make_dataset(root)
            vpath = root / "videos" / "clip.mp4"
            _make_video(vpath, n_frames=30, w=64, h=48)
            registry = VideoRegistry(root)
            from lava_labeler.core.video_io import VideoReader
            reader = VideoReader(vpath); info = reader.info; reader.close()
            entry, _ = registry.register(info)
            store = MetadataStore(root)
            rec = FrameRecord(
                sample_id="s_edge", video_path=str(vpath), frame_index=0,
                source_width=64, source_height=48, fps=info.fps,
                video_id=entry.video_id, video_filename=entry.video_filename,
                label_status="complete",
            )
            _add_labeled_sample(ds, store, rec, 64, 48)
            store.save()

            out = tmp / "export"
            cfg = TemporalExportConfig(
                output_root=out, temporal_radius=2, edge_policy="skip")
            summary = export_temporal_dataset(ds, store, registry, cfg)
            assert summary["samples_exported"] == 0
            assert summary["samples_skipped"] == 1

    def test_edge_policy_replicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = tmp / "workspace"
            ds = _make_dataset(root)
            vpath = root / "videos" / "clip.mp4"
            _make_video(vpath, n_frames=30, w=64, h=48)
            registry = VideoRegistry(root)
            from lava_labeler.core.video_io import VideoReader
            reader = VideoReader(vpath); info = reader.info; reader.close()
            entry, _ = registry.register(info)
            store = MetadataStore(root)
            rec = FrameRecord(
                sample_id="s_edge", video_path=str(vpath), frame_index=0,
                source_width=64, source_height=48, fps=info.fps,
                video_id=entry.video_id, video_filename=entry.video_filename,
                label_status="complete",
            )
            _add_labeled_sample(ds, store, rec, 64, 48)
            store.save()

            out = tmp / "export"
            cfg = TemporalExportConfig(
                output_root=out, temporal_radius=2, edge_policy="replicate")
            summary = export_temporal_dataset(ds, store, registry, cfg)
            assert summary["samples_exported"] == 1

            import csv
            with open(out / "temporal_manifest.csv") as f:
                row = next(csv.DictReader(f))
            import json
            valid = json.loads(row["temporal_valid_json"])
            # offsets -2,-1 are invalid (replicated) at frame 0.
            assert valid == [False, False, True, True, True]
            assert row["all_temporal_frames_valid"] in ("False", "false")

    def test_export_does_not_mutate_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ds, store, registry, _ = self._build_roi_workspace(tmp)
            frames_csv = ds.root / "metadata" / "frames.csv"
            before = hashlib.md5(frames_csv.read_bytes()).hexdigest()

            out = tmp / "export"
            cfg = TemporalExportConfig(output_root=out, temporal_radius=2)
            export_temporal_dataset(ds, store, registry, cfg)

            after = hashlib.md5(frames_csv.read_bytes()).hexdigest()
            assert before == after

    def test_overwrite_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ds, store, registry, _ = self._build_roi_workspace(tmp)
            out = tmp / "export"
            out.mkdir()
            (out / "existing.txt").write_text("keep me")
            cfg = TemporalExportConfig(output_root=out, overwrite_existing=False)
            with pytest.raises(FileExistsError):
                export_temporal_dataset(ds, store, registry, cfg)
