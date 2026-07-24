from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np

from app.minre import (
    MinREOptions,
    MinREPipeline,
    _alignment_plan,
    _clean_intervals,
    _verify_pair,
    _write_pair,
)


class MinRETests(unittest.TestCase):
    def test_index_only_creates_resumable_dataset_index(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            video = source / "episode.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
            for index in range(4):
                writer.write(np.full((24, 32, 3), index * 30, dtype=np.uint8))
            writer.release()
            result = MinREPipeline(MinREOptions(
                source=source,
                output=output,
                vlm_config=root / "unused.json",
                index_only=True,
                quiet=True,
            )).run()
            self.assertEqual("indexed", result["status"])
            self.assertEqual(1, result["episode_count"])
            self.assertTrue((output / "minre-state.json").is_file())
            self.assertTrue((output / "dataset.json").is_file())

    def test_clean_intervals_remove_vlm_phases_and_alignment_gaps(self) -> None:
        intervals, summary = _clean_intervals(
            12,
            {"segments": [{"start_frame": 0, "end_frame": 11, "state": "valid"}]},
            {"segments": [
                {"start_frame": 0, "end_frame": 1, "phase_label": "idle"},
                {"start_frame": 2, "end_frame": 3, "phase_label": "reach"},
                {"start_frame": 4, "end_frame": 11, "phase_label": "grasp"},
            ]},
            np.asarray([True, True, True, True, True, True, False, True, True, True, True, True]),
            {"idle", "reach", "observe", "withdraw", "unknown"},
            2,
        )
        self.assertEqual([(4, 5), (7, 11)], intervals)
        self.assertEqual(4, summary["phase_removals"]["idle"] + summary["phase_removals"]["reach"])
        self.assertEqual(1, summary["alignment_invalid_frame_count"])

    def test_alignment_plan_uses_hdf5_frame_index(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "episode.h5"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("state", data=np.arange(20, dtype=np.float32).reshape(10, 2))
            manifest = {
                "root_path": str(root),
                "files": [{
                    "relative_path": "episode.h5",
                    "episode_id": "ep-1",
                    "extension": ".h5",
                }],
            }
            episode = {"id": "ep-1", "episode_key": "episode", "frame_count": 10, "fps": 10.0}
            valid, plans = _alignment_plan(
                manifest,
                episode,
                {"streams": [{
                    "relative_path": "episode.h5",
                    "data_count": 10,
                    "mode": "paired_frame_index",
                }]},
                10,
                10.0,
            )
            self.assertTrue(valid.all())
            self.assertEqual(np.arange(10).tolist(), plans["episode.h5"]["rows"].tolist())

    def test_write_pair_reopens_and_matches_video_and_hdf5_counts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "smooth.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
            for index in range(8):
                writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
            writer.release()
            source = root / "episode.h5"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("state", data=np.arange(16, dtype=np.float32).reshape(8, 2))
            manifest = {
                "id": "dataset-1",
                "root_path": str(root),
                "files": [{"relative_path": "episode.h5", "episode_id": "ep-1", "extension": ".h5"}],
            }
            episode = {"id": "ep-1", "name": "episode-1", "episode_key": "episode", "frame_count": 8, "fps": 10.0}
            behavior = {
                "task_label": "pick object",
                "segments": [{"start_frame": 0, "end_frame": 7, "phase_label": "grasp"}],
            }
            pair = _write_pair(
                manifest,
                episode,
                {"path": str(video), "relative_path": "smooth.mp4", "frame_count": 8, "fps": 10.0},
                behavior,
                {"episode.h5": {
                    "record": manifest["files"][0],
                    "path": source,
                    "stream": {"data_count": 8, "mode": "paired_frame_index"},
                    "rows": np.arange(8),
                }},
                root / "pick_object",
                2,
                5,
            )
            self.assertTrue(_verify_pair(pair))
            with h5py.File(pair["hdf5"], "r") as handle:
                source_group = next(iter(handle["sources"].values()))
                self.assertEqual([4, 6, 8, 10], source_group["state"][:, 0].tolist())
                self.assertEqual([2, 3, 4, 5], handle["minre/source_frame_index"][:].tolist())


if __name__ == "__main__":
    unittest.main()
