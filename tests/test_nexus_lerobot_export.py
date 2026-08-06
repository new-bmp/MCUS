from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import pyarrow.parquet as parquet

from app.nexus_lerobot_export import (
    NEXUS_SKELETON_NODE_NAMES,
    convert_nexus_to_lerobot,
)


class NexusLeRobotExportTests(unittest.TestCase):
    @staticmethod
    def _write_video(path: Path, frame_count: int, width: int = 16, height: int = 12) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError("test video writer failed")
        try:
            for index in range(frame_count):
                frame = np.full((height, width, 3), index * 10, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    @classmethod
    def _write_episode(cls, root: Path, name: str = "ep_0001_20260101_120000", count: int = 6) -> Path:
        episode = root / name
        (episode / "meta").mkdir(parents=True)
        (episode / "mocap").mkdir()
        (episode / "tactile").mkdir()
        (episode / "sensor").mkdir()
        cls._write_video(episode / "camera" / "head_rgb.mp4", count)
        cls._write_video(episode / "camera" / "wrist_left.mp4", count)
        cls._write_video(episode / "camera" / "wrist_right.mp4", count)
        (episode / "meta" / "metadata.json").write_text(json.dumps({
            "schema_version": "4.0",
            "files": {
                "camera": {"head_rgb": "camera/head_rgb.mp4"},
                "mocap": {
                    "dexweaveg1_left": "mocap/dexweaveg1_left.h5",
                    "dexweaveg1_right": "mocap/dexweaveg1_right.h5",
                },
            },
        }), encoding="utf-8")
        table = {
            "frame_idx": np.arange(count, dtype=np.int64),
            "master_ts": 1000.0 + np.arange(count, dtype=np.float64) / 30.0,
            "partial": np.asarray([False, False, True, False, False, False][:count]),
            "partial_reason": ["", "", "test_partial", "", "", ""][:count],
            "head_frame_idx": np.arange(count, dtype=np.int64),
            "wrist_left_frame_idx": np.arange(count, dtype=np.int64),
            "wrist_right_frame_idx": np.arange(count, dtype=np.int64),
            "depth_frame_idx": np.arange(count, dtype=np.int64),
            "tactile_left_source_seq": 100 + np.arange(count, dtype=np.int64),
            "tactile_right_source_seq": 200 + np.arange(count, dtype=np.int64),
            "mocap_dexweaveg1_source_seq": 300 + np.arange(count, dtype=np.int64),
        }
        import pyarrow as pa

        parquet.write_table(pa.table(table), episode / "meta" / "sync.parquet")
        for side, offset in (("left", 0.0), ("right", 100.0)):
            skeleton = np.zeros((count, 20, 7), dtype=np.float32)
            skeleton[..., 0] = np.arange(count, dtype=np.float32)[:, None] + offset
            skeleton[..., 6] = 1.0
            with h5py.File(episode / "mocap" / f"dexweaveg1_{side}.h5", "w") as output:
                output.create_dataset("skeleton", data=skeleton)
                output.create_dataset("joints", data=np.tile(np.arange(6, dtype=np.uint8), (count, 1)))
                output.create_dataset("wrist_quat", data=np.tile(np.asarray([0, 0, 0, 1], dtype=np.float32), (count, 1)))
                output.create_dataset("partial", data=np.zeros(count, dtype=bool))
            tactile = np.zeros((count, 225), dtype=np.uint16)
            tactile[:, 0] = np.arange(1, count + 1, dtype=np.uint16)
            with h5py.File(episode / "tactile" / f"{side}.h5", "w") as output:
                output.create_dataset("adc", data=tactile)
                output.create_dataset("partial", data=np.zeros(count, dtype=bool))
        imu_count = count * 4
        with h5py.File(episode / "sensor" / "head_imu.h5", "w") as output:
            group = output.require_group("imu")
            group.create_dataset("accel", data=np.tile(np.asarray([1, 2, 3], dtype=np.float32), (imu_count, 1)))
            group.create_dataset("gyro", data=np.tile(np.asarray([4, 5, 6], dtype=np.float32), (imu_count, 1)))
            imu_times = 1000.0 + (np.arange(imu_count, dtype=np.float64) + 0.5) / (count * 4 / (count / 30.0))
            group.create_dataset("host_arrival_ts_ns", data=np.rint(imu_times * 1_000_000_000).astype(np.int64))
        return episode

    def test_preserves_nexus_shapes_and_aligns_video_to_master_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "nexus"
            output = root / "lerobot"
            self._write_episode(source)

            with patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}):
                result = convert_nexus_to_lerobot(source, output, cameras=("head",))

            self.assertEqual(1, result["output_episode_count"])
            data_path = output / "data" / "chunk-000" / "episode_000000.parquet"
            data = parquet.read_table(data_path)
            self.assertEqual(6, data.num_rows)
            self.assertEqual(20 * 7, data.schema.field("observation.left_hand.skeleton").type.list_size)
            self.assertEqual(225, data.schema.field("observation.left_hand.tactile").type.list_size)
            self.assertNotIn("action", data.column_names)
            self.assertEqual([0, 1, 2, 3, 4, 5], data["source.master_frame_index"].to_pylist())
            self.assertEqual([False, False, True, False, False, False], data["quality.partial"].to_pylist())
            tactile_features = np.asarray(
                data["observation.left_hand.tactile_features"].combine_chunks().values
            ).reshape(6, 7)
            self.assertEqual([1.0] * 6, tactile_features[:, 0].tolist())
            self.assertEqual([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], tactile_features[:, 2].tolist())
            video = output / "videos" / "chunk-000" / "observation.images.head" / "episode_000000.mp4"
            capture = cv2.VideoCapture(str(video))
            try:
                self.assertEqual(6, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                self.assertEqual(30.0, capture.get(cv2.CAP_PROP_FPS))
            finally:
                capture.release()
            info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual("nexus_dexweaveg1_bimanual_multimodal", info["robot_type"])
            self.assertEqual(list(NEXUS_SKELETON_NODE_NAMES), info["features"]["observation.left_hand.skeleton"]["names"])
            self.assertIn("observation.images.head", info["features"])
            self.assertNotIn("action", info["features"])
            tasks = parquet.read_table(output / "meta" / "tasks.parquet")
            self.assertEqual(["ep_0001_20260101_120000"], tasks["task"].to_pylist())

    def test_valid_curation_ranges_split_episodes_and_vlm_sentences_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "nexus"
            output = root / "lerobot"
            episode = self._write_episode(source)
            sidecar = root / ".alicePD" / "dataset-id"
            curation = sidecar / "curation" / "opaque-id.curation.alice"
            curation.parent.mkdir(parents=True)
            curation.write_text(json.dumps({
                "episode_id": episode.name,
                "episode_name": episode.name,
                "segments": [
                    {"start_frame": 0, "end_frame": 1, "state": "valid"},
                    {"start_frame": 2, "end_frame": 3, "state": "invalid"},
                    {"start_frame": 4, "end_frame": 5, "state": "valid"},
                ],
            }), encoding="utf-8")
            annotations = sidecar / "behavior-annotations" / "opaque-id.behavior.alice"
            annotations.parent.mkdir(parents=True)
            annotations.write_text(json.dumps({
                "source_video": {"relative_path": f"{episode.name}/camera/head_rgb.mp4"},
                "coarse": {"summary": "sort cups"},
                "fine": [
                    {"start_frame": 0, "end_frame": 1, "skill": "Grasp", "description": "The right hand grasps a cup."},
                    {"start_frame": 4, "end_frame": 5, "skill": "Place", "description": "The right hand places the cup on the shelf."},
                ],
            }), encoding="utf-8")

            with patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}):
                result = convert_nexus_to_lerobot(
                    source,
                    output,
                    cameras=("head",),
                    alice_sidecar=sidecar,
                )

            self.assertEqual(2, result["output_episode_count"])
            self.assertEqual(4, result["frame_count"])
            first = parquet.read_table(output / "data" / "chunk-000" / "episode_000000.parquet")
            second = parquet.read_table(output / "data" / "chunk-000" / "episode_000001.parquet")
            self.assertEqual(["Grasp", "Grasp"], first["annotation.phase_label"].to_pylist())
            self.assertEqual(["Place", "Place"], second["annotation.phase_label"].to_pylist())
            self.assertEqual(
                ["The right hand places the cup on the shelf."] * 2,
                second["annotation.action_description"].to_pylist(),
            )
            tasks = parquet.read_table(output / "meta" / "tasks.parquet")
            self.assertEqual(["sort cups"], tasks["task"].to_pylist())


if __name__ == "__main__":
    unittest.main()
