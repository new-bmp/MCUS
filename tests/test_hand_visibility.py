from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from app.egodex_mano import required_egodex_mano_names
from app.full_export import MANO_44_JOINT_NAMES
from app.hand_visibility import _classify_mano_visibility, inspect_full_hand_visibility


class HandVisibilityTests(unittest.TestCase):
    def test_mano_visibility_uses_three_level_thresholds(self) -> None:
        passed, review, invalid = _classify_mano_visibility({
            "right": np.asarray([21, 20, 9, 8, 0], dtype=np.int64),
        })

        self.assertEqual([True, False, False, False, False], passed.tolist())
        self.assertEqual([False, True, True, False, False], review.tolist())
        self.assertEqual([False, False, False, True, True], invalid.tolist())

    def test_missing_required_hand_is_not_diluted_by_visible_other_hand(self) -> None:
        passed, review, invalid = _classify_mano_visibility({
            "left": np.asarray([21], dtype=np.int64),
            "right": np.asarray([0], dtype=np.int64),
        })

        self.assertFalse(passed[0])
        self.assertFalse(review[0])
        self.assertTrue(invalid[0])

    @staticmethod
    def _fixture(root: Path) -> tuple[dict, dict, dict]:
        path = root / "episode.hdf5"
        frame_count = 4
        with h5py.File(path, "w") as output:
            output.create_dataset(
                "camera/intrinsic",
                data=np.array([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            )
            camera = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
            output.create_dataset("transforms/camera", data=camera)
            transform_names = tuple(dict.fromkeys((
                *MANO_44_JOINT_NAMES,
                *required_egodex_mano_names("left"),
                *required_egodex_mano_names("right"),
            )))
            for index, name in enumerate(transform_names):
                transforms = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
                transforms[:, 0, 3] = ((index % 5) - 2) * 0.04
                transforms[:, 1, 3] = ((index % 4) - 1.5) * 0.04
                transforms[:, 2, 3] = 1.0
                if name.startswith("right"):
                    transforms[2, 0, 3] += 2.0
                output.create_dataset(f"transforms/{name}", data=transforms)
        episode = {"id": "ep", "name": "episode_0", "frame_count": frame_count, "fps": 30.0, "width": 100, "height": 100}
        manifest = {
            "id": "dataset",
            "root_path": str(root),
            "files": [{"id": "h5", "relative_path": path.name, "extension": ".hdf5", "episode_id": "ep"}],
            "episode_resolution": {"file_episode_assignments": {"h5": "ep"}},
        }
        media = {"frame_count": frame_count, "fps": 30.0, "width": 100, "height": 100}
        return manifest, episode, media

    def test_checks_only_the_hand_required_by_the_action_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, episode, media = self._fixture(Path(temporary))
            right = inspect_full_hand_visibility(manifest, episode, media, ["right"])
            left = inspect_full_hand_visibility(manifest, episode, media, ["left"])
            both = inspect_full_hand_visibility(manifest, episode, media, ["left", "right"])

        self.assertTrue(right["available"])
        self.assertEqual([False, False, True, False], right["invalid_mask"].tolist())
        self.assertFalse(left["invalid_mask"].any())
        self.assertEqual(right["invalid_mask"].tolist(), both["invalid_mask"].tolist())
        self.assertEqual(21, right["metrics"]["sides"]["right"]["joint_count"])
        self.assertEqual("egodex_full_skeleton_fk", right["metrics"]["hand_geometry_source"])

    def test_missing_intrinsics_skips_without_rejecting_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, episode, media = self._fixture(Path(temporary))
            source = Path(manifest["root_path"]) / "episode.hdf5"
            with h5py.File(source, "r+") as output:
                del output["camera/intrinsic"]
            result = inspect_full_hand_visibility(manifest, episode, media, ["right"])

        self.assertFalse(result["available"])
        self.assertFalse(result["invalid_mask"].any())


if __name__ == "__main__":
    unittest.main()
