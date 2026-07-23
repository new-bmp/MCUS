from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.preview_proxy import _collect_source_pts, _normalise_pts, _source_codec_info


class PreviewProxyHelpersTest(unittest.TestCase):
    def _video(self, root: Path, name: str, fourcc: str = "mp4v") -> Path:
        path = root / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 20.0, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(12):
            writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
        writer.release()
        return path

    def test_mpeg4_part_2_mp4_requires_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._video(Path(directory), "source.mp4")
            info = _source_codec_info(source, {})
            self.assertFalse(info["browser_compatible"])
            self.assertEqual("video/mp4", info["mime_type"])
            self.assertIn(info["codec"], {"fmp4", "mp4v"})

    def test_source_pts_are_monotonic_and_frame_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._video(Path(directory), "source.mp4")
            points = _collect_source_pts(source, 20.0, 12)
            self.assertEqual(12, len(points))
            self.assertEqual(0.0, points[0])
            self.assertTrue(all(left <= right for left, right in zip(points, points[1:])))
            self.assertAlmostEqual(0.55, points[-1], places=2)

    def test_timestamp_regression_is_clamped(self) -> None:
        self.assertEqual([0.0, 0.1, 0.1, 0.4], _normalise_pts([3.0, 3.1, 3.05, 3.4], 10.0))


if __name__ == "__main__":
    unittest.main()
