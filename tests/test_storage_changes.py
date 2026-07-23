from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app.storage as storage
from app.curation_pipeline import CURATION_PIPELINE_VERSION, source_signature
from app.qwen_trim import _source_video_fingerprint


class StorageChangeBackfillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary.name)
        self.source_root = temporary_root / "dataset"
        self.source_root.mkdir()
        self.source_file = self.source_root / "episode_1.mp4"
        self.source_file.write_bytes(b"source-media-placeholder")
        source_fingerprint = _source_video_fingerprint({"path": str(self.source_file)})
        self.sidecar = temporary_root / ".alicePD" / "fixture"
        artifact_root = self.sidecar / "qwen-action-trim"
        artifact_root.mkdir(parents=True)
        self.artifact = artifact_root / "episode_1.trim.alice"
        self.artifact.write_text(
            json.dumps(
                {
                    "schema": "alice/qwen-action-trim/v1",
                    "dataset_id": "fixture",
                    "episode_id": "episode_1",
                    "source_video": {"relative_path": "episode_1.mp4", "fingerprint": source_fingerprint},
                    "summary": {
                        "segment_count": 2,
                        "valid_frame_count": 3,
                        "invalid_frame_count": 1,
                    },
                    "segments": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.manifest = {
            "id": "fixture",
            "name": "fixture",
            "root_path": str(self.source_root),
            "sidecar_path": str(self.sidecar),
            "created_at": "2026-01-01T00:00:00+00:00",
            "episodes": [
                {
                    "id": "episode_1",
                    "name": "Episode 1",
                    "relative_path": "episode_1.mp4",
                    "frame_count": 4,
                }
            ],
            "files": [
                {
                    "id": "video-1",
                    "relative_path": "episode_1.mp4",
                    "episode_id": "episode_1",
                }
            ],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_qwen_artifact_is_backfilled_once_without_mutating_source(self) -> None:
        source_before = (self.source_file.read_bytes(), self.source_file.stat().st_mtime_ns)

        with patch.object(storage, "get_manifest", return_value=self.manifest):
            first = storage.list_changes("fixture")
            second = storage.list_changes("fixture")

        self.assertEqual(1, first["pending_count"])
        self.assertEqual(1, len(first["items"]))
        item = first["items"][0]
        self.assertEqual("qwen_action_trim", item["kind"])
        self.assertEqual("episode_1", item["episode_id"])
        self.assertEqual("pending", item["status"])
        self.assertEqual(1, item["revision"])
        self.assertEqual("episode_1.mp4", item["source_paths"][0])
        self.assertEqual(item["id"], second["items"][0]["id"])
        self.assertEqual(item["revision"], second["items"][0]["revision"])
        self.assertEqual(1, len(list((self.sidecar / "changes" / "records").glob("*.alice"))))
        self.assertEqual(source_before, (self.source_file.read_bytes(), self.source_file.stat().st_mtime_ns))

    def test_apply_qwen_change_keeps_unchanged_source_read_only(self) -> None:
        source_before = (self.source_file.read_bytes(), self.source_file.stat().st_mtime_ns)
        with patch.object(storage, "get_manifest", return_value=self.manifest):
            catalog = storage.list_changes("fixture")
            result = storage.apply_changes("fixture", [catalog["items"][0]["id"]])
        self.assertEqual(1, result["application"]["change_count"])
        self.assertFalse(result["application"]["source_mutated"])
        self.assertEqual(source_before, (self.source_file.read_bytes(), self.source_file.stat().st_mtime_ns))

    def test_apply_qwen_change_rejects_replaced_source(self) -> None:
        with patch.object(storage, "get_manifest", return_value=self.manifest):
            catalog = storage.list_changes("fixture")
            self.source_file.write_bytes(b"externally-replaced-source-media")
            with self.assertRaisesRegex(ValueError, "Source video changed"):
                storage.apply_changes("fixture", [catalog["items"][0]["id"]])

    def test_apply_paper_curation_activates_invalid_frame_index(self) -> None:
        curation_root = self.sidecar / "curation"
        curation_root.mkdir(parents=True, exist_ok=True)
        signature = source_signature(self.source_root, self.source_file.name)
        artifact = curation_root / "episode_1.curation.alice"
        artifact.write_text(
            json.dumps({
                "schema": "alice/paper-curation/v1",
                "pipeline_version": CURATION_PIPELINE_VERSION,
                "dataset_id": "fixture",
                "episode_id": "episode_1",
                "source_signatures": [signature],
                "source_video": {"relative_path": self.source_file.name, "frame_count": 8, "fps": 8.0},
                "segments": [{"start_frame": 2, "end_frame": 4, "state": "invalid"}],
                "samples": [],
                "summary": {"invalid_frame_count": 3},
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest = {**self.manifest, "episodes": [{**self.manifest["episodes"][0], "frame_count": 8}], "files": [{**self.manifest["files"][0], "relative_path": self.source_file.name}]}
        with patch.object(storage, "get_manifest", return_value=manifest):
            storage.record_change("fixture", "paper_curation", "episode_1", "Paper curation", [artifact], {"invalid_frame_count": 3}, [self.source_file.name])
            catalog = storage.list_changes("fixture")
            paper = next(item for item in catalog["items"] if item["kind"] == "paper_curation")
            self.assertEqual(CURATION_PIPELINE_VERSION, paper["pipeline_version"])
            self.assertFalse(paper["requires_rerun"])
            self.assertEqual(2, catalog["runnable_pending_count"])
            self.assertEqual(0, catalog["requires_rerun_count"])
            result = storage.apply_changes("fixture", [paper["id"]])
            index = storage.load_invalid_frame_index("fixture", "episode_1")
            invalid, _ = storage.is_frame_invalid("fixture", "episode_1", 3)

        self.assertEqual(1, result["application"]["change_count"])
        self.assertIsNotNone(index)
        self.assertTrue(invalid)
        snapshot_paths = [item["relative_path"] for item in result["application"]["changes"][0]["snapshot_artifacts"]]
        self.assertTrue(any(path.endswith(".invalid.bin") for path in snapshot_paths))

    def test_apply_legacy_paper_curation_requires_rerun(self) -> None:
        curation_root = self.sidecar / "curation"
        curation_root.mkdir(parents=True, exist_ok=True)
        artifact = curation_root / "legacy.curation.alice"
        artifact.write_text(json.dumps({
            "schema": "alice/paper-curation/v1",
            "dataset_id": "fixture",
            "episode_id": "episode_1",
            "source_signatures": [source_signature(self.source_root, self.source_file.name)],
        }), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "algorithm has changed"):
            storage._verify_paper_curation_source(self.manifest, artifact, "episode_1")

    def test_legacy_paper_change_is_marked_and_not_runnable(self) -> None:
        curation_root = self.sidecar / "curation"
        curation_root.mkdir(parents=True, exist_ok=True)
        artifact = curation_root / "legacy.curation.alice"
        artifact.write_text(json.dumps({
            "schema": "alice/paper-curation/v1",
            "dataset_id": "fixture",
            "episode_id": "episode_1",
            "source_signatures": [source_signature(self.source_root, self.source_file.name)],
            "summary": {"invalid_frame_count": 1},
        }), encoding="utf-8")

        with patch.object(storage, "get_manifest", return_value=self.manifest):
            catalog = storage.list_changes("fixture")

        paper = next(item for item in catalog["items"] if item["kind"] == "paper_curation")
        self.assertEqual(1, paper["pipeline_version"])
        self.assertTrue(paper["requires_rerun"])
        self.assertEqual(1, catalog["runnable_pending_count"])
        self.assertEqual(1, catalog["requires_rerun_count"])

    def test_future_paper_curation_version_requires_rerun(self) -> None:
        curation_root = self.sidecar / "curation"
        curation_root.mkdir(parents=True, exist_ok=True)
        artifact = curation_root / "future.curation.alice"
        artifact.write_text(json.dumps({
            "schema": "alice/paper-curation/v1",
            "pipeline_version": CURATION_PIPELINE_VERSION + 1,
            "dataset_id": "fixture",
            "episode_id": "episode_1",
            "source_signatures": [source_signature(self.source_root, self.source_file.name)],
        }), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "algorithm has changed"):
            storage._verify_paper_curation_source(self.manifest, artifact, "episode_1")

    def test_storage_slug_preserves_long_id_uniqueness(self) -> None:
        first = "a" * 70 + "x"
        second = "a" * 70 + "y"

        self.assertNotEqual(storage.storage_slug(first), storage.storage_slug(second))
        self.assertLessEqual(len(storage.storage_slug(first)), 64)
        self.assertNotEqual(storage.storage_slug("episode a"), storage.storage_slug("episode-a"))

    def test_get_manifest_reads_matching_legacy_long_id_registry(self) -> None:
        dataset_id = "legacy-" + "x" * 70
        registry_root = Path(self.temporary.name) / "registry"
        registry_root.mkdir()
        sidecar = Path(self.temporary.name) / "legacy-sidecar"
        sidecar.mkdir()
        manifest = {
            "id": dataset_id,
            "name": "legacy",
            "root_path": str(self.source_root),
            "sidecar_path": str(sidecar),
            "created_at": "2026-01-01T00:00:00+00:00",
            "episodes": [],
            "files": [],
        }
        manifest_path = sidecar / "dataset.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        legacy_registry = registry_root / f"{storage.slugify(dataset_id)}.json"
        legacy_registry.write_text(json.dumps({
            "id": dataset_id,
            "manifest_path": str(manifest_path),
        }), encoding="utf-8")

        with patch.object(storage, "MANIFESTS", registry_root):
            loaded = storage.get_manifest(dataset_id)

        self.assertEqual(dataset_id, loaded["id"])
        self.assertEqual(sidecar, Path(loaded["sidecar_path"]))


if __name__ == "__main__":
    unittest.main()
