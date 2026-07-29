from __future__ import annotations

import io
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from app.cli import _balanced_full_shards, _select_full_episodes, build_parser, command_full, command_robots
from app import main as main_module
from app.episode_resolver import build_sampled_episode_framework, episode_key
from app.models import ModelRegistry
from app.schema_profiler import MAX_FILES, MAX_FILES_PER_FOLDER, MAX_FOLDERS, build_inventory, sample_profile_paths
from app.schemas import LocalModelConfig
from app.storage import _iter_dataset_files, discover_dataset_roots


class RuntimeCliTests(unittest.TestCase):
    def test_dataset_root_discovery_is_first_level_and_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "dataset_a"
            second = root / "dataset_b"
            first.mkdir()
            second.mkdir()
            (root / ".alicePD").mkdir()
            (first / "deep" / "episode").mkdir(parents=True)
            discovery = discover_dataset_roots(root)

        self.assertEqual("collection", discovery["mode"])
        self.assertEqual(2, discovery["dataset_count"])
        self.assertEqual(["dataset_a", "dataset_b"], [item["name"] for item in discovery["datasets"]])
        self.assertTrue(all(item["status"] == "unloaded" for item in discovery["datasets"]))

    def test_dataset_root_without_child_folder_stays_single(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "episode.mp4").write_bytes(b"fixture")
            discovery = discover_dataset_roots(root)

        self.assertEqual("single", discovery["mode"])
        self.assertEqual(1, discovery["dataset_count"])
        self.assertEqual(str(root.resolve()), discovery["datasets"][0]["path"])

    def test_alice_full_output_is_one_dataset_instead_of_modality_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("body", "data", "meta", "videos"):
                (root / name).mkdir()
            (root / "dataset.json").write_text(
                '{"schema":"alice/full-dataset/v2"}',
                encoding="utf-8",
            )
            discovery = discover_dataset_roots(root)

        self.assertEqual("single", discovery["mode"])
        self.assertEqual([str(root.resolve())], [item["path"] for item in discovery["datasets"]])

    def test_standard_lerobot_root_is_one_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            (root / "videos").mkdir()
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
            discovery = discover_dataset_roots(root)

        self.assertEqual("single", discovery["mode"])

    def test_alice_lerobot_body_file_uses_same_episode_as_data_and_video(self) -> None:
        root = Path("dataset")
        paths = (
            root / "data" / "chunk-000" / "episode_000007.parquet",
            root / "body" / "chunk-000" / "episode_000007.parquet",
            root / "videos" / "chunk-000" / "observation.images.main" / "episode_000007.mp4",
        )

        self.assertEqual(["episode_7"] * 3, [episode_key(path, root) for path in paths])

    def test_full_output_folder_is_ignored_as_source_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "episode.mp4").write_bytes(b"fixture")
            output = root / "output" / "make_sandwich" / "ep1"
            output.mkdir(parents=True)
            (output / "video.mp4").write_bytes(b"generated")
            (output / "data.hdf5").write_bytes(b"generated")

            discovery = discover_dataset_roots(root)
            relative_paths = [path.relative_to(root).as_posix() for path, _ in _iter_dataset_files(root)]

        self.assertEqual("single", discovery["mode"])
        self.assertEqual(["episode.mp4"], relative_paths)

    def test_folder_dialog_collection_does_not_scan_child_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dataset_a").mkdir()
            (root / "dataset_b").mkdir()
            with (
                patch.object(main_module, "choose_folder", return_value=str(root)),
                patch.object(main_module, "scan_dataset", side_effect=AssertionError("collection discovery must stay lazy")),
            ):
                payload = main_module.open_dataset_folder()

        self.assertEqual("collection", payload["mode"])
        self.assertIsNone(payload["dataset"])
        self.assertEqual(2, payload["dataset_count"])

    def test_cli_exposes_service_and_dataset_commands(self) -> None:
        parser = build_parser()
        for command in ("serve", "start", "status", "stop", "doctor", "datasets", "open", "schema", "full", "robots"):
            with self.subTest(command=command):
                arguments = [command]
                if command == "open":
                    arguments.append(".")
                elif command == "schema":
                    arguments.append("dataset-id")
                elif command == "full":
                    arguments.extend(["dataset-id", "--all"])
                parsed = parser.parse_args(arguments)
                self.assertTrue(callable(parsed.handler))

    def test_full_episode_selection_supports_names_ids_and_globs(self) -> None:
        manifest = {"episodes": [
            {"id": "episode-1", "name": "pick_usb"},
            {"id": "episode-2", "name": "place_usb"},
            {"id": "trial-3", "name": "inspect"},
        ]}

        selected = _select_full_episodes(manifest, ["episode-*", "inspect"], False)

        self.assertEqual(["episode-1", "episode-2", "trial-3"], [item["id"] for item in selected])

    def test_full_shards_balance_by_frame_count(self) -> None:
        episodes = [
            {"id": "a", "frame_count": 100},
            {"id": "b", "frame_count": 90},
            {"id": "c", "frame_count": 20},
            {"id": "d", "frame_count": 10},
        ]

        shards = _balanced_full_shards(episodes, 2)
        loads = sorted(sum(int(item["frame_count"]) for item in shard) for shard in shards)

        self.assertEqual([110, 110], loads)

    def test_linux_full_launcher_configures_gpu_workers(self) -> None:
        launcher = (Path(__file__).resolve().parents[1] / "full.sh").read_text(encoding="utf-8")

        self.assertIn("ALICE_GPU_DEVICES", launcher)
        self.assertIn("ALICE_FULL_WORKERS", launcher)
        self.assertIn("ALICE_FULL_PARALLEL", launcher)
        self.assertIn("torch.cuda.is_available()", launcher)
        self.assertIn('= "--robots"', launcher)
        self.assertIn("--expect-workers", launcher)
        self.assertIn("-m app.cli full", launcher)

    def test_full_command_submits_balanced_detached_jobs(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "full", "dataset", "--all", "--parallel", "2", "--detach",
            "--url", "http://127.0.0.1:8000",
        ])
        manifest = {
            "id": "dataset",
            "name": "dataset",
            "episodes": [
                {"id": "ep-1", "name": "ep-1", "frame_count": 100, "primary_media_file_id": "v1", "media_streams": [{"file_id": "v1"}]},
                {"id": "ep-2", "name": "ep-2", "frame_count": 50, "primary_media_file_id": "v2", "media_streams": [{"file_id": "v2"}]},
            ],
        }
        submitted = []

        def request(_url, method="GET", payload=None, timeout=3.0):
            submitted.append(payload)
            return {"id": f"job-{len(submitted)}", "status": "queued", "episode_count": len(payload["episode_ids"])}

        with (
            patch("app.cli._health", return_value={"ok": True, "runtime": {"full_pipeline": {"workers": 2, "gpu_devices": []}}}),
            patch("app.cli._resolve_full_dataset", return_value=manifest),
            patch("app.cli._request_json", side_effect=request),
            patch("app.cli._print"),
        ):
            result = command_full(args)

        self.assertEqual(0, result)
        self.assertEqual(2, len(submitted))
        self.assertTrue(all(payload["full_pipeline"] for payload in submitted))
        self.assertTrue(all(payload["full_output_format"] == "lerobot" for payload in submitted))
        self.assertTrue(all("full_action_profile_id" not in payload for payload in submitted))

        action_args = parser.parse_args([
            "full", "dataset", "--all", "--parallel", "2", "--detach",
            "--robot", "so100_so101", "--source-hand", "right",
            "--url", "http://127.0.0.1:8000",
        ])
        submitted.clear()
        with (
            patch("app.cli._health", return_value={"ok": True, "runtime": {"full_pipeline": {"workers": 2, "gpu_devices": []}}}),
            patch("app.cli._resolve_full_dataset", return_value=manifest),
            patch("app.cli._request_json", side_effect=request),
            patch("app.cli._print"),
        ):
            action_result = command_full(action_args)

        self.assertEqual(0, action_result)
        self.assertEqual(2, len(submitted))
        self.assertTrue(all(payload["full_action_profile_id"] == "so100_so101" for payload in submitted))

    def test_robot_type_is_optional_listed_and_validated(self) -> None:
        parser = build_parser()
        no_robot = parser.parse_args(["full", "dataset", "--all"])
        selected = parser.parse_args(["full", "dataset", "--all", "--robot", "franka_panda"])

        self.assertIsNone(no_robot.action_profile)
        self.assertEqual("lerobot", no_robot.output_format)
        self.assertEqual("franka_panda", selected.action_profile)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["full", "dataset", "--all", "--robot", "unknown_robot"])
        with patch("app.cli._print") as output:
            result = command_robots(parser.parse_args(["robots", "--json"]))
        self.assertEqual(0, result)
        profile_ids = [item["id"] for item in output.call_args.args[0]["items"]]
        self.assertIn("so100_so101", profile_ids)
        self.assertIn("aloha_bimanual", profile_ids)

    def test_async_model_loading_reports_loading_before_worker_finishes(self) -> None:
        registry = ModelRegistry()
        started = threading.Event()
        release = threading.Event()

        def worker(_config):
            started.set()
            release.wait(2)
            return registry.status()

        config = LocalModelConfig(kind="yolo", model_path="model.pt", device="cpu", confidence=0.25)
        with patch.object(registry, "configure_local", side_effect=worker):
            status = registry.configure_local_async(config)
            self.assertTrue(status["local"]["loading"])
            self.assertTrue(started.wait(1))
            release.set()
            registry._loader_thread.join(2)

    def test_schema_sampling_is_spread_across_folders_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = []
            for index in range(MAX_FOLDERS + 40):
                folder = root / f"episode_{index:04d}"
                folder.mkdir()
                for extension in (".json", ".csv", ".npy", ".txt", ".bin"):
                    path = folder / f"sample{extension}"
                    path.write_text("{}" if extension == ".json" else "value", encoding="utf-8")
                    files.append(path)

            sampled, metadata = sample_profile_paths(root, files)

        self.assertLessEqual(len(sampled), MAX_FILES)
        self.assertLessEqual(metadata["folders_sampled"], MAX_FOLDERS)
        folder_counts = {}
        for path in sampled:
            folder_counts[path.parent.name] = folder_counts.get(path.parent.name, 0) + 1
        self.assertTrue(folder_counts)
        self.assertLessEqual(max(folder_counts.values()), MAX_FILES_PER_FOLDER)
        self.assertGreater(len(folder_counts), 1)

    def test_inventory_uses_supplied_scan_paths_without_second_tree_walk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "episode_1").mkdir()
            source = root / "episode_1" / "metadata.json"
            source.write_text('{"state": [1, 2, 3]}', encoding="utf-8")
            with patch.object(Path, "rglob", side_effect=AssertionError("unexpected second walk")):
                inventory = build_inventory(root, [], {"episode_1/metadata.json"})

        self.assertEqual(1, inventory["file_count"])
        self.assertEqual("folder_extension_stratified_v1", inventory["sampling"]["strategy"])
        self.assertEqual(1, inventory["files_profiled"])

    def test_episode_audit_framework_keeps_real_ids_and_samples_folders(self) -> None:
        files = [
            {
                "id": f"id-{index}",
                "relative_path": f"task_{index:04d}/episode_{index:04d}.h5",
                "kind": "structured",
                "category": "sensor",
                "episode_token": str(index),
                "episode_key": f"task_{index:04d}",
                "episode_id": None,
                "size_bytes": 10,
            }
            for index in range(800)
        ]

        framework = build_sampled_episode_framework(files, max_files=120, max_folders=60, max_per_folder=2)

        self.assertLessEqual(len(framework["files"]), 120)
        self.assertEqual(800, framework["sampling"]["full_file_count"])
        self.assertTrue({item["file_id"] for item in framework["files"]} <= {item["id"] for item in files})


if __name__ == "__main__":
    unittest.main()
