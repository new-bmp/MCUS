from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import PropertyMock, patch

from fastapi import HTTPException

from app import main as main_module
from app.camera_profiles import NEXUS_OAKD_PRO_W9_PROFILE_ID
from app.cli import _preflight_and_open_path, build_parser, command_format
from app.schemas import PathOpenRequest


def sample_report(root: Path, *, token: str = "format-token", status: str = "warning") -> dict:
    return {
        "schema": "alice/dataset-format-map/v1",
        "root_path": str(root),
        "root_mode": "dataset",
        "format_family": "nexus_multimodal",
        "format_confidence": 0.99,
        "status": status,
        "confirmation_token": token,
        "episode_layout": "episode_directories",
        "episode_count_hint": 12,
        "episode_samples": ["ep_0001", "ep_0012"],
        "declared_streams": [
            {"kind": "vision", "modality": "rgb"},
            {"kind": "vision", "modality": "depth"},
            {"kind": "sensor", "modality": "tactile"},
            {"kind": "sensor", "modality": "imu"},
        ],
        "modality_counts": {"rgb": 3, "depth": 1, "tactile": 2, "imu": 1},
        "kind_counts": {"vision": 4, "sensor": 3},
        "capabilities": {"can_import": status != "blocked", "can_vlm": True, "can_full_export": False},
        "camera_calibration": {
            "requires_profile_selection": True,
            "recommended_profile_id": NEXUS_OAKD_PRO_W9_PROFILE_ID,
            "selected_profile_id": None,
            "profiles": [{
                "id": NEXUS_OAKD_PRO_W9_PROFILE_ID,
                "label": "OAK-D Pro W9 · Nexus 当前相机",
                "description": "左目到右目 X=-7.5 cm；右目到 RGB X=+3.75 cm。",
            }],
        },
        "issues": [
            {"severity": "warning", "code": "camera_extrinsics_missing", "message": "相机外参未应用。"},
            {"severity": "info", "code": "action_missing", "message": "未发现原生 Action。"},
        ],
    }


class FormatPreflightApiTests(unittest.TestCase):
    def test_preflight_endpoint_returns_episode_modalities_warnings_and_capabilities(self) -> None:
        root = Path("dataset").resolve()
        report = sample_report(root)
        with patch.object(main_module, "inspect_dataset_format", return_value=report):
            payload = main_module.dataset_format_preflight(PathOpenRequest(path=str(root)))

        self.assertEqual("nexus_multimodal", payload["format_family"])
        self.assertEqual(12, payload["episode_summary"]["count_hint"])
        self.assertEqual(1, len(payload["warnings"]))
        self.assertEqual("camera_extrinsics_missing", payload["warnings"][0]["code"])
        self.assertEqual(1, payload["modality_counts"]["depth"])
        self.assertTrue(payload["capabilities"]["can_import"])

    def test_folder_dialog_preflight_does_not_scan_or_register_single_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            discovery = {
                "mode": "single",
                "root_path": str(root),
                "dataset_count": 1,
                "datasets": [{"key": "one", "name": root.name, "path": str(root), "status": "loading"}],
            }
            with (
                patch.object(main_module, "choose_folder", return_value=str(root)),
                patch.object(main_module, "discover_dataset_roots", return_value=discovery),
                patch.object(main_module, "inspect_dataset_format", return_value=sample_report(root)),
                patch.object(main_module, "scan_dataset", side_effect=AssertionError("preflight must not import")),
            ):
                payload = main_module.open_dataset_folder(preflight_only=True)

        self.assertIsNone(payload["dataset"])
        self.assertEqual("format-token", payload["preflight"]["confirmation_token"])

    def test_confirmed_import_rechecks_the_preflight_token(self) -> None:
        root = Path("dataset").resolve()
        request = PathOpenRequest(
            path=str(root),
            analyze_schema=False,
            camera_profile_id=NEXUS_OAKD_PRO_W9_PROFILE_ID,
        )
        manifest = {"id": "dataset", "episodes": []}
        with (
            patch.object(main_module, "inspect_dataset_format", return_value=sample_report(root)),
            patch.object(main_module, "scan_dataset", return_value=manifest) as scan,
        ):
            payload = main_module.open_path(request, confirmation_token="format-token")
        self.assertEqual(manifest, payload)
        scan.assert_called_once_with(
            str(root),
            None,
            camera_profile_id=NEXUS_OAKD_PRO_W9_PROFILE_ID,
        )

        with (
            patch.object(main_module, "inspect_dataset_format", return_value=sample_report(root)),
            patch.object(main_module, "scan_dataset") as scan,
            self.assertRaises(HTTPException) as raised,
        ):
            main_module.open_path(request, confirmation_token="stale-token")
        self.assertEqual(409, raised.exception.status_code)
        scan.assert_not_called()

    def test_import_without_confirmation_is_rejected_before_scan(self) -> None:
        root = Path("dataset").resolve()
        request = PathOpenRequest(path=str(root), analyze_schema=False)
        with patch.object(main_module, "scan_dataset") as scan, self.assertRaises(HTTPException) as raised:
            main_module.open_path(request)
        self.assertEqual(428, raised.exception.status_code)
        scan.assert_not_called()

    def test_collection_root_is_not_merged_by_open_path(self) -> None:
        root = Path("collection").resolve()
        report = {**sample_report(root), "root_mode": "collection"}
        request = PathOpenRequest(path=str(root), analyze_schema=False)
        with (
            patch.object(main_module, "inspect_dataset_format", return_value=report),
            patch.object(main_module, "scan_dataset") as scan,
            self.assertRaises(HTTPException) as raised,
        ):
            main_module.open_path(request, confirmation_token="format-token")
        self.assertEqual(409, raised.exception.status_code)
        scan.assert_not_called()

    def test_qwen_cannot_replace_preflight_format_family_or_capabilities(self) -> None:
        manifest = {
            "id": "dataset",
            "format_map": {
                "format_family": "nexus_multimodal",
                "format_confidence": 0.99,
                "capabilities": {"can_vlm": True, "can_full_export": False},
            },
            "schema_profile": {"inventory": {"candidate_streams": []}},
            "episode_resolution": {"requires_api": False},
        }
        qwen_understanding = {
            "format_family": "egodex",
            "format_confidence": 0.4,
            "summary": "wrong family",
            "episode_organization": "unknown",
            "streams": [],
            "associations": [],
        }
        with (
            patch.object(type(main_module.registry), "has_vlm", new_callable=PropertyMock, return_value=True),
            patch.object(main_module.registry, "understand_dataset_schema", return_value={}),
            patch.object(main_module, "validate_understanding", return_value=(qwen_understanding, [])),
            patch.object(main_module.registry, "status", return_value={"vlm": {"kind": "qwen"}}),
            patch.object(main_module, "save_manifest"),
        ):
            result = main_module._understand_manifest(manifest)

        understanding = result["schema_profile"]["understanding"]
        self.assertEqual("nexus_multimodal", understanding["format_family"])
        self.assertEqual(0.99, understanding["format_confidence"])
        self.assertEqual({"can_vlm": True, "can_full_export": False}, understanding["capabilities"])
        self.assertTrue(any("local preflight confirmed nexus_multimodal" in item for item in result["schema_profile"]["warnings"]))


class FormatCliAndFrontendContractTests(unittest.TestCase):
    def test_alice_format_is_offline_and_supports_json(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["format", "dataset", "--json"])
        report = sample_report(Path("dataset").resolve())
        output = io.StringIO()
        with patch("app.dataset_format.inspect_dataset_format", return_value=report), redirect_stdout(output):
            result = command_format(args)

        self.assertEqual(0, result)
        self.assertEqual("nexus_multimodal", json.loads(output.getvalue())["format_family"])

    def test_frontend_confirms_single_and_collection_imports(self) -> None:
        root = Path(__file__).resolve().parents[1]
        app = (root / "static" / "app.js").read_text(encoding="utf-8")
        index = (root / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="formatPreflightModal"', index)
        self.assertIn('id="formatPreflightModalities"', index)
        self.assertIn('id="formatPreflightCapabilities"', index)
        self.assertIn('id="formatPreflightCameraProfile"', index)
        self.assertIn('/api/system/open-dataset-folder?preflight_only=true', app)
        self.assertIn('requestDatasetPreflight(item.path, item.name)', app)
        self.assertIn('confirmDatasetPreflight(report', app)
        self.assertIn('confirmation_token=${encodeURIComponent(report.confirmation_token)}', app)
        self.assertIn('camera_profile_id: report.selected_camera_profile_id || null', app)
        self.assertIn('R → RGB X=+3.75 cm', app)
        self.assertIn("RGB、深度、触觉、压力、IMU、关节与 Action 将保持为独立数据流", index)

    def test_cli_path_import_preflights_and_forwards_token(self) -> None:
        root = str(Path("dataset").resolve())
        report = sample_report(Path(root), token="cli-token", status="ready")
        manifest = {"id": "dataset"}
        with patch("app.cli._request_json", side_effect=[report, manifest]) as request:
            payload = _preflight_and_open_path(
                "http://127.0.0.1:8000",
                root,
                name="sample",
                analyze_schema=False,
                timeout=30,
            )
        self.assertEqual(manifest, payload)
        self.assertEqual(2, request.call_count)
        self.assertIn("confirmation_token=cli-token", request.call_args_list[1].args[0])

    def test_cli_rejects_collection_before_open(self) -> None:
        report = {**sample_report(Path("collection").resolve(), status="ready"), "root_mode": "collection"}
        with patch("app.cli._request_json", return_value=report) as request, self.assertRaises(RuntimeError):
            _preflight_and_open_path(
                "http://127.0.0.1:8000",
                str(Path("collection").resolve()),
                name=None,
                analyze_schema=False,
                timeout=30,
            )
        self.assertEqual(1, request.call_count)


if __name__ == "__main__":
    unittest.main()
