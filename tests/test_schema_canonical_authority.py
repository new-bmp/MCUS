from __future__ import annotations

import unittest

from app.models import ModelRegistry
from app.schema_profiler import validate_understanding


def candidate(
    source_id: str,
    *,
    kind: str,
    modality: str,
    side: str = "unknown",
    role: str = "",
    variant: str = "primary",
    extraction: str = "",
    evidence: str = "recorder_metadata",
    confidence: float = 0.99,
    shape: list[int] | None = None,
    dtype: str = "float32",
) -> dict:
    return {
        "id": source_id,
        "source_path": f"episode/{source_id}.h5",
        "field": source_id,
        "kind": kind,
        "modality": modality,
        "side_hint": side,
        "shape": shape or [30, 6],
        "dtype": dtype,
        "role": role,
        "variant": variant,
        "extraction": extraction,
        "evidence": evidence,
        "canonical_kind": kind,
        "canonical_evidence": evidence,
        "canonical_confidence": confidence,
        "side": side,
    }


def proposal(source_id: str, **overrides) -> dict:
    payload = {
        "source_id": source_id,
        "kind": "action",
        "modality": "command",
        "side": "right",
        "role": "robot_action",
        "representation": "delta",
        "dimension_names": ["semantic_name"],
        "gripper_indices": [0],
        "embodiment_id": "qwen-name",
        "confidence": 0.51,
        "evidence": "qwen_guess",
    }
    payload.update(overrides)
    return payload


class CanonicalSchemaAuthorityTests(unittest.TestCase):
    def test_recorder_metadata_locks_processing_fields_against_qwen(self) -> None:
        inventory = {"candidate_streams": [
            candidate("depth", kind="vision", modality="depth", role="raw_depth_frames"),
            candidate("tactile", kind="sensor", modality="tactile", side="left", role="contact_sensor", variant="synchronized"),
            candidate("imu", kind="sensor", modality="imu", role="inertial_sensor"),
            candidate("device_joints", kind="sensor", modality="hand_device", side="left", role="device_channels", shape=[30, 6], dtype="uint8"),
            candidate("skeleton", kind="joint", modality="pose", side="right", role="hand_skeleton", variant="synchronized", extraction="skeleton_xyz", shape=[30, 60]),
        ]}
        raw = {
            "streams": [
                proposal("depth", side="left", role="joint_state"),
                proposal("tactile", kind="joint", modality="position"),
                proposal("imu", kind="joint", modality="position"),
                proposal("device_joints"),
                proposal("skeleton", kind="action", modality="command"),
            ],
            "associations": [],
        }

        understanding, warnings = validate_understanding(inventory, raw)
        streams = {item["source_id"]: item for item in understanding["streams"]}

        self.assertEqual(("vision", "depth", "unknown", "raw_depth_frames"), tuple(streams["depth"][key] for key in ("kind", "modality", "side", "role")))
        self.assertEqual(("sensor", "tactile", "left", "contact_sensor", "synchronized"), tuple(streams["tactile"][key] for key in ("kind", "modality", "side", "role", "variant")))
        self.assertEqual(("sensor", "imu"), (streams["imu"]["kind"], streams["imu"]["modality"]))
        self.assertEqual(("sensor", "hand_device"), (streams["device_joints"]["kind"], streams["device_joints"]["modality"]))
        self.assertEqual(("joint", "pose", "skeleton_xyz"), (streams["skeleton"]["kind"], streams["skeleton"]["modality"], streams["skeleton"]["extraction"]))
        self.assertEqual("unknown", streams["skeleton"]["representation"])
        self.assertEqual(["semantic_name"], streams["skeleton"]["dimension_names"])
        self.assertTrue(all(item["canonical_locked"] for item in streams.values()))
        self.assertTrue(any("authoritative canonical stream depth" in warning for warning in warnings))

    def test_qwen_omissions_restore_all_authoritative_modalities(self) -> None:
        inventory = {"candidate_streams": [
            candidate("rgb", kind="vision", modality="rgb", role="camera_stream"),
            candidate("depth", kind="vision", modality="depth", role="raw_depth_frames"),
            candidate("tactile", kind="sensor", modality="tactile", side="left", role="contact_sensor"),
            candidate("imu", kind="sensor", modality="imu", role="inertial_sensor"),
            candidate("clock", kind="timestamp", modality="time", role="alignment_table"),
        ]}
        raw = {
            "streams": [proposal("rgb", kind="vision", modality="rgb", side="unknown", role="camera_stream")],
            "associations": [{
                "vision_id": "rgb",
                "joint_ids": [],
                "sensor_ids": ["tactile", "imu"],
                "side": "unknown",
                "time_alignment": "recorder sync",
                "timestamp_id": "clock",
                "confidence": 0.8,
                "reason": "metadata",
            }],
        }

        understanding, warnings = validate_understanding(inventory, raw)
        streams = {item["source_id"]: item for item in understanding["streams"]}

        self.assertEqual({"rgb", "depth", "tactile", "imu", "clock"}, set(streams))
        self.assertEqual("depth", streams["depth"]["modality"])
        self.assertEqual("tactile", streams["tactile"]["modality"])
        self.assertEqual("imu", streams["imu"]["modality"])
        self.assertEqual("timestamp", streams["clock"]["kind"])
        self.assertEqual(["tactile", "imu"], understanding["associations"][0]["sensor_ids"])
        self.assertEqual("clock", understanding["associations"][0]["timestamp_id"])
        self.assertTrue(any("Restored 4 authoritative canonical stream" in warning for warning in warnings))

    def test_unknown_generic_stream_can_still_be_classified_by_qwen(self) -> None:
        unknown = candidate(
            "unknown_vector",
            kind="other",
            modality="unknown",
            role="",
            evidence="path_field_shape_heuristic",
            confidence=0.4,
        )
        inventory = {"candidate_streams": [unknown]}
        raw = {"streams": [proposal("unknown_vector", kind="action", modality="command", side="right", role="robot_action")], "associations": []}

        understanding, _ = validate_understanding(inventory, raw)
        stream = understanding["streams"][0]

        self.assertEqual(("action", "command", "right", "robot_action"), tuple(stream[key] for key in ("kind", "modality", "side", "role")))
        self.assertFalse(stream["canonical_locked"])
        self.assertEqual("qwen_guess", stream["evidence"])

    def test_high_confidence_canonical_mapping_is_authoritative_without_recorder_label(self) -> None:
        mapped = candidate(
            "mapped_pressure",
            kind="sensor",
            modality="pressure",
            side="left",
            role="contact_sensor",
            evidence="verified_adapter",
            confidence=0.97,
        )
        inventory = {"candidate_streams": [mapped]}
        raw = {"streams": [proposal("mapped_pressure", kind="action", modality="command")], "associations": []}

        understanding, _ = validate_understanding(inventory, raw)
        stream = understanding["streams"][0]

        self.assertEqual(("sensor", "pressure", "left", "contact_sensor"), tuple(stream[key] for key in ("kind", "modality", "side", "role")))
        self.assertTrue(stream["canonical_locked"])

    def test_compact_qwen_inventory_keeps_unknown_generic_but_not_metadata(self) -> None:
        unknown = candidate(
            "unknown_vector",
            kind="other",
            modality="unknown",
            evidence="path_field_shape_heuristic",
            confidence=0.4,
        )
        metadata = candidate(
            "metadata_field",
            kind="other",
            modality="metadata",
            evidence="recorder_metadata",
            confidence=0.99,
        )
        metadata["canonical_kind"] = "metadata"

        compact = ModelRegistry._compact_schema_inventory({"candidate_streams": [unknown, metadata]})
        source_ids = {item["id"] for item in compact["candidate_streams"]}

        self.assertEqual({"unknown_vector"}, source_ids)


if __name__ == "__main__":
    unittest.main()
