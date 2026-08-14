from __future__ import annotations

"""Resolve explicit dataset processing modes without shape-based guessing."""

from typing import Any


KNOWN_DATASET_FAMILIES = {
    "egodex", "nexus_multimodal", "openxr", "lerobot", "alice_full",
    "generic_multimodal", "vision_only", "unknown",
}

_FAMILY_ALIASES = {
    "nexus": "nexus_multimodal",
    "nexus_multimodal": "nexus_multimodal",
    "egodex": "egodex",
    "openxr": "openxr",
    "lerobot": "lerobot",
    "alice_full": "alice_full",
    "generic_multimodal": "generic_multimodal",
    "vision_only": "vision_only",
    "unknown": "unknown",
}

_STRATEGY_FAMILIES = {
    "egodex_joint_centric_v1": "egodex",
    "nexus_sensor_fusion_v1": "nexus_multimodal",
    "openxr_hand_tracking_v1": "openxr",
}

_MODE_CONTRACTS = {
    "egodex": {
        "joint_adapter": "egodex_full_skeleton_to_mano21",
        "joint_coordinate_space": "episode_world",
        "camera_projection_mode": "embedded_camera_pose",
        "hand_visibility_backend": "egodex_embedded_camera_v1",
        "projection_correction_backend": "egodex_mano_prior_v1",
        "requires_external_hand_camera_extrinsics": False,
    },
    "nexus_multimodal": {
        "joint_adapter": "nexus_dexweaveg1_20_to_mano21",
        "joint_coordinate_space": "mocap_tracking",
        "camera_projection_mode": "calibrated_external_extrinsics",
        "hand_visibility_backend": "nexus_calibrated_tracking_v1",
        "projection_correction_backend": None,
        "requires_external_hand_camera_extrinsics": True,
    },
    "openxr": {
        "joint_adapter": "openxr_hand_26_to_mano21",
        "joint_coordinate_space": "openxr_base_space",
        "camera_projection_mode": "calibrated_external_extrinsics",
        "hand_visibility_backend": "openxr_calibrated_base_v1",
        "projection_correction_backend": None,
        "requires_external_hand_camera_extrinsics": True,
    },
    "lerobot": {
        "joint_adapter": "declared_lerobot_schema",
        "joint_coordinate_space": "declared_metadata",
        "camera_projection_mode": "declared_metadata",
        "hand_visibility_backend": None,
        "projection_correction_backend": None,
        "requires_external_hand_camera_extrinsics": False,
    },
    "alice_full": {
        "joint_adapter": "declared_alice_full_schema",
        "joint_coordinate_space": "declared_metadata",
        "camera_projection_mode": "declared_metadata",
        "hand_visibility_backend": None,
        "projection_correction_backend": None,
        "requires_external_hand_camera_extrinsics": False,
    },
}

_CONSERVATIVE_CONTRACT = {
    "joint_adapter": None,
    "joint_coordinate_space": "unknown",
    "camera_projection_mode": "disabled_until_confirmed",
    "hand_visibility_backend": None,
    "projection_correction_backend": None,
    "requires_external_hand_camera_extrinsics": None,
}


def _normal(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _family(value: Any) -> str | None:
    normalized = _normal(value)
    if not normalized:
        return None
    return _FAMILY_ALIASES.get(normalized, normalized if normalized in KNOWN_DATASET_FAMILIES else None)


def dataset_mode(manifest: dict) -> dict:
    """Return one explicit semantic and spatial contract for a manifest."""

    format_map = manifest.get("format_map") or {}
    raw_family_values = (
        manifest.get("format_family"),
        format_map.get("format_family"),
    )
    explicit_family_present = any(str(value or "").strip() for value in raw_family_values)
    families = list(dict.fromkeys(
        item for item in (
            _family(manifest.get("format_family")),
            _family(format_map.get("format_family")),
        ) if item and item != "unknown"
    ))
    strategy = format_map.get("processing_strategy") or manifest.get("processing_strategy") or {}
    strategy_id = _normal(strategy.get("id"))
    strategy_family = _STRATEGY_FAMILIES.get(strategy_id)
    conflicts: list[str] = []
    if len(families) > 1:
        conflicts.append("manifest_and_format_map_disagree")
    explicit_family = families[0] if len(families) == 1 else None
    if explicit_family and strategy_family and explicit_family != strategy_family:
        conflicts.append("format_family_and_processing_strategy_disagree")
    family = explicit_family or strategy_family or "unknown"
    if conflicts:
        family = "unknown"
    return {
        "family": family,
        "strategy_id": strategy_id or None,
        "explicit_family_present": explicit_family_present,
        "resolved_from": "format_family" if explicit_family and not conflicts else "processing_strategy" if strategy_family and not conflicts else "unresolved",
        "conflict": bool(conflicts),
        "conflict_reasons": conflicts,
        **dict(_MODE_CONTRACTS.get(family, _CONSERVATIVE_CONTRACT)),
    }


def mode_contract_for_family(family: str, strategy: dict | None = None) -> dict:
    return dataset_mode({"format_family": family, "processing_strategy": dict(strategy or {})})
