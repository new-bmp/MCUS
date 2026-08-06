from __future__ import annotations

from copy import deepcopy


NEXUS_OAKD_PRO_W9_PROFILE_ID = "oakd_pro_w9_lr75_r_rgb375_v1"


def _translation_matrix(x_m: float, y_m: float, z_m: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, float(x_m)],
        [0.0, 1.0, 0.0, float(y_m)],
        [0.0, 0.0, 1.0, float(z_m)],
        [0.0, 0.0, 0.0, 1.0],
    ]


_NEXUS_CAMERA_PROFILES = {
    NEXUS_OAKD_PRO_W9_PROFILE_ID: {
        "id": NEXUS_OAKD_PRO_W9_PROFILE_ID,
        "label": "OAK-D Pro W9 · Nexus 当前相机",
        "camera_type": "oakd_pro_w9",
        "description": "左目到右目 X=-7.5 cm；右目到 RGB X=+3.75 cm；旋转按同轴处理。",
        "frame_convention": "OpenCV (x-right, y-down, z-forward)",
        "unit": "m",
        "depth_reference": "right",
        "rotation_assumption": "identity",
        "transforms": {
            "T_right__left": _translation_matrix(-0.075, 0.0, 0.0),
            "T_head_rgb__right": _translation_matrix(0.0375, 0.0, 0.0),
            "T_head_rgb__left": _translation_matrix(-0.0375, 0.0, 0.0),
            "T_head_rgb__head_depth": _translation_matrix(0.0375, 0.0, 0.0),
        },
        "safe_for": [
            "depth_camera_point_cloud",
            "head_rgb_depth_registration",
            "depth_assisted_visible_arm_localization",
        ],
        "not_safe_for": [
            "mocap_to_head_camera_projection",
            "wrist_camera_to_head_camera_projection",
            "joint_overlay",
        ],
        "source": "user_confirmed_camera_geometry",
    },
}


def nexus_camera_profiles() -> list[dict]:
    """Return frontend-safe copies of the available Nexus camera presets."""
    return [deepcopy(profile) for profile in _NEXUS_CAMERA_PROFILES.values()]


def nexus_camera_profile(profile_id: str | None) -> dict | None:
    if not profile_id:
        return None
    profile = _NEXUS_CAMERA_PROFILES.get(str(profile_id))
    if profile is None:
        raise ValueError(f"Unknown Nexus camera profile: {profile_id}")
    return deepcopy(profile)

