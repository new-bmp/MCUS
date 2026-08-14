from __future__ import annotations

import hashlib
import json
import math
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

from .full_export import _aligned_rows, _find_transform_source, _rot6d, _take_rows
from .job_control import CancellableJobMixin, JobCancelled
from .schemas import ActionMappingRequest
from .sensor_alignment import ensure_episode_time_sync
from .storage import dataset_artifact_dir, dataset_sidecar_root, get_manifest, storage_slug


ACTION_MAPPING_SCHEMA = "alice/action-mapping/v1"
ACTION_REPORT_SCHEMA = "alice/action-mapping-report/v1"
ACTION_INDEX_SCHEMA = "alice/action-mapping-index/v1"

OBSERVATION_FIELDS = [
    *[f"left_{name}" for name in ("x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "grip")],
    *[f"right_{name}" for name in ("x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "grip")],
]

BIMANUAL_POSE_FIELDS = list(OBSERVATION_FIELDS)
BIMANUAL_DELTA_FIELDS = [
    *[f"left_{name}" for name in ("dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "grip")],
    *[f"right_{name}" for name in ("dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "grip")],
]
SINGLE_DELTA_FIELDS = ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "grip"]


ACTION_MAPPING_PROFILES: dict[str, dict] = {
    "generic_bimanual_pose": {
        "id": "generic_bimanual_pose",
        "name": "通用双臂 · 绝对末端 20D",
        "robot_family": "generic_bimanual",
        "description": "左右腕未来 xyz + rot6d + grip；适合先训练与具体机器人无关的双手目标策略。",
        "representation": "absolute_pose_rot6d",
        "control_space": "cartesian_pose_target",
        "sides": 2,
        "action_dim": 20,
        "fields": BIMANUAL_POSE_FIELDS,
        "requires_calibration": True,
        "requires_ik": False,
        "target_native_action_dim": None,
    },
    "generic_bimanual_delta": {
        "id": "generic_bimanual_delta",
        "name": "通用双臂 · 末端增量 14D",
        "robot_family": "generic_bimanual",
        "description": "左右腕 Δxyz + 旋转向量 + grip；适合支持笛卡尔增量控制的双臂机器人。",
        "representation": "delta_pose_axis_angle",
        "control_space": "cartesian_delta",
        "sides": 2,
        "action_dim": 14,
        "fields": BIMANUAL_DELTA_FIELDS,
        "requires_calibration": True,
        "requires_ik": False,
        "target_native_action_dim": None,
    },
    "aloha_bimanual": {
        "id": "aloha_bimanual",
        "name": "ALOHA · 双臂末端代理 14D",
        "robot_family": "aloha",
        "description": "生成双臂笛卡尔代理动作；转成 ALOHA 原生双臂关节目标前还需标定与 IK。",
        "representation": "delta_pose_axis_angle",
        "control_space": "cartesian_delta_proxy",
        "sides": 2,
        "action_dim": 14,
        "fields": BIMANUAL_DELTA_FIELDS,
        "requires_calibration": True,
        "requires_ik": True,
        "target_native_action_dim": 14,
    },
    "so100_so101": {
        "id": "so100_so101",
        "name": "SO-100 / SO-101 · 单臂末端代理 7D",
        "robot_family": "lerobot_so100_so101",
        "description": "面向 LeRobot SO-100/SO-101 的单手末端代理；转成舵机位置前还需机械臂标定与 IK。",
        "representation": "delta_pose_axis_angle",
        "control_space": "cartesian_delta_proxy",
        "sides": 1,
        "action_dim": 7,
        "fields": SINGLE_DELTA_FIELDS,
        "requires_calibration": True,
        "requires_ik": True,
        "target_native_action_dim": 6,
    },
    "franka_panda": {
        "id": "franka_panda",
        "name": "Franka Panda · 单臂末端代理 7D",
        "robot_family": "franka_panda",
        "description": "所选人手映射为 Δxyz + 旋转向量 + grip；原生 7 关节与夹爪目标需 IK。",
        "representation": "delta_pose_axis_angle",
        "control_space": "cartesian_delta_proxy",
        "sides": 1,
        "action_dim": 7,
        "fields": SINGLE_DELTA_FIELDS,
        "requires_calibration": True,
        "requires_ik": True,
        "target_native_action_dim": 8,
    },
    "ur5e": {
        "id": "ur5e",
        "name": "UR5 / UR5e · 单臂末端代理 7D",
        "robot_family": "universal_robots_ur5e",
        "description": "所选人手映射为笛卡尔增量与夹爪；转为 6 轴关节控制前还需 IK。",
        "representation": "delta_pose_axis_angle",
        "control_space": "cartesian_delta_proxy",
        "sides": 1,
        "action_dim": 7,
        "fields": SINGLE_DELTA_FIELDS,
        "requires_calibration": True,
        "requires_ik": True,
        "target_native_action_dim": 7,
    },
    "xarm7": {
        "id": "xarm7",
        "name": "xArm 7 · 单臂末端代理 7D",
        "robot_family": "ufactory_xarm7",
        "description": "所选人手映射为笛卡尔增量与夹爪；转为 7 轴关节目标前还需 IK。",
        "representation": "delta_pose_axis_angle",
        "control_space": "cartesian_delta_proxy",
        "sides": 1,
        "action_dim": 7,
        "fields": SINGLE_DELTA_FIELDS,
        "requires_calibration": True,
        "requires_ik": True,
        "target_native_action_dim": 8,
    },
    "agilex_piper": {
        "id": "agilex_piper",
        "name": "AgileX Piper · 单臂末端代理 7D",
        "robot_family": "agilex_piper",
        "description": "所选人手映射为笛卡尔增量与夹爪；转为 Piper 关节目标前还需 IK。",
        "representation": "delta_pose_axis_angle",
        "control_space": "cartesian_delta_proxy",
        "sides": 1,
        "action_dim": 7,
        "fields": SINGLE_DELTA_FIELDS,
        "requires_calibration": True,
        "requires_ik": True,
        "target_native_action_dim": 7,
    },
}

_INDEX_LOCK = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def action_mapping_profiles() -> list[dict]:
    return [deepcopy(profile) for profile in ACTION_MAPPING_PROFILES.values()]


def _profile(profile_id: str) -> dict:
    profile = ACTION_MAPPING_PROFILES.get(str(profile_id))
    if profile is None:
        raise ValueError(f"不支持的机器人映射: {profile_id}")
    return profile


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_signature(path: Path, relative: str, source_count: int, video_count: int) -> dict:
    stat = path.stat()
    return {
        "relative_path": relative.replace("\\", "/"),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "source_row_count": int(source_count),
        "video_frame_count": int(video_count),
    }


def _config_payload(request: ActionMappingRequest) -> dict:
    return {
        "profile_id": request.profile_id,
        "source_hand": request.source_hand,
        "coordinate_frame": request.coordinate_frame,
        "horizon_frames": int(request.horizon_frames),
    }


def _action_paths(dataset_id: str, profile_id: str, episode_id: str) -> tuple[Path, Path, Path]:
    root = dataset_artifact_dir(dataset_id, "actions") / profile_id
    root.mkdir(parents=True, exist_ok=True)
    stem = storage_slug(episode_id)
    return root / f"{stem}.action.hdf5", root / f"{stem}.action.alice", root / "dataset.json"


def _pinch_value(transforms: dict[str, np.ndarray], side: str) -> np.ndarray:
    thumb = transforms[f"{side}ThumbTip"][:, :3, 3]
    index = transforms[f"{side}IndexFingerTip"][:, :3, 3]
    wrist = transforms[f"{side}Hand"][:, :3, 3]
    palm = transforms[f"{side}MiddleFingerKnuckle"][:, :3, 3]
    normalized = np.linalg.norm(thumb - index, axis=1) / np.maximum(np.linalg.norm(wrist - palm, axis=1), 1e-5)
    grip = np.clip((0.75 - normalized) / 0.50, 0.0, 1.0).astype(np.float32)
    return median_filter(grip, size=5, mode="nearest").astype(np.float32)


def _pose10(transform: np.ndarray, grip: np.ndarray) -> np.ndarray:
    return np.concatenate((transform[:, :3, 3], _rot6d(transform), grip[:, None]), axis=1).astype(np.float32)


def _delta7(current: np.ndarray, target: np.ndarray, target_grip: np.ndarray) -> np.ndarray:
    translation = target[:, :3, 3] - current[:, :3, 3]
    relative_rotation = np.swapaxes(current[:, :3, :3], 1, 2) @ target[:, :3, :3]
    rotation_vector = Rotation.from_matrix(relative_rotation).as_rotvec().astype(np.float32)
    return np.concatenate((translation, rotation_vector, target_grip[:, None]), axis=1).astype(np.float32)


def _relative_pair(
    current: np.ndarray,
    target: np.ndarray,
    camera_current: np.ndarray,
    coordinate_frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    if coordinate_frame == "world":
        return current, target
    reference_inverse = np.linalg.inv(camera_current)
    return reference_inverse @ current, reference_inverse @ target


def build_action_arrays(
    transforms: dict[str, np.ndarray],
    profile: dict,
    source_hand: str,
    coordinate_frame: str,
    horizon_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    count = int(transforms["camera"].shape[0])
    horizon = int(horizon_frames)
    if count <= horizon:
        raise ValueError(f"Episode 只有 {count} 帧，无法生成未来 {horizon} 帧 Action")
    camera_current = transforms["camera"][:-horizon]
    grip = {side: _pinch_value(transforms, side) for side in ("left", "right")}
    current_pose: dict[str, np.ndarray] = {}
    target_pose: dict[str, np.ndarray] = {}
    observation_parts: list[np.ndarray] = []
    for side in ("left", "right"):
        current_pose[side], target_pose[side] = _relative_pair(
            transforms[f"{side}Hand"][:-horizon],
            transforms[f"{side}Hand"][horizon:],
            camera_current,
            coordinate_frame,
        )
        observation_parts.append(_pose10(current_pose[side], grip[side][:-horizon]))
    observation = np.concatenate(observation_parts, axis=1).astype(np.float32)
    if profile["representation"] == "absolute_pose_rot6d":
        action = np.concatenate([
            _pose10(target_pose["left"], grip["left"][horizon:]),
            _pose10(target_pose["right"], grip["right"][horizon:]),
        ], axis=1).astype(np.float32)
    elif int(profile["sides"]) == 2:
        action = np.concatenate([
            _delta7(current_pose["left"], target_pose["left"], grip["left"][horizon:]),
            _delta7(current_pose["right"], target_pose["right"], grip["right"][horizon:]),
        ], axis=1).astype(np.float32)
    else:
        action = _delta7(current_pose[source_hand], target_pose[source_hand], grip[source_hand][horizon:])
    if observation.shape != (count - horizon, 20):
        raise RuntimeError(f"Observation shape invalid: {observation.shape}")
    if action.shape != (count - horizon, int(profile["action_dim"])):
        raise RuntimeError(f"Action shape invalid: {action.shape}")
    if not np.isfinite(observation).all() or not np.isfinite(action).all():
        raise RuntimeError("Action 映射包含 NaN 或 Inf")
    return observation, action, {"left": grip["left"], "right": grip["right"]}


def _load_episode_transforms(
    manifest: dict,
    episode: dict,
    reference_media_file_id: str | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, Path, str, int]:
    source_path, source_relative, source_count = _find_transform_source(manifest, episode)
    from .projection_correction import active_projection_source

    applied = active_projection_source(manifest, episode)
    projection_timeline = bool(applied and Path(applied["path"]).resolve() == source_path.resolve())
    selected_media = next(
        (
            item
            for item in episode.get("media_streams") or []
            if str(item.get("file_id") or "") == str(reference_media_file_id or "")
        ),
        None,
    )
    if selected_media is None and str(reference_media_file_id or "") == str(episode.get("primary_media_file_id") or ""):
        selected_media = episode
    video_count = source_count if projection_timeline else int(
        (selected_media or {}).get("source_frame_count")
        or (selected_media or {}).get("frame_count")
        or episode.get("frame_count")
        or source_count
    )
    rows = np.arange(source_count, dtype=np.int64) if projection_timeline else _aligned_rows(
        manifest,
        episode,
        source_relative,
        source_count,
        video_count,
        reference_media_file_id=reference_media_file_id,
    )
    required = [
        "camera",
        "leftHand", "leftThumbTip", "leftIndexFingerTip", "leftMiddleFingerKnuckle",
        "rightHand", "rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerKnuckle",
    ]
    with h5py.File(source_path, "r") as source:
        group = source.get("transforms")
        if not isinstance(group, h5py.Group):
            raise ValueError("HDF5 缺少 transforms 组")
        missing = [name for name in required if name not in group]
        if missing:
            raise ValueError(f"HDF5 缺少 Action 所需腕部/手指/相机字段: {', '.join(missing)}")
        transforms = {name: _take_rows(group[name], rows).astype(np.float32) for name in required}
    if any(value.shape != (video_count, 4, 4) for value in transforms.values()):
        raise ValueError("Action 源变换不是统一的 T×4×4")
    if any(not np.isfinite(value).all() for value in transforms.values()):
        raise ValueError("Action 源变换包含 NaN 或 Inf")
    return transforms, rows, source_path, source_relative, source_count


def _reference_media_timing(
    episode: dict,
    reference_media_file_id: str | None = None,
) -> tuple[str | None, float, int]:
    requested = str(reference_media_file_id or "").strip()
    selected = next(
        (
            item
            for item in episode.get("media_streams") or []
            if requested and str(item.get("file_id") or "") == requested
        ),
        None,
    )
    if selected is None and requested == str(episode.get("primary_media_file_id") or ""):
        selected = episode
    selected = selected or episode
    return (
        str(selected.get("file_id") or requested or episode.get("primary_media_file_id") or "") or None,
        float(selected.get("fps") or episode.get("fps") or 30.0),
        int(
            selected.get("source_frame_count")
            or selected.get("frame_count")
            or episode.get("frame_count")
            or 0
        ),
    )


def _write_hdf5_atomic(
    path: Path,
    observation: np.ndarray,
    action: np.ndarray,
    source_rows: np.ndarray,
    grips: dict[str, np.ndarray],
    profile: dict,
    request: ActionMappingRequest,
    manifest: dict,
    episode: dict,
    source_relative: str,
    reference_media_file_id: str | None,
    reference_fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    horizon = int(request.horizon_frames)
    count = int(action.shape[0])
    fps = float(reference_fps or episode.get("fps") or 30.0)
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs.update({
                "schema": ACTION_MAPPING_SCHEMA,
                "dataset_id": str(manifest["id"]),
                "episode_id": str(episode["id"]),
                "episode_name": str(episode.get("name") or episode["id"]),
                "profile_id": str(profile["id"]),
                "robot_family": str(profile["robot_family"]),
                "control_space": str(profile["control_space"]),
                "representation": str(profile["representation"]),
                "coordinate_frame": request.coordinate_frame,
                "coordinate_semantics": "current_camera_reference" if request.coordinate_frame == "camera" else "source_world_reference",
                "source_hand": request.source_hand,
                "source_hdf5": source_relative,
                "reference_media_file_id": str(reference_media_file_id or ""),
                "horizon_frames": horizon,
                "fps": fps,
                "action_dim": int(profile["action_dim"]),
                "requires_calibration": bool(profile["requires_calibration"]),
                "requires_ik": bool(profile["requires_ik"]),
                "observation_fields": json.dumps(OBSERVATION_FIELDS, ensure_ascii=False),
                "action_fields": json.dumps(profile["fields"], ensure_ascii=False),
                "created_at": _utc_now(),
            })
            output.create_dataset("observation/state", data=observation, compression="lzf", shuffle=True)
            output.create_dataset("action", data=action, compression="lzf", shuffle=True)
            output.create_dataset("frame_index", data=np.arange(count, dtype=np.int64), compression="lzf")
            output.create_dataset("target_frame_index", data=np.arange(horizon, horizon + count, dtype=np.int64), compression="lzf")
            output.create_dataset("source_hdf5_row", data=source_rows[:-horizon], compression="lzf")
            output.create_dataset("target_hdf5_row", data=source_rows[horizon:], compression="lzf")
            output.create_dataset("timestamp", data=np.arange(count, dtype=np.float64) / max(0.01, fps), compression="lzf")
            output.create_dataset("gripper/left", data=grips["left"][:-horizon], compression="lzf")
            output.create_dataset("gripper/right", data=grips["right"][:-horizon], compression="lzf")
        with h5py.File(temporary, "r") as check:
            if check["observation/state"].shape != observation.shape or check["action"].shape != action.shape:
                raise RuntimeError("Action HDF5 写入校验失败")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _update_action_index(index_path: Path, manifest: dict, profile: dict, report: dict) -> None:
    with _INDEX_LOCK:
        try:
            existing = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            existing = {}
        by_episode = {
            str(item.get("episode_id")): item
            for item in existing.get("episodes") or []
            if isinstance(item, dict) and item.get("episode_id")
        }
        by_episode[str(report["episode_id"])] = {
            "episode_id": report["episode_id"],
            "episode_name": report["episode_name"],
            "action_count": report["summary"]["action_count"],
            "action_dim": report["summary"]["action_dim"],
            "coordinate_frame": report["config"]["coordinate_frame"],
            "horizon_frames": report["config"]["horizon_frames"],
            "source_hand": report["config"]["source_hand"],
            "artifact_path": report["artifact_path"],
            "report_path": report["report_path"],
            "config_signature": report["config_signature"],
        }
        episodes = sorted(by_episode.values(), key=lambda item: (str(item.get("episode_name") or ""), str(item.get("episode_id") or "")))
        _write_json_atomic(index_path, {
            "schema": ACTION_INDEX_SCHEMA,
            "dataset_id": manifest["id"],
            "source_root": manifest.get("root_path"),
            "profile": deepcopy(profile),
            "updated_at": _utc_now(),
            "episode_count": len(episodes),
            "action_row_count": sum(int(item.get("action_count") or 0) for item in episodes),
            "episodes": episodes,
        })


def generate_episode_action(
    dataset_id: str,
    manifest: dict,
    episode: dict,
    request: ActionMappingRequest,
    reference_media_file_id: str | None = None,
) -> dict:
    profile = _profile(request.profile_id)
    artifact_path, report_path, index_path = _action_paths(dataset_id, request.profile_id, str(episode["id"]))
    selected_media_id, reference_fps, _ = _reference_media_timing(episode, reference_media_file_id)
    transforms, source_rows, source_path, source_relative, source_count = _load_episode_transforms(
        manifest,
        episode,
        reference_media_file_id=selected_media_id,
    )
    source_signature = _source_signature(source_path, source_relative, source_count, int(transforms["camera"].shape[0]))
    config = _config_payload(request)
    config["reference_media_file_id"] = selected_media_id
    config["reference_fps"] = reference_fps
    config_signature = _digest({"config": config, "source": source_signature, "schema": ACTION_MAPPING_SCHEMA})
    if not request.force and report_path.is_file() and artifact_path.is_file():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if existing.get("config_signature") == config_signature and existing.get("schema") == ACTION_REPORT_SCHEMA:
                reused = {**existing, "reused": True}
                _update_action_index(index_path, manifest, profile, reused)
                return reused
        except (OSError, json.JSONDecodeError):
            pass
    observation, action, grips = build_action_arrays(
        transforms,
        profile,
        request.source_hand,
        request.coordinate_frame,
        request.horizon_frames,
    )
    _write_hdf5_atomic(
        artifact_path,
        observation,
        action,
        source_rows,
        grips,
        profile,
        request,
        manifest,
        episode,
        source_relative,
        selected_media_id,
        reference_fps,
    )
    warnings = ["人手轨迹已转换为机器人末端代理 Action；部署前必须完成尺度、轴向、基座和工作空间标定。"]
    if profile["requires_ik"]:
        warnings.append("该品牌机器人的原生关节 Action 尚未伪造；必须加载对应 URDF、关节限位和 IK 后再转换。")
    report = {
        "schema": ACTION_REPORT_SCHEMA,
        "dataset_id": manifest["id"],
        "episode_id": episode["id"],
        "episode_name": episode.get("name") or episode["id"],
        "created_at": _utc_now(),
        "source_policy": "源数据保持只读；Action 派生文件仅写入 .alicePD/actions。",
        "profile": deepcopy(profile),
        "config": config,
        "source": source_signature,
        "config_signature": config_signature,
        "artifact_path": str(artifact_path),
        "report_path": str(report_path),
        "index_path": str(index_path),
        "summary": {
            "source_frame_count": int(transforms["camera"].shape[0]),
            "action_count": int(action.shape[0]),
            "observation_dim": int(observation.shape[1]),
            "action_dim": int(action.shape[1]),
            "fps": reference_fps,
            "finite": True,
            "left_grip_mean": round(float(grips["left"].mean()), 6),
            "right_grip_mean": round(float(grips["right"].mean()), 6),
        },
        "warnings": warnings,
        "reused": False,
    }
    _write_json_atomic(report_path, report)
    _update_action_index(index_path, manifest, profile, report)
    return report


def load_episode_action_mapping(
    dataset_id: str,
    episode_id: str,
    profile_id: str | None = None,
    manifest: dict | None = None,
) -> dict | None:
    manifest = manifest or get_manifest(dataset_id)
    sidecar_path = manifest.get("sidecar_path")
    if not sidecar_path and not manifest.get("root_path"):
        return None
    root = Path(sidecar_path or dataset_sidecar_root(manifest["root_path"], manifest["id"])) / "actions"
    stem = storage_slug(episode_id)
    if profile_id is not None:
        _profile(profile_id)
        paths = [root / profile_id / f"{stem}.action.alice"]
    else:
        paths = list(root.glob(f"*/{stem}.action.alice")) if root.is_dir() else []
    candidates: list[tuple[int, dict]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") == ACTION_REPORT_SCHEMA and str(payload.get("episode_id")) == str(episode_id):
                candidates.append((path.stat().st_mtime_ns, payload))
        except (OSError, json.JSONDecodeError):
            continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def validate_episode_action_mapping(
    dataset_id: str,
    manifest: dict,
    episode: dict,
    frame_count: int | None = None,
) -> dict | None:
    report = load_episode_action_mapping(dataset_id, str(episode["id"]), manifest=manifest)
    if report is None:
        return None
    target_frame_count = int(frame_count if frame_count is not None else episode.get("frame_count") or 0)
    invalid = np.zeros(max(0, target_frame_count), dtype=bool)
    config = report.get("config") or {}
    profile_id = str(config.get("profile_id") or (report.get("profile") or {}).get("id") or "")
    try:
        profile = _profile(profile_id)
        request = ActionMappingRequest(
            episode_ids=[str(episode["id"])],
            profile_id=profile_id,
            source_hand=str(config.get("source_hand") or "right"),
            coordinate_frame=str(config.get("coordinate_frame") or "camera"),
            horizon_frames=int(config.get("horizon_frames") or 1),
        )
        reference_media_file_id = str(config.get("reference_media_file_id") or "") or None
        _, reference_fps, _ = _reference_media_timing(episode, reference_media_file_id)
        transforms, source_rows, source_path, source_relative, source_count = _load_episode_transforms(
            manifest,
            episode,
            reference_media_file_id=reference_media_file_id,
        )
        expected_observation, expected_action, _ = build_action_arrays(
            transforms,
            profile,
            request.source_hand,
            request.coordinate_frame,
            request.horizon_frames,
        )
        artifact_path, _, _ = _action_paths(dataset_id, profile_id, str(episode["id"]))
        if not artifact_path.is_file():
            raise ValueError("Action 派生文件不存在")
        with h5py.File(artifact_path, "r") as artifact:
            actual_observation = np.asarray(artifact["observation/state"][:], dtype=np.float32)
            actual_action = np.asarray(artifact["action"][:], dtype=np.float32)
            actual_frames = np.asarray(artifact["frame_index"][:], dtype=np.int64)
            actual_targets = np.asarray(artifact["target_frame_index"][:], dtype=np.int64)
            actual_source_rows = np.asarray(artifact["source_hdf5_row"][:], dtype=np.int64)
            actual_target_rows = np.asarray(artifact["target_hdf5_row"][:], dtype=np.int64)
            actual_timestamps = np.asarray(artifact["timestamp"][:], dtype=np.float64)
            attribute_match = all([
                str(artifact.attrs.get("profile_id") or "") == profile_id,
                str(artifact.attrs.get("representation") or "") == str(profile["representation"]),
                str(artifact.attrs.get("coordinate_frame") or "") == request.coordinate_frame,
                str(artifact.attrs.get("source_hand") or "") == request.source_hand,
                int(artifact.attrs.get("horizon_frames") or 0) == request.horizon_frames,
                str(artifact.attrs.get("reference_media_file_id") or "") == str(reference_media_file_id or ""),
                math.isclose(float(artifact.attrs.get("fps") or 0.0), reference_fps, rel_tol=1e-9, abs_tol=1e-9),
            ])

        count = int(expected_action.shape[0])
        horizon = request.horizon_frames
        expected_frames = np.arange(count, dtype=np.int64)
        expected_targets = np.arange(horizon, horizon + count, dtype=np.int64)
        expected_timestamps = expected_frames.astype(np.float64) / max(0.01, reference_fps)
        shape_match = actual_observation.shape == expected_observation.shape and actual_action.shape == expected_action.shape
        index_shape_match = all(array.shape == (count,) for array in (
            actual_frames, actual_targets, actual_source_rows, actual_target_rows, actual_timestamps,
        ))
        row_ok = np.zeros(count, dtype=bool)
        max_observation_error = None
        max_action_error = None
        if shape_match and index_shape_match:
            observation_delta = np.abs(actual_observation.astype(np.float64) - expected_observation.astype(np.float64))
            action_delta = np.abs(actual_action.astype(np.float64) - expected_action.astype(np.float64))
            finite = np.isfinite(actual_observation).all(axis=1) & np.isfinite(actual_action).all(axis=1)
            row_ok = (
                finite
                & np.isclose(actual_observation, expected_observation, rtol=1e-5, atol=1e-5).all(axis=1)
                & np.isclose(actual_action, expected_action, rtol=1e-5, atol=1e-5).all(axis=1)
                & (actual_frames == expected_frames)
                & (actual_targets == expected_targets)
                & (actual_source_rows == source_rows[:-horizon])
                & (actual_target_rows == source_rows[horizon:])
                & np.isclose(actual_timestamps, expected_timestamps, rtol=1e-7, atol=1e-7)
            )
            max_observation_error = round(float(np.nanmax(observation_delta, initial=0.0)), 9)
            max_action_error = round(float(np.nanmax(action_delta, initial=0.0)), 9)

        current_signature = _source_signature(
            source_path,
            source_relative,
            source_count,
            int(transforms["camera"].shape[0]),
        )
        signature_match = current_signature == report.get("source")
        if not attribute_match or not signature_match:
            row_ok[:] = False
        bad_rows = np.flatnonzero(~row_ok)
        for row in bad_rows:
            current_frame = int(expected_frames[row])
            target_frame = int(expected_targets[row])
            if 0 <= current_frame < invalid.size:
                invalid[current_frame] = True
            if 0 <= target_frame < invalid.size:
                invalid[target_frame] = True
        if not shape_match or not index_shape_match:
            invalid[:] = True

        mismatch_count = int((~row_ok).sum())
        return {
            "verdict": "pass" if mismatch_count == 0 else "reject_candidate",
            "invalid_mask": invalid,
            "source": "generated_action",
            "profile_id": profile_id,
            "representation": profile["representation"],
            "coordinate_frame": request.coordinate_frame,
            "source_hand": request.source_hand,
            "required_sides": [request.source_hand] if int(profile["sides"]) == 1 else ["left", "right"],
            "horizon_frames": horizon,
            "checked_row_count": count,
            "mismatch_row_count": mismatch_count,
            "shape_match": shape_match,
            "index_shape_match": index_shape_match,
            "attribute_match": attribute_match,
            "source_signature_match": signature_match,
            "max_observation_error": max_observation_error,
            "max_action_error": max_action_error,
            "artifact_path": str(artifact_path),
        }
    except (OSError, KeyError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        invalid[:] = True
        profile = report.get("profile") or {}
        sides = [str(config.get("source_hand") or "right")] if int(profile.get("sides") or 0) == 1 else ["left", "right"]
        return {
            "verdict": "reject_candidate",
            "invalid_mask": invalid,
            "source": "generated_action",
            "profile_id": profile_id,
            "required_sides": sides,
            "checked_row_count": 0,
            "mismatch_row_count": target_frame_count,
            "error": str(exc)[:240],
            "artifact_path": str(report.get("artifact_path") or ""),
        }


class ActionMappingJobManager(CancellableJobMixin):
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="alice-action-mapping")
        self._jobs: dict[str, dict] = {}
        self._reservations: dict[tuple[str, str, str], str] = {}
        self._lock = threading.RLock()
        self._init_cancellation()

    def submit(self, dataset_id: str, request: ActionMappingRequest) -> dict:
        manifest = get_manifest(dataset_id)
        profile = _profile(request.profile_id)
        episodes = {str(item["id"]): item for item in manifest.get("episodes") or []}
        episode_ids = list(dict.fromkeys(map(str, request.episode_ids)))
        if not episode_ids:
            raise ValueError("至少选择一个 Episode")
        missing = [episode_id for episode_id in episode_ids if episode_id not in episodes]
        if missing:
            raise KeyError(missing[0])
        job_id = uuid.uuid4().hex
        with self._lock:
            conflicts = [
                episode_id for episode_id in episode_ids
                if (dataset_id, episode_id, request.profile_id) in self._reservations
            ]
            if conflicts:
                raise RuntimeError(f"{episodes[conflicts[0]]['name']} 的同类 Action 映射任务已在运行")
            job = {
                "id": job_id,
                "kind": "action_mapping",
                "operation": "action_mapping",
                "dataset_id": dataset_id,
                "profile_id": request.profile_id,
                "profile_name": profile["name"],
                "status": "queued",
                "progress": 0,
                "message": f"已提交 Action 生成 · {len(episode_ids)} Episodes",
                "episode_count": len(episode_ids),
                "completed_count": 0,
                "result": None,
                "error": None,
            }
            self._jobs[job_id] = job
            self._register_cancellation(job_id)
            for episode_id in episode_ids:
                self._reservations[(dataset_id, episode_id, request.profile_id)] = job_id
        self._executor.submit(self._run, job_id, dataset_id, episode_ids, request)
        return deepcopy(job)

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return deepcopy(self._jobs[job_id])

    def list(self, dataset_id: str, active_only: bool = False) -> list[dict]:
        with self._lock:
            items = [deepcopy(item) for item in self._jobs.values() if item.get("dataset_id") == dataset_id]
        if active_only:
            items = [item for item in items if item.get("status") in {"queued", "running", "cancelling"}]
        return items

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def _on_cancelled_before_run(self, job_id: str) -> None:
        with self._lock:
            for key, owner in list(self._reservations.items()):
                if owner == job_id:
                    self._reservations.pop(key, None)
            self._forget_cancellation(job_id)

    def _run(self, job_id: str, dataset_id: str, episode_ids: list[str], request: ActionMappingRequest) -> None:
        results: list[dict] = []
        failures: list[dict] = []
        try:
            manifest = get_manifest(dataset_id)
            episodes = {str(item["id"]): item for item in manifest.get("episodes") or []}
            total = len(episode_ids)
            self._start_unless_cancelled(job_id, status="running", progress=1, message=f"正在生成 Action · 0/{total}")
            for position, episode_id in enumerate(episode_ids):
                self._raise_if_cancelled(job_id)
                episode = episodes[episode_id]
                self._update(job_id, message=f"{episode['name']} · Action 映射 {position + 1}/{total}")
                try:
                    self._update(job_id, message=f"{episode['name']} · T0 时间同步 {position + 1}/{total}")
                    ensure_episode_time_sync(manifest, episode)
                    report = generate_episode_action(dataset_id, manifest, episode, request)
                    results.append({
                        "episode_id": episode_id,
                        "episode_name": episode.get("name"),
                        "status": "skipped" if report.get("reused") else "completed",
                        "reused": bool(report.get("reused")),
                        "action_count": report.get("summary", {}).get("action_count", 0),
                        "action_dim": report.get("summary", {}).get("action_dim", 0),
                        "artifact_path": report.get("artifact_path"),
                        "report_path": report.get("report_path"),
                    })
                except JobCancelled:
                    raise
                except Exception as exc:
                    failures.append({"episode_id": episode_id, "episode_name": episode.get("name"), "error": str(exc)})
                self._raise_if_cancelled(job_id)
                self._update(
                    job_id,
                    completed_count=position + 1,
                    progress=round((position + 1) / total * 100, 1),
                    message=f"正在生成 Action · {position + 1}/{total}",
                )
            result = {
                "dataset_id": dataset_id,
                "operation": "action_mapping",
                "profile_id": request.profile_id,
                "profile": deepcopy(_profile(request.profile_id)),
                "episode_count": total,
                "completed_count": len(results),
                "skipped_count": sum(item.get("status") == "skipped" for item in results),
                "failure_count": len(failures),
                "action_row_count": sum(int(item.get("action_count") or 0) for item in results),
                "items": results,
                "failures": failures,
                "output_root": str(dataset_artifact_dir(dataset_id, "actions") / request.profile_id),
            }
            if failures and not results:
                self._update(job_id, status="failed", progress=100, message=f"全部 {total} 个 Episode Action 生成失败", result=result, error=failures[0]["error"])
            else:
                message = f"Action 生成完成 · {len(results)}/{total}"
                if failures:
                    message += f" · {len(failures)} 个失败"
                self._update(job_id, status="complete", progress=100, message=message, result=result, error=None)
        except JobCancelled:
            self._mark_cancelled(job_id)
        except Exception as exc:
            self._update(job_id, status="failed", progress=100, message=str(exc), error=str(exc), trace=traceback.format_exc(limit=8))
        finally:
            with self._lock:
                for episode_id in episode_ids:
                    key = (dataset_id, episode_id, request.profile_id)
                    if self._reservations.get(key) == job_id:
                        self._reservations.pop(key, None)
            self._forget_cancellation(job_id)


action_mapping_jobs = ActionMappingJobManager()
