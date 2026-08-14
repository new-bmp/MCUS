from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .dataset_modes import dataset_mode
from .egodex_mano import (
    EGODEX_MANO_REVISION,
    direct_mano21_transforms,
    egodex_mano_source_names,
    fit_egodex_mano_template,
    retarget_egodex_mano_frame,
    source_is_retargeted,
)
from .full_export import _find_transform_source
from .mano21 import side_hand_joint_names
from .sensor_alignment import aligned_sensor_positions
from .temporal_resampling import sample_hdf5_numeric, sample_hdf5_transforms
from .video_smoothing import target_stabilization_matrices


HAND_JOINT_NAMES = {
    side: side_hand_joint_names(side)
    for side in ("left", "right")
}

EXTERNAL_HAND_VISIBILITY_BACKENDS = {
    "nexus_calibrated_tracking_v1",
    "openxr_calibrated_base_v1",
}

_EXTERNAL_PROJECTION_CONTRACTS = {
    "nexus_multimodal": {
        "backend": "nexus_calibrated_tracking_v1",
        "source_space": "mocap_tracking",
        "transform_key": "T_rgb__mocap_tracking",
        "extraction": "nexus_dexweaveg1_20_to_mano21",
        "geometry_revision": "nexus20_to_mano21",
    },
    "openxr": {
        "backend": "openxr_calibrated_base_v1",
        "source_space": "openxr_base_space",
        "transform_key": "T_rgb__openxr_base",
        "extraction": "openxr_hand_26_to_mano21",
        "geometry_revision": "openxr26_to_mano21",
    },
}


def _classify_mano_visibility(side_joint_visibility: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pass, review and reject masks for required MANO hands.

    Each hand is graded independently so a fully visible hand cannot dilute a
    missing required hand.  Exactly 60% invisible remains review; only more
    than 60% invisible (or no visible point) is rejected.
    """
    if not side_joint_visibility:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty
    all_visible_masks: list[np.ndarray] = []
    review_masks: list[np.ndarray] = []
    invalid_masks: list[np.ndarray] = []
    for side, values in side_joint_visibility.items():
        visible_joint_count = np.asarray(values, dtype=np.int64)
        joint_count = len(HAND_JOINT_NAMES[side])
        all_visible = visible_joint_count == joint_count
        invalid = (visible_joint_count == 0) | (visible_joint_count < joint_count * 0.4)
        review = (visible_joint_count > 0) & ~all_visible & ~invalid
        all_visible_masks.append(all_visible)
        review_masks.append(review)
        invalid_masks.append(invalid)
    invalid = np.logical_or.reduce(invalid_masks)
    review = np.logical_or.reduce(review_masks) & ~invalid
    passed = np.logical_and.reduce(all_visible_masks) & ~review & ~invalid
    return passed, review, invalid


def _empty_result(frame_count: int, message: str, required_sides: list[str]) -> dict:
    return {
        "available": False,
        "message": message,
        "invalid_mask": np.zeros(max(0, frame_count), dtype=bool),
        "review_mask": np.zeros(max(0, frame_count), dtype=bool),
        "metrics": {
            "available": False,
            "required_sides": required_sides,
            "message": message,
        },
    }


def _finite_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != shape or not np.isfinite(matrix).all():
        return None
    return matrix


def external_hand_projection_calibration(
    manifest: dict,
    media: dict | None = None,
) -> tuple[dict | None, str | None]:
    """Validate the explicit tracking/base-space to RGB projection contract.

    RGB/depth camera presets are intentionally ignored here.  Nexus and
    OpenXR C3 require a family-specific hand-space transform, its direction
    and unit, RGB intrinsics, and a declared target media identity.
    """
    mode = dataset_mode(manifest)
    contract = _EXTERNAL_PROJECTION_CONTRACTS.get(str(mode["family"]))
    if contract is None:
        return None, f"数据集模式 {mode['family']} 不使用外部手部投影标定"

    calibration = manifest.get("camera_calibration") or {}
    projection = calibration.get("hand_projection") or {}
    if not isinstance(projection, dict):
        return None, "camera_calibration.hand_projection 必须是对象"
    if not bool(calibration.get("source_extrinsics_applied")) or not bool(projection.get("applied")):
        return None, "手部空间到 RGB 的外参尚未明确应用"
    if str(projection.get("source_space") or "") != contract["source_space"]:
        return None, f"外参 source_space 必须为 {contract['source_space']}"
    if str(projection.get("target_space") or "") not in {"rgb_camera", "head_rgb_camera", "rgb_optical"}:
        return None, "外参 target_space 必须明确指向 RGB 相机"
    if str(projection.get("transform_direction") or "") != "source_to_rgb_camera":
        return None, "外参方向必须明确为 source_to_rgb_camera"
    if str(projection.get("unit") or "").casefold() not in {"m", "meter", "meters"}:
        return None, "手部坐标和外参平移单位必须明确为米"

    transform = _finite_matrix(projection.get(contract["transform_key"]), (4, 4))
    if transform is None or not np.allclose(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        return None, f"缺少有效的 4x4 {contract['transform_key']}"

    intrinsics_payload = projection.get("intrinsics") or {}
    if not isinstance(intrinsics_payload, dict):
        return None, "hand_projection.intrinsics 必须是对象"
    intrinsics = _finite_matrix(
        intrinsics_payload.get("K") if intrinsics_payload.get("K") is not None else intrinsics_payload.get("matrix"),
        (3, 3),
    )
    if intrinsics is None or intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        return None, "缺少有效的 RGB 3x3 相机内参"

    target_media_file_id = str(projection.get("target_media_file_id") or "").strip()
    target_stream_name = str(projection.get("target_stream_name") or "").strip()
    if not target_media_file_id and not target_stream_name:
        return None, "外参必须绑定 target_media_file_id 或 target_stream_name"
    if media is not None:
        actual_file_id = str(media.get("file_id") or "").strip()
        actual_stream_name = str(media.get("stream_name") or media.get("relative_path") or "").replace("\\", "/").strip()
        file_matches = bool(target_media_file_id and actual_file_id and target_media_file_id == actual_file_id)
        stream_matches = bool(
            target_stream_name
            and actual_stream_name
            and target_stream_name.replace("\\", "/").casefold() == actual_stream_name.casefold()
        )
        if not file_matches and not stream_matches:
            return None, "手部投影标定绑定的 RGB 媒体与当前分析视频不一致"

    declared_width = int(intrinsics_payload.get("width") or 0)
    declared_height = int(intrinsics_payload.get("height") or 0)
    if media is not None:
        actual_width = int(media.get("width") or 0)
        actual_height = int(media.get("height") or 0)
        if declared_width and actual_width and declared_width != actual_width:
            return None, "RGB 内参标定宽度与当前视频不一致"
        if declared_height and actual_height and declared_height != actual_height:
            return None, "RGB 内参标定高度与当前视频不一致"

    return {
        **contract,
        "transform": transform,
        "intrinsics": intrinsics,
        "target_media_file_id": target_media_file_id or None,
        "target_stream_name": target_stream_name or None,
        "source_relative_path": str(projection.get("source_relative_path") or "") or None,
    }, None


def _camera_intrinsics(source: h5py.File, positions: np.ndarray) -> np.ndarray | None:
    for key in ("camera/intrinsic", "camera/intrinsics", "intrinsic", "intrinsics"):
        if key not in source:
            continue
        dataset = source[key]
        value = np.asarray(dataset[()], dtype=np.float64)
        if value.shape == (3, 3):
            return np.repeat(value[None], len(positions), axis=0)
        if value.ndim == 3 and value.shape[1:] == (3, 3):
            return sample_hdf5_numeric(dataset, positions).astype(np.float64)
    return None


def _mano_points(
    transforms: h5py.Group,
    positions: np.ndarray,
    side: str,
    *,
    already_retargeted: bool,
) -> np.ndarray:
    source_names = (
        HAND_JOINT_NAMES[side]
        if already_retargeted
        else egodex_mano_source_names(transforms, side)
    )
    missing = [name for name in source_names if name not in transforms]
    if missing:
        raise KeyError(f"缺少 MANO 运动学重建所需关节：{', '.join(missing[:4])}")

    template = None if already_retargeted else fit_egodex_mano_template(transforms, side)
    sampled = {
        name: sample_hdf5_transforms(transforms[name], positions).astype(np.float64)
        for name in source_names
    }
    points: list[np.ndarray] = []
    for frame_index in range(len(positions)):
        named = {name: values[frame_index] for name, values in sampled.items()}
        matrices = (
            direct_mano21_transforms(named, side)
            if already_retargeted
            else retarget_egodex_mano_frame(named, template)
        )
        points.append(matrices[:, :3, 3])
    return np.stack(points, axis=0)


def _external_mano_points(
    bundle: dict,
    sides: list[str],
    extraction: str,
    frame_count: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    joint_values = bundle.get("joint")
    if joint_values is None:
        raise ValueError("未读取到可投影的 MANO21 关节流")
    joint_values = np.asarray(joint_values, dtype=np.float64)
    if joint_values.ndim != 2 or joint_values.shape[0] != frame_count:
        raise ValueError("MANO21 关节流与视频时间轴不一致")

    points: dict[str, np.ndarray] = {}
    source_paths: list[str] = []
    for binding in bundle.get("bindings") or []:
        if binding.get("kind") != "joint" or str(binding.get("extraction") or "") != extraction:
            continue
        side = str(binding.get("side") or "").casefold()
        if side not in sides or side in points:
            continue
        start = int(binding.get("column_start") or 0)
        end = int(binding.get("column_end") or 0)
        if end - start != 63 or start < 0 or end > joint_values.shape[1]:
            continue
        values = joint_values[:, start:end].reshape(frame_count, 21, 3)
        points[side] = values
        relative = str(binding.get("relative_path") or "")
        if relative:
            source_paths.append(relative)
    missing = [side for side in sides if side not in points]
    if missing:
        raise ValueError(f"缺少明确标注手侧的 MANO21 流：{', '.join(missing)}")
    return points, list(dict.fromkeys(source_paths))


def _project_and_classify(
    points: dict[str, np.ndarray],
    camera_points_transform: np.ndarray | None,
    intrinsics: np.ndarray,
    media: dict,
    sides: list[str],
) -> dict:
    frame_count = int(media.get("frame_count") or 0)
    if frame_count <= 0 and points:
        frame_count = int(next(iter(points.values())).shape[0])
    width = int(media.get("width") or 0)
    height = int(media.get("height") or 0)
    if intrinsics.shape == (3, 3):
        intrinsics = np.repeat(intrinsics[None], frame_count, axis=0)
    if intrinsics.shape != (frame_count, 3, 3):
        raise ValueError("RGB 相机内参与视频时间轴不一致")

    margin = max(2.0, min(width, height) * 0.005)
    pixel_transforms = target_stabilization_matrices(media, frame_count)
    side_metrics: dict[str, dict] = {}
    side_joint_visibility: dict[str, np.ndarray] = {}
    for side, source_points in points.items():
        if source_points.shape != (frame_count, 21, 3):
            raise ValueError(f"{side} MANO21 点维度无效")
        if camera_points_transform is None:
            camera_points = source_points
        else:
            homogeneous = np.concatenate((source_points, np.ones((*source_points.shape[:2], 1))), axis=2)
            camera_points = np.einsum("ij,fkj->fki", camera_points_transform, homogeneous)[..., :3]
        depth = camera_points[..., 2]
        safe_depth = np.where(np.abs(depth) > 1e-9, depth, np.nan)
        x = intrinsics[:, None, 0, 0] * camera_points[..., 0] / safe_depth + intrinsics[:, None, 0, 2]
        y = intrinsics[:, None, 1, 1] * camera_points[..., 1] / safe_depth + intrinsics[:, None, 1, 2]
        if pixel_transforms is not None:
            pixels = np.stack((x, y, np.ones_like(x)), axis=2)
            stabilized = np.einsum("fij,fkj->fki", pixel_transforms, pixels)
            safe_scale = np.where(np.abs(stabilized[..., 2]) > 1e-9, stabilized[..., 2], np.nan)
            x = stabilized[..., 0] / safe_scale
            y = stabilized[..., 1] / safe_scale
        in_frame = (
            np.isfinite(x)
            & np.isfinite(y)
            & (depth > 1e-6)
            & (x >= margin)
            & (x < width - margin)
            & (y >= margin)
            & (y < height - margin)
        )
        visible_joint_count = in_frame.sum(axis=1).astype(np.int64)
        visible = visible_joint_count == in_frame.shape[1]
        side_joint_visibility[side] = visible_joint_count
        side_metrics[side] = {
            "joint_count": len(HAND_JOINT_NAMES[side]),
            "full_visible_frame_count": int(visible.sum()),
            "full_visible_ratio": round(float(visible.mean()) if visible.size else 0.0, 6),
            "minimum_visible_joint_count": int(visible_joint_count.min(initial=len(HAND_JOINT_NAMES[side]))),
            "visible_joint_count_min": int(visible_joint_count.min(initial=len(HAND_JOINT_NAMES[side]))),
            "visible_joint_count_max": int(visible_joint_count.max(initial=0)),
            "partial_visible_frame_count": int(((visible_joint_count > 0) & (visible_joint_count < in_frame.shape[1])).sum()),
            "mostly_invisible_frame_count": int((visible_joint_count < in_frame.shape[1] * 0.4).sum()),
        }

    total_joint_count = sum(len(HAND_JOINT_NAMES[side]) for side in sides)
    visible_joint_count = np.sum(
        np.stack([side_joint_visibility[side] for side in sides], axis=0),
        axis=0,
        dtype=np.int64,
    )
    all_visible, review, invalid = _classify_mano_visibility(side_joint_visibility)
    return {
        "all_visible": all_visible,
        "review": review,
        "invalid": invalid,
        "safe_margin_pixels": round(margin, 3),
        "eis_pixel_transform_applied": pixel_transforms is not None,
        "total_joint_count": int(total_joint_count),
        "visible_joint_count": visible_joint_count,
        "sides": side_metrics,
    }


def inspect_full_hand_visibility(
    manifest: dict,
    episode: dict,
    media: dict,
    required_sides: list[str] | tuple[str, ...] = ("left", "right"),
    signal_bundle: dict | None = None,
) -> dict:
    frame_count = int(media.get("frame_count") or episode.get("frame_count") or 0)
    sides = list(dict.fromkeys(str(side) for side in required_sides if str(side) in HAND_JOINT_NAMES))
    if not sides:
        sides = ["left", "right"]
    mode = dataset_mode(manifest)
    backend = str(mode.get("hand_visibility_backend") or "")
    if backend not in {"egodex_embedded_camera_v1", *EXTERNAL_HAND_VISIBILITY_BACKENDS}:
        return _empty_result(
            frame_count,
            f"整手可见性未运行：数据集模式 {mode['family']} 没有可用的手部投影器",
            sides,
        )
    width = int(media.get("width") or episode.get("width") or 0)
    height = int(media.get("height") or episode.get("height") or 0)
    if frame_count <= 0 or width <= 0 or height <= 0:
        return _empty_result(frame_count, "视频尺寸或帧数无效，无法检查整手可见性", sides)

    if backend in EXTERNAL_HAND_VISIBILITY_BACKENDS:
        calibration, calibration_error = external_hand_projection_calibration(manifest, media)
        if calibration is None:
            return _empty_result(frame_count, f"整手可见性检查不可用：{calibration_error}", sides)
        if signal_bundle is None:
            return _empty_result(frame_count, "整手可见性检查不可用：未提供已完成 T0 对齐的 MANO21 关节流", sides)
        try:
            points, source_paths = _external_mano_points(
                signal_bundle,
                sides,
                str(calibration["extraction"]),
                frame_count,
            )
            projected = _project_and_classify(
                points,
                np.asarray(calibration["transform"], dtype=np.float64),
                np.asarray(calibration["intrinsics"], dtype=np.float64),
                media,
                sides,
            )
        except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            return _empty_result(frame_count, f"整手可见性检查不可用：{str(exc)[:180]}", sides)

        all_visible = projected["all_visible"]
        review = projected["review"]
        invalid = projected["invalid"]
        visible_joint_count = projected["visible_joint_count"]
        total_joint_count = int(projected["total_joint_count"])
        metrics = {
            "available": True,
            "source_relative_paths": source_paths,
            "required_sides": sides,
            "frame_count": frame_count,
            "hand_geometry_schema": "mano21_kinematic_positions",
            "hand_geometry_revision": calibration["geometry_revision"],
            "hand_geometry_source": calibration["extraction"],
            "dataset_mode": mode,
            "projection_backend": backend,
            "projection_source_space": calibration["source_space"],
            "projection_transform_direction": "source_to_rgb_camera",
            "calibration_source_relative_path": calibration.get("source_relative_path"),
            "safe_margin_pixels": projected["safe_margin_pixels"],
            "fractional_sensor_retiming": bool(media.get("source_frame_positions")),
            "eis_pixel_transform_applied": projected["eis_pixel_transform_applied"],
            "total_joint_count": total_joint_count,
            "all_visible_frame_count": int(all_visible.sum()),
            "all_visible_ratio": round(float(all_visible.mean()) if all_visible.size else 0.0, 6),
            "partial_visible_frame_count": int(review.sum()),
            "mostly_invisible_frame_count": int(invalid.sum()),
            "visible_joint_count_min": int(visible_joint_count.min(initial=total_joint_count)),
            "visible_joint_count_max": int(visible_joint_count.max(initial=0)),
            "visible_joint_ratio_min": round(float((visible_joint_count / max(1, total_joint_count)).min(initial=1.0)), 6),
            "visible_joint_ratio_max": round(float((visible_joint_count / max(1, total_joint_count)).max(initial=0.0)), 6),
            "required_visible_frame_count": int(all_visible.sum()),
            "invalid_frame_count": int(invalid.sum()),
            "review_frame_count": int(review.sum()),
            "required_visible_ratio": round(float(all_visible.mean()) if all_visible.size else 0.0, 6),
            "frame_state_counts": {
                "pass": int((all_visible & ~invalid & ~review).sum()),
                "review": int(review.sum()),
                "reject": int(invalid.sum()),
            },
            "sides": projected["sides"],
        }
        return {
            "available": True,
            "message": (
                f"MANO 全点可见 {metrics['all_visible_frame_count']}/{frame_count} 帧；"
                f"部分可见 {metrics['partial_visible_frame_count']} 帧；"
                f"超过 60% 点不可见 {metrics['mostly_invisible_frame_count']} 帧"
            ),
            "source_relative_path": source_paths[0] if len(source_paths) == 1 else None,
            "source_relative_paths": source_paths,
            "invalid_mask": invalid,
            "review_mask": review,
            "metrics": metrics,
        }

    try:
        # C3 is a source-quality gate that runs before projection correction.
        # Always reconstruct from the immutable EgoDex episode here; an
        # applied correction may contain inserted frames on a different
        # timeline and must not be aligned back to the raw video by index.
        source_path, source_relative, source_count = _find_transform_source(
            manifest,
            episode,
            prefer_applied=False,
        )
        positions = aligned_sensor_positions(
            manifest,
            episode,
            source_relative,
            source_count,
            frame_count,
            source_frame_positions=media.get("source_frame_positions") or None,
            reference_media_file_id=str(media.get("file_id") or "") or None,
        )
        with h5py.File(source_path, "r") as source:
            transforms = source.get("transforms")
            if not isinstance(transforms, h5py.Group):
                return _empty_result(frame_count, "HDF5 缺少 transforms，无法检查整手可见性", sides)
            intrinsics = _camera_intrinsics(source, positions)
            if intrinsics is None:
                return _empty_result(frame_count, "缺少相机内参，无法可靠判断整手是否位于画面内", sides)
            camera = sample_hdf5_transforms(transforms["camera"], positions).astype(np.float64)
            already_retargeted = source_is_retargeted(source)
            points = {
                side: _mano_points(
                    transforms,
                    positions,
                    side,
                    already_retargeted=already_retargeted,
                )
                for side in sides
            }
    except (OSError, KeyError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return _empty_result(frame_count, f"整手可见性检查不可用：{str(exc)[:180]}", sides)

    try:
        camera_inverse = np.linalg.inv(camera)
    except np.linalg.LinAlgError:
        return _empty_result(frame_count, "相机位姿不可逆，无法检查整手可见性", sides)

    camera_points_by_side: dict[str, np.ndarray] = {}
    for side, world_points in points.items():
        homogeneous = np.concatenate((world_points, np.ones((*world_points.shape[:2], 1))), axis=2)
        camera_points_by_side[side] = np.einsum("fij,fkj->fki", camera_inverse, homogeneous)[..., :3]
    projected = _project_and_classify(camera_points_by_side, None, intrinsics, media, sides)
    total_joint_count = int(projected["total_joint_count"])
    visible_joint_count = projected["visible_joint_count"]
    all_visible = projected["all_visible"]
    review = projected["review"]
    invalid = projected["invalid"]
    metrics = {
        "available": True,
        "source_relative_path": source_relative,
        "required_sides": sides,
        "frame_count": frame_count,
        "hand_geometry_schema": "mano21_kinematic_retarget",
        "hand_geometry_revision": EGODEX_MANO_REVISION,
        "hand_geometry_source": "retargeted_snapshot" if already_retargeted else "egodex_full_skeleton_fk",
        "dataset_mode": mode,
        "safe_margin_pixels": projected["safe_margin_pixels"],
        "fractional_sensor_retiming": bool(media.get("source_frame_positions")),
        "eis_pixel_transform_applied": projected["eis_pixel_transform_applied"],
        "total_joint_count": int(total_joint_count),
        "all_visible_frame_count": int(all_visible.sum()),
        "all_visible_ratio": round(float(all_visible.mean()) if all_visible.size else 0.0, 6),
        "partial_visible_frame_count": int(review.sum()),
        "mostly_invisible_frame_count": int(invalid.sum()),
        "visible_joint_count_min": int(visible_joint_count.min(initial=total_joint_count)),
        "visible_joint_count_max": int(visible_joint_count.max(initial=0)),
        "visible_joint_ratio_min": round(float((visible_joint_count / max(1, total_joint_count)).min(initial=1.0)), 6),
        "visible_joint_ratio_max": round(float((visible_joint_count / max(1, total_joint_count)).max(initial=0.0)), 6),
        "required_visible_frame_count": int(all_visible.sum()),
        "invalid_frame_count": int(invalid.sum()),
        "review_frame_count": int(review.sum()),
        "required_visible_ratio": round(float(all_visible.mean()) if all_visible.size else 0.0, 6),
        "frame_state_counts": {
            "pass": int((all_visible & ~invalid & ~review).sum()),
            "review": int(review.sum()),
            "reject": int(invalid.sum()),
        },
        "sides": projected["sides"],
    }
    return {
        "available": True,
        "message": (
            f"MANO 全点可见 {metrics['all_visible_frame_count']}/{frame_count} 帧；"
            f"部分可见 {metrics['partial_visible_frame_count']} 帧；"
            f"超过 60% 点不可见 {metrics['mostly_invisible_frame_count']} 帧"
        ),
        "source_relative_path": source_relative,
        "invalid_mask": invalid,
        "review_mask": review,
        "metrics": metrics,
    }
