from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .egodex_mano import (
    EGODEX_MANO_REVISION,
    direct_mano21_transforms,
    egodex_mano_source_names,
    fit_egodex_mano_template,
    retarget_egodex_mano_frame,
    source_is_retargeted,
)
from .full_export import _aligned_rows, _find_transform_source, _take_rows
from .mano21 import side_hand_joint_names


HAND_JOINT_NAMES = {
    side: side_hand_joint_names(side)
    for side in ("left", "right")
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


def _camera_intrinsics(source: h5py.File, rows: np.ndarray) -> np.ndarray | None:
    for key in ("camera/intrinsic", "camera/intrinsics", "intrinsic", "intrinsics"):
        if key not in source:
            continue
        dataset = source[key]
        value = np.asarray(dataset[()], dtype=np.float64)
        if value.shape == (3, 3):
            return np.repeat(value[None], len(rows), axis=0)
        if value.ndim == 3 and value.shape[1:] == (3, 3):
            return _take_rows(dataset, rows).astype(np.float64)
    return None


def _mano_points(
    transforms: h5py.Group,
    rows: np.ndarray,
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
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    unique_points: list[np.ndarray] = []
    for row in unique_rows:
        named = {
            name: np.asarray(transforms[name][int(row)], dtype=np.float64)
            for name in source_names
        }
        matrices = (
            direct_mano21_transforms(named, side)
            if already_retargeted
            else retarget_egodex_mano_frame(named, template)
        )
        unique_points.append(matrices[:, :3, 3])
    return np.stack(unique_points, axis=0)[inverse]


def inspect_full_hand_visibility(
    manifest: dict,
    episode: dict,
    media: dict,
    required_sides: list[str] | tuple[str, ...] = ("left", "right"),
) -> dict:
    frame_count = int(media.get("frame_count") or episode.get("frame_count") or 0)
    sides = list(dict.fromkeys(str(side) for side in required_sides if str(side) in HAND_JOINT_NAMES))
    if not sides:
        sides = ["left", "right"]
    width = int(media.get("width") or episode.get("width") or 0)
    height = int(media.get("height") or episode.get("height") or 0)
    if frame_count <= 0 or width <= 0 or height <= 0:
        return _empty_result(frame_count, "视频尺寸或帧数无效，无法检查整手可见性", sides)

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
        rows = _aligned_rows(manifest, episode, source_relative, source_count, frame_count)
        with h5py.File(source_path, "r") as source:
            transforms = source.get("transforms")
            if not isinstance(transforms, h5py.Group):
                return _empty_result(frame_count, "HDF5 缺少 transforms，无法检查整手可见性", sides)
            intrinsics = _camera_intrinsics(source, rows)
            if intrinsics is None:
                return _empty_result(frame_count, "缺少相机内参，无法可靠判断整手是否位于画面内", sides)
            camera = _take_rows(transforms["camera"], rows).astype(np.float64)
            already_retargeted = source_is_retargeted(source)
            points = {
                side: _mano_points(
                    transforms,
                    rows,
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

    margin = max(2.0, min(width, height) * 0.005)
    side_metrics: dict[str, dict] = {}
    side_joint_visibility: dict[str, np.ndarray] = {}
    for side, world_points in points.items():
        homogeneous = np.concatenate((world_points, np.ones((*world_points.shape[:2], 1))), axis=2)
        camera_points = np.einsum("fij,fkj->fki", camera_inverse, homogeneous)[..., :3]
        depth = camera_points[..., 2]
        safe_depth = np.where(np.abs(depth) > 1e-9, depth, np.nan)
        x = intrinsics[:, None, 0, 0] * camera_points[..., 0] / safe_depth + intrinsics[:, None, 0, 2]
        y = intrinsics[:, None, 1, 1] * camera_points[..., 1] / safe_depth + intrinsics[:, None, 1, 2]
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
    metrics = {
        "available": True,
        "source_relative_path": source_relative,
        "required_sides": sides,
        "frame_count": frame_count,
        "hand_geometry_schema": "mano21_kinematic_retarget",
        "hand_geometry_revision": EGODEX_MANO_REVISION,
        "hand_geometry_source": "retargeted_snapshot" if already_retargeted else "egodex_full_skeleton_fk",
        "safe_margin_pixels": round(margin, 3),
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
        "sides": side_metrics,
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
