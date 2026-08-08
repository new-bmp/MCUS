from __future__ import annotations

"""Apply full-frame hand observations as constrained EgoDex 3D corrections.

The source HDF5 stays read-only.  A corrected, video-aligned transform HDF5 is
staged in ``.alicePD`` and becomes active only after the normal change-review
flow marks it applied.  MediaPipe supplies image observations; it never gets to
resize fingers or directly overwrite unconstrained XYZ values.
"""

import hashlib
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .egodex_mano import (
    EGODEX_MANO_REVISION,
    MANO21_RETARGETED_ATTRIBUTE,
    MANO21_RETARGET_REVISION_ATTRIBUTE,
    direct_mano21_transforms,
    egodex_mano_source_names,
    fit_egodex_mano_template,
    has_egodex_mano_source,
    retarget_egodex_mano_frame,
    source_is_retargeted,
)
from .lerobot_export import HAND_21_JOINT_NAMES, scaled_egodex_camera_intrinsic, side_hand_joint_names
from .sensor_alignment import load_sensor_alignment, scan_episode_sensor_alignment
from .storage import change_is_applied, dataset_artifact_dir, read_frame, record_change, slugify


PROJECTION_CORRECTION_SCHEMA = "alice/projection-correction/v1"
PROJECTION_CORRECTION_KIND = "projection_correction"
PROJECTION_CORRECTION_ALGORITHM_REVISION = "egodex-mano-prior-then-model-v13"
CONFIDENCE_BLEND_SAFETY_CAP = 0.85
PROJECTION_SMOOTHING_SECONDS = 0.20
PROJECTION_EDGE_TAPER_SECONDS = 0.15
MAXIMUM_NORMALIZED_CORRECTION_SPEED_PER_SECOND = 1.0
PROJECTION_S1_SIGMA = 6.0
FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
PALM_BASE_INDICES = (1, 5, 9, 13, 17)
VISUAL_PALM_BASE_INDICES = (5, 9, 13, 17)
VISUAL_WRIST_PALM_WIDTH_RATIO = 0.80
HAND_BONES = tuple(edge for chain in FINGER_CHAINS for edge in zip(chain, chain[1:]))


def _artifact_paths(dataset_id: str, episode_id: str) -> tuple[Path, Path]:
    root = dataset_artifact_dir(dataset_id, "projection-correction")
    root.mkdir(parents=True, exist_ok=True)
    stem = slugify(episode_id)
    return root / f"{stem}.projection.alice", root / f"{stem}.projection.hdf5"


def _retimed_video_path(dataset_id: str, episode_id: str) -> Path:
    root = dataset_artifact_dir(dataset_id, "projection-correction")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{slugify(episode_id)}.projection.mp4"


def _episode_records(manifest: dict, episode: dict) -> list[dict]:
    episode_id = str(episode.get("id") or "")
    episode_key = str(episode.get("episode_key") or "")
    assignments = (manifest.get("episode_resolution") or {}).get("file_episode_assignments") or {}
    return [
        item
        for item in manifest.get("files") or []
        if str(assignments.get(str(item.get("id") or "")) or item.get("episode_id") or "") == episode_id
        or (
            not assignments.get(str(item.get("id") or ""))
            and not item.get("episode_id")
            and str(item.get("episode_key") or "") == episode_key
        )
    ]


def _raw_transform_source(manifest: dict, episode: dict) -> tuple[Path, str, int]:
    import h5py

    root = Path(str(manifest["root_path"])).expanduser().resolve()
    required = (*side_hand_joint_names("left"), *side_hand_joint_names("right"), "camera")
    for record in _episode_records(manifest, episode):
        relative = str(record.get("relative_path") or "").replace("\\", "/")
        if Path(relative).suffix.casefold() not in {".h5", ".hdf5", ".h5df"}:
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
            with h5py.File(path, "r") as source:
                transforms = source.get("transforms")
                if not isinstance(transforms, h5py.Group):
                    continue
                if any(name not in transforms for name in required):
                    continue
                counts = {int(transforms[name].shape[0]) for name in required}
                if len(counts) != 1 or any(transforms[name].shape[1:] != (4, 4) for name in required):
                    continue
                return path, relative, counts.pop()
        except (OSError, KeyError, ValueError):
            continue
    raise RuntimeError("Episode has no EgoDex left/right 21-joint transforms plus camera transform")


def _media_path(manifest: dict, media: dict) -> Path:
    declared = str(media.get("path") or "").strip()
    if declared:
        path = Path(declared).expanduser().resolve()
    else:
        path = (Path(manifest["root_path"]).expanduser().resolve() / str(media.get("relative_path") or "")).resolve()
    if not path.is_file():
        raise RuntimeError("Selected RGB video is unavailable")
    return path


def _aligned_rows(manifest: dict, episode: dict, relative: str, source_count: int, video_count: int) -> np.ndarray:
    if source_count == video_count:
        return np.arange(video_count, dtype=np.int64)
    alignment = load_sensor_alignment(manifest, str(episode["id"])) or scan_episode_sensor_alignment(manifest, episode)
    stream = next((
        item for item in alignment.get("streams") or []
        if str(item.get("relative_path") or "").replace("\\", "/").casefold() == relative.casefold()
    ), None)
    if stream is None:
        raise RuntimeError(f"Projection source cannot be aligned to video frames: {relative}")
    lookup = stream.get("frame_to_sensor_index")
    if isinstance(lookup, list) and len(lookup) >= video_count:
        rows = np.asarray(lookup[:video_count], dtype=np.int64)
    elif stream.get("mode") in {"prealigned_master_clock", "paired_frame_index"}:
        rows = np.arange(video_count, dtype=np.int64)
    elif stream.get("index_multiplier") is not None:
        rows = np.rint(np.arange(video_count) * float(stream["index_multiplier"])).astype(np.int64)
    else:
        raise RuntimeError(f"Projection alignment has no frame mapping: {relative}")
    if (rows < 0).any() or (rows >= source_count).any():
        raise RuntimeError(f"Projection alignment contains out-of-range rows: {relative}")
    return rows


def _take_rows(dataset: Any, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return np.asarray(dataset[unique.tolist()])[inverse]


def _intrinsic(handle: Any, row: int, width: int, height: int) -> np.ndarray:
    for key in ("camera/intrinsic", "camera/intrinsics", "intrinsic", "intrinsics"):
        value = handle.get(key)
        if value is None:
            continue
        intrinsic = np.asarray(value[()], dtype=np.float64)
        if intrinsic.ndim == 3:
            intrinsic = intrinsic[min(row, intrinsic.shape[0] - 1)]
        if intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all():
            return intrinsic
    return scaled_egodex_camera_intrinsic(width, height).astype(np.float64)


def _world_to_camera(points: np.ndarray, camera: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((np.asarray(points, dtype=np.float64), np.ones((len(points), 1))), axis=1)
    return (np.linalg.inv(camera) @ homogeneous.T).T[:, :3]


def _camera_to_world(points: np.ndarray, camera: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((np.asarray(points, dtype=np.float64), np.ones((len(points), 1))), axis=1)
    return (camera @ homogeneous.T).T[:, :3]


def _project_camera(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    projected = np.full((len(xyz), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-8)
    if valid.any():
        normalized = xyz[valid, :2] / xyz[valid, 2:3]
        projected[valid, 0] = intrinsic[0, 0] * normalized[:, 0] + intrinsic[0, 2]
        projected[valid, 1] = intrinsic[1, 1] * normalized[:, 1] + intrinsic[1, 2]
    return projected


def _pixel_ray(pixel: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    return np.linalg.inv(intrinsic) @ np.asarray([pixel[0], pixel[1], 1.0], dtype=np.float64)


def _ray_sphere_candidates(parent: np.ndarray, length: float, pixel: np.ndarray, intrinsic: np.ndarray) -> list[np.ndarray]:
    ray = _pixel_ray(pixel, intrinsic)
    a = float(ray @ ray)
    b = float(-2.0 * (parent @ ray))
    c = float(parent @ parent - length * length)
    discriminant = b * b - 4.0 * a * c
    if not np.isfinite(discriminant) or discriminant < 0.0:
        return []
    root = math.sqrt(max(0.0, discriminant))
    candidates = []
    for depth in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
        point = ray * depth
        if depth > 1e-8 and np.isfinite(point).all():
            candidates.append(point)
    return candidates


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=np.float64) / max(norm, 1e-12)


def _angle_between(left: np.ndarray, right: np.ndarray) -> float:
    return float(math.acos(float(np.clip(_unit(left) @ _unit(right), -1.0, 1.0))))


def _rotation_between(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    source, target = _unit(left), _unit(right)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = _unit(np.cross(source, np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])))
        return cv2.Rodrigues(axis * math.pi)[0]
    axis = cross / sine
    return cv2.Rodrigues(axis * math.atan2(sine, cosine))[0]


def _limited_rotation(rotation: np.ndarray, maximum_radians: float, blend: float = 1.0) -> np.ndarray:
    vector = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0].reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        return np.eye(3, dtype=np.float64)
    target = min(angle, maximum_radians) * max(0.0, min(1.0, blend))
    return cv2.Rodrigues(vector / angle * target)[0]


def _weighted_rigid_rotation(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1) & (weights > 0.0)
    if valid.sum() < 2:
        return np.eye(3, dtype=np.float64)
    left, right, weight = source[valid], target[valid], weights[valid]
    covariance = (left * weight[:, None]).T @ right
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return rotation


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _crop_box(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    valid = np.asarray(points, dtype=np.float64)
    valid = valid[np.isfinite(valid).all(axis=1)]
    low, high = valid.min(axis=0), valid.max(axis=0)
    center = (low + high) / 2.0
    side = min(800, int(max(460.0, float(np.max(high - low)) * 3.0)))
    side = min(side, width, height)
    x = max(0, min(width - side, int(round(center[0] - side / 2.0))))
    y = max(0, min(height - side, int(round(center[1] - side / 2.0))))
    return x, y, x + side, y + side


def _map_pose_to_frame(result: dict | None, box: tuple[int, int, int, int], flipped: bool) -> dict | None:
    if not result:
        return None
    x0, y0, x1, _ = box
    points = np.asarray(result.get("keypoints"), dtype=np.float64).copy()
    confidence = np.asarray(result.get("confidence"), dtype=np.float64).reshape(-1)
    if points.shape != (21, 2) or confidence.shape != (21,):
        return None
    if flipped:
        points[:, 0] = (x1 - x0 - 1) - points[:, 0]
    points[:, 0] += x0
    points[:, 1] += y0
    return {
        "keypoints": points,
        "confidence": confidence,
        "box_confidence": float(result.get("box_confidence") or 0.0),
        "flipped_input": bool(flipped),
    }


def _visual_wrist_anchor(points: np.ndarray) -> np.ndarray:
    """Return a palm-proportional wrist anchor without moving the real joint 0.

    EgoDex joint 0 is a rigid wrist root attached to the forearm.  It can sit
    farther behind the palm than the visual wrist emitted by MediaPipe or
    AlicePose.  Using it directly as the scale centre makes that anatomical
    offset inflate the apparent hand radius and distort every normalized
    finger correction.  The virtual anchor stays on the segment from the four
    non-thumb knuckles to the real wrist, capped by palm width, and is only a
    2D/3D measurement aid.  The source transform and forearm topology remain
    untouched.
    """
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 21:
        raise ValueError("Visual wrist anchor requires 21 points")
    bases = values[list(VISUAL_PALM_BASE_INDICES)]
    finite_bases = np.isfinite(bases).all(axis=1)
    if int(finite_bases.sum()) < 3:
        return values[0].copy() if np.isfinite(values[0]).all() else np.zeros(values.shape[1], dtype=np.float64)

    palm_center = np.median(bases[finite_bases], axis=0)
    wrist = values[0]
    if not np.isfinite(wrist).all():
        return palm_center
    pairwise = np.linalg.norm(
        bases[finite_bases, None, :] - bases[None, finite_bases, :],
        axis=2,
    )
    palm_width = float(np.max(pairwise)) if pairwise.size else 0.0
    wrist_vector = wrist - palm_center
    wrist_distance = float(np.linalg.norm(wrist_vector))
    if palm_width <= 1e-8 or wrist_distance <= 1e-8:
        return wrist.copy()
    visual_distance = min(wrist_distance, palm_width * VISUAL_WRIST_PALM_WIDTH_RATIO)
    return palm_center + wrist_vector * (visual_distance / wrist_distance)


def _projected_hand_radius(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 2):
        return 1.0
    anchor = _visual_wrist_anchor(points)
    distances = np.linalg.norm(points - anchor, axis=1)
    valid = distances[np.isfinite(distances) & (distances > 1e-6)]
    return max(1.0, float(np.percentile(valid, 75))) if valid.size else 1.0


def _normalized_pose_displacement(points: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, float]:
    """Express 2D correction in hand-radius units and constrain proportions."""
    points = np.asarray(points, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if points.shape != (21, 2) or source.shape != (21, 2):
        raise ValueError("Hand displacement normalization requires 21 source and target image points")
    radius = _projected_hand_radius(source)
    raw = (points - source) / radius
    anchor_indices = VISUAL_PALM_BASE_INDICES
    raw_anchor = np.median(raw[list(anchor_indices)], axis=0)
    global_shift = raw_anchor.copy()
    global_norm = float(np.linalg.norm(global_shift))
    if global_norm > 0.45:
        global_shift *= 0.45 / global_norm

    normalized = np.empty_like(raw)
    source_anchor = _visual_wrist_anchor(source)
    for index in range(21):
        reach = float(np.clip(np.linalg.norm(source[index] - source_anchor) / radius, 0.0, 1.25))
        local = raw[index] - raw_anchor
        local_limit = 0.10 + 0.32 * reach
        local_norm = float(np.linalg.norm(local))
        if local_norm > local_limit:
            local *= local_limit / local_norm
        combined = global_shift + local
        total_limit = 0.50 + 0.20 * reach
        combined_norm = float(np.linalg.norm(combined))
        if combined_norm > total_limit:
            combined *= total_limit / combined_norm
        normalized[index] = combined
    return normalized, radius


def _pose_candidate_score(candidate: dict, previous: dict | None = None) -> float | None:
    """Reject a confident prediction when it is spatially the other hand.

    A detector confidence describes how hand-like a detection is; it does not
    prove that the hand belongs to the requested left/right crop.  Source joint
    projection proximity, scale, projected palm orientation and short-term
    displacement continuity therefore gate confidence before it can affect 3D.
    """
    points = np.asarray(candidate.get("keypoints"), dtype=np.float64)
    source = np.asarray(candidate.get("source_pixels"), dtype=np.float64)
    confidence = np.asarray(candidate.get("confidence"), dtype=np.float64).reshape(-1)
    if points.shape != (21, 2) or source.shape != (21, 2) or confidence.shape != (21,):
        return None
    valid = np.isfinite(points).all(axis=1) & np.isfinite(source).all(axis=1) & (confidence >= 0.18)
    if int(valid.sum()) < 12:
        return None

    hand_radius = _projected_hand_radius(source)
    displacement = points - source
    residual = np.linalg.norm(displacement[valid], axis=1)
    median_residual = float(np.median(residual))
    p90_residual = float(np.percentile(residual, 90))
    visual_wrist_residual = float(np.linalg.norm(_visual_wrist_anchor(points) - _visual_wrist_anchor(source)))
    palm_valid = [index for index in PALM_BASE_INDICES if valid[index]]
    palm_residual = (
        float(np.linalg.norm(np.mean(points[palm_valid], axis=0) - np.mean(source[palm_valid], axis=0)))
        if len(palm_valid) >= 3
        else median_residual
    )

    if median_residual > max(90.0, hand_radius * 1.15):
        return None
    if p90_residual > max(150.0, hand_radius * 1.8):
        return None
    if visual_wrist_residual > max(120.0, hand_radius * 1.4):
        return None
    if palm_residual > max(100.0, hand_radius * 1.2):
        return None

    source_bones = []
    target_bones = []
    for parent, child in HAND_BONES:
        if valid[parent] and valid[child]:
            source_bones.append(float(np.linalg.norm(source[child] - source[parent])))
            target_bones.append(float(np.linalg.norm(points[child] - points[parent])))
    source_scale = float(np.median(source_bones)) if source_bones else 0.0
    target_scale = float(np.median(target_bones)) if target_bones else 0.0
    if source_scale <= 1e-6 or target_scale <= 1e-6:
        return None
    scale_ratio = target_scale / source_scale
    if not 0.45 <= scale_ratio <= 2.2:
        return None

    temporal_score = 1.0
    if previous:
        previous_points = np.asarray(previous.get("keypoints"), dtype=np.float64)
        previous_source = np.asarray(previous.get("source_pixels"), dtype=np.float64)
        previous_confidence = np.asarray(previous.get("confidence"), dtype=np.float64).reshape(-1)
        if previous_points.shape == (21, 2) and previous_source.shape == (21, 2) and previous_confidence.shape == (21,):
            common = valid & np.isfinite(previous_points).all(axis=1) & np.isfinite(previous_source).all(axis=1) & (previous_confidence >= 0.18)
            if int(common.sum()) >= 8:
                temporal_delta = np.linalg.norm(displacement[common] - (previous_points - previous_source)[common], axis=1)
                median_temporal_delta = float(np.median(temporal_delta))
                if median_temporal_delta > max(100.0, hand_radius * 1.25):
                    return None
                temporal_score = math.exp(-((median_temporal_delta / max(20.0, hand_radius * 0.65)) ** 2))

    orientation_score = 1.0
    if all(valid[index] for index in (5, 17)):
        source_wrist = _visual_wrist_anchor(source)
        target_wrist = _visual_wrist_anchor(points)
        source_index = source[5] - source_wrist
        source_little = source[17] - source_wrist
        target_index = points[5] - target_wrist
        target_little = points[17] - target_wrist
        source_orientation = float(source_index[0] * source_little[1] - source_index[1] * source_little[0])
        target_orientation = float(target_index[0] * target_little[1] - target_index[1] * target_little[0])
        orientation_area = hand_radius * hand_radius
        if abs(source_orientation) > orientation_area * 0.02 and abs(target_orientation) > orientation_area * 0.02:
            orientation_score = 1.0 if source_orientation * target_orientation > 0.0 else 0.45

    mean_confidence = float(np.mean(confidence[valid]))
    detector_score = float(candidate.get("box_confidence") or 0.0) * (0.5 + 0.5 * mean_confidence)
    proximity_scale = max(24.0, hand_radius * 0.55)
    proximity_score = math.exp(-((median_residual / proximity_scale) ** 2))
    scale_score = math.exp(-1.5 * abs(math.log(max(scale_ratio, 1e-8))))
    return detector_score * proximity_score * scale_score * temporal_score * orientation_score


def _choose_pose(original: dict | None, mirrored: dict | None, previous: dict | None = None) -> dict | None:
    scored = [
        (score, item)
        for item in (original, mirrored)
        if item and (score := _pose_candidate_score(item, previous)) is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda pair: pair[0])[1]


def _prepare_full_frame_hand_candidate(candidate: dict, width: int, height: int) -> dict | None:
    """Prepare a full-frame hand while keeping MediaPipe landmarks equal-weight.

    MediaPipe's handedness score is a hand-level cue, not a confidence value
    for each landmark.  In full-approval mode it must never attenuate the 3D
    solve (or make a numerically valid hand disappear).  Geometry checks here
    are only finite/degenerate-data guards, not confidence weighting.
    """
    points = np.asarray(candidate.get("keypoints"), dtype=np.float64)
    confidence = np.asarray(candidate.get("confidence"), dtype=np.float64).reshape(-1)
    if points.shape != (21, 2) or confidence.shape != (21,) or not np.isfinite(points).all():
        return None
    margin_x, margin_y = max(4.0, width * 0.04), max(4.0, height * 0.04)
    in_frame = (
        (points[:, 0] >= -margin_x)
        & (points[:, 0] <= width - 1 + margin_x)
        & (points[:, 1] >= -margin_y)
        & (points[:, 1] <= height - 1 + margin_y)
    )
    if int(in_frame.sum()) < 17:
        return None
    radius = _projected_hand_radius(points)
    if radius < 10.0 or radius > max(width, height) * 0.65:
        return None
    bone_lengths = np.asarray([
        np.linalg.norm(points[child] - points[parent])
        for parent, child in HAND_BONES
    ], dtype=np.float64)
    if not np.isfinite(bone_lengths).all() or float(np.median(bone_lengths)) < 2.0:
        return None
    p10, p90 = np.percentile(bone_lengths, (10, 90))
    if p10 <= 0.5 or p90 / max(p10, 1e-6) > 8.0:
        return None
    palm = points[list(PALM_BASE_INDICES)]
    palm_spread = float(np.max(np.linalg.norm(palm[:, None, :] - palm[None, :, :], axis=2)))
    if palm_spread < max(6.0, radius * 0.12):
        return None

    is_mediapipe = str(candidate.get("backend") or "").casefold() == "mediapipe"
    detector_score = float(candidate.get("box_confidence") or 0.0)
    if not math.isfinite(detector_score):
        detector_score = 0.0
    topology_score = float(np.clip((p10 / max(2.0, p90)) * 3.0, 0.45, 1.0))
    # Full approval: all 21 finite landmarks enter the 3D solve with exactly
    # the same weight, regardless of MediaPipe's hand-level score.
    effective_confidence = (
        np.ones(21, dtype=np.float64)
        if is_mediapipe
        else np.clip(confidence, 0.0, 1.0) * topology_score * np.where(in_frame, 1.0, 0.35)
    )
    prepared = dict(candidate)
    prepared["keypoints"] = points
    prepared["raw_confidence"] = np.clip(confidence, 0.0, 1.0)
    prepared["confidence"] = effective_confidence
    prepared["geometry_quality"] = 1.0 if is_mediapipe else topology_score
    prepared["full_approval"] = is_mediapipe
    prepared["projected_radius"] = radius
    prepared["in_frame_joint_count"] = int(in_frame.sum())
    return prepared


def _camera_handedness(candidate: dict) -> str | None:
    """Convert MediaPipe's selfie-view label for an unmirrored camera frame."""
    label = str(candidate.get("handedness") or "").strip().casefold()
    if label == "left":
        return "right"
    if label == "right":
        return "left"
    return None


def _full_frame_side_score(
    candidate: dict,
    source_pixels: np.ndarray,
    side: str,
    previous: dict | None = None,
) -> float | None:
    points = np.asarray(candidate.get("keypoints"), dtype=np.float64)
    source = np.asarray(source_pixels, dtype=np.float64)
    confidence = np.asarray(candidate.get("confidence"), dtype=np.float64).reshape(-1)
    if points.shape != (21, 2) or source.shape != (21, 2) or confidence.shape != (21,):
        return None
    full_approval = bool(candidate.get("full_approval")) or str(candidate.get("backend") or "").casefold() == "mediapipe"
    valid = np.isfinite(points).all(axis=1) & np.isfinite(source).all(axis=1)
    if not full_approval:
        valid &= confidence >= 0.18
    if int(valid.sum()) < 12:
        return None
    anchors = np.asarray(VISUAL_PALM_BASE_INDICES, dtype=np.int64)
    anchor_valid = anchors[valid[anchors]]
    comparison = anchor_valid if len(anchor_valid) >= 3 else np.flatnonzero(valid)
    median_residual = float(np.median(np.linalg.norm(points[comparison] - source[comparison], axis=1)))
    source_radius = _projected_hand_radius(source)
    candidate_radius = float(candidate.get("projected_radius") or _projected_hand_radius(points))
    proximity_scale = max(90.0, source_radius * 3.0, candidate_radius * 2.5)
    proximity_score = math.exp(-((median_residual / proximity_scale) ** 2))

    handedness = _camera_handedness(candidate)
    handedness_score = float(candidate.get("handedness_score") or 0.0)
    if handedness is None:
        handedness_factor = 1.0
    elif handedness == side:
        handedness_factor = 1.0 + 0.18 * max(0.0, min(1.0, handedness_score))
    else:
        # Handedness is deliberately a weak cue: egocentric exports and
        # mirrored streams do not always follow MediaPipe's selfie convention.
        handedness_factor = 1.0 - 0.18 * max(0.0, min(1.0, handedness_score))

    temporal_score = 1.0
    if previous:
        previous_points = np.asarray(previous.get("keypoints"), dtype=np.float64)
        if previous_points.shape == (21, 2) and np.isfinite(previous_points).all():
            current_center = _visual_wrist_anchor(points)
            previous_center = _visual_wrist_anchor(previous_points)
            center_delta = float(np.linalg.norm(current_center - previous_center))
            temporal_scale = max(80.0, candidate_radius * 2.5)
            temporal_score = 0.25 + 0.75 * math.exp(-((center_delta / temporal_scale) ** 2))

    detector_score = 1.0 if full_approval else float(candidate.get("box_confidence") or 0.0)
    geometry_score = 1.0 if full_approval else float(candidate.get("geometry_quality") or 0.0)
    # Keep a proximity floor so a bad source projection cannot make the true
    # full-frame detection impossible to select.  Proximity still dominates
    # which of two visible hands is assigned to each side.
    return detector_score * geometry_score * (0.18 + 0.82 * proximity_score) * handedness_factor * temporal_score


def _assign_full_frame_hands(
    candidates: list[dict],
    source_pixels: dict[str, np.ndarray],
    previous: dict[str, dict | None],
    width: int,
    height: int,
) -> dict[str, dict | None]:
    prepared = [item for candidate in candidates if (item := _prepare_full_frame_hand_candidate(candidate, width, height)) is not None]
    if not prepared:
        return {"left": None, "right": None}
    scores = {
        side: [
            _full_frame_side_score(candidate, source_pixels[side], side, previous.get(side))
            for candidate in prepared
        ]
        for side in ("left", "right")
    }
    choices: list[tuple[float, int | None, int | None]] = []
    indices: list[int | None] = [None, *range(len(prepared))]
    for left_index in indices:
        for right_index in indices:
            if left_index is not None and left_index == right_index:
                continue
            left_score = scores["left"][left_index] if left_index is not None else None
            right_score = scores["right"][right_index] if right_index is not None else None
            if left_index is not None and left_score is None:
                continue
            if right_index is not None and right_score is None:
                continue
            assigned = int(left_index is not None) + int(right_index is not None)
            total = float(left_score or 0.0) + float(right_score or 0.0) + assigned * 0.015
            choices.append((total, left_index, right_index))
    if not choices:
        return {"left": None, "right": None}
    _, left_index, right_index = max(choices, key=lambda item: item[0])
    output: dict[str, dict | None] = {"left": None, "right": None}
    for side, index in (("left", left_index), ("right", right_index)):
        if index is None:
            continue
        selected = dict(prepared[index])
        selected["source_pixels"] = np.asarray(source_pixels[side], dtype=np.float64)
        selected["assigned_side"] = side
        selected["camera_handedness"] = _camera_handedness(selected)
        output[side] = selected
    return output


def _interpolate_observations(
    frame_count: int,
    samples: dict[int, tuple[np.ndarray, np.ndarray]],
    maximum_gap_frames: int,
    *,
    smoothing_frames: int = 3,
    taper_frames: int = 0,
    maximum_step: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    displacement = np.zeros((frame_count, 21, 2), dtype=np.float64)
    confidence = np.zeros((frame_count, 21), dtype=np.float64)
    available = np.zeros((frame_count, 21), dtype=bool)
    if not samples:
        return displacement, confidence, available
    sample_frames = sorted(samples)
    for joint in range(21):
        valid_frames = [frame for frame in sample_frames if samples[frame][1][joint] > 0.0]
        for frame in valid_frames:
            displacement[frame, joint] = samples[frame][0][joint]
            confidence[frame, joint] = samples[frame][1][joint]
            available[frame, joint] = True
        for start, end in zip(valid_frames, valid_frames[1:]):
            if end - start > maximum_gap_frames:
                continue
            left_disp, left_conf = samples[start][0][joint], samples[start][1][joint]
            right_disp, right_conf = samples[end][0][joint], samples[end][1][joint]
            for frame in range(start + 1, end):
                ratio = (frame - start) / max(1, end - start)
                displacement[frame, joint] = left_disp * (1.0 - ratio) + right_disp * ratio
                confidence[frame, joint] = left_conf * (1.0 - ratio) + right_conf * ratio
                available[frame, joint] = True
    filtered = displacement.copy()
    window = max(1, int(smoothing_frames))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    for joint in range(21):
        indices = np.flatnonzero(available[:, joint])
        if not len(indices):
            continue
        run_start = 0
        runs: list[np.ndarray] = []
        for offset in range(1, len(indices)):
            if indices[offset] != indices[offset - 1] + 1:
                runs.append(indices[run_start:offset])
                run_start = offset
        runs.append(indices[run_start:])
        for run in runs:
            values = displacement[run, joint]
            robust = values.copy()
            for local_index in range(len(run)):
                low, high = max(0, local_index - radius), min(len(run), local_index + radius + 1)
                robust[local_index] = np.median(values[low:high], axis=0)
            smoothed = robust.copy()
            for local_index in range(len(run)):
                low, high = max(0, local_index - radius), min(len(run), local_index + radius + 1)
                offsets = np.arange(low, high) - local_index
                weights = (radius + 1 - np.abs(offsets)).astype(np.float64)
                smoothed[local_index] = np.average(robust[low:high], axis=0, weights=weights)

            taper = min(max(0, int(taper_frames)), len(run) // 2)
            if taper and int(run[0]) > 0:
                for local_index in range(taper):
                    gain = math.sin(math.pi * 0.5 * (local_index + 1) / (taper + 1)) ** 2
                    smoothed[local_index] *= gain
            if taper and int(run[-1]) < frame_count - 1:
                for offset in range(taper):
                    gain = math.sin(math.pi * 0.5 * (offset + 1) / (taper + 1)) ** 2
                    smoothed[-offset - 1] *= gain

            if maximum_step is not None and maximum_step > 0.0 and len(run) > 1:
                forward = smoothed.copy()
                for local_index in range(1, len(run)):
                    delta = forward[local_index] - forward[local_index - 1]
                    norm = float(np.linalg.norm(delta))
                    if norm > maximum_step:
                        forward[local_index] = forward[local_index - 1] + delta * (maximum_step / norm)
                backward = smoothed.copy()
                for local_index in range(len(run) - 2, -1, -1):
                    delta = backward[local_index] - backward[local_index + 1]
                    norm = float(np.linalg.norm(delta))
                    if norm > maximum_step:
                        backward[local_index] = backward[local_index + 1] + delta * (maximum_step / norm)
                smoothed = (forward + backward) * 0.5
            filtered[run, joint] = smoothed
    return filtered, confidence, available


def _confidence_adjustment_multiplier(
    confidence: float,
    minimum_confidence: float = 0.18,
    *,
    middle_confidence: float = 0.60,
    low_multiplier: float = 0.4,
    middle_multiplier: float = 1.0,
    high_multiplier: float = 2.0,
) -> float:
    """Return a monotonic two-segment smoothstep confidence multiplier."""
    confidence = float(np.clip(confidence, 0.0, 1.0))
    low_x = float(np.clip(minimum_confidence, 0.0, 0.99))
    middle_x = float(np.clip(middle_confidence, low_x + 1e-6, 0.999999))
    low_y = max(0.0, float(low_multiplier))
    middle_y = max(low_y, float(middle_multiplier))
    high_y = max(middle_y, float(high_multiplier))
    if confidence < low_x:
        return 0.0
    if confidence <= middle_x:
        t = float(np.clip((confidence - low_x) / max(1e-9, middle_x - low_x), 0.0, 1.0))
        smooth = t * t * (3.0 - 2.0 * t)
        return low_y + (middle_y - low_y) * smooth
    t = float(np.clip((confidence - middle_x) / max(1e-9, 1.0 - middle_x), 0.0, 1.0))
    smooth = t * t * (3.0 - 2.0 * t)
    return middle_y + (high_y - middle_y) * smooth


def _confidence_scaled_blend(
    confidence: float,
    minimum_confidence: float,
    base_blend: float,
    safety_cap: float = CONFIDENCE_BLEND_SAFETY_CAP,
    *,
    middle_confidence: float = 0.60,
    low_multiplier: float = 0.4,
    middle_multiplier: float = 1.0,
    high_multiplier: float = 2.0,
) -> float:
    """Apply the smooth confidence multiplier while retaining a hard safety cap."""
    multiplier = _confidence_adjustment_multiplier(
        confidence,
        minimum_confidence,
        middle_confidence=middle_confidence,
        low_multiplier=low_multiplier,
        middle_multiplier=middle_multiplier,
        high_multiplier=high_multiplier,
    )
    if multiplier <= 0.0:
        return 0.0
    return max(0.0, min(float(safety_cap), float(base_blend) * multiplier))


def _constrained_hand_correction(
    original_camera: np.ndarray,
    original_rotations_world: np.ndarray,
    camera_transform: np.ndarray,
    intrinsic: np.ndarray,
    target_pixels: np.ndarray,
    confidence: np.ndarray,
    *,
    forearm_camera: np.ndarray | None = None,
    minimum_confidence: float = 0.18,
    maximum_root_angle_degrees: float = 14.0,
    maximum_palm_angle_degrees: float = 20.0,
    maximum_bone_angle_degrees: float = 24.0,
    local_blend: float = 0.58,
    confidence_policy: str = "scaled",
    wrist_point_source: str = "model",
    dynamic_mid_confidence: float = 0.60,
    dynamic_low_multiplier: float = 0.4,
    dynamic_mid_multiplier: float = 1.0,
    dynamic_high_multiplier: float = 2.0,
) -> dict:
    original = np.asarray(original_camera, dtype=np.float64)
    target = np.asarray(target_pixels, dtype=np.float64).copy()
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if original.shape != (21, 3) or target.shape != (21, 2) or confidence.shape != (21,):
        raise ValueError("Constrained hand correction requires 21 camera XYZ and 21 image points")
    source_pixels = _project_camera(original, intrinsic)
    wrist_point_source = str(wrist_point_source or "model").strip().casefold()
    if wrist_point_source not in {"egodex", "model"}:
        raise ValueError(f"Unsupported wrist point source: {wrist_point_source}")
    if wrist_point_source == "egodex":
        # Joint 0 remains the original rigid EgoDex wrist root.  Palm and
        # finger observations still enter the solve, while model landmark 0
        # cannot rotate or translate the forearm attachment.
        target[0] = source_pixels[0]
    policy = str(confidence_policy).casefold()
    full_approval = policy in {"full_approval_uniform", "mediapipe_full_approval", "mediapipe_full_approval_uniform"}
    uniform_adjustment = policy in {"uniform", "full_approval_uniform", "mediapipe_full_approval", "mediapipe_full_approval_uniform"}
    valid = np.isfinite(source_pixels).all(axis=1) & np.isfinite(target).all(axis=1)
    if not full_approval:
        valid &= confidence >= minimum_confidence
    if valid.sum() < 12:
        return {"applied": False, "reason": "insufficient_confident_keypoints"}
    bone_lengths = np.asarray([np.linalg.norm(original[child] - original[parent]) for parent, child in HAND_BONES])
    hand_unit = float(np.median(bone_lengths[bone_lengths > 1e-9])) if (bone_lengths > 1e-9).any() else 1.0
    maximum_depth_change = max(1e-8, hand_unit * 2.25)
    hand_pixel_radius = _projected_hand_radius(source_pixels)
    proportional_residual = float(np.median(np.linalg.norm(target[valid] - source_pixels[valid], axis=1))) / hand_pixel_radius
    angle_scale = float(np.clip(0.20 + 0.75 * proportional_residual, 0.25, 1.0))
    root_angle_limit = maximum_root_angle_degrees * angle_scale
    palm_angle_limit = maximum_palm_angle_degrees * angle_scale
    bone_angle_limit = maximum_bone_angle_degrees * angle_scale
    corrected = original.copy()
    uniform_rate = float(np.clip(local_blend, 0.0, 1.0))

    wrist_weight = (
        0.0
        if wrist_point_source == "egodex"
        else (
            (uniform_rate if valid[0] else 0.0)
            if uniform_adjustment
            else _confidence_scaled_blend(
                confidence[0],
                minimum_confidence,
                local_blend,
                middle_confidence=dynamic_mid_confidence,
                low_multiplier=dynamic_low_multiplier,
                middle_multiplier=dynamic_mid_multiplier,
                high_multiplier=dynamic_high_multiplier,
            )
        )
    )
    if wrist_point_source == "egodex":
        corrected[0] = original[0]
    elif forearm_camera is not None and np.isfinite(forearm_camera).all():
        root_vector = original[0] - forearm_camera
        root_length = float(np.linalg.norm(root_vector))
        candidates = _ray_sphere_candidates(forearm_camera, root_length, target[0], intrinsic)
        if candidates:
            candidate = min(candidates, key=lambda item: np.linalg.norm(item - original[0]))
            rotation = _rotation_between(root_vector, candidate - forearm_camera)
            rotation = _limited_rotation(rotation, math.radians(root_angle_limit), wrist_weight)
            corrected[0] = forearm_camera + rotation @ root_vector
        else:
            corrected[0] = original[0]
    else:
        ray = _pixel_ray(target[0], intrinsic)
        candidate = ray * (original[0, 2] / max(ray[2], 1e-8))
        shift = candidate - original[0]
        norm = float(np.linalg.norm(shift))
        if norm > maximum_depth_change:
            shift *= maximum_depth_change / norm
        corrected[0] = original[0] + wrist_weight * shift

    palm_source = original[list(PALM_BASE_INDICES)] - original[0]
    provisional = []
    palm_weights = []
    for index in PALM_BASE_INDICES:
        length = float(np.linalg.norm(original[index] - original[0]))
        candidates = _ray_sphere_candidates(corrected[0], length, target[index], intrinsic) if valid[index] else []
        candidate = min(candidates, key=lambda item: np.linalg.norm(item - original[index])) if candidates else corrected[0] + (original[index] - original[0])
        provisional.append(candidate - corrected[0])
        palm_weights.append(
            (1.0 if valid[index] else 0.0)
            if uniform_adjustment
            else float(np.clip(confidence[index], 0.0, 1.0))
            * _confidence_adjustment_multiplier(
                confidence[index],
                minimum_confidence,
                middle_confidence=dynamic_mid_confidence,
                low_multiplier=dynamic_low_multiplier,
                middle_multiplier=dynamic_mid_multiplier,
                high_multiplier=dynamic_high_multiplier,
            )
        )
    palm_rotation = _weighted_rigid_rotation(palm_source, np.asarray(provisional), np.asarray(palm_weights))
    palm_confidence = float(np.mean(np.clip(confidence[list(PALM_BASE_INDICES)], 0.0, 1.0)))
    palm_rotation = _limited_rotation(
        palm_rotation,
        math.radians(palm_angle_limit),
        (
            uniform_rate
            if uniform_adjustment
            else _confidence_scaled_blend(
                palm_confidence,
                minimum_confidence,
                local_blend,
                middle_confidence=dynamic_mid_confidence,
                low_multiplier=dynamic_low_multiplier,
                middle_multiplier=dynamic_mid_multiplier,
                high_multiplier=dynamic_high_multiplier,
            )
        ),
    )
    for index, vector in zip(PALM_BASE_INDICES, palm_source):
        corrected[index] = corrected[0] + palm_rotation @ vector

    parent_delta: dict[int, np.ndarray] = {0: palm_rotation}
    for chain in FINGER_CHAINS:
        for parent, child in zip(chain[1:], chain[2:]):
            original_vector = original[child] - original[parent]
            length = float(np.linalg.norm(original_vector))
            baseline_vector = parent_delta.get(parent, palm_rotation) @ original_vector
            baseline = corrected[parent] + baseline_vector
            weight = (
                (uniform_rate if valid[child] else 0.0)
                if uniform_adjustment
                else _confidence_scaled_blend(
                    confidence[child],
                    minimum_confidence,
                    local_blend,
                    middle_confidence=dynamic_mid_confidence,
                    low_multiplier=dynamic_low_multiplier,
                    middle_multiplier=dynamic_mid_multiplier,
                    high_multiplier=dynamic_high_multiplier,
                )
            )
            candidates = _ray_sphere_candidates(corrected[parent], length, target[child], intrinsic) if valid[child] else []
            candidate = min(candidates, key=lambda item: np.linalg.norm(item - baseline)) if candidates else baseline
            candidate_vector = candidate - corrected[parent]
            delta = _rotation_between(baseline_vector, candidate_vector)
            delta = _limited_rotation(delta, math.radians(bone_angle_limit), weight)
            corrected_vector = delta @ baseline_vector
            point = corrected[parent] + corrected_vector
            expected_depth = baseline[2]
            if abs(float(point[2] - expected_depth)) > maximum_depth_change:
                point = baseline
                delta = np.eye(3, dtype=np.float64)
            corrected[child] = point
            parent_delta[child] = delta @ parent_delta.get(parent, palm_rotation)

    corrected_pixels = _project_camera(corrected, intrinsic)
    before = np.linalg.norm(source_pixels[valid] - target[valid], axis=1)
    after = np.linalg.norm(corrected_pixels[valid] - target[valid], axis=1)
    before_median = float(np.median(before))
    after_median = float(np.median(after))
    length_errors = []
    for parent, child in HAND_BONES:
        expected = float(np.linalg.norm(original[child] - original[parent]))
        actual = float(np.linalg.norm(corrected[child] - corrected[parent]))
        length_errors.append(abs(actual - expected) / max(expected, 1e-12))
    palm_before = original[list(PALM_BASE_INDICES)]
    palm_after = corrected[list(PALM_BASE_INDICES)]
    palm_errors = []
    for left in range(len(PALM_BASE_INDICES)):
        for right in range(left + 1, len(PALM_BASE_INDICES)):
            expected = float(np.linalg.norm(palm_before[left] - palm_before[right]))
            actual = float(np.linalg.norm(palm_after[left] - palm_after[right]))
            palm_errors.append(abs(actual - expected) / max(expected, 1e-12))
    maximum_length_error = max(length_errors, default=0.0)
    maximum_palm_error = max(palm_errors, default=0.0)
    if not np.isfinite(corrected).all() or maximum_length_error > 1e-5 or maximum_palm_error > 1e-5:
        return {"applied": False, "reason": "kinematic_constraint_failed"}
    if after_median >= before_median * 0.97:
        return {
            "applied": False,
            "reason": "no_reprojection_improvement",
            "before_median_px": round(before_median, 4),
            "after_median_px": round(after_median, 4),
        }

    rotation_deltas_camera: list[np.ndarray] = [np.eye(3, dtype=np.float64) for _ in range(21)]
    rotation_deltas_camera[0] = palm_rotation
    children = {parent: child for parent, child in HAND_BONES if parent != 0}
    parent_lookup = {child: parent for parent, child in HAND_BONES}
    for index in range(1, 21):
        if index in children:
            child = children[index]
            rotation_deltas_camera[index] = _rotation_between(
                original[child] - original[index], corrected[child] - corrected[index],
            )
        else:
            rotation_deltas_camera[index] = rotation_deltas_camera[parent_lookup.get(index, 0)]
    camera_rotation = np.asarray(camera_transform[:3, :3], dtype=np.float64)
    corrected_rotations = np.empty_like(original_rotations_world, dtype=np.float64)
    for index, (source_rotation, delta_camera) in enumerate(zip(original_rotations_world, rotation_deltas_camera)):
        delta_world = camera_rotation @ delta_camera @ camera_rotation.T
        corrected_rotations[index] = _orthonormalize(delta_world @ source_rotation)
    return {
        "applied": True,
        "camera_points": corrected,
        "world_rotations": corrected_rotations,
        "before_median_px": round(before_median, 4),
        "after_median_px": round(after_median, 4),
        "maximum_bone_length_relative_error": float(maximum_length_error),
        "maximum_palm_distance_relative_error": float(maximum_palm_error),
        "valid_keypoint_count": int(valid.sum()),
    }


def _projection_xyz_matrix(handle: Any, names: tuple[str, ...], rows: np.ndarray | None = None) -> np.ndarray:
    transforms = handle["transforms"]
    if rows is None:
        count = min(int(transforms[name].shape[0]) for name in names)
        values = [np.asarray(transforms[name][:count, :3, 3], dtype=np.float64) for name in names]
    else:
        values = [np.asarray(_take_rows(transforms[name], rows)[:, :3, 3], dtype=np.float64) for name in names]
    return np.concatenate(values, axis=1)


def _s1_insertion_plan(frame_count: int, trigger_mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    mask = np.asarray(trigger_mask, dtype=bool).reshape(-1)
    if mask.size != frame_count:
        raise ValueError("S1 insertion mask length does not match the corrected video")
    positions: list[float] = []
    insertions: list[dict] = []
    for source_frame in range(frame_count):
        if source_frame > 0 and bool(mask[source_frame]):
            output_frame = len(positions)
            positions.append(source_frame - 0.5)
            insertions.append({
                "output_frame": output_frame,
                "left_source_frame": source_frame - 1,
                "right_source_frame": source_frame,
                "source_position": source_frame - 0.5,
            })
        positions.append(float(source_frame))
    return np.asarray(positions, dtype=np.float64), insertions


def _interpolate_numeric_rows(values: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    positions = np.asarray(source_positions, dtype=np.float64).reshape(-1)
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    alpha = positions - left
    if source.dtype.kind not in "biufc":
        return source[np.rint(positions).astype(np.int64)]
    reshape = (len(alpha), *([1] * (source.ndim - 1)))
    interpolated = source[left].astype(np.float64) * (1.0 - alpha.reshape(reshape))
    interpolated += source[right].astype(np.float64) * alpha.reshape(reshape)
    if source.dtype.kind == "b":
        return interpolated >= 0.5
    return interpolated.astype(source.dtype, copy=False)


def _interpolate_transform_rows(values: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    positions = np.asarray(source_positions, dtype=np.float64).reshape(-1)
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    alpha = positions - left
    output = source[left].copy()
    output[:, :3, 3] = (
        source[left, :3, 3] * (1.0 - alpha[:, None])
        + source[right, :3, 3] * alpha[:, None]
    )
    for output_index in np.flatnonzero((right != left) & (alpha > 0.0)):
        rotations = Rotation.from_matrix(source[[left[output_index], right[output_index]], :3, :3])
        output[output_index, :3, :3] = Slerp([0.0, 1.0], rotations)([float(alpha[output_index])]).as_matrix()[0]
    output[:, 3, :] = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    return output.astype(values.dtype, copy=False)


def _copy_retimed_hdf5_group(source: Any, target: Any, source_positions: np.ndarray, source_frame_count: int) -> None:
    import h5py

    for key, value in source.attrs.items():
        target.attrs[key] = value
    for name, value in source.items():
        if isinstance(value, h5py.Group):
            child = target.create_group(name)
            _copy_retimed_hdf5_group(value, child, source_positions, source_frame_count)
            continue
        raw = value[()]
        if value.ndim and int(value.shape[0]) == source_frame_count:
            raw = (
                _interpolate_transform_rows(raw, source_positions)
                if value.ndim == 3 and tuple(value.shape[1:]) == (4, 4)
                else _interpolate_numeric_rows(raw, source_positions)
            )
        options: dict[str, Any] = {}
        if np.asarray(raw).ndim and np.asarray(raw).size:
            if value.compression:
                options["compression"] = value.compression
                if value.compression_opts is not None:
                    options["compression_opts"] = value.compression_opts
            if value.shuffle and np.asarray(raw).dtype.kind in "biufc":
                options["shuffle"] = True
            if value.chunks:
                options["chunks"] = tuple(
                    min(int(chunk), int(size)) if int(size) > 0 else int(chunk)
                    for chunk, size in zip(value.chunks, np.asarray(raw).shape)
                )
        created = target.create_dataset(name, data=raw, dtype=value.dtype, **options)
        for key, item in value.attrs.items():
            created.attrs[key] = item


def _intermediate_hand_points(left: np.ndarray, right: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    output = left * (1.0 - alpha) + right * alpha
    wrist = output[0].copy()
    left_palm = left[list(PALM_BASE_INDICES)] - left[0]
    right_palm = right[list(PALM_BASE_INDICES)] - right[0]
    palm_delta = _weighted_rigid_rotation(left_palm, right_palm, np.ones(len(PALM_BASE_INDICES)))
    palm_half = _limited_rotation(palm_delta, math.pi, alpha)
    palm_shape = left_palm * (1.0 - alpha) + (palm_delta.T @ right_palm.T).T * alpha
    for local_index, joint_index in enumerate(PALM_BASE_INDICES):
        vector = palm_half @ palm_shape[local_index]
        expected_length = (
            np.linalg.norm(left[joint_index] - left[0]) * (1.0 - alpha)
            + np.linalg.norm(right[joint_index] - right[0]) * alpha
        )
        output[joint_index] = wrist + _unit(vector) * expected_length
    for chain in FINGER_CHAINS:
        for parent, child in zip(chain[1:], chain[2:]):
            left_vector = left[child] - left[parent]
            right_vector = right[child] - right[parent]
            half_rotation = _limited_rotation(_rotation_between(left_vector, right_vector), math.pi, alpha)
            expected_length = np.linalg.norm(left_vector) * (1.0 - alpha) + np.linalg.norm(right_vector) * alpha
            output[child] = output[parent] + _unit(half_rotation @ left_vector) * expected_length
    return output


def _enforce_inserted_hand_proportions(output: Any, source_positions: np.ndarray) -> dict:
    transforms = output["transforms"]
    maximum_bone_error = 0.0
    maximum_palm_error = 0.0
    source_row_lookup = {
        int(round(float(position))): output_index
        for output_index, position in enumerate(source_positions)
        if abs(float(position) - round(float(position))) <= 1e-9
    }
    for output_index in np.flatnonzero(np.abs(source_positions - np.rint(source_positions)) > 1e-9):
        source_position = float(source_positions[output_index])
        left_source, right_source = int(math.floor(source_position)), int(math.ceil(source_position))
        left_index, right_index = source_row_lookup[left_source], source_row_lookup[right_source]
        alpha = source_position - left_source
        for side in ("left", "right"):
            names = side_hand_joint_names(side)
            left = np.asarray([transforms[name][left_index, :3, 3] for name in names], dtype=np.float64)
            right = np.asarray([transforms[name][right_index, :3, 3] for name in names], dtype=np.float64)
            midpoint = _intermediate_hand_points(left, right, alpha)
            for joint_index, name in enumerate(names):
                matrix = np.asarray(transforms[name][output_index], dtype=np.float64)
                matrix[:3, 3] = midpoint[joint_index]
                transforms[name][output_index] = matrix.astype(transforms[name].dtype)
            for parent, child in HAND_BONES:
                expected = (
                    np.linalg.norm(left[child] - left[parent]) * (1.0 - alpha)
                    + np.linalg.norm(right[child] - right[parent]) * alpha
                )
                actual = np.linalg.norm(midpoint[child] - midpoint[parent])
                maximum_bone_error = max(maximum_bone_error, abs(actual - expected) / max(expected, 1e-12))
            for first in range(len(PALM_BASE_INDICES)):
                for second in range(first + 1, len(PALM_BASE_INDICES)):
                    first_index, second_index = PALM_BASE_INDICES[first], PALM_BASE_INDICES[second]
                    expected = (
                        np.linalg.norm(left[first_index] - left[second_index]) * (1.0 - alpha)
                        + np.linalg.norm(right[first_index] - right[second_index]) * alpha
                    )
                    actual = np.linalg.norm(midpoint[first_index] - midpoint[second_index])
                    maximum_palm_error = max(maximum_palm_error, abs(actual - expected) / max(expected, 1e-12))
    return {
        "maximum_inserted_bone_length_relative_error": float(maximum_bone_error),
        "maximum_inserted_palm_distance_relative_error": float(maximum_palm_error),
    }


def _retime_corrected_hdf5(path: Path, source_positions: np.ndarray, source_frame_count: int) -> dict:
    import h5py

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.retime.tmp")
    try:
        with h5py.File(path, "r") as source, h5py.File(temporary, "w") as output:
            _copy_retimed_hdf5_group(source, output, source_positions, source_frame_count)
            proportion_metrics = _enforce_inserted_hand_proportions(output, source_positions)
            if "timing" in output:
                del output["timing"]
            timing = output.create_group("timing")
            timing.create_dataset("source_frame_position", data=source_positions, compression="lzf")
            inserted = np.abs(source_positions - np.rint(source_positions)) > 1e-9
            timing.create_dataset("inserted_intermediate_frame", data=inserted, compression="lzf")
            timing.create_dataset("nearest_source_frame", data=np.rint(source_positions).astype(np.int64), compression="lzf")
            output.attrs["source_video_frame_count"] = int(source_frame_count)
            output.attrs["video_frame_count"] = int(source_positions.size)
            output.attrs["alice_temporal_retiming"] = "insert_one_midpoint_before_projection_introduced_s1"
            output.attrs["inserted_hand_bone_policy"] = "average_endpoint_lengths_with_spherical_direction_interpolation"
            output.attrs["inserted_palm_policy"] = "rigidly_aligned_average_shape_with_wrist_base_length_constraint"
        temporary.replace(path)
        return proportion_metrics
    finally:
        temporary.unlink(missing_ok=True)


def _write_retimed_video(
    source_path: Path,
    target_path: Path,
    source_positions: np.ndarray,
    frame_count: int,
    fps: float,
    width: int,
    height: int,
) -> dict:
    from .video_smoothing import _create_frame_writer

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot decode video for synchronized S1 frame insertion: {source_path}")
    temporary = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.tmp.mp4")
    writer = _create_frame_writer(temporary, fps, width, height)
    failed = True
    previous: np.ndarray | None = None
    written = 0
    positions = np.asarray(source_positions, dtype=np.float64).reshape(-1)
    position_index = 0
    try:
        for source_frame in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            while previous is not None and position_index < positions.size and positions[position_index] < source_frame - 1e-9:
                alpha = float(positions[position_index] - (source_frame - 1))
                writer.write(cv2.addWeighted(previous, 1.0 - alpha, frame, alpha, 0.0))
                written += 1
                position_index += 1
            if position_index < positions.size and abs(float(positions[position_index]) - source_frame) <= 1e-9:
                writer.write(frame)
                written += 1
                position_index += 1
            previous = frame
        writer.close()
        failed = False
    finally:
        capture.release()
        if failed:
            writer.abort()
            temporary.unlink(missing_ok=True)
    expected = int(positions.size)
    if written != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Synchronized S1 video insertion stopped at {written}/{expected} frames")
    temporary.replace(target_path)
    return {
        "frame_count": written,
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "encoder": writer.name,
        "encoder_gpu": getattr(writer, "gpu_device", None),
        "audio_preserved": False,
    }


def _detect_projection_introduced_s1(
    source: Any,
    corrected_path: Path,
    rows: np.ndarray,
    sigma: float = PROJECTION_S1_SIGMA,
    raw_values: np.ndarray | None = None,
) -> dict:
    from .curation_pipeline import detect_sudden_changes

    names = (*side_hand_joint_names("left"), *side_hand_joint_names("right"))
    raw = (
        np.asarray(raw_values, dtype=np.float64)
        if raw_values is not None
        else _projection_xyz_matrix(source, names, rows)
    )
    import h5py
    with h5py.File(corrected_path, "r") as corrected:
        result = _projection_xyz_matrix(corrected, names)
    raw_detected = detect_sudden_changes(raw, sigma)
    corrected_detected = detect_sudden_changes(result, sigma)
    introduced = np.asarray(corrected_detected["mask"], dtype=bool) & ~np.asarray(raw_detected["mask"], dtype=bool)
    return {
        "mask": introduced,
        "raw_mask": np.asarray(raw_detected["mask"], dtype=bool),
        "corrected_mask": np.asarray(corrected_detected["mask"], dtype=bool),
        "raw": raw,
        "corrected": result,
        "raw_event_count": int(raw_detected["event_count"]),
        "corrected_event_count": int(corrected_detected["event_count"]),
        "introduced_event_count": int(introduced.sum()),
    }


def _retimed_projection_xyz(values: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    positions = np.asarray(source_positions, dtype=np.float64).reshape(-1)
    output = _interpolate_numeric_rows(source, positions).astype(np.float64)
    hand_width = 21 * 3
    for output_index in np.flatnonzero(np.abs(positions - np.rint(positions)) > 1e-9):
        position = float(positions[output_index])
        left_index, right_index = int(math.floor(position)), int(math.ceil(position))
        alpha = position - left_index
        for side_offset in (0, hand_width):
            left = source[left_index, side_offset:side_offset + hand_width].reshape(21, 3)
            right = source[right_index, side_offset:side_offset + hand_width].reshape(21, 3)
            output[output_index, side_offset:side_offset + hand_width] = _intermediate_hand_points(left, right, alpha).reshape(-1)
    return output


def _map_original_s1_mask(original_mask: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    """Project the immutable pre-correction S1 record onto a retimed frame axis."""
    baseline = np.asarray(original_mask, dtype=bool).reshape(-1)
    positions = np.asarray(source_positions, dtype=np.float64).reshape(-1)
    if baseline.size == 0:
        return np.zeros(positions.shape, dtype=bool)
    if not np.isfinite(positions).all():
        raise ValueError("S1 source positions must be finite")
    if positions.size and (float(positions.min()) < -1e-9 or float(positions.max()) > baseline.size - 1 + 1e-9):
        raise ValueError("S1 source positions exceed the original frame range")
    clipped = np.clip(positions, 0.0, float(baseline.size - 1))
    left = np.floor(clipped).astype(np.int64)
    right = np.ceil(clipped).astype(np.int64)
    return baseline[left] | baseline[right]


def _refine_s1_insertion_positions(
    raw: np.ndarray,
    corrected: np.ndarray,
    sigma: float = PROJECTION_S1_SIGMA,
    *,
    original_raw_mask: np.ndarray | None = None,
    maximum_passes: int = 4,
    maximum_inserted_fraction: float = 0.35,
) -> dict:
    from .curation_pipeline import detect_sudden_changes

    source_count = int(np.asarray(raw).shape[0])
    if original_raw_mask is None:
        original_raw_mask = np.asarray(detect_sudden_changes(raw, sigma)["mask"], dtype=bool)
    else:
        original_raw_mask = np.asarray(original_raw_mask, dtype=bool).reshape(-1)
    if original_raw_mask.shape != (source_count,):
        raise ValueError("Original S1 mask length must match the unmodified source timeline")
    positions = np.arange(source_count, dtype=np.float64)
    insertions: list[dict] = []
    initial_mask = np.zeros(source_count, dtype=bool)
    pass_reports: list[dict] = []
    maximum_insertions = max(1, int(math.ceil(source_count * max(0.0, maximum_inserted_fraction))))
    for pass_index in range(max(0, int(maximum_passes))):
        corrected_timeline = _retimed_projection_xyz(corrected, positions)
        corrected_result = detect_sudden_changes(corrected_timeline, sigma)
        baseline_guard = _map_original_s1_mask(original_raw_mask, positions)
        introduced = np.asarray(corrected_result["mask"], dtype=bool) & ~baseline_guard
        if pass_index == 0:
            initial_mask = introduced.copy()
        trigger_indices = [int(index) for index in np.flatnonzero(introduced) if index > 0]
        remaining_capacity = maximum_insertions - len(insertions)
        if remaining_capacity <= 0:
            pass_reports.append({"pass": pass_index + 1, "trigger_count": len(trigger_indices), "inserted_count": 0, "capacity_reached": True})
            break
        trigger_indices = trigger_indices[:remaining_capacity]
        if not trigger_indices:
            pass_reports.append({"pass": pass_index + 1, "trigger_count": 0, "inserted_count": 0})
            break
        trigger_set = set(trigger_indices)
        refined: list[float] = []
        pass_inserted = 0
        for current_index, position in enumerate(positions):
            if current_index in trigger_set:
                midpoint = 0.5 * (float(positions[current_index - 1]) + float(position))
                if midpoint > float(positions[current_index - 1]) + 1e-9 and midpoint < float(position) - 1e-9:
                    output_frame = len(refined)
                    refined.append(midpoint)
                    insertions.append({
                        "pass": pass_index + 1,
                        "output_frame_at_insertion": output_frame,
                        "left_source_position": round(float(positions[current_index - 1]), 9),
                        "right_source_position": round(float(position), 9),
                        "source_position": round(midpoint, 9),
                    })
                    pass_inserted += 1
            refined.append(float(position))
        positions = np.asarray(refined, dtype=np.float64)
        pass_reports.append({
            "pass": pass_index + 1,
            "trigger_count": int(introduced.sum()),
            "inserted_count": pass_inserted,
        })
        if pass_inserted == 0:
            break
    corrected_final = _retimed_projection_xyz(corrected, positions)
    corrected_result = detect_sudden_changes(corrected_final, sigma)
    baseline_guard = _map_original_s1_mask(original_raw_mask, positions)
    remaining = np.asarray(corrected_result["mask"], dtype=bool) & ~baseline_guard
    return {
        "source_positions": positions,
        "insertions": insertions,
        "initial_mask": initial_mask,
        "initial_event_count": int(initial_mask.sum()),
        "remaining_mask": remaining,
        "remaining_event_count": int(remaining.sum()),
        "passes": pass_reports,
        "maximum_insertions": maximum_insertions,
        "original_raw_mask": original_raw_mask,
        "baseline_guard": baseline_guard,
    }


def _sample_digest(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as source:
        digest.update(source.read(1024 * 1024))
        if size > 1024 * 1024:
            source.seek(max(0, size - 1024 * 1024))
            digest.update(source.read(1024 * 1024))
    return digest.hexdigest()


def _source_fingerprint(root: Path, relative: str) -> dict:
    path = (root / relative).resolve()
    path.relative_to(root)
    stat = path.stat()
    return {
        "relative_path": relative.replace("\\", "/"),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_sha256": _sample_digest(path),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_projection_correction(
    dataset_id: str,
    manifest: dict,
    episode: dict,
    media: dict,
    registry: Any,
    progress: Callable[[float, str], None] | None = None,
    *,
    sample_fps: float = 15.0,
    maximum_interpolation_gap_seconds: float = 0.75,
    adjustment_rate: float = 0.58,
    adjustment_mode: str = "uniform",
    wrist_point_source: str = "egodex",
    dynamic_low_confidence: float = 0.18,
    dynamic_mid_confidence: float = 0.60,
    dynamic_low_multiplier: float = 0.4,
    dynamic_mid_multiplier: float = 1.0,
    dynamic_high_multiplier: float = 2.0,
) -> dict:
    import h5py

    if not registry.has_hand_pose:
        raise RuntimeError("Hand-pose detector is not loaded")
    report_progress = progress or (lambda _value, _message: None)
    adjustment_rate = float(np.clip(adjustment_rate, 0.0, 1.0))
    adjustment_mode = str(adjustment_mode or "uniform").strip().casefold()
    if adjustment_mode not in {"uniform", "dynamic"}:
        raise ValueError(f"Unsupported projection adjustment mode: {adjustment_mode}")
    wrist_point_source = str(wrist_point_source or "egodex").strip().casefold()
    if wrist_point_source not in {"egodex", "model"}:
        raise ValueError(f"Unsupported wrist point source: {wrist_point_source}")
    dynamic_low_confidence = float(np.clip(dynamic_low_confidence, 0.01, 0.95))
    dynamic_mid_confidence = float(np.clip(dynamic_mid_confidence, dynamic_low_confidence + 1e-6, 0.99))
    dynamic_low_multiplier = max(0.0, float(dynamic_low_multiplier))
    dynamic_mid_multiplier = max(dynamic_low_multiplier, float(dynamic_mid_multiplier))
    dynamic_high_multiplier = max(dynamic_mid_multiplier, float(dynamic_high_multiplier))
    minimum_confidence = dynamic_low_confidence if adjustment_mode == "dynamic" else 0.18
    source_path, source_relative, source_count = _raw_transform_source(manifest, episode)
    video_path = _media_path(manifest, media)
    frame_count = int(media.get("frame_count") or episode.get("frame_count") or 0)
    fps = max(0.01, float(media.get("fps") or episode.get("fps") or 30.0))
    width = int(media.get("width") or episode.get("width") or 0)
    height = int(media.get("height") or episode.get("height") or 0)
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Selected video geometry is invalid")
    rows = _aligned_rows(manifest, episode, source_relative, source_count, frame_count)
    step = max(1, int(round(fps / max(0.1, float(sample_fps)))))
    sample_frames = sorted(set([*range(0, frame_count, step), frame_count - 1]))
    maximum_gap_frames = max(0, int(round(maximum_interpolation_gap_seconds * fps)))
    observations: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {"left": {}, "right": {}}
    accepted_samples = {"left": 0, "right": 0}
    rejected_samples = {"left": 0, "right": 0}
    previous_selected: dict[str, dict | None] = {"left": None, "right": None}
    pose_status = registry.status().get("hand_pose") or {}
    pose_backend = str(pose_status.get("backend") or "hand_pose")
    pose_label = "MediaPipe" if pose_backend == "mediapipe" else "AlicePose" if pose_backend == "alicepose" else "Hand pose"
    pose_device = str(pose_status.get("device") or "cpu")
    default_batch_size = "6" if pose_backend == "mediapipe" else ("1" if pose_device == "cpu" else "4")
    batch_size = max(1, int(os.getenv("ALICE_POSE_BATCH_FRAMES", default_batch_size)))

    with h5py.File(source_path, "r") as source:
        transforms = source["transforms"]
        already_retargeted = source_is_retargeted(source)
        mano_templates = {
            side: (
                None
                if already_retargeted or not has_egodex_mano_source(transforms, side)
                else fit_egodex_mano_template(transforms, side)
            )
            for side in ("left", "right")
        }

        def source_mano_frame(side: str, row: int) -> np.ndarray:
            template = mano_templates[side]
            required = side_hand_joint_names(side) if template is None else egodex_mano_source_names(transforms, side)
            named = {name: np.asarray(transforms[name][row], dtype=np.float64) for name in required}
            return direct_mano21_transforms(named, side) if template is None else retarget_egodex_mano_frame(named, template)

        for batch_start in range(0, len(sample_frames), batch_size):
            batch_frames = sample_frames[batch_start:batch_start + batch_size]
            frame_jobs: list[dict] = []
            for frame_index in batch_frames:
                frame = read_frame({**media, "path": str(video_path)}, frame_index)
                if frame is None:
                    rejected_samples["left"] += 1
                    rejected_samples["right"] += 1
                    continue
                row = int(rows[frame_index])
                camera = np.asarray(transforms["camera"][row], dtype=np.float64)
                intrinsic = _intrinsic(source, row, frame.shape[1], frame.shape[0])
                projected: dict[str, np.ndarray] = {}
                for side in ("left", "right"):
                    world = source_mano_frame(side, row)[:, :3, 3]
                    camera_points = _world_to_camera(world, camera)
                    source_pixels = _project_camera(camera_points, intrinsic)
                    if not np.isfinite(source_pixels).all():
                        rejected_samples[side] += 1
                        continue
                    projected[side] = source_pixels
                if len(projected) == 2:
                    frame_jobs.append({"frame": frame_index, "image": frame, "source_pixels": projected})
                else:
                    for side in ("left", "right"):
                        if side in projected:
                            rejected_samples[side] += 1
            if not frame_jobs:
                continue
            results = registry.infer_hand_pose_full_frame_batch([item["image"] for item in frame_jobs])
            for job, candidates in zip(frame_jobs, results):
                frame_index = int(job["frame"])
                assigned = _assign_full_frame_hands(
                    list(candidates or []),
                    job["source_pixels"],
                    previous_selected,
                    int(job["image"].shape[1]),
                    int(job["image"].shape[0]),
                )
                for side in ("left", "right"):
                    selected = assigned[side]
                    full_approval = bool(selected and selected.get("full_approval"))
                    if selected is None or (not full_approval and int((np.asarray(selected["confidence"]) >= minimum_confidence).sum()) < 12):
                        rejected_samples[side] += 1
                        continue
                    displacement, _hand_radius = _normalized_pose_displacement(selected["keypoints"], selected["source_pixels"])
                    if full_approval and adjustment_mode == "uniform":
                        confidence = np.ones(21, dtype=np.float64)
                    else:
                        source_confidence = np.asarray(
                            selected.get("raw_confidence") if full_approval else selected["confidence"],
                            dtype=np.float64,
                        )
                        confidence = np.where(source_confidence >= minimum_confidence, source_confidence, 0.0)
                    observations[side][frame_index] = (displacement, confidence)
                    previous_selected[side] = selected
                    accepted_samples[side] += 1
            complete = min(len(sample_frames), batch_start + len(batch_frames))
            report_progress(5.0 + 45.0 * complete / max(1, len(sample_frames)), f"{pose_label} full-frame samples {complete}/{len(sample_frames)}")

        smoothing_frames = max(3, int(round(PROJECTION_SMOOTHING_SECONDS * fps)))
        taper_frames = max(1, int(round(PROJECTION_EDGE_TAPER_SECONDS * fps)))
        maximum_normalized_step = MAXIMUM_NORMALIZED_CORRECTION_SPEED_PER_SECOND / fps
        interpolated = {
            side: _interpolate_observations(
                frame_count,
                observations[side],
                maximum_gap_frames,
                smoothing_frames=smoothing_frames,
                taper_frames=taper_frames,
                maximum_step=maximum_normalized_step,
            )
            for side in ("left", "right")
        }
        metadata_path, corrected_path = _artifact_paths(dataset_id, str(episode["id"]))
        retimed_video_path = _retimed_video_path(dataset_id, str(episode["id"]))
        temporary_hdf5 = corrected_path.with_name(f".{corrected_path.name}.{uuid.uuid4().hex}.tmp")
        applied_frames = {"left": 0, "right": 0}
        rejected_frames = {"left": 0, "right": 0}
        before_residuals: list[float] = []
        after_residuals: list[float] = []
        maximum_bone_error = 0.0
        maximum_palm_error = 0.0
        raw_mano_xyz = np.empty((frame_count, 2 * 21 * 3), dtype=np.float64)
        try:
            with h5py.File(temporary_hdf5, "w") as output:
                for key, value in source.attrs.items():
                    output.attrs[key] = value
                if "camera" in source:
                    source.copy("camera", output)
                output_transforms = output.create_group("transforms")
                output_transforms.attrs.update(dict(transforms.attrs))
                transform_names = [name for name, value in transforms.items() if isinstance(value, h5py.Dataset) and value.shape == (source_count, 4, 4)]
                chunk = min(256, max(1, frame_count))
                for name in transform_names:
                    output_transforms.create_dataset(name, shape=(frame_count, 4, 4), dtype=np.float32, chunks=(chunk, 4, 4), compression="lzf", shuffle=True)
                output_confidences = output.create_group("confidences") if "confidences" in source else None
                if output_confidences is not None:
                    for name, value in source["confidences"].items():
                        if isinstance(value, h5py.Dataset) and value.ndim and int(value.shape[0]) == source_count:
                            output_confidences.create_dataset(name, shape=(frame_count, *value.shape[1:]), dtype=value.dtype, chunks=True, compression="lzf", shuffle=True)
                for start in range(0, frame_count, chunk):
                    end = min(frame_count, start + chunk)
                    selected_rows = rows[start:end]
                    transform_block = {name: _take_rows(transforms[name], selected_rows).astype(np.float64) for name in transform_names}
                    confidence_block = {}
                    if output_confidences is not None:
                        for name, value in source["confidences"].items():
                            if name in output_confidences:
                                confidence_block[name] = _take_rows(value, selected_rows)
                    for local_index, frame_index in enumerate(range(start, end)):
                        camera = transform_block["camera"][local_index]
                        intrinsic = _intrinsic(source, int(selected_rows[local_index]), width, height)
                        for side_offset, side in enumerate(("left", "right")):
                            template = mano_templates[side]
                            required = side_hand_joint_names(side) if template is None else egodex_mano_source_names(transform_block, side)
                            named = {name: np.asarray(transform_block[name][local_index], dtype=np.float64).copy() for name in required}
                            matrices = (
                                direct_mano21_transforms(named, side)
                                if template is None
                                else retarget_egodex_mano_frame(named, template)
                            )
                            names = side_hand_joint_names(side)
                            raw_mano_xyz[frame_index, side_offset * 63:(side_offset + 1) * 63] = matrices[:, :3, 3].reshape(-1)
                            # The immutable source remains EgoDex.  The staged
                            # snapshot stores the kinematically retargeted
                            # MANO21 prior in the canonical 21 named fields.
                            for joint_index, name in enumerate(names):
                                transform_block[name][local_index] = matrices[joint_index]
                            displacement, pose_confidence, available = interpolated[side]
                            if int(available[frame_index].sum()) < 12:
                                rejected_frames[side] += 1
                                continue
                            world_points = matrices[:, :3, 3]
                            camera_points = _world_to_camera(world_points, camera)
                            source_pixels = _project_camera(camera_points, intrinsic)
                            hand_radius = _projected_hand_radius(source_pixels)
                            target_pixels = source_pixels + displacement[frame_index] * hand_radius
                            forearm_name = f"{side}Forearm"
                            forearm_camera = None
                            if forearm_name in transform_block:
                                forearm_camera = _world_to_camera(transform_block[forearm_name][local_index, :3, 3][None, :], camera)[0]
                            confidence_policy = (
                                "mediapipe_full_approval_uniform"
                                if pose_backend == "mediapipe" and adjustment_mode == "uniform"
                                else adjustment_mode
                            )
                            correction = _constrained_hand_correction(
                                camera_points,
                                matrices[:, :3, :3],
                                camera,
                                intrinsic,
                                target_pixels,
                                np.where(available[frame_index], pose_confidence[frame_index], 0.0),
                                forearm_camera=forearm_camera,
                                minimum_confidence=minimum_confidence,
                                confidence_policy=confidence_policy,
                                wrist_point_source=wrist_point_source,
                                local_blend=adjustment_rate,
                                dynamic_mid_confidence=dynamic_mid_confidence,
                                dynamic_low_multiplier=dynamic_low_multiplier,
                                dynamic_mid_multiplier=dynamic_mid_multiplier,
                                dynamic_high_multiplier=dynamic_high_multiplier,
                            )
                            if not correction.get("applied"):
                                rejected_frames[side] += 1
                                continue
                            corrected_world = _camera_to_world(correction["camera_points"], camera)
                            for joint_index, name in enumerate(names):
                                transform_block[name][local_index, :3, 3] = corrected_world[joint_index]
                                transform_block[name][local_index, :3, :3] = correction["world_rotations"][joint_index]
                            applied_frames[side] += 1
                            before_residuals.append(float(correction["before_median_px"]))
                            after_residuals.append(float(correction["after_median_px"]))
                            maximum_bone_error = max(maximum_bone_error, float(correction["maximum_bone_length_relative_error"]))
                            maximum_palm_error = max(maximum_palm_error, float(correction["maximum_palm_distance_relative_error"]))
                    for name, values in transform_block.items():
                        output_transforms[name][start:end] = values.astype(np.float32)
                    if output_confidences is not None:
                        for name, values in confidence_block.items():
                            output_confidences[name][start:end] = values
                    report_progress(50.0 + 45.0 * end / max(1, frame_count), f"3D constrained solve {end}/{frame_count}")
                output.attrs.update({
                    "alice_schema": PROJECTION_CORRECTION_SCHEMA,
                    "alice_algorithm_revision": PROJECTION_CORRECTION_ALGORITHM_REVISION,
                    MANO21_RETARGETED_ATTRIBUTE: True,
                    MANO21_RETARGET_REVISION_ATTRIBUTE: EGODEX_MANO_REVISION,
                    "source_relative_path": source_relative,
                    "video_relative_path": str(media.get("relative_path") or ""),
                    "source_frame_count": int(source_count),
                    "video_frame_count": int(frame_count),
                    "bone_length_policy": "exact_preservation",
                    "palm_geometry_policy": "rigid_pairwise_distance_preservation",
                    "source_3d_mutated": False,
                })
            temporary_hdf5.replace(corrected_path)
        finally:
            temporary_hdf5.unlink(missing_ok=True)

        report_progress(96.0, "Strict S1 check and synchronized intermediate-frame insertion")
        s1_detection = _detect_projection_introduced_s1(
            source,
            corrected_path,
            rows,
            raw_values=raw_mano_xyz,
        )
        refinement = _refine_s1_insertion_positions(
            s1_detection["raw"],
            s1_detection["corrected"],
            PROJECTION_S1_SIGMA,
            original_raw_mask=s1_detection["raw_mask"],
        )
        source_positions = np.asarray(refinement["source_positions"], dtype=np.float64)
        insertions = list(refinement["insertions"])
        retimed_video: dict | None = None
        retiming_proportion_metrics = {
            "maximum_inserted_bone_length_relative_error": 0.0,
            "maximum_inserted_palm_distance_relative_error": 0.0,
        }
        remaining_introduced_s1 = int(refinement["remaining_event_count"])
        if insertions:
            retiming_proportion_metrics = _retime_corrected_hdf5(corrected_path, source_positions, frame_count)
            retimed_video = _write_retimed_video(
                video_path,
                retimed_video_path,
                source_positions,
                frame_count,
                fps,
                width,
                height,
            )
            from .curation_pipeline import detect_sudden_changes

            with h5py.File(corrected_path, "r") as corrected:
                corrected_retimed = _projection_xyz_matrix(
                    corrected,
                    (*side_hand_joint_names("left"), *side_hand_joint_names("right")),
                )
            corrected_after = detect_sudden_changes(corrected_retimed, PROJECTION_S1_SIGMA)
            original_s1_guard = _map_original_s1_mask(s1_detection["raw_mask"], source_positions)
            remaining_introduced_s1 = int((
                np.asarray(corrected_after["mask"], dtype=bool)
                & ~original_s1_guard
            ).sum())
        else:
            retimed_video_path.unlink(missing_ok=True)

    output_frame_count = int(source_positions.size)

    root = Path(manifest["root_path"]).expanduser().resolve()
    source_signatures = [_source_fingerprint(root, source_relative)]
    media_relative = str(media.get("relative_path") or "").replace("\\", "/")
    if media_relative:
        source_signatures.append(_source_fingerprint(root, media_relative))
    model_status = registry.status().get("hand_pose") or {}
    document = {
        "schema": PROJECTION_CORRECTION_SCHEMA,
        "algorithm_revision": PROJECTION_CORRECTION_ALGORITHM_REVISION,
        "dataset_id": dataset_id,
        "episode_id": str(episode["id"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Source HDF5/video remain read-only; applying activates the reviewed corrected and synchronized retimed snapshots.",
        "source_transform": {"relative_path": source_relative, "frame_count": source_count},
        "source_video": {"relative_path": media_relative, "file_id": media.get("file_id"), "frame_count": frame_count, "fps": fps},
        "source_signatures": source_signatures,
        "model": {key: model_status.get(key) for key in ("backend", "model_path", "device", "family", "keypoint_count")},
        "sampling": {
            "sample_fps": float(sample_fps),
            "sample_count": len(sample_frames),
            "maximum_interpolation_gap_seconds": float(maximum_interpolation_gap_seconds),
            "adjustment_rate": adjustment_rate,
            "adjustment_mode": adjustment_mode,
            "wrist_point_source": wrist_point_source,
            "dynamic_curve": {
                "low_confidence": dynamic_low_confidence,
                "mid_confidence": dynamic_mid_confidence,
                "low_multiplier": dynamic_low_multiplier,
                "mid_multiplier": dynamic_mid_multiplier,
                "high_multiplier": dynamic_high_multiplier,
            },
        },
        "retiming": {
            "mode": "insert_one_midpoint_before_projection_introduced_s1",
            "s1_detector": "original_residual_and_acceleration_or_jerk",
            "baseline_s1_policy": "immutable_original_s1_record",
            "sigma": PROJECTION_S1_SIGMA,
            "source_frame_count": frame_count,
            "output_frame_count": output_frame_count,
            "inserted_frame_count": len(insertions),
            "original_s1_frame_count": int(s1_detection["raw_event_count"]),
            "model_introduced_s1_frame_count": int(s1_detection["introduced_event_count"]),
            "trigger_frame_count": int(s1_detection["introduced_event_count"]),
            "remaining_projection_introduced_s1_frame_count": remaining_introduced_s1,
            "source_frame_positions": [round(float(value), 6) for value in source_positions],
            "insertions": insertions,
            "insertion_passes": refinement["passes"],
            "maximum_insertions": int(refinement["maximum_insertions"]),
            "video": retimed_video,
            **retiming_proportion_metrics,
            "duration_seconds": output_frame_count / fps,
            "source_duration_seconds": frame_count / fps,
        },
        "constraints": {
            "model_input": f"entire_rgb_frame_no_se3_source_skeleton_crop:{pose_backend}",
            "adjustment_rate": adjustment_rate,
            "adjustment_mode": adjustment_mode,
            "wrist_point_source": wrist_point_source,
            "wrist_point_policy": (
                "model_landmark_0_with_forearm_sphere_and_angle_limits"
                if wrist_point_source == "model"
                else "preserve_egodex_estimated_mano_visual_wrist"
            ),
            "adjustment_rate_policy": (
                f"smooth_confidence_curve_{dynamic_low_confidence:.3f}_{dynamic_low_multiplier:.3f}x_{dynamic_mid_confidence:.3f}_{dynamic_mid_multiplier:.3f}x_1.000_{dynamic_high_multiplier:.3f}x"
                if adjustment_mode == "dynamic"
                else "direct_uniform_blend_for_wrist_palm_and_fingers"
            ),
            "source_prior": f"{EGODEX_MANO_REVISION}:full_palm_rigid_fit_plus_relative_joint_rotation_fk_before_{pose_backend}",
            "bone_lengths": "fixed_from_episode_mano_template_then_preserved_exactly_during_model_correction",
            "palm_pairwise_distances": "fixed_rigid_mano_palm_template_then_preserved_exactly_during_model_correction",
            "per_finger_scale": "forbidden",
            "depth": "ray_sphere_solution_with_hand_scale_bound",
            "maximum_root_angle_degrees": 14.0,
            "maximum_palm_angle_degrees": 20.0,
            "maximum_bone_angle_degrees": 24.0,
            "candidate_selection": f"whole-frame {pose_label} detection plus unique left/right association using source proximity, weak handedness and temporal continuity; source SE3 is never used to crop model input",
            "projection_displacement": "normalized by per-frame projected hand radius with palm/finger proportion limits",
            "temporal_smoothing": {
                "mode": "centered robust triangular filter with bidirectional speed limiting",
                "window_seconds": PROJECTION_SMOOTHING_SECONDS,
                "edge_taper_seconds": PROJECTION_EDGE_TAPER_SECONDS,
                "maximum_hand_radii_per_second": MAXIMUM_NORMALIZED_CORRECTION_SPEED_PER_SECOND,
                "phase_delay_frames": 0,
            },
            "angle_limit_scaling": "maximum correction angles scale with reprojection residual / projected hand radius",
            "confidence_policy": (
                "smooth_per_keypoint_confidence_multiplier"
                if adjustment_mode == "dynamic"
                else "uniform_adjustment_rate_without_confidence_multiplier"
            ),
            "confidence_multiplier": "enabled" if adjustment_mode == "dynamic" else "disabled",
            "dynamic_confidence_blend_safety_cap": CONFIDENCE_BLEND_SAFETY_CAP,
            "dynamic_confidence_curve": {
                "low_confidence": dynamic_low_confidence,
                "mid_confidence": dynamic_mid_confidence,
                "high_confidence": 1.0,
                "low_multiplier": dynamic_low_multiplier,
                "mid_multiplier": dynamic_mid_multiplier,
                "high_multiplier": dynamic_high_multiplier,
                "interpolation": "piecewise_smoothstep",
            },
            "low_confidence_policy": (
                "all_finite_mediapipe_landmarks_are_accepted"
                if pose_backend == "mediapipe" and adjustment_mode == "uniform"
                else f"minimum_{minimum_confidence:.3f}_confidence_for_3d_adjustment"
            ),
        },
        "summary": {
            "frame_count": output_frame_count,
            "source_frame_count": frame_count,
            "inserted_frame_count": len(insertions),
            "original_s1_frame_count": int(s1_detection["raw_event_count"]),
            "projection_introduced_s1_frame_count_before_insertion": int(s1_detection["introduced_event_count"]),
            "projection_introduced_s1_frame_count_after_insertion": remaining_introduced_s1,
            **retiming_proportion_metrics,
            "sample_count": len(sample_frames),
            "adjustment_rate": adjustment_rate,
            "adjustment_mode": adjustment_mode,
            "wrist_point_source": wrist_point_source,
            "accepted_samples": accepted_samples,
            "rejected_samples": rejected_samples,
            "applied_frames": applied_frames,
            "rejected_frames": rejected_frames,
            "applied_hand_frame_count": int(sum(applied_frames.values())),
            "median_reprojection_before_px": round(float(np.median(before_residuals)), 4) if before_residuals else None,
            "median_reprojection_after_px": round(float(np.median(after_residuals)), 4) if after_residuals else None,
            "maximum_bone_length_relative_error": maximum_bone_error,
            "maximum_palm_distance_relative_error": maximum_palm_error,
        },
        "corrected_hdf5": str(corrected_path),
        "retimed_video": str(retimed_video_path) if retimed_video is not None else None,
    }
    _write_json_atomic(metadata_path, document)
    change = record_change(
        dataset_id,
        PROJECTION_CORRECTION_KIND,
        str(episode["id"]),
        f"{pose_label} constrained projection correction: {episode.get('name') or episode['id']}",
        [metadata_path, corrected_path, *([retimed_video_path] if retimed_video is not None else [])],
        document["summary"],
        [source_relative, media_relative],
    )
    document["artifact_path"] = str(metadata_path)
    document["change"] = {"id": change["id"], "status": change["status"], "revision": change["revision"]}
    report_progress(100.0, "Projection correction staged for review")
    return document


def projection_correction_status(dataset_id: str, manifest: dict, episode: dict) -> dict:
    metadata_path, corrected_path = _artifact_paths(dataset_id, str(episode["id"]))
    retimed_path = _retimed_video_path(dataset_id, str(episode["id"]))
    payload = None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    except (OSError, json.JSONDecodeError):
        payload = None
    inserted = int(((payload or {}).get("retiming") or {}).get("inserted_frame_count") or 0)
    artifact_exists = bool(payload and corrected_path.is_file() and (inserted == 0 or retimed_path.is_file()))
    current_algorithm = bool(payload and payload.get("algorithm_revision") == PROJECTION_CORRECTION_ALGORITHM_REVISION)
    return {
        "available": bool(artifact_exists and current_algorithm),
        "stale": bool(artifact_exists and not current_algorithm),
        "algorithm_revision": PROJECTION_CORRECTION_ALGORITHM_REVISION,
        "applied": change_is_applied(dataset_id, PROJECTION_CORRECTION_KIND, str(episode["id"])),
        "artifact_path": str(metadata_path),
        "corrected_hdf5": str(corrected_path),
        "retimed_video": str(retimed_path) if inserted and retimed_path.is_file() else None,
        "summary": (payload or {}).get("summary") or {},
    }


def _applied_entry(manifest: dict, episode_id: str) -> dict | None:
    sidecar = Path(str(manifest.get("sidecar_path") or "")).expanduser().resolve()
    current_path = sidecar / "changes" / "current.alice"
    try:
        current = json.loads(current_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return (current.get("entries") or {}).get(f"{PROJECTION_CORRECTION_KIND}:{episode_id}")


def applied_projection_source(manifest: dict, episode: dict) -> dict | None:
    episode_id = str(episode.get("id") or "")
    if not episode_id:
        return None
    entry = _applied_entry(manifest, episode_id)
    if not entry:
        return None
    sidecar = Path(str(manifest.get("sidecar_path") or "")).expanduser().resolve()
    artifacts = entry.get("artifacts") or []
    hdf5_record = next((item for item in artifacts if Path(str(item.get("snapshot_path") or "")).suffix.casefold() in {".h5", ".hdf5", ".h5df"}), None)
    video_record = next((item for item in artifacts if Path(str(item.get("snapshot_path") or "")).suffix.casefold() in {".mp4", ".mov", ".mkv", ".avi"}), None)
    metadata_record = next((item for item in artifacts if item.get("schema") == PROJECTION_CORRECTION_SCHEMA), None)
    if not hdf5_record or not metadata_record:
        return None
    hdf5_path = (sidecar / str(hdf5_record["snapshot_path"])).resolve()
    video_path = (sidecar / str(video_record["snapshot_path"])).resolve() if video_record else None
    metadata_path = (sidecar / str(metadata_record["snapshot_path"])).resolve()
    try:
        hdf5_path.relative_to(sidecar)
        if video_path is not None:
            video_path.relative_to(sidecar)
        metadata_path.relative_to(sidecar)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    inserted = int((metadata.get("retiming") or {}).get("inserted_frame_count") or 0)
    if not hdf5_path.is_file() or metadata.get("schema") != PROJECTION_CORRECTION_SCHEMA:
        return None
    if inserted > 0 and (video_path is None or not video_path.is_file()):
        return None
    return {
        "path": hdf5_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "video_path": video_path,
        "source_relative_path": str((metadata.get("source_transform") or {}).get("relative_path") or ""),
        "frame_count": int((metadata.get("summary") or {}).get("frame_count") or 0),
        "application_id": entry.get("application_id"),
        "change_id": entry.get("change_id"),
    }


def staged_projection_source(manifest: dict, episode: dict) -> dict | None:
    """Return the latest review artifact without activating it for processing."""
    episode_id = str(episode.get("id") or "")
    sidecar_value = str(manifest.get("sidecar_path") or "").strip()
    if not episode_id or not sidecar_value:
        return None
    sidecar = Path(sidecar_value).expanduser().resolve()
    root = sidecar / "projection-correction"
    stem = slugify(episode_id)
    metadata_path = root / f"{stem}.projection.alice"
    hdf5_path = root / f"{stem}.projection.hdf5"
    video_path = root / f"{stem}.projection.mp4"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_path.relative_to(sidecar)
        hdf5_path.relative_to(sidecar)
        video_path.relative_to(sidecar)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        metadata.get("schema") != PROJECTION_CORRECTION_SCHEMA
        or metadata.get("algorithm_revision") != PROJECTION_CORRECTION_ALGORITHM_REVISION
        or str(metadata.get("dataset_id") or "") != str(manifest.get("id") or "")
        or str(metadata.get("episode_id") or "") != episode_id
        or not hdf5_path.is_file()
    ):
        return None
    retiming = metadata.get("retiming") or {}
    inserted = int(retiming.get("inserted_frame_count") or 0)
    output_count = int(retiming.get("output_frame_count") or (metadata.get("summary") or {}).get("frame_count") or 0)
    positions = retiming.get("source_frame_positions") or []
    if output_count <= 0 or len(positions) != output_count:
        return None
    if inserted > 0 and not video_path.is_file():
        return None
    return {
        "path": hdf5_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "video_path": video_path if inserted > 0 else None,
        "source_relative_path": str((metadata.get("source_transform") or {}).get("relative_path") or ""),
        "frame_count": output_count,
        "application_id": None,
        "change_id": (metadata.get("change") or {}).get("id"),
        "review_status": "pending",
        "applied": False,
    }


def review_projection_source(manifest: dict, episode: dict) -> dict | None:
    """Prefer a pending correction for visual review, then an applied snapshot."""
    episode_id = str(episode.get("id") or "")
    applied_state = False
    if episode_id:
        try:
            applied_state = change_is_applied(str(manifest.get("id") or ""), PROJECTION_CORRECTION_KIND, episode_id)
        except KeyError:
            # Preview/status helpers are also used with temporary manifests that
            # are not registered in storage.  Their sidecar current snapshot is
            # still sufficient to determine whether a correction is applied.
            applied_state = _applied_entry(manifest, episode_id) is not None
    if episode_id and not applied_state:
        staged = staged_projection_source(manifest, episode)
        if staged is not None:
            return staged
    applied = applied_projection_source(manifest, episode)
    if applied is not None:
        return {**applied, "review_status": "applied", "applied": True}
    return staged_projection_source(manifest, episode)


def preferred_projection_media(manifest: dict, episode: dict, media: dict) -> tuple[dict, dict | None]:
    """Use the applied S1-retimed video together with its corrected HDF5 timeline."""
    applied = applied_projection_source(manifest, episode)
    if applied is None or applied.get("video_path") is None:
        return media, None
    metadata = applied.get("metadata") or {}
    source_video = metadata.get("source_video") or {}
    if source_video.get("file_id") and source_video.get("file_id") != media.get("file_id"):
        return media, None
    retiming = metadata.get("retiming") or {}
    positions = retiming.get("source_frame_positions") or []
    output_count = int(retiming.get("output_frame_count") or applied.get("frame_count") or 0)
    fps = max(0.01, float(source_video.get("fps") or media.get("fps") or episode.get("fps") or 30.0))
    if output_count <= 0 or len(positions) != output_count:
        return media, None
    return {
        **media,
        "path": str(applied["video_path"]),
        "frame_count": output_count,
        "fps": fps,
        "duration": output_count / fps,
        "source_frame_positions": positions,
        "projection_retimed": True,
        "preview_variant": "projection-retimed",
        "projection_application_id": applied.get("application_id"),
    }, metadata


def preferred_projection_review_media(manifest: dict, episode: dict, media: dict) -> tuple[dict, dict | None]:
    """Present a staged or applied correction without changing processing state."""
    review = review_projection_source(manifest, episode)
    if review is None:
        return media, None
    metadata = review.get("metadata") or {}
    source_video = metadata.get("source_video") or {}
    if source_video.get("file_id") and source_video.get("file_id") != media.get("file_id"):
        return media, None
    retiming = metadata.get("retiming") or {}
    positions = retiming.get("source_frame_positions") or []
    output_count = int(retiming.get("output_frame_count") or review.get("frame_count") or 0)
    fps = max(0.01, float(source_video.get("fps") or media.get("fps") or episode.get("fps") or 30.0))
    if output_count <= 0 or len(positions) != output_count:
        return media, None
    video_path = review.get("video_path")
    return {
        **media,
        **({"path": str(video_path)} if video_path is not None else {}),
        "frame_count": output_count,
        "fps": fps,
        "duration": output_count / fps,
        "source_frame_positions": positions,
        "projection_retimed": bool(video_path is not None),
        "projection_review": True,
        "projection_review_status": review.get("review_status") or "pending",
        "preview_variant": "projection-review",
        "projection_application_id": review.get("application_id"),
    }, metadata


def verify_projection_correction_for_apply(manifest: dict, metadata_path: Path) -> dict:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Projection correction metadata is not readable") from exc
    if payload.get("schema") != PROJECTION_CORRECTION_SCHEMA:
        raise ValueError("Projection correction schema is invalid")
    if str(payload.get("dataset_id") or "") != str(manifest.get("id") or ""):
        raise ValueError("Projection correction belongs to a different dataset")
    root = Path(manifest["root_path"]).expanduser().resolve()
    for expected in payload.get("source_signatures") or []:
        relative = str(expected.get("relative_path") or "")
        try:
            current = _source_fingerprint(root, relative)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Projection correction source is unavailable: {relative}") from exc
        for key in ("size_bytes", "mtime_ns", "sample_sha256"):
            if current.get(key) != expected.get(key):
                raise ValueError(f"Projection correction source changed after review: {relative}")
    summary = payload.get("summary") or {}
    if float(summary.get("maximum_bone_length_relative_error") or 0.0) > 1e-5:
        raise ValueError("Projection correction violates the 3D bone-length constraint")
    if float(summary.get("maximum_palm_distance_relative_error") or 0.0) > 1e-5:
        raise ValueError("Projection correction violates the rigid-palm constraint")
    retiming = payload.get("retiming") or {}
    source_count = int(retiming.get("source_frame_count") or 0)
    output_count = int(retiming.get("output_frame_count") or 0)
    inserted_count = int(retiming.get("inserted_frame_count") or 0)
    positions = np.asarray(retiming.get("source_frame_positions") or [], dtype=np.float64).reshape(-1)
    if source_count <= 0 or output_count != source_count + inserted_count or positions.size != output_count:
        raise ValueError("Projection correction retiming frame counts are inconsistent")
    if not np.isfinite(positions).all() or (np.diff(positions) <= 0.0).any():
        raise ValueError("Projection correction retiming positions are invalid")
    fractional = positions[np.abs(positions - np.rint(positions)) > 1e-9]
    original_positions = positions[np.abs(positions - np.rint(positions)) <= 1e-9]
    if fractional.size != inserted_count or not np.allclose(original_positions, np.arange(source_count), atol=1e-9):
        raise ValueError("Projection correction retiming does not preserve every original frame")
    declared_insertions = retiming.get("insertions") or []
    if len(declared_insertions) != inserted_count:
        raise ValueError("Projection correction insertion audit length is inconsistent")
    for item in declared_insertions:
        left = float(item.get("left_source_position"))
        right = float(item.get("right_source_position"))
        midpoint = float(item.get("source_position"))
        if not left < midpoint < right or abs(midpoint - 0.5 * (left + right)) > 1e-8:
            raise ValueError("Projection correction contains an invalid midpoint insertion")
    if inserted_count:
        if float(retiming.get("maximum_inserted_bone_length_relative_error") or 0.0) > 1e-5:
            raise ValueError("Projection correction inserted frame violates finger bone-length interpolation")
        if float(retiming.get("maximum_inserted_palm_distance_relative_error") or 0.0) > 0.05:
            raise ValueError("Projection correction inserted frame distorts palm proportions")
        declared_video = str(payload.get("retimed_video") or "").strip()
        retimed_path = Path(declared_video).expanduser().resolve() if declared_video else metadata_path.with_name(metadata_path.name.removesuffix(".projection.alice") + ".projection.mp4")
        sidecar = Path(str(manifest.get("sidecar_path") or metadata_path.parent)).expanduser().resolve()
        try:
            retimed_path.relative_to(sidecar)
        except ValueError as exc:
            raise ValueError("Projection correction retimed video escaped .alicePD") from exc
        capture = cv2.VideoCapture(str(retimed_path))
        video_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if capture.isOpened() else 0
        capture.release()
        if not retimed_path.is_file() or video_count != output_count:
            raise ValueError("Projection correction retimed video frame count is inconsistent")
    return payload
