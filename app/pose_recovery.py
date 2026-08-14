from __future__ import annotations

"""Recover short missing mocap prefixes using an anchored visual-odometry check."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .storage import dataset_artifact_dir, record_change, slugify


POSE_RECOVERY_SCHEMA = "alice/pose-recovery/v1"


def _artifact_path(dataset_id: str, episode_id: str) -> Path:
    root = dataset_artifact_dir(dataset_id, "pose-recovery")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{slugify(episode_id)}.slam.alice"


def _episode_records(manifest: dict, episode: dict) -> list[dict]:
    episode_id = str(episode.get("id") or "")
    episode_key = str(episode.get("episode_key") or "")
    assignments = (manifest.get("episode_resolution") or {}).get("file_episode_assignments") or {}
    return [
        item
        for item in manifest.get("files", [])
        if str(assignments.get(str(item.get("id") or "")) or item.get("episode_id") or "") == episode_id
        or (
            not assignments.get(str(item.get("id") or ""))
            and not item.get("episode_id")
            and str(item.get("episode_key") or "") == episode_key
        )
    ]


def _side_sources(manifest: dict, episode: dict) -> list[dict[str, Any]]:
    root = Path(manifest["root_path"]).resolve()
    records = _episode_records(manifest, episode)
    sources = []
    for side in ("left", "right"):
        mocap = next((item for item in records if item.get("extension") in {".h5", ".hdf5", ".h5df"} and "mocap" in str(item.get("relative_path", "")).lower() and side in str(item.get("relative_path", "")).lower()), None)
        video = next((item for item in records if item.get("kind") == "video" and f"wrist_{side}" in str(item.get("relative_path", "")).lower()), None)
        if mocap:
            sources.append({
                "side": side,
                "mocap_relative_path": mocap["relative_path"],
                "mocap_path": root / mocap["relative_path"],
                "video_relative_path": video["relative_path"] if video else None,
                "video_path": root / video["relative_path"] if video else None,
            })
    return sources


def _mocap_gap(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        if "skeleton" not in handle:
            return {"frame_count": 0, "first_valid_frame": None, "initial_missing_frames": 0, "invalid_frames": 0}
        xyz = np.asarray(handle["skeleton"][:, :, :3], dtype=np.float64)
        valid = np.isfinite(xyz).all(axis=(1, 2)) & (np.linalg.norm(xyz, axis=2).max(axis=1) > 1e-6)
    indices = np.flatnonzero(valid)
    first = int(indices[0]) if len(indices) else None
    return {
        "frame_count": int(len(valid)),
        "first_valid_frame": first,
        "initial_missing_frames": first or 0,
        "invalid_frames": int((~valid).sum()),
    }


def _read_video_frame(path: Path, index: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def _visual_odometry_evidence(path: Path | None, missing_frame: int, anchor_frame: int) -> dict:
    if path is None or not path.is_file():
        return {"available": False, "reason": "wrist video is unavailable"}
    before = _read_video_frame(path, missing_frame)
    anchor = _read_video_frame(path, anchor_frame)
    if before is None or anchor is None:
        return {"available": False, "reason": "video frames could not be decoded"}
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=8)
    keypoints_a, descriptors_a = orb.detectAndCompute(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), None)
    keypoints_b, descriptors_b = orb.detectAndCompute(cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY), None)
    if descriptors_a is None or descriptors_b is None:
        return {"available": False, "reason": "insufficient visual features"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors_a, descriptors_b, k=2)
    matches = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(matches) < 8:
        return {"available": False, "feature_count": len(keypoints_a), "matches": len(matches), "reason": "insufficient matched features"}
    source = np.float32([keypoints_a[item.queryIdx].pt for item in matches])
    target = np.float32([keypoints_b[item.trainIdx].pt for item in matches])
    homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    ratio = float(inliers / max(1, len(matches)))
    median_flow = np.median(target - source, axis=0)
    return {
        "available": homography is not None,
        "feature_count": len(keypoints_a),
        "matches": len(matches),
        "inliers": inliers,
        "inlier_ratio": round(ratio, 6),
        "median_flow_px": [round(float(value), 4) for value in median_flow],
        "homography": np.asarray(homography).round(9).tolist() if homography is not None else None,
    }


def _recover_side(source: dict) -> dict:
    import h5py

    path = Path(source["mocap_path"])
    gap = _mocap_gap(path)
    first = gap["first_valid_frame"]
    result = {
        "side": source["side"],
        "source_path": source["mocap_relative_path"],
        "video_path": source.get("video_relative_path"),
        **gap,
        "recovered_frames": [],
    }
    if first is None or first == 0:
        result["status"] = "not_needed" if first == 0 else "unrecoverable"
        return result
    with h5py.File(path, "r") as handle:
        skeleton = np.asarray(handle["skeleton"], dtype=np.float64)
        quaternions = np.asarray(handle["wrist_quat"], dtype=np.float64) if "wrist_quat" in handle else None
        timestamps = np.asarray(handle["timestamps"], dtype=np.float64) if "timestamps" in handle else None
    next_index = min(len(skeleton) - 1, first + 1)
    while next_index < len(skeleton) - 1 and np.linalg.norm(skeleton[next_index, :, :3]) <= 1e-6:
        next_index += 1
    velocity = skeleton[next_index, :, :3] - skeleton[first, :, :3] if next_index > first else np.zeros_like(skeleton[first, :, :3])
    evidence = _visual_odometry_evidence(source.get("video_path"), first - 1, first)
    visual_confidence = float(evidence.get("inlier_ratio", 0.0)) if evidence.get("available") else 0.0
    confidence = max(0.45, min(0.98, 0.62 + 0.36 * visual_confidence))
    for frame in range(first - 1, -1, -1):
        steps = first - frame
        points = skeleton[first, :, :3] - velocity * steps
        quaternion = quaternions[first] if quaternions is not None else None
        result["recovered_frames"].append({
            "frame": frame,
            "timestamp": round(float(timestamps[frame]), 9) if timestamps is not None and np.isfinite(timestamps[frame]) else None,
            "skeleton_xyz": np.asarray(points).round(7).tolist(),
            "wrist_quat": np.asarray(quaternion).round(7).tolist() if quaternion is not None and np.isfinite(quaternion).all() else None,
            "anchor_frame": first,
            "next_valid_frame": next_index,
            "confidence": round(confidence * max(0.7, 1.0 - 0.08 * (steps - 1)), 6),
        })
    result["visual_odometry"] = evidence
    result["status"] = "recovered"
    result["method"] = "mocap_backward_extrapolation_with_orb_ransac_validation"
    return result


def pose_recovery_status(dataset_id: str, manifest: dict, episode: dict) -> dict:
    sides = [{"side": source["side"], "source_path": source["mocap_relative_path"], **_mocap_gap(Path(source["mocap_path"]))} for source in _side_sources(manifest, episode)]
    path = _artifact_path(dataset_id, episode["id"])
    source_gap_exists = any(item["initial_missing_frames"] > 0 for item in sides)
    return {
        "available": bool(sides),
        "needed": source_gap_exists and not path.is_file(),
        "source_gap_exists": source_gap_exists,
        "artifact_exists": path.is_file(),
        "artifact_path": str(path),
        "sides": sides,
    }


def recover_episode_pose(dataset_id: str, manifest: dict, episode: dict) -> dict:
    sources = _side_sources(manifest, episode)
    if not sources:
        raise ValueError("当前 Episode 没有可读取的手部 mocap 数据")
    sides = [_recover_side(source) for source in sources]
    document = {
        "schema": POSE_RECOVERY_SCHEMA,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Source files remain read-only; recovered poses are sidecar estimates.",
        "absolute_anchor": "first_valid_mocap_frame",
        "limitations": [
            "Visual odometry validates relative image motion but does not independently recover global position.",
            "Camera intrinsics and inter-camera extrinsics are null in this dataset, so metric visual SLAM is not claimed.",
        ],
        "sides": sides,
    }
    path = _artifact_path(dataset_id, episode["id"])
    document["artifact_path"] = str(path)
    document["recovered_frame_count"] = sum(len(item.get("recovered_frames", [])) for item in sides)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    change = record_change(
        dataset_id,
        "pose_recovery",
        episode["id"],
        f"Pose recovery: {episode['name']}",
        [path],
        {"recovered_frame_count": document["recovered_frame_count"], "method": "slam_visual_odometry"},
        [str(item.get("source_path") or "") for item in sides],
    )
    document["change"] = {"id": change["id"], "status": change["status"], "revision": change["revision"]}
    return document


def load_recovered_points(dataset_id: str, episode_id: str, source_path: str, frame: int) -> tuple[np.ndarray, list[str]] | None:
    path = _artifact_path(dataset_id, episode_id)
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    side = next((item for item in document.get("sides", []) if item.get("source_path") == source_path), None)
    if side is None:
        return None
    recovered = next((item for item in side.get("recovered_frames", []) if int(item.get("frame", -1)) == int(frame)), None)
    if recovered is None:
        return None
    points = np.asarray(recovered.get("skeleton_xyz"), dtype=np.float64)
    side_name = str(side.get("side") or "")
    if len(points) == 20 and side_name in {"left", "right"}:
        labels = [f"{side_name}Hand"]
        labels.extend(f"{side_name}Thumb{i}" for i in range(1, 4))
        for finger in ("Index", "Middle", "Ring", "Little"):
            labels.extend(f"{side_name}{finger}Finger{i}" for i in range(1, 5))
    else:
        labels = [f"joint_{index:02d}" for index in range(len(points))]
    return points, labels
