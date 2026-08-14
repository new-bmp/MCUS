from __future__ import annotations

"""Shared fractional-row sampling for retimed sensor trajectories."""

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def _validated_positions(positions: np.ndarray, source_count: int) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64).reshape(-1)
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > source_count - 1):
        raise ValueError("fractional source positions are outside the available rows")
    snapped = np.rint(values)
    return np.where(np.abs(values - snapped) <= 1e-9, snapped, values)


def interpolate_numeric_rows(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    resolved = _validated_positions(positions, int(source.shape[0]))
    left = np.floor(resolved).astype(np.int64)
    right = np.ceil(resolved).astype(np.int64)
    alpha = resolved - left
    if source.dtype.kind not in "biufc":
        return source[np.rint(resolved).astype(np.int64)]
    reshape = (len(alpha), *([1] * (source.ndim - 1)))
    output = source[left].astype(np.float64) * (1.0 - alpha.reshape(reshape))
    output += source[right].astype(np.float64) * alpha.reshape(reshape)
    if source.dtype.kind == "b":
        return output >= 0.5
    return output.astype(source.dtype, copy=False)


def _slerp_quaternions(left: np.ndarray, right: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    left_q = Rotation.from_matrix(left.reshape(-1, 3, 3)).as_quat().reshape(*left.shape[:-2], 4)
    right_q = Rotation.from_matrix(right.reshape(-1, 3, 3)).as_quat().reshape(*right.shape[:-2], 4)
    dot = np.sum(left_q * right_q, axis=-1, keepdims=True)
    right_q = np.where(dot < 0.0, -right_q, right_q)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    alpha_shape = (len(alpha), *([1] * (left_q.ndim - 2)), 1)
    weight = alpha.reshape(alpha_shape)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-7
    safe = np.where(near, 1.0, sin_theta)
    blended = (
        np.sin((1.0 - weight) * theta) / safe * left_q
        + np.sin(weight * theta) / safe * right_q
    )
    linear = (1.0 - weight) * left_q + weight * right_q
    blended = np.where(near, linear, blended)
    norm = np.linalg.norm(blended, axis=-1, keepdims=True)
    return blended / np.where(norm > 1e-12, norm, 1.0)


def interpolate_transform_rows(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Interpolate ``[T,...,4,4]`` transforms with translation lerp + rotation SLERP."""

    source = np.asarray(values)
    if source.ndim < 3 or tuple(source.shape[-2:]) != (4, 4):
        raise ValueError("transform rows must have shape [T,...,4,4]")
    resolved = _validated_positions(positions, int(source.shape[0]))
    left_index = np.floor(resolved).astype(np.int64)
    right_index = np.ceil(resolved).astype(np.int64)
    alpha = resolved - left_index
    left = source[left_index].astype(np.float64)
    right = source[right_index].astype(np.float64)
    output = left.copy()
    translation_shape = (len(alpha), *([1] * (source.ndim - 3)), 1)
    weight = alpha.reshape(translation_shape)
    output[..., :3, 3] = left[..., :3, 3] * (1.0 - weight) + right[..., :3, 3] * weight
    output[..., :3, :3] = Rotation.from_quat(
        _slerp_quaternions(left[..., :3, :3], right[..., :3, :3], alpha).reshape(-1, 4)
    ).as_matrix().reshape(*output.shape[:-2], 3, 3)
    output[..., 3, :] = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    return output.astype(source.dtype, copy=False)


def sample_hdf5_numeric(dataset: h5py.Dataset, positions: np.ndarray) -> np.ndarray:
    resolved = _validated_positions(positions, int(dataset.shape[0]))
    left = np.floor(resolved).astype(np.int64)
    right = np.ceil(resolved).astype(np.int64)
    rows = np.unique(np.concatenate((left, right)))
    loaded = np.asarray(dataset[rows.tolist()])
    left_local = np.searchsorted(rows, left)
    right_local = np.searchsorted(rows, right)
    local_positions = left_local.astype(np.float64) * (1.0 - (resolved - left))
    local_positions += right_local.astype(np.float64) * (resolved - left)
    return interpolate_numeric_rows(loaded, local_positions)


def sample_hdf5_transforms(dataset: h5py.Dataset, positions: np.ndarray) -> np.ndarray:
    resolved = _validated_positions(positions, int(dataset.shape[0]))
    left = np.floor(resolved).astype(np.int64)
    right = np.ceil(resolved).astype(np.int64)
    rows = np.unique(np.concatenate((left, right)))
    loaded = np.asarray(dataset[rows.tolist()])
    left_local = np.searchsorted(rows, left)
    right_local = np.searchsorted(rows, right)
    local_positions = left_local.astype(np.float64) * (1.0 - (resolved - left))
    local_positions += right_local.astype(np.float64) * (resolved - left)
    return interpolate_transform_rows(loaded, local_positions)
