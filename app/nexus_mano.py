from __future__ import annotations

"""Experimental DexWeaveG1/Nexus 20-node hand to canonical MANO21 adapter.

The Nexus files inspected by Alice expose ``skeleton[..., 20, 7]`` records but
do not declare anatomical node names.  This adapter therefore makes one
bounded, auditable profile assumption: five contiguous proximal-to-distal
four-node finger chains ordered thumb, index, middle, ring, little.  It never
applies to a bare 20-node tensor without an explicit Nexus/DexWeave source
hint.

MANO0 is reconstructed rather than measured.  Four non-thumb proximal nodes
define the palm base, palm width provides scale, and their proximal segments
define the distal palm direction.  When ``wrist_quat`` is present, the most
consistent quaternion basis axis is fused with that geometric direction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .mano21 import HAND_21_JOINT_NAMES, side_hand_joint_names


NEXUS_MANO21_SCHEMA = "alice/nexus-dexweaveg1-20-to-mano21/v1"
NEXUS_MANO21_REVISION = "dexweaveg1-5x4-assumed-order-geometric-wrist-v1"
NEXUS_NODE_COUNT = 20
NEXUS_FINGER_COUNT = 5
NEXUS_NODES_PER_FINGER = 4
NEXUS_ASSUMED_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
MANO_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
DEFAULT_WRIST_TO_PALM_WIDTH_RATIO = 0.60
DEFAULT_WRIST_ORIENTATION_BLEND = 0.35
_EPSILON = 1e-9


@dataclass(frozen=True)
class NexusMano21Series:
    points: np.ndarray
    labels: tuple[str, ...]
    valid: np.ndarray
    root_confidence: np.ndarray
    palm_width: np.ndarray
    side: str | None
    source_field: str
    finger_order: tuple[str, ...]
    wrist_axis: int | None
    wrist_axis_sign: int | None
    wrist_to_palm_width_ratio: float


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _normal(value: Any) -> str:
    return "".join(character for character in _text(value).casefold() if character.isalnum())


def _side(value: Any) -> str | None:
    text = _text(value).replace("\\", "/").casefold()
    left = any(token in text for token in ("/left/", "_left", "left_", "lefthand", "hand=left"))
    right = any(token in text for token in ("/right/", "_right", "right_", "righthand", "hand=right"))
    if left != right:
        return "left" if left else "right"
    return None


def _attributes(obj: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        for key in getattr(obj, "attrs", {}).keys():
            value = obj.attrs[key]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            result[str(key)] = value
    except Exception:
        pass
    return result


def detect_nexus20_schema(
    *,
    shape: Sequence[int] | None,
    source_path: str = "",
    field: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dimensions = tuple(int(value) for value in (shape or ()))
    shape_matches = len(dimensions) >= 2 and dimensions[-2] == 20 and dimensions[-1] in {3, 7}
    text = f"{source_path}/{field}".replace("\\", "/").casefold()
    metadata_text = " ".join(
        f"{_text(key)} {_text(value)}" for key, value in (metadata or {}).items()
    ).casefold()
    dexweave = any(token in text or token in metadata_text for token in (
        "dexweaveg1", "dexweave_g1", "dexweave g1",
    ))
    nexus = "nexus" in text or "nexus" in metadata_text
    skeleton_field = _normal(field).endswith("skeleton")
    detected = bool(shape_matches and dexweave and (skeleton_field or nexus))
    return {
        "detected": detected,
        "confidence": 0.94 if detected else 0.0,
        "evidence": "dexweaveg1_path_and_20x3_or_7_skeleton" if detected else (
            "bare_20_node_shape_is_not_sufficient" if shape_matches else "no_nexus20_evidence"
        ),
        "experimental_node_order": detected,
    }


def _validated_finger_order(value: Sequence[str]) -> tuple[str, ...]:
    order = tuple(_text(item).strip().casefold() for item in value)
    if len(order) != 5 or set(order) != set(MANO_FINGER_ORDER):
        raise ValueError("finger_order must contain thumb, index, middle, ring, little exactly once")
    return order


def _validate_ratio(value: float) -> float:
    ratio = float(value)
    if not np.isfinite(ratio) or not 0.1 <= ratio <= 1.5:
        raise ValueError("wrist_to_palm_width_ratio must be finite and within [0.1, 1.5]")
    return ratio


def _validate_blend(value: float) -> float:
    blend = float(value)
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("wrist_orientation_blend must be finite and within [0, 1]")
    return blend


def _joint_validity(points: np.ndarray, validity: Any | None) -> np.ndarray:
    finite = np.isfinite(points).all(axis=-1)
    if validity is None:
        return finite
    supplied = np.asarray(validity)
    if supplied.shape == points.shape[:-2]:
        supplied = supplied[..., None]
    try:
        supplied = np.broadcast_to(supplied, points.shape[:-1])
    except ValueError as exc:
        raise ValueError(
            f"Nexus validity shape {np.asarray(validity).shape} does not match {points.shape[:-1]}"
        ) from exc
    mask = supplied if supplied.dtype == np.bool_ else np.isfinite(supplied) & (supplied > 0)
    return finite & mask


def _normalize_vectors(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(values).all(axis=-1) & np.isfinite(norms) & (norms > _EPSILON)
    normalized = np.divide(
        values,
        norms[..., None],
        out=np.zeros_like(values, dtype=np.float64),
        where=valid[..., None],
    )
    return normalized, valid


def _quaternion_matrices_xyzw(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(quaternions, dtype=np.float64)
    if values.shape[-1:] != (4,):
        raise ValueError(f"Expected wrist quaternion [..., 4] in XYZW order, got {values.shape}")
    norms = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(values).all(axis=-1) & np.isfinite(norms) & (norms > _EPSILON)
    q = np.divide(values, norms[..., None], out=np.zeros_like(values), where=valid[..., None])
    x, y, z, w = (q[..., index] for index in range(4))
    matrices = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrices[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[..., 0, 1] = 2.0 * (x * y - z * w)
    matrices[..., 0, 2] = 2.0 * (x * z + y * w)
    matrices[..., 1, 0] = 2.0 * (x * y + z * w)
    matrices[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[..., 1, 2] = 2.0 * (y * z - x * w)
    matrices[..., 2, 0] = 2.0 * (x * z - y * w)
    matrices[..., 2, 1] = 2.0 * (y * z + x * w)
    matrices[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    matrices[~valid] = np.nan
    return matrices, valid


def _orientation_direction(
    wrist_quaternion: Any | None,
    geometry_direction: np.ndarray,
    geometry_valid: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, int | None, int | None]:
    if wrist_quaternion is None:
        return None, np.zeros(geometry_valid.shape, dtype=np.float64), None, None
    matrices, quaternion_valid = _quaternion_matrices_xyzw(np.asarray(wrist_quaternion, dtype=np.float64))
    try:
        matrices = np.broadcast_to(matrices, geometry_direction.shape[:-1] + (3, 3))
        quaternion_valid = np.broadcast_to(quaternion_valid, geometry_valid.shape)
    except ValueError as exc:
        raise ValueError("wrist_quaternion leading dimensions do not match Nexus skeleton") from exc
    calibration_valid = geometry_valid & quaternion_valid
    if not calibration_valid.any():
        return None, np.zeros(geometry_valid.shape, dtype=np.float64), None, None
    dots = np.stack([
        np.sum(matrices[..., :, axis] * geometry_direction, axis=-1)
        for axis in range(3)
    ], axis=-1)
    flat_valid = calibration_valid.reshape(-1)
    flat_dots = dots.reshape(-1, 3)[flat_valid]
    scores = np.nanmedian(np.abs(flat_dots), axis=0)
    axis = int(np.nanargmax(scores))
    sign = 1 if float(np.nanmedian(flat_dots[:, axis])) >= 0.0 else -1
    direction = matrices[..., :, axis] * float(sign)
    alignment = np.where(
        calibration_valid,
        np.clip(np.sum(direction * geometry_direction, axis=-1), 0.0, 1.0),
        0.0,
    )
    return direction, alignment, axis, sign


def convert_nexus20_positions(
    positions: Any,
    *,
    wrist_quaternion: Any | None = None,
    validity: Any | None = None,
    finger_order: Sequence[str] = NEXUS_ASSUMED_FINGER_ORDER,
    wrist_to_palm_width_ratio: float = DEFAULT_WRIST_TO_PALM_WIDTH_RATIO,
    wrist_orientation_blend: float = DEFAULT_WRIST_ORIENTATION_BLEND,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert ``[..., 20, 3]`` Nexus positions into MANO21 positions."""

    values = np.asarray(positions, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2:] != (NEXUS_NODE_COUNT, 3):
        raise ValueError(f"Expected Nexus positions [..., 20, 3], got {values.shape}")
    order = _validated_finger_order(finger_order)
    ratio = _validate_ratio(wrist_to_palm_width_ratio)
    blend = _validate_blend(wrist_orientation_blend)
    source_valid = _joint_validity(values, validity)

    source_chains = values.reshape(values.shape[:-2] + (5, 4, 3))
    source_chain_valid = source_valid.reshape(source_valid.shape[:-1] + (5, 4))
    canonical_indices = np.asarray([order.index(finger) for finger in MANO_FINGER_ORDER], dtype=np.int64)
    chains = np.take(source_chains, canonical_indices, axis=-3)
    chain_valid = np.take(source_chain_valid, canonical_indices, axis=-2)

    output = np.empty(values.shape[:-2] + (21, 3), dtype=np.float64)
    output_valid = np.zeros(values.shape[:-2] + (21,), dtype=bool)
    output[..., 1:, :] = chains.reshape(values.shape[:-2] + (20, 3))
    output_valid[..., 1:] = chain_valid.reshape(values.shape[:-2] + (20,))

    non_thumb_bases = chains[..., 1:, 0, :]
    non_thumb_base_valid = chain_valid[..., 1:, 0]
    base_count = non_thumb_base_valid.sum(axis=-1)
    palm_center = np.divide(
        np.where(non_thumb_base_valid[..., None], non_thumb_bases, 0.0).sum(axis=-2),
        base_count[..., None],
        out=np.full(values.shape[:-2] + (3,), np.nan, dtype=np.float64),
        where=base_count[..., None] > 0,
    )

    index_base = chains[..., 1, 0, :]
    little_base = chains[..., 4, 0, :]
    width_vector = little_base - index_base
    palm_width = np.linalg.norm(width_vector, axis=-1)
    width_valid = (
        chain_valid[..., 1, 0] & chain_valid[..., 4, 0]
        & np.isfinite(palm_width) & (palm_width > _EPSILON)
    )

    proximal_segments = chains[..., 1:, 1, :] - chains[..., 1:, 0, :]
    segment_directions, segment_nonzero = _normalize_vectors(proximal_segments)
    segment_valid = chain_valid[..., 1:, 0] & chain_valid[..., 1:, 1] & segment_nonzero
    segment_count = segment_valid.sum(axis=-1)
    geometry_sum = np.where(segment_valid[..., None], segment_directions, 0.0).sum(axis=-2)
    geometry_direction, geometry_nonzero = _normalize_vectors(geometry_sum)
    geometry_valid = (base_count >= 3) & (segment_count >= 3) & width_valid & geometry_nonzero

    orientation_direction, orientation_alignment, wrist_axis, wrist_axis_sign = _orientation_direction(
        wrist_quaternion,
        geometry_direction,
        geometry_valid,
    )
    root_direction = geometry_direction
    if orientation_direction is not None and blend > 0.0:
        quaternion_available = np.isfinite(orientation_direction).all(axis=-1)
        fused = (1.0 - blend) * geometry_direction + blend * np.where(
            quaternion_available[..., None], orientation_direction, geometry_direction,
        )
        root_direction, fused_valid = _normalize_vectors(fused)
        geometry_valid &= fused_valid

    root = palm_center - ratio * palm_width[..., None] * root_direction
    output[..., 0, :] = root
    output_valid[..., 0] = geometry_valid & np.isfinite(root).all(axis=-1)
    output = np.where(output_valid[..., None], output, np.nan)

    direction_coherence = np.divide(
        np.linalg.norm(geometry_sum, axis=-1),
        segment_count,
        out=np.zeros_like(palm_width),
        where=segment_count > 0,
    )
    root_confidence = np.where(
        output_valid[..., 0],
        np.clip(0.55 + 0.25 * direction_coherence + 0.20 * orientation_alignment, 0.0, 1.0),
        0.0,
    )
    diagnostics = {
        "root_confidence": root_confidence,
        "palm_width": palm_width,
        "wrist_axis": wrist_axis,
        "wrist_axis_sign": wrist_axis_sign,
        "finger_order": order,
        "wrist_to_palm_width_ratio": ratio,
        "wrist_orientation_blend": blend,
        "node_order_assumed": True,
        "revision": NEXUS_MANO21_REVISION,
    }
    return output, output_valid, diagnostics


def convert_nexus20_pose7(
    poses: Any,
    *,
    wrist_quaternion: Any | None = None,
    validity: Any | None = None,
    finger_order: Sequence[str] = NEXUS_ASSUMED_FINGER_ORDER,
    wrist_to_palm_width_ratio: float = DEFAULT_WRIST_TO_PALM_WIDTH_RATIO,
    wrist_orientation_blend: float = DEFAULT_WRIST_ORIENTATION_BLEND,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2:] != (20, 7):
        raise ValueError(f"Expected Nexus pose records [..., 20, 7], got {values.shape}")
    points, valid, diagnostics = convert_nexus20_positions(
        values[..., :3],
        wrist_quaternion=wrist_quaternion,
        validity=validity,
        finger_order=finger_order,
        wrist_to_palm_width_ratio=wrist_to_palm_width_ratio,
        wrist_orientation_blend=wrist_orientation_blend,
    )
    order = _validated_finger_order(finger_order)
    chains = values.reshape(values.shape[:-2] + (5, 4, 7))
    indices = np.asarray([order.index(finger) for finger in MANO_FINGER_ORDER], dtype=np.int64)
    canonical = np.take(chains, indices, axis=-3).reshape(values.shape[:-2] + (20, 7))
    output = np.empty(values.shape[:-2] + (21, 7), dtype=np.float64)
    output[..., 1:, :] = canonical
    output[..., :, :3] = points
    if wrist_quaternion is None:
        output[..., 0, 3:7] = np.nan
    else:
        quaternions = np.asarray(wrist_quaternion, dtype=np.float64)
        try:
            output[..., 0, 3:7] = np.broadcast_to(quaternions, values.shape[:-2] + (4,))
        except ValueError as exc:
            raise ValueError("wrist_quaternion leading dimensions do not match Nexus poses") from exc
    output = np.where(valid[..., None], output, np.nan)
    return output, valid, diagnostics


def mano21_points_from_nexus_value(
    value: Any,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(value)
    if values.ndim >= 2 and values.shape[-2:] == (20, 7):
        converted, valid, diagnostics = convert_nexus20_pose7(values, **kwargs)
        return converted[..., :3], valid, diagnostics
    if values.ndim >= 2 and values.shape[-2:] == (20, 3):
        return convert_nexus20_positions(values, **kwargs)
    raise ValueError(f"Unsupported Nexus/DexWeaveG1 tensor shape: {values.shape}")


def mano21_labels(side: str | None) -> tuple[str, ...]:
    return side_hand_joint_names(side) if side in {"left", "right"} else HAND_21_JOINT_NAMES


def read_nexus20_hdf5_series(
    handle: Any,
    *,
    source_path: str = "",
    side: str | None = None,
) -> NexusMano21Series | None:
    dataset = handle.get("skeleton")
    if dataset is None:
        return None
    metadata = {**_attributes(handle), **_attributes(dataset)}
    detection = detect_nexus20_schema(
        shape=getattr(dataset, "shape", ()),
        source_path=source_path,
        field="skeleton",
        metadata=metadata,
    )
    if not detection["detected"]:
        return None
    values = np.asarray(dataset[()])
    wrist_dataset = handle.get("wrist_quat")
    wrist_quaternion = np.asarray(wrist_dataset[()]) if wrist_dataset is not None else None
    partial_dataset = handle.get("partial")
    validity = None
    if partial_dataset is not None and getattr(partial_dataset, "shape", ()) == values.shape[:-2]:
        validity = ~np.asarray(partial_dataset[()], dtype=bool)
    points, valid, diagnostics = mano21_points_from_nexus_value(
        values,
        wrist_quaternion=wrist_quaternion,
        validity=validity,
    )
    resolved_side = side if side in {"left", "right"} else _side(source_path)
    return NexusMano21Series(
        points=points,
        labels=mano21_labels(resolved_side),
        valid=valid,
        root_confidence=np.asarray(diagnostics["root_confidence"], dtype=np.float64),
        palm_width=np.asarray(diagnostics["palm_width"], dtype=np.float64),
        side=resolved_side,
        source_field="skeleton",
        finger_order=tuple(diagnostics["finger_order"]),
        wrist_axis=diagnostics["wrist_axis"],
        wrist_axis_sign=diagnostics["wrist_axis_sign"],
        wrist_to_palm_width_ratio=float(diagnostics["wrist_to_palm_width_ratio"]),
    )


def source_path_has_nexus20_hint(path: str | Path) -> bool:
    return bool(detect_nexus20_schema(
        shape=(1, 20, 7),
        source_path=str(path),
        field="skeleton",
    )["detected"])
