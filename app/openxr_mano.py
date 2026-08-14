from __future__ import annotations

"""OpenXR ``XR_EXT_hand_tracking`` 26-joint to canonical MANO21 adapter.

OpenXR is a runtime API rather than a storage format.  This module therefore
requires explicit OpenXR evidence (schema metadata, standard joint names, or an
OpenXR-named field) before interpreting a 26-joint tensor.  A bare tensor with
26 rows is deliberately not guessed to be OpenXR.

The canonical output is a MANO-compatible 21-joint layout, not MANO mesh or
shape-parameter fitting.  The OpenXR palm and the four non-thumb metacarpal
joints are omitted.  MANO joint 0 is placed between OpenXR wrist and palm so a
runtime wrist located on the forearm does not become an overly proximal root.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .mano21 import HAND_21_JOINT_NAMES, side_hand_joint_names


OPENXR_HAND_SCHEMA = "alice/openxr-hand-26-to-mano21/v1"
OPENXR_JOINT_SET = "XR_HAND_JOINT_SET_DEFAULT_EXT"
OPENXR_JOINT_COUNT = 26
OPENXR_MANO21_REVISION = "openxr26-drop-palm-metacarpals-wrist-palm-root-v1"
DEFAULT_WRIST_TO_PALM_RATIO = 0.35

# XrSpaceLocationFlags bits used by XR_EXT_hand_tracking joint locations.
XR_SPACE_ORIENTATION_VALID_BIT = 0x00000001
XR_SPACE_POSITION_VALID_BIT = 0x00000002
XR_SPACE_ORIENTATION_TRACKED_BIT = 0x00000004
XR_SPACE_POSITION_TRACKED_BIT = 0x00000008

OPENXR_JOINT_NAMES = (
    "palm",
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_metacarpal",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
)

OPENXR_ENUM_NAMES = tuple(f"XR_HAND_JOINT_{name.upper()}_EXT" for name in OPENXR_JOINT_NAMES)

# OpenXR palm and non-thumb metacarpals (6, 11, 16, 21) are not separate
# joints in the common MANO21 keypoint convention.
OPENXR_TO_MANO21_INDICES = (
    1,
    2, 3, 4, 5,
    7, 8, 9, 10,
    12, 13, 14, 15,
    17, 18, 19, 20,
    22, 23, 24, 25,
)
OPENXR_OMITTED_INDICES = (0, 6, 11, 16, 21)


@dataclass(frozen=True)
class OpenXRHandFrame:
    points: np.ndarray
    labels: tuple[str, ...]
    valid: np.ndarray
    sides: tuple[str, ...]
    source_fields: tuple[str, ...]
    wrist_to_palm_ratio: float


@dataclass(frozen=True)
class OpenXRHandSeries:
    points: np.ndarray
    labels: tuple[str, ...]
    valid: np.ndarray
    sides: tuple[str, ...]
    source_fields: tuple[str, ...]
    wrist_to_palm_ratio: float


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _normal(value: Any) -> str:
    return "".join(character for character in _text(value).casefold() if character.isalnum())


def _metadata_text(metadata: Mapping[str, Any] | None) -> str:
    if not metadata:
        return ""
    parts: list[str] = []
    for key, value in metadata.items():
        parts.append(_text(key))
        if isinstance(value, (str, bytes, int, float, bool, np.generic)):
            parts.append(_text(value))
        elif isinstance(value, (list, tuple)) and len(value) <= 64:
            parts.extend(_text(item) for item in value)
        elif isinstance(value, np.ndarray) and value.ndim == 1 and value.size <= 64:
            parts.extend(_text(item) for item in value.tolist())
    return " ".join(parts).casefold()


def _canonical_joint_name(value: Any) -> str:
    normalized = _normal(value)
    for prefix in (
        "xrhandjointext",
        "xrhandjoint",
        "openxrhandjoint",
        "openxrjoint",
        "lefthandjoint",
        "righthandjoint",
        "left",
        "right",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    if normalized.endswith("ext"):
        normalized = normalized[:-3]
    return normalized


def openxr_joint_order_matches(labels: Sequence[Any] | None) -> bool:
    if labels is None or len(labels) != OPENXR_JOINT_COUNT:
        return False
    return tuple(_canonical_joint_name(label) for label in labels) == tuple(
        _canonical_joint_name(name) for name in OPENXR_JOINT_NAMES
    )


def _has_openxr_joint_axis(shape: Sequence[int] | None) -> bool:
    if not shape:
        return False
    dimensions = tuple(int(value) for value in shape)
    return (
        len(dimensions) >= 2
        and dimensions[-2] == OPENXR_JOINT_COUNT
        and dimensions[-1] in {3, 7}
    ) or (
        len(dimensions) >= 3
        and dimensions[-3:] == (OPENXR_JOINT_COUNT, 4, 4)
    )


def detect_openxr_schema(
    *,
    metadata: Mapping[str, Any] | None = None,
    labels: Sequence[Any] | None = None,
    shape: Sequence[int] | None = None,
    source_path: str = "",
    field: str = "",
) -> dict[str, Any]:
    """Return bounded OpenXR schema evidence without guessing from count alone."""

    metadata_text = _metadata_text(metadata)
    path_text = f"{source_path}/{field}".replace("\\", "/").casefold()
    shape_matches = _has_openxr_joint_axis(shape)
    standard_labels = openxr_joint_order_matches(labels)
    explicit_metadata = any(token in metadata_text for token in (
        "xr_ext_hand_tracking",
        "xr_hand_joint_set_default_ext",
        "openxr-hand-26",
        "openxr_hand_26",
        "openxr",
    ))
    explicit_path = any(token in path_text for token in (
        "openxr",
        "xr_hand",
        "xr-hand",
        "xrhand",
    ))

    if standard_labels and shape_matches:
        return {"detected": True, "confidence": 1.0, "evidence": "standard_joint_names_and_shape"}
    if explicit_metadata and (shape_matches or standard_labels):
        return {"detected": True, "confidence": 0.99, "evidence": "openxr_metadata"}
    if explicit_path and shape_matches:
        return {"detected": True, "confidence": 0.96, "evidence": "openxr_path_and_shape"}
    return {
        "detected": False,
        "confidence": 0.0,
        "evidence": "bare_26_joint_shape_is_not_sufficient" if shape_matches else "no_openxr_evidence",
    }


def _position_validity(points: np.ndarray, validity: Any | None) -> np.ndarray:
    finite = np.isfinite(points).all(axis=-1)
    if validity is None:
        return finite
    supplied = np.asarray(validity)
    expected = points.shape[:-1]
    try:
        supplied = np.broadcast_to(supplied, expected)
    except ValueError as exc:
        raise ValueError(f"OpenXR validity shape {supplied.shape} does not match {expected}") from exc
    if supplied.dtype == np.bool_:
        mask = supplied
    elif np.issubdtype(supplied.dtype, np.integer):
        # Integer inputs are OpenXR XrSpaceLocationFlags. Recorders exposing
        # a 0/1 mask should cast/store it as bool (the HDF5 adapter does this
        # automatically for datasets named validity/valid/tracked).
        mask = (supplied.astype(np.int64) & XR_SPACE_POSITION_VALID_BIT) != 0
    else:
        mask = np.isfinite(supplied) & (supplied > 0)
    return finite & mask


def _validate_ratio(value: float) -> float:
    ratio = float(value)
    if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("wrist_to_palm_ratio must be finite and within [0, 1]")
    return ratio


def convert_openxr_positions(
    positions: Any,
    *,
    validity: Any | None = None,
    wrist_to_palm_ratio: float = DEFAULT_WRIST_TO_PALM_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``[..., 26, 3]`` OpenXR positions to MANO21 positions."""

    values = np.asarray(positions, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2:] != (OPENXR_JOINT_COUNT, 3):
        raise ValueError(f"Expected OpenXR positions [..., 26, 3], got {values.shape}")
    ratio = _validate_ratio(wrist_to_palm_ratio)
    source_valid = _position_validity(values, validity)
    indices = np.asarray(OPENXR_TO_MANO21_INDICES, dtype=np.int64)
    output = np.take(values, indices, axis=-2).copy()
    output_valid = np.take(source_valid, indices, axis=-1).copy()

    palm = values[..., 0, :]
    wrist = values[..., 1, :]
    palm_valid = source_valid[..., 0]
    wrist_valid = source_valid[..., 1]
    both = palm_valid & wrist_valid
    root = np.where(wrist_valid[..., None], wrist, palm)
    blended = wrist + ratio * (palm - wrist)
    root = np.where(both[..., None], blended, root)
    output[..., 0, :] = root
    output_valid[..., 0] = palm_valid | wrist_valid
    output = np.where(output_valid[..., None], output, np.nan)
    return output, output_valid


def convert_openxr_pose7(
    poses: Any,
    *,
    validity: Any | None = None,
    wrist_to_palm_ratio: float = DEFAULT_WRIST_TO_PALM_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``[..., 26, 7]`` position + XYZW quaternion records."""

    values = np.asarray(poses, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2:] != (OPENXR_JOINT_COUNT, 7):
        raise ValueError(f"Expected OpenXR pose records [..., 26, 7], got {values.shape}")
    points, output_valid = convert_openxr_positions(
        values[..., :3], validity=validity, wrist_to_palm_ratio=wrist_to_palm_ratio,
    )
    indices = np.asarray(OPENXR_TO_MANO21_INDICES, dtype=np.int64)
    output = np.take(values, indices, axis=-2).copy()
    output[..., :, :3] = points
    palm_valid = _position_validity(values[..., :3], validity)[..., 0]
    wrist_valid = _position_validity(values[..., :3], validity)[..., 1]
    root_quaternion = np.where(
        wrist_valid[..., None], values[..., 1, 3:7], values[..., 0, 3:7],
    )
    output[..., 0, 3:7] = root_quaternion
    output = np.where(output_valid[..., None], output, np.nan)
    return output, output_valid


def convert_openxr_transforms(
    transforms: Any,
    *,
    validity: Any | None = None,
    wrist_to_palm_ratio: float = DEFAULT_WRIST_TO_PALM_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``[..., 26, 4, 4]`` OpenXR transforms to MANO21 transforms."""

    values = np.asarray(transforms, dtype=np.float64)
    if values.ndim < 3 or values.shape[-3:] != (OPENXR_JOINT_COUNT, 4, 4):
        raise ValueError(f"Expected OpenXR transforms [..., 26, 4, 4], got {values.shape}")
    positions = values[..., :3, 3]
    points, output_valid = convert_openxr_positions(
        positions, validity=validity, wrist_to_palm_ratio=wrist_to_palm_ratio,
    )
    indices = np.asarray(OPENXR_TO_MANO21_INDICES, dtype=np.int64)
    output = np.take(values, indices, axis=-3).copy()
    output[..., :, :3, 3] = points
    source_valid = _position_validity(positions, validity)
    root = np.where(
        source_valid[..., 1, None, None], values[..., 1, :, :], values[..., 0, :, :],
    ).copy()
    root[..., :3, 3] = points[..., 0, :]
    output[..., 0, :, :] = root
    output = np.where(output_valid[..., None, None], output, np.nan)
    return output, output_valid


def convert_openxr_value(
    value: Any,
    *,
    validity: Any | None = None,
    wrist_to_palm_ratio: float = DEFAULT_WRIST_TO_PALM_RATIO,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Convert a supported OpenXR tensor and report its representation."""

    values = np.asarray(value)
    if values.ndim >= 3 and values.shape[-3:] == (OPENXR_JOINT_COUNT, 4, 4):
        converted, valid = convert_openxr_transforms(
            values, validity=validity, wrist_to_palm_ratio=wrist_to_palm_ratio,
        )
        return converted, valid, "transform"
    if values.ndim >= 2 and values.shape[-2:] == (OPENXR_JOINT_COUNT, 7):
        converted, valid = convert_openxr_pose7(
            values, validity=validity, wrist_to_palm_ratio=wrist_to_palm_ratio,
        )
        return converted, valid, "pose7_xyzw"
    if values.ndim >= 2 and values.shape[-2:] == (OPENXR_JOINT_COUNT, 3):
        converted, valid = convert_openxr_positions(
            values, validity=validity, wrist_to_palm_ratio=wrist_to_palm_ratio,
        )
        return converted, valid, "position"
    raise ValueError(f"Unsupported OpenXR hand tensor shape: {values.shape}")


def mano21_points_from_openxr_value(
    value: Any,
    *,
    validity: Any | None = None,
    wrist_to_palm_ratio: float = DEFAULT_WRIST_TO_PALM_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    converted, valid, representation = convert_openxr_value(
        value, validity=validity, wrist_to_palm_ratio=wrist_to_palm_ratio,
    )
    if representation == "transform":
        return converted[..., :3, 3], valid
    return converted[..., :3], valid


def mano21_labels(side: str | None) -> tuple[str, ...]:
    normalized = _text(side).strip().casefold()
    return side_hand_joint_names(normalized) if normalized in {"left", "right"} else HAND_21_JOINT_NAMES


def _attributes(obj: Any) -> dict[str, Any]:
    attrs = getattr(obj, "attrs", {})
    result: dict[str, Any] = {}
    try:
        for key in attrs.keys():
            value = attrs[key]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            result[str(key)] = value
    except Exception:
        return result
    return result


def _parse_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            value = parsed
        except json.JSONDecodeError:
            value = [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [_text(item) for item in value] if isinstance(value, (list, tuple)) else []


def _side_from_context(text: str, *metadata: Mapping[str, Any]) -> str | None:
    combined = text.replace("\\", "/").casefold()
    for attrs in metadata:
        combined += " " + _metadata_text(attrs)
    left = any(token in combined for token in ("/left/", "left_hand", "lefthand", "hand=left", "hand left"))
    right = any(token in combined for token in ("/right/", "right_hand", "righthand", "hand=right", "hand right"))
    if left != right:
        return "left" if left else "right"
    return None


def _frame_value(dataset: Any, index: int, representation: str) -> np.ndarray:
    static_dimensions = 3 if representation == "transform" else 2
    if int(dataset.ndim) == static_dimensions:
        return np.asarray(dataset[()])
    frame = min(max(0, int(index)), max(0, int(dataset.shape[0]) - 1))
    return np.asarray(dataset[frame])


def _sibling_validity(handle: Any, dataset: Any, index: int) -> np.ndarray | None:
    parent = str(dataset.name).rsplit("/", 1)[0]
    for leaf in ("location_flags", "flags", "validity", "valid", "tracked"):
        path = f"{parent}/{leaf}" if parent else f"/{leaf}"
        candidate = handle.get(path)
        if candidate is None or not getattr(candidate, "shape", None):
            continue
        if int(candidate.ndim) == 1 and int(candidate.shape[0]) == OPENXR_JOINT_COUNT:
            value = np.asarray(candidate[()])
            return value.astype(bool) if leaf in {"validity", "valid", "tracked"} else value
        if int(candidate.ndim) >= 2 and int(candidate.shape[-1]) == OPENXR_JOINT_COUNT:
            frame = min(max(0, int(index)), max(0, int(candidate.shape[0]) - 1))
            return np.asarray(candidate[frame])
    return None


def _sibling_validity_series(handle: Any, dataset: Any) -> np.ndarray | None:
    parent = str(dataset.name).rsplit("/", 1)[0]
    for leaf in ("location_flags", "flags", "validity", "valid", "tracked"):
        path = f"{parent}/{leaf}" if parent else f"/{leaf}"
        candidate = handle.get(path)
        if candidate is None or not getattr(candidate, "shape", None):
            continue
        shape = tuple(int(value) for value in candidate.shape)
        if shape == (OPENXR_JOINT_COUNT,) or (len(shape) >= 2 and shape[-1] == OPENXR_JOINT_COUNT):
            value = np.asarray(candidate[()])
            return value.astype(bool) if leaf in {"validity", "valid", "tracked"} else value
    return None


def read_openxr_hdf5_frame(
    handle: Any,
    index: int,
    *,
    source_path: str = "",
    wrist_to_palm_ratio: float = DEFAULT_WRIST_TO_PALM_RATIO,
) -> OpenXRHandFrame | None:
    """Read explicitly identified OpenXR joint tensors from an open HDF5 file."""

    file_metadata = _attributes(handle)
    candidates: list[tuple[float, int, str, Any, str | None, dict[str, Any]]] = []

    def visitor(name: str, obj: Any) -> None:
        shape = tuple(int(value) for value in getattr(obj, "shape", ()) or ())
        if not _has_openxr_joint_axis(shape):
            return
        dataset_metadata = {**file_metadata, **_attributes(getattr(obj, "parent", None)), **_attributes(obj)}
        labels = []
        for key in ("joint_names", "joints", "joint_order"):
            labels = _parse_labels(dataset_metadata.get(key))
            if labels:
                break
        detection = detect_openxr_schema(
            metadata=dataset_metadata,
            labels=labels,
            shape=shape,
            source_path=source_path,
            field=name,
        )
        if not detection["detected"]:
            return
        representation = "transform" if shape[-3:] == (26, 4, 4) else "pose7_xyzw" if shape[-1] == 7 else "position"
        rank = {"transform": 0, "pose7_xyzw": 1, "position": 2}[representation]
        side = _side_from_context(f"/{name}", dataset_metadata)
        candidates.append((float(detection["confidence"]), rank, name, obj, side, dataset_metadata))

    handle.visititems(visitor)
    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    points: list[np.ndarray] = []
    labels: list[str] = []
    validity: list[np.ndarray] = []
    sides: list[str] = []
    fields: list[str] = []
    used_sides: set[str] = set()
    unknown_used = False
    for _, _, name, dataset, side, _ in candidates:
        side_key = side or "unknown"
        if side in used_sides or (side is None and unknown_used):
            continue
        value = _frame_value(dataset, index, "transform" if dataset.shape[-3:] == (26, 4, 4) else "value")
        frame_validity = _sibling_validity(handle, dataset, index)
        converted, valid = mano21_points_from_openxr_value(
            value,
            validity=frame_validity,
            wrist_to_palm_ratio=wrist_to_palm_ratio,
        )
        if converted.shape != (21, 3):
            continue
        points.append(converted)
        labels.extend(mano21_labels(side))
        validity.append(valid)
        fields.append(name)
        if side:
            sides.append(side)
            used_sides.add(side)
        else:
            unknown_used = True
        if used_sides == {"left", "right"}:
            break
    if not points:
        return None
    return OpenXRHandFrame(
        points=np.concatenate(points, axis=0),
        labels=tuple(labels),
        valid=np.concatenate(validity, axis=0),
        sides=tuple(sides),
        source_fields=tuple(fields),
        wrist_to_palm_ratio=float(wrist_to_palm_ratio),
    )


def read_openxr_hdf5_series(
    handle: Any,
    *,
    source_path: str = "",
    wrist_to_palm_ratio: float = DEFAULT_WRIST_TO_PALM_RATIO,
) -> OpenXRHandSeries | None:
    """Load selected OpenXR hand streams once for low-latency playback."""

    selected = read_openxr_hdf5_frame(
        handle,
        0,
        source_path=source_path,
        wrist_to_palm_ratio=wrist_to_palm_ratio,
    )
    if selected is None:
        return None

    hand_points: list[np.ndarray] = []
    hand_validity: list[np.ndarray] = []
    for field in selected.source_fields:
        dataset = handle.get(field)
        if dataset is None:
            return None
        values = np.asarray(dataset[()])
        validity = _sibling_validity_series(handle, dataset)
        points, valid = mano21_points_from_openxr_value(
            values,
            validity=validity,
            wrist_to_palm_ratio=wrist_to_palm_ratio,
        )
        if points.ndim == 2:
            points = points[None, ...]
            valid = valid[None, ...]
        if points.ndim != 3 or points.shape[-2:] != (21, 3):
            return None
        hand_points.append(points)
        hand_validity.append(valid)

    if not hand_points:
        return None
    non_static_counts = [int(item.shape[0]) for item in hand_points if int(item.shape[0]) > 1]
    frame_count = min(non_static_counts) if non_static_counts else 1
    aligned_points = [
        np.repeat(item, frame_count, axis=0) if item.shape[0] == 1 and frame_count > 1 else item[:frame_count]
        for item in hand_points
    ]
    aligned_validity = [
        np.repeat(item, frame_count, axis=0) if item.shape[0] == 1 and frame_count > 1 else item[:frame_count]
        for item in hand_validity
    ]
    return OpenXRHandSeries(
        points=np.concatenate(aligned_points, axis=1),
        labels=selected.labels,
        valid=np.concatenate(aligned_validity, axis=1),
        sides=selected.sides,
        source_fields=selected.source_fields,
        wrist_to_palm_ratio=float(wrist_to_palm_ratio),
    )


def source_path_has_openxr_hint(path: str | Path, field: str = "") -> bool:
    detection = detect_openxr_schema(
        source_path=str(path), field=field, shape=(OPENXR_JOINT_COUNT, 3),
    )
    return bool(detection["detected"])
