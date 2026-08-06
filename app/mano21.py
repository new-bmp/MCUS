from __future__ import annotations

"""Canonical MANO-compatible 21-keypoint hand layout.

EgoDex stores a full named kinematic skeleton, including body joints and an
extra metacarpal transform on each non-thumb finger. A MANO 21-keypoint hand
contains one wrist plus four points per finger. These helpers perform that
layout conversion explicitly instead of renaming an unfiltered EgoDex tree.

This is a joint-layout conversion, not MANO mesh/shape-parameter fitting.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


MANO21_LAYOUT_VERSION = "egodex-to-mano21-v1"

HAND_21_JOINT_NAMES = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
)

MANO21_FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
MANO21_EDGES = tuple(edge for chain in MANO21_FINGER_CHAINS for edge in zip(chain, chain[1:]))


def side_hand_joint_names(side: str) -> tuple[str, ...]:
    normalized = str(side).strip().casefold()
    if normalized not in {"left", "right"}:
        raise ValueError(f"Unsupported hand side: {side}")
    return tuple(f"{normalized}{name}" for name in HAND_21_JOINT_NAMES)


def mano21_transforms_from_named(
    values: Mapping[str, Any],
    side: str,
    *,
    axis: int = 1,
) -> np.ndarray:
    """Stack named EgoDex transforms into canonical MANO21 order.

    Non-thumb EgoDex metacarpals are intentionally omitted. The source
    ``leftHand``/``rightHand`` rigid wrist is joint 0; body and forearm
    transforms remain separate data.
    """

    names = side_hand_joint_names(side)
    missing = [name for name in names if name not in values]
    if missing:
        raise KeyError(f"Missing MANO21 source transforms: {', '.join(missing)}")
    return np.stack([np.asarray(values[name]) for name in names], axis=axis)


def _leaf_key(label: str) -> str:
    leaf = str(label).replace("\\", "/").rsplit("/", 1)[-1]
    return "".join(character for character in leaf.casefold() if character.isalnum())


def select_mano21_points(
    points: np.ndarray,
    labels: Sequence[str],
) -> tuple[np.ndarray, list[str], np.ndarray, tuple[str, ...]]:
    """Select complete named hands and return them in MANO21 order.

    If no complete named hand is present, the input is returned unchanged and
    ``sides`` is empty. Generic joint arrays therefore remain usable without
    being misrepresented as MANO.
    """

    values = np.asarray(points)
    names = [str(label) for label in labels]
    if values.ndim < 2 or values.shape[0] != len(names):
        return values, names, np.arange(len(names), dtype=np.int64), ()

    buckets: dict[str, list[int]] = {}
    for index, label in enumerate(names):
        buckets.setdefault(_leaf_key(label), []).append(index)

    selected_indices: list[int] = []
    selected_labels: list[str] = []
    selected_sides: list[str] = []
    for side in ("left", "right"):
        required = side_hand_joint_names(side)
        resolved: list[int] = []
        for name in required:
            matches = buckets.get(_leaf_key(name), [])
            if len(matches) != 1:
                resolved = []
                break
            resolved.append(matches[0])
        if not resolved:
            continue
        selected_indices.extend(resolved)
        selected_labels.extend(required)
        selected_sides.append(side)

    if not selected_indices:
        return values, names, np.arange(len(names), dtype=np.int64), ()
    indices = np.asarray(selected_indices, dtype=np.int64)
    return values[indices], selected_labels, indices, tuple(selected_sides)


def mano21_local_index(label: str) -> int | None:
    key = _leaf_key(label)
    for side in ("left", "right"):
        if not key.startswith(side):
            continue
        suffix = key[len(side):]
        for index, name in enumerate(HAND_21_JOINT_NAMES):
            if suffix == _leaf_key(name):
                return index
    return None
