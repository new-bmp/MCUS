from __future__ import annotations

"""Kinematic EgoDex -> MANO21 retargeting.

The source EgoDex skeleton contains a rigid hand node, explicit metacarpals and
globally oriented transforms for every phalanx.  A MANO/MediaPipe-style 21
joint hand has a visual wrist, five palm anchors and three articulated bones
per finger.  Retargeting therefore has to estimate a palm frame and reconstruct
the fingers from joint rotations; selecting 21 similarly named translations is
not sufficient.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .mano21 import MANO21_EDGES, side_hand_joint_names


EGODEX_MANO_SCHEMA = "alice/egodex-mano21-retarget/v1"
EGODEX_MANO_REVISION = "rigid-palm-relative-rotation-fk-v1"
MANO21_RETARGETED_ATTRIBUTE = "alice_mano21_retargeted"
MANO21_RETARGET_REVISION_ATTRIBUTE = "alice_mano21_retarget_revision"
WRIST_BACK_PALM_WIDTH_RATIO = 0.45

_FINGER_STEMS = ("Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")
_MCP_INDICES = (1, 5, 9, 13, 17)
_NON_THUMB_STEMS = ("IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")


@dataclass(frozen=True)
class EgoDexManoTemplate:
    side: str
    mcp_offsets: np.ndarray
    bone_offsets: np.ndarray
    wrist_back_ratio: float
    sample_count: int

    def metadata(self) -> dict:
        return {
            "schema": EGODEX_MANO_SCHEMA,
            "revision": EGODEX_MANO_REVISION,
            "side": self.side,
            "sample_count": int(self.sample_count),
            "wrist_back_palm_width_ratio": float(self.wrist_back_ratio),
            "mcp_offsets": np.asarray(self.mcp_offsets, dtype=float).round(9).tolist(),
            "bone_offsets": np.asarray(self.bone_offsets, dtype=float).round(9).tolist(),
        }


def required_egodex_mano_names(side: str) -> tuple[str, ...]:
    normalized = str(side).strip().casefold()
    direct = side_hand_joint_names(normalized)
    metacarpals = tuple(f"{normalized}{stem}Metacarpal" for stem in _NON_THUMB_STEMS)
    return tuple(dict.fromkeys((*direct, *metacarpals)))


def has_egodex_mano_source(values: Mapping[str, Any], side: str) -> bool:
    return all(name in values for name in required_egodex_mano_names(side))


def source_is_retargeted(source: Any) -> bool:
    attrs = getattr(source, "attrs", source)
    try:
        return bool(attrs.get(MANO21_RETARGETED_ATTRIBUTE, False))
    except (AttributeError, TypeError, ValueError):
        return False


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if np.isfinite(norm) and norm > 1e-9:
        return value / norm
    replacement = np.asarray(fallback, dtype=np.float64)
    return replacement / max(float(np.linalg.norm(replacement)), 1e-12)


def _matrix(named: Mapping[str, Any], name: str) -> np.ndarray:
    value = np.asarray(named[name], dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"Invalid EgoDex transform: {name}")
    return value


def estimate_mano_palm_frame(
    named: Mapping[str, Any],
    side: str,
    *,
    wrist_back_ratio: float = WRIST_BACK_PALM_WIDTH_RATIO,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate visual wrist translation and palm rotation from full palm bones."""

    normalized = str(side).strip().casefold()
    metacarpals = np.stack([
        _matrix(named, f"{normalized}{stem}Metacarpal")[:3, 3]
        for stem in _NON_THUMB_STEMS
    ])
    knuckles = np.stack([
        _matrix(named, f"{normalized}{stem}Knuckle")[:3, 3]
        for stem in _NON_THUMB_STEMS
    ])
    meta_center = np.median(metacarpals, axis=0)
    knuckle_center = np.median(knuckles, axis=0)
    hand_rotation = _orthonormalize(_matrix(named, f"{normalized}Hand")[:3, :3])

    thumbward = _unit(metacarpals[0] - metacarpals[-1], hand_rotation[:, 0])
    distal = knuckle_center - meta_center
    distal = distal - thumbward * float(distal @ thumbward)
    distal = _unit(distal, hand_rotation[:, 1])
    normal = _unit(np.cross(thumbward, distal), hand_rotation[:, 2])
    distal = _unit(np.cross(normal, thumbward), distal)
    rotation = _orthonormalize(np.column_stack((thumbward, distal, normal)))

    palm_width = float(np.linalg.norm(metacarpals[0] - metacarpals[-1]))
    if not np.isfinite(palm_width) or palm_width <= 1e-8:
        pairwise = np.linalg.norm(metacarpals[:, None, :] - metacarpals[None, :, :], axis=2)
        palm_width = float(np.max(pairwise)) if pairwise.size else 0.08
    wrist = meta_center - rotation[:, 1] * max(1e-6, palm_width * float(wrist_back_ratio))
    return wrist, rotation, {
        "palm_width": palm_width,
        "meta_center": meta_center,
        "knuckle_center": knuckle_center,
    }


def _sample_rows(values: Mapping[str, Any], names: tuple[str, ...], maximum: int) -> np.ndarray:
    counts = [int(values[name].shape[0]) for name in names]
    count = min(counts)
    if count <= 0:
        raise ValueError("EgoDex hand transforms are empty")
    sample_count = min(max(1, int(maximum)), count)
    return np.unique(np.rint(np.linspace(0, count - 1, sample_count)).astype(np.int64))


def fit_egodex_mano_template(
    transforms: Mapping[str, Any],
    side: str,
    *,
    sample_rows: np.ndarray | None = None,
    maximum_samples: int = 64,
    wrist_back_ratio: float = WRIST_BACK_PALM_WIDTH_RATIO,
) -> EgoDexManoTemplate:
    """Fit fixed palm anchors and local bone vectors from an EgoDex episode."""

    normalized = str(side).strip().casefold()
    names = required_egodex_mano_names(normalized)
    missing = [name for name in names if name not in transforms]
    if missing:
        raise KeyError(f"Missing EgoDex MANO source transforms: {', '.join(missing)}")
    rows = (
        _sample_rows(transforms, names, maximum_samples)
        if sample_rows is None
        else np.unique(np.asarray(sample_rows, dtype=np.int64).reshape(-1))
    )
    if not len(rows):
        raise ValueError("MANO template fitting requires at least one source row")
    sampled = {name: np.asarray(transforms[name][rows.tolist()], dtype=np.float64) for name in names}

    mcp_samples: list[np.ndarray] = []
    bone_samples: list[np.ndarray] = []
    direct_names = side_hand_joint_names(normalized)
    for sample_index in range(len(rows)):
        named = {name: sampled[name][sample_index] for name in names}
        wrist, palm_rotation, _ = estimate_mano_palm_frame(
            named,
            normalized,
            wrist_back_ratio=wrist_back_ratio,
        )
        mcp_samples.append(np.stack([
            palm_rotation.T @ (_matrix(named, direct_names[index])[:3, 3] - wrist)
            for index in _MCP_INDICES
        ]))
        offsets: list[np.ndarray] = []
        for parent, child in MANO21_EDGES:
            if parent == 0:
                offsets.append(np.zeros(3, dtype=np.float64))
                continue
            parent_matrix = _matrix(named, direct_names[parent])
            child_matrix = _matrix(named, direct_names[child])
            parent_rotation = _orthonormalize(parent_matrix[:3, :3])
            offsets.append(parent_rotation.T @ (child_matrix[:3, 3] - parent_matrix[:3, 3]))
        bone_samples.append(np.stack(offsets))

    mcp_offsets = np.median(np.stack(mcp_samples), axis=0)
    bone_offsets = np.median(np.stack(bone_samples), axis=0)
    return EgoDexManoTemplate(
        side=normalized,
        mcp_offsets=np.asarray(mcp_offsets, dtype=np.float64),
        bone_offsets=np.asarray(bone_offsets, dtype=np.float64),
        wrist_back_ratio=float(wrist_back_ratio),
        sample_count=int(len(rows)),
    )


def direct_mano21_transforms(named: Mapping[str, Any], side: str) -> np.ndarray:
    names = side_hand_joint_names(side)
    return np.stack([_matrix(named, name) for name in names])


def retarget_egodex_mano_frame(
    named: Mapping[str, Any],
    template: EgoDexManoTemplate,
) -> np.ndarray:
    """Reconstruct a MANO21 transform tree from palm pose and joint angles."""

    side = template.side
    names = side_hand_joint_names(side)
    wrist, palm_rotation, _ = estimate_mano_palm_frame(
        named,
        side,
        wrist_back_ratio=template.wrist_back_ratio,
    )
    output = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 21, axis=0)
    output[0, :3, :3] = palm_rotation
    output[0, :3, 3] = wrist

    for anchor_index, joint_index in enumerate(_MCP_INDICES):
        output[joint_index, :3, 3] = wrist + palm_rotation @ template.mcp_offsets[anchor_index]
        output[joint_index, :3, :3] = _orthonormalize(_matrix(named, names[joint_index])[:3, :3])

    for edge_index, (parent, child) in enumerate(MANO21_EDGES):
        if parent == 0:
            continue
        parent_rotation = output[parent, :3, :3]
        output[child, :3, 3] = output[parent, :3, 3] + parent_rotation @ template.bone_offsets[edge_index]
        output[child, :3, :3] = _orthonormalize(_matrix(named, names[child])[:3, :3])

    if not np.isfinite(output).all():
        raise ValueError("EgoDex to MANO21 retargeting produced non-finite transforms")
    return output


def retarget_or_read_mano_frame(
    named: Mapping[str, Any],
    template: EgoDexManoTemplate,
    *,
    already_retargeted: bool = False,
) -> np.ndarray:
    return direct_mano21_transforms(named, template.side) if already_retargeted else retarget_egodex_mano_frame(named, template)
