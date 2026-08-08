from __future__ import annotations

import unittest

import numpy as np

from app.egodex_mano import (
    EGODEX_MANO_REVISION,
    MAX_WRIST_EXTENSION_PALM_WIDTH_RATIO,
    MIN_WRIST_EXTENSION_PALM_WIDTH_RATIO,
    estimate_mano_palm_frame,
    fit_egodex_mano_template,
    required_egodex_mano_names,
    retarget_egodex_mano_frame,
)
from app.mano21 import MANO21_EDGES, side_hand_joint_names


def _translation(x: float, y: float, z: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = (x, y, z)
    return value


def _synthetic_hand(side: str, frame_count: int = 3) -> dict[str, np.ndarray]:
    sign = -1.0 if side == "left" else 1.0
    result = {name: np.repeat(np.eye(4, dtype=np.float64)[None, ...], frame_count, axis=0) for name in required_egodex_mano_names(side)}
    result[f"{side}Forearm"] = np.repeat(np.eye(4, dtype=np.float64)[None, ...], frame_count, axis=0)
    for frame in range(frame_count):
        shift = np.array([0.01 * frame, 0.0, 1.0])
        result[f"{side}Hand"][frame, :3, 3] = shift + (0.0, -0.04, 0.0)
        result[f"{side}Forearm"][frame, :3, 3] = shift + (0.0, -0.30, 0.0)
        for finger_index, stem in enumerate(("IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")):
            x = sign * (0.03 - finger_index * 0.02)
            meta = shift + (x, 0.0, 0.0)
            knuckle = shift + (x, 0.06, 0.0)
            result[f"{side}{stem}Metacarpal"][frame, :3, 3] = meta
            result[f"{side}{stem}Knuckle"][frame, :3, 3] = knuckle
        result[f"{side}ThumbKnuckle"][frame, :3, 3] = shift + (sign * 0.05, 0.025, 0.0)
        names = side_hand_joint_names(side)
        for chain in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
            for parent, child in zip(chain, chain[1:]):
                parent_name, child_name = names[parent], names[child]
                parent_position = result[parent_name][frame, :3, 3]
                result[child_name][frame, :3, 3] = parent_position + (0.0, 0.03, 0.0)
    return result


class EgoDexManoRetargetTests(unittest.TestCase):
    def test_palm_frame_estimates_visual_wrist_without_copying_hand_node(self) -> None:
        arrays = _synthetic_hand("left", 1)
        named = {name: value[0] for name, value in arrays.items()}
        wrist, rotation, details = estimate_mano_palm_frame(named, "left")

        self.assertGreater(float(np.linalg.norm(wrist - named["leftHand"][:3, 3])), 1e-3)
        self.assertAlmostEqual(1.0, float(np.linalg.det(rotation)), places=6)
        self.assertGreater(details["palm_width"], 0.01)

    def test_retarget_uses_fixed_palm_and_forward_kinematics(self) -> None:
        arrays = _synthetic_hand("right")
        template = fit_egodex_mano_template(arrays, "right")
        named = {name: value[2] for name, value in arrays.items()}
        retargeted = retarget_egodex_mano_frame(named, template)

        self.assertEqual((21, 4, 4), retargeted.shape)
        self.assertEqual(EGODEX_MANO_REVISION, template.metadata()["revision"])
        self.assertGreater(float(np.linalg.norm(retargeted[0, :3, 3] - named["rightHand"][:3, 3])), 1e-3)
        for parent, child in MANO21_EDGES:
            self.assertGreater(float(np.linalg.norm(retargeted[child, :3, 3] - retargeted[parent, :3, 3])), 1e-5)

    def test_retarget_places_wrist_on_forearm_hand_axis(self) -> None:
        arrays = _synthetic_hand("right", 1)
        template = fit_egodex_mano_template(arrays, "right")
        named = {name: value[0] for name, value in arrays.items()}
        wrist, _rotation, details = estimate_mano_palm_frame(named, "right")

        hand = named["rightHand"][:3, 3]
        forearm = named["rightForearm"][:3, 3]
        axis = (hand - forearm) / np.linalg.norm(hand - forearm)
        extension = float(np.linalg.norm(wrist - hand))

        self.assertGreater(float((wrist - hand) @ axis), 0.0)
        self.assertAlmostEqual(extension, details["wrist_extension"], places=8)
        self.assertGreaterEqual(extension, details["palm_width"] * MIN_WRIST_EXTENSION_PALM_WIDTH_RATIO - 1e-8)
        self.assertLessEqual(extension, details["palm_width"] * MAX_WRIST_EXTENSION_PALM_WIDTH_RATIO + 1e-8)
        self.assertGreater(template.metadata()["forearm_hand_wrist_extension_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
