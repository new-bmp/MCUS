from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.curation_pipeline import _load_signal_bundle, _signal_candidates
from app.full_export import _find_transform_source
from app.joint_overlay import _CANDIDATE_CACHE, _candidate_sources
from app.lerobot_export import side_hand_joint_names
from app.projection_correction import (
    HAND_BONES,
    PALM_BASE_INDICES,
    PROJECTION_CORRECTION_SCHEMA,
    VISUAL_PALM_BASE_INDICES,
    VISUAL_WRIST_PALM_WIDTH_RATIO,
    _assign_full_frame_hands,
    _choose_pose,
    _confidence_adjustment_multiplier,
    _confidence_scaled_blend,
    _constrained_hand_correction,
    _intermediate_hand_points,
    _interpolate_observations,
    _interpolate_transform_rows,
    _map_original_s1_mask,
    _normalized_pose_displacement,
    _project_camera,
    _prepare_full_frame_hand_candidate,
    _refine_s1_insertion_positions,
    _s1_insertion_plan,
    _visual_wrist_anchor,
    _write_retimed_video,
    applied_projection_source,
    preferred_projection_media,
    preferred_projection_review_media,
)
from app.schemas import BatchAnalysisRequest, HandPoseModelConfig


def synthetic_hand() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = (0.0, 0.0, 0.65)
    bases = (
        (-0.035, 0.010, 0.650),
        (-0.025, 0.035, 0.650),
        (0.000, 0.042, 0.650),
        (0.024, 0.035, 0.650),
        (0.045, 0.025, 0.650),
    )
    chains = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    for chain, base in zip(chains, bases):
        points[chain[1]] = base
        direction = np.asarray(base, dtype=np.float64) - points[0]
        direction[2] = 0.004
        direction /= np.linalg.norm(direction)
        for parent, child in zip(chain[1:], chain[2:]):
            points[child] = points[parent] + direction * 0.025
    return points


class ProjectionCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = synthetic_hand()
        self.intrinsic = np.asarray(((900.0, 0.0, 960.0), (0.0, 900.0, 540.0), (0.0, 0.0, 1.0)))
        self.rotations = np.repeat(np.eye(3, dtype=np.float64)[None, ...], 21, axis=0)

    def test_constrained_correction_improves_projection_and_preserves_3d_scale(self) -> None:
        targets = _project_camera(self.points, self.intrinsic) + np.asarray((18.0, -10.0))
        result = _constrained_hand_correction(
            self.points,
            self.rotations,
            np.eye(4, dtype=np.float64),
            self.intrinsic,
            targets,
            np.ones(21, dtype=np.float64),
        )

        self.assertTrue(result["applied"])
        self.assertLess(result["after_median_px"], result["before_median_px"])
        corrected = result["camera_points"]
        for parent, child in HAND_BONES:
            self.assertAlmostEqual(
                float(np.linalg.norm(self.points[child] - self.points[parent])),
                float(np.linalg.norm(corrected[child] - corrected[parent])),
                delta=1e-8,
            )
        before_palm = self.points[list(PALM_BASE_INDICES)]
        after_palm = corrected[list(PALM_BASE_INDICES)]
        for left in range(len(PALM_BASE_INDICES)):
            for right in range(left + 1, len(PALM_BASE_INDICES)):
                self.assertAlmostEqual(
                    float(np.linalg.norm(before_palm[left] - before_palm[right])),
                    float(np.linalg.norm(after_palm[left] - after_palm[right])),
                    delta=1e-8,
                )

    def test_low_confidence_and_no_improvement_do_not_modify_3d(self) -> None:
        targets = _project_camera(self.points, self.intrinsic) + np.asarray((12.0, 4.0))
        low = _constrained_hand_correction(
            self.points, self.rotations, np.eye(4), self.intrinsic, targets, np.full(21, 0.1)
        )
        self.assertFalse(low["applied"])
        self.assertEqual("insufficient_confident_keypoints", low["reason"])

        unchanged = _constrained_hand_correction(
            self.points,
            self.rotations,
            np.eye(4),
            self.intrinsic,
            _project_camera(self.points, self.intrinsic),
            np.ones(21),
        )
        self.assertFalse(unchanged["applied"])
        self.assertEqual("no_reprojection_improvement", unchanged["reason"])

    def test_mediapipe_full_approval_uses_all_points_without_confidence_scaling(self) -> None:
        source = _project_camera(self.points, self.intrinsic)
        candidate = {
            "keypoints": source + np.asarray((18.0, -10.0)),
            "confidence": np.full(21, 0.01),
            "box_confidence": 0.01,
            "backend": "mediapipe",
        }
        prepared = _prepare_full_frame_hand_candidate(candidate, 1920, 1080)
        self.assertIsNotNone(prepared)
        np.testing.assert_allclose(prepared["confidence"], np.ones(21))
        self.assertTrue(prepared["full_approval"])

        equal = _constrained_hand_correction(
            self.points,
            self.rotations,
            np.eye(4),
            self.intrinsic,
            candidate["keypoints"],
            np.zeros(21),
            confidence_policy="mediapipe_full_approval",
        )
        reference = _constrained_hand_correction(
            self.points,
            self.rotations,
            np.eye(4),
            self.intrinsic,
            candidate["keypoints"],
            np.ones(21),
            confidence_policy="mediapipe_full_approval",
        )
        self.assertTrue(equal["applied"])
        self.assertTrue(reference["applied"])
        np.testing.assert_allclose(equal["camera_points"], reference["camera_points"], atol=1e-10)

        low_rate = _constrained_hand_correction(
            self.points,
            self.rotations,
            np.eye(4),
            self.intrinsic,
            candidate["keypoints"],
            np.zeros(21),
            confidence_policy="mediapipe_full_approval",
            local_blend=0.2,
        )
        high_rate = _constrained_hand_correction(
            self.points,
            self.rotations,
            np.eye(4),
            self.intrinsic,
            candidate["keypoints"],
            np.zeros(21),
            confidence_policy="mediapipe_full_approval",
            local_blend=0.9,
        )
        self.assertTrue(low_rate["applied"])
        self.assertTrue(high_rate["applied"])
        low_motion = float(np.mean(np.linalg.norm(low_rate["camera_points"] - self.points, axis=1)))
        high_motion = float(np.mean(np.linalg.norm(high_rate["camera_points"] - self.points, axis=1)))
        self.assertLess(low_motion, high_motion)

    def test_node_confidence_uses_smooth_multiplier_curve(self) -> None:
        self.assertAlmostEqual(0.4, _confidence_adjustment_multiplier(0.18), places=8)
        self.assertAlmostEqual(1.0, _confidence_adjustment_multiplier(0.60), places=8)
        self.assertAlmostEqual(2.0, _confidence_adjustment_multiplier(1.00), places=8)
        samples = np.linspace(0.18, 1.0, 200)
        multipliers = np.asarray([_confidence_adjustment_multiplier(value) for value in samples])
        self.assertTrue((np.diff(multipliers) > 0.0).all())
        self.assertLess(float(np.max(np.abs(np.diff(multipliers, n=2)))), 0.001)

        self.assertAlmostEqual(0.58 * 0.4, _confidence_scaled_blend(0.18, 0.18, 0.58))
        self.assertAlmostEqual(0.58, _confidence_scaled_blend(0.60, 0.18, 0.58))
        self.assertAlmostEqual(0.85, _confidence_scaled_blend(1.00, 0.18, 0.58))
        self.assertEqual(0.0, _confidence_scaled_blend(0.17, 0.18, 0.58))

        targets = _project_camera(self.points, self.intrinsic) + np.asarray((30.0, -15.0))
        high = _constrained_hand_correction(
            self.points, self.rotations, np.eye(4), self.intrinsic, targets, np.ones(21)
        )
        medium = _constrained_hand_correction(
            self.points, self.rotations, np.eye(4), self.intrinsic, targets, np.full(21, 0.6)
        )
        self.assertTrue(high["applied"])
        self.assertTrue(medium["applied"])
        high_motion = float(np.mean(np.linalg.norm(high["camera_points"] - self.points, axis=1)))
        medium_motion = float(np.mean(np.linalg.norm(medium["camera_points"] - self.points, axis=1)))
        self.assertLess(medium_motion, high_motion)

    def test_dynamic_curve_parameters_change_the_adjustment_strength(self) -> None:
        default = _confidence_adjustment_multiplier(0.9)
        conservative = _confidence_adjustment_multiplier(
            0.9,
            0.25,
            middle_confidence=0.7,
            low_multiplier=0.2,
            middle_multiplier=0.6,
            high_multiplier=1.0,
        )
        self.assertLess(conservative, default)
        self.assertAlmostEqual(
            0.6,
            _confidence_adjustment_multiplier(
                0.7,
                0.25,
                middle_confidence=0.7,
                low_multiplier=0.2,
                middle_multiplier=0.6,
                high_multiplier=1.0,
            ),
        )

    def test_uniform_adjustment_ignores_confidence_magnitude_after_eligibility(self) -> None:
        targets = _project_camera(self.points, self.intrinsic) + np.asarray((30.0, -15.0))
        low_confidence = _constrained_hand_correction(
            self.points,
            self.rotations,
            np.eye(4),
            self.intrinsic,
            targets,
            np.full(21, 0.2),
            confidence_policy="uniform",
            local_blend=0.58,
        )
        high_confidence = _constrained_hand_correction(
            self.points,
            self.rotations,
            np.eye(4),
            self.intrinsic,
            targets,
            np.ones(21),
            confidence_policy="uniform",
            local_blend=0.58,
        )
        self.assertTrue(low_confidence["applied"])
        self.assertTrue(high_confidence["applied"])
        np.testing.assert_allclose(low_confidence["camera_points"], high_confidence["camera_points"], atol=1e-10)

    def test_pose_selection_rejects_confident_other_hand_candidate(self) -> None:
        source = _project_camera(self.points, self.intrinsic)
        good = {
            "keypoints": source + np.asarray((24.0, -12.0)),
            "source_pixels": source,
            "confidence": np.full(21, 0.82),
            "box_confidence": 0.78,
        }
        wrong = {
            "keypoints": source + np.asarray((280.0, 40.0)),
            "source_pixels": source,
            "confidence": np.full(21, 0.99),
            "box_confidence": 0.99,
        }
        self.assertIs(good, _choose_pose(wrong, good))
        self.assertIsNone(_choose_pose(wrong, None))

    def test_pose_selection_uses_temporal_displacement_continuity(self) -> None:
        source = _project_camera(self.points, self.intrinsic)
        previous = {
            "keypoints": source + np.asarray((20.0, -8.0)),
            "source_pixels": source,
            "confidence": np.full(21, 0.9),
            "box_confidence": 0.9,
        }
        continuous = {
            "keypoints": source + np.asarray((23.0, -10.0)),
            "source_pixels": source,
            "confidence": np.full(21, 0.8),
            "box_confidence": 0.8,
        }
        jump = {
            "keypoints": source + np.asarray((115.0, 0.0)),
            "source_pixels": source,
            "confidence": np.full(21, 0.99),
            "box_confidence": 0.99,
        }
        self.assertIs(continuous, _choose_pose(jump, continuous, previous))

    def test_full_frame_assignment_keeps_two_hands_unique(self) -> None:
        base = _project_camera(self.points, self.intrinsic)
        left_source = base + np.asarray((-150.0, 0.0))
        right_source = base + np.asarray((150.0, 0.0))
        candidates = [
            {
                "keypoints": right_source + np.asarray((18.0, -5.0)),
                "confidence": np.full(21, 0.92),
                "box_confidence": 0.92,
                "handedness": "left",
                "handedness_score": 0.9,
            },
            {
                "keypoints": left_source + np.asarray((-12.0, 7.0)),
                "confidence": np.full(21, 0.88),
                "box_confidence": 0.88,
                "handedness": "right",
                "handedness_score": 0.86,
            },
        ]
        assigned = _assign_full_frame_hands(
            candidates,
            {"left": left_source, "right": right_source},
            {"left": None, "right": None},
            1920,
            1080,
        )
        self.assertIsNotNone(assigned["left"])
        self.assertIsNotNone(assigned["right"])
        self.assertLess(float(np.mean(assigned["left"]["keypoints"][:, 0])), 960.0)
        self.assertGreater(float(np.mean(assigned["right"]["keypoints"][:, 0])), 960.0)
        self.assertEqual("left", assigned["left"]["camera_handedness"])
        self.assertEqual("right", assigned["right"]["camera_handedness"])

    def test_full_frame_quality_rejects_collapsed_landmarks(self) -> None:
        collapsed = {
            "keypoints": np.repeat(np.asarray(((900.0, 500.0),)), 21, axis=0),
            "confidence": np.ones(21),
            "box_confidence": 0.99,
        }
        self.assertIsNone(_prepare_full_frame_hand_candidate(collapsed, 1920, 1080))

    def test_interpolation_does_not_cross_long_rejected_gap(self) -> None:
        displacement = np.ones((21, 2), dtype=np.float64)
        confidence = np.ones(21, dtype=np.float64)
        _, _, available = _interpolate_observations(
            12,
            {1: (displacement, confidence), 10: (displacement * 2.0, confidence)},
            maximum_gap_frames=4,
        )
        self.assertTrue(available[1].all())
        self.assertTrue(available[10].all())
        self.assertFalse(available[2:10].any())

    def test_normalized_displacement_is_scale_invariant_and_proportion_limited(self) -> None:
        source = _project_camera(self.points, self.intrinsic)
        radius = np.percentile(np.linalg.norm(source - _visual_wrist_anchor(source), axis=1)[1:], 75)
        target = source + np.asarray((radius * 0.9, -radius * 0.4))
        target[4] += np.asarray((radius * 0.8, radius * 0.5))
        normalized, measured_radius = _normalized_pose_displacement(target, source)

        scaled_source = source[0] + (source - source[0]) * 2.0
        scaled_target = scaled_source + (target - source) * 2.0
        scaled_normalized, scaled_radius = _normalized_pose_displacement(scaled_target, scaled_source)
        np.testing.assert_allclose(normalized, scaled_normalized, atol=1e-10)
        self.assertAlmostEqual(measured_radius * 2.0, scaled_radius, places=8)
        self.assertLessEqual(float(np.linalg.norm(normalized[0])), 0.450001)
        self.assertLessEqual(float(np.linalg.norm(normalized[4])), 0.750001)

    def test_visual_wrist_anchor_caps_a_forearm_like_root_by_palm_width(self) -> None:
        source = _project_camera(self.points, self.intrinsic)
        palm = source[list(VISUAL_PALM_BASE_INDICES)]
        palm_center = np.median(palm, axis=0)
        palm_width = float(np.max(np.linalg.norm(palm[:, None, :] - palm[None, :, :], axis=2)))
        source[0] = palm_center + np.asarray((0.0, palm_width * 4.0))

        anchor = _visual_wrist_anchor(source)

        self.assertAlmostEqual(
            palm_width * VISUAL_WRIST_PALM_WIDTH_RATIO,
            float(np.linalg.norm(anchor - palm_center)),
            places=8,
        )
        self.assertLess(float(np.linalg.norm(anchor - palm_center)), float(np.linalg.norm(source[0] - palm_center)))
        np.testing.assert_allclose(
            (anchor - palm_center) / np.linalg.norm(anchor - palm_center),
            (source[0] - palm_center) / np.linalg.norm(source[0] - palm_center),
            atol=1e-10,
        )

    def test_forearm_like_root_no_longer_changes_finger_normalization_or_radius(self) -> None:
        source = _project_camera(self.points, self.intrinsic)
        palm_center = np.median(source[list(VISUAL_PALM_BASE_INDICES)], axis=0)
        near = source.copy()
        far = source.copy()
        near[0] = palm_center + np.asarray((0.0, 240.0))
        far[0] = palm_center + np.asarray((0.0, 720.0))
        target_near = near + np.asarray((20.0, -12.0))
        target_far = far + np.asarray((20.0, -12.0))

        normalized_near, radius_near = _normalized_pose_displacement(target_near, near)
        normalized_far, radius_far = _normalized_pose_displacement(target_far, far)

        self.assertAlmostEqual(radius_near, radius_far, places=8)
        np.testing.assert_allclose(normalized_near[1:], normalized_far[1:], atol=1e-10)

    def test_forearm_constraint_keeps_the_real_wrist_root_length(self) -> None:
        forearm = self.points[0] + np.asarray((0.0, -0.265, 0.0))
        targets = _project_camera(self.points, self.intrinsic) + np.asarray((18.0, -10.0))
        result = _constrained_hand_correction(
            self.points,
            self.rotations,
            np.eye(4),
            self.intrinsic,
            targets,
            np.ones(21),
            forearm_camera=forearm,
        )

        self.assertTrue(result["applied"])
        self.assertAlmostEqual(
            float(np.linalg.norm(self.points[0] - forearm)),
            float(np.linalg.norm(result["camera_points"][0] - forearm)),
            places=10,
        )

    def test_egodex_wrist_option_keeps_joint_zero_exactly_unchanged(self) -> None:
        targets = _project_camera(self.points, self.intrinsic) + np.asarray((24.0, -14.0))
        result = _constrained_hand_correction(
            self.points,
            self.rotations,
            np.eye(4),
            self.intrinsic,
            targets,
            np.ones(21),
            wrist_point_source="egodex",
        )

        self.assertTrue(result["applied"])
        np.testing.assert_allclose(result["camera_points"][0], self.points[0], atol=1e-12)

    def test_model_wrist_option_allows_landmark_zero_to_adjust_the_root(self) -> None:
        targets = _project_camera(self.points, self.intrinsic) + np.asarray((24.0, -14.0))
        result = _constrained_hand_correction(
            self.points,
            self.rotations,
            np.eye(4),
            self.intrinsic,
            targets,
            np.ones(21),
            wrist_point_source="model",
        )

        self.assertTrue(result["applied"])
        self.assertGreater(float(np.linalg.norm(result["camera_points"][0] - self.points[0])), 1e-6)

    def test_interpolation_smooths_spikes_without_crossing_gaps_or_adding_lag(self) -> None:
        confidence = np.ones(21, dtype=np.float64)
        zero = np.zeros((21, 2), dtype=np.float64)
        spike = np.zeros((21, 2), dtype=np.float64)
        spike[:, 0] = 0.8
        filtered, _, available = _interpolate_observations(
            13,
            {2: (zero, confidence), 6: (spike, confidence), 10: (zero, confidence)},
            maximum_gap_frames=4,
            smoothing_frames=7,
            taper_frames=2,
            maximum_step=0.04,
        )
        self.assertFalse(available[:2].any())
        self.assertFalse(available[11:].any())
        steps = np.linalg.norm(np.diff(filtered[2:11, 0], axis=0), axis=1)
        self.assertLessEqual(float(steps.max()), 0.0400001)
        self.assertAlmostEqual(float(filtered[4, 0, 0]), float(filtered[8, 0, 0]), places=10)
        self.assertLess(float(filtered[2, 0, 0]), float(filtered[5, 0, 0]))
        self.assertLess(float(filtered[10, 0, 0]), float(filtered[7, 0, 0]))

    def test_s1_insertion_plan_adds_one_midpoint_before_each_trigger(self) -> None:
        mask = np.zeros(8, dtype=bool)
        mask[[3, 6]] = True
        positions, insertions = _s1_insertion_plan(8, mask)

        np.testing.assert_allclose(positions, [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 5.5, 6.0, 7.0])
        self.assertEqual([3, 7], [item["output_frame"] for item in insertions])
        self.assertEqual([(2, 3), (5, 6)], [
            (item["left_source_frame"], item["right_source_frame"])
            for item in insertions
        ])

    def test_original_s1_failure_is_never_given_an_inserted_frame(self) -> None:
        values = np.zeros((8, 42 * 3), dtype=np.float64)
        original_mask = np.zeros(8, dtype=bool)
        original_mask[4] = True

        def corrected_detection(matrix: np.ndarray, _sigma: float) -> dict:
            mask = np.zeros(matrix.shape[0], dtype=bool)
            mask[4] = True
            return {"mask": mask, "event_count": int(mask.sum())}

        with patch("app.curation_pipeline.detect_sudden_changes", side_effect=corrected_detection):
            result = _refine_s1_insertion_positions(
                values,
                values,
                original_raw_mask=original_mask,
            )

        self.assertEqual([], result["insertions"])
        self.assertEqual(0, result["initial_event_count"])
        self.assertEqual(0, result["remaining_event_count"])

    def test_model_introduced_s1_failure_receives_an_inserted_frame(self) -> None:
        values = np.zeros((8, 42 * 3), dtype=np.float64)
        original_mask = np.zeros(8, dtype=bool)

        def corrected_detection(matrix: np.ndarray, _sigma: float) -> dict:
            mask = np.zeros(matrix.shape[0], dtype=bool)
            if matrix.shape[0] == 8:
                mask[4] = True
            return {"mask": mask, "event_count": int(mask.sum())}

        with patch("app.curation_pipeline.detect_sudden_changes", side_effect=corrected_detection):
            result = _refine_s1_insertion_positions(
                values,
                values,
                original_raw_mask=original_mask,
            )

        self.assertEqual(1, len(result["insertions"]))
        self.assertEqual(9, len(result["source_positions"]))
        self.assertAlmostEqual(3.5, result["insertions"][0]["source_position"])

    def test_original_s1_record_remains_protected_after_other_insertions(self) -> None:
        values = np.zeros((8, 42 * 3), dtype=np.float64)
        original_mask = np.zeros(8, dtype=bool)
        original_mask[4] = True

        def corrected_detection(matrix: np.ndarray, _sigma: float) -> dict:
            mask = np.zeros(matrix.shape[0], dtype=bool)
            if matrix.shape[0] == 8:
                mask[6] = True
            else:
                mapped_original = _map_original_s1_mask(
                    original_mask,
                    np.asarray((0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 7.0)),
                )
                mask[mapped_original] = True
            return {"mask": mask, "event_count": int(mask.sum())}

        with patch("app.curation_pipeline.detect_sudden_changes", side_effect=corrected_detection):
            result = _refine_s1_insertion_positions(
                values,
                values,
                original_raw_mask=original_mask,
            )

        self.assertEqual(1, len(result["insertions"]))
        self.assertAlmostEqual(5.5, result["insertions"][0]["source_position"])
        self.assertEqual(0, result["remaining_event_count"])

    def test_inserted_transform_uses_midpoint_translation_and_rotation_slerp(self) -> None:
        values = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        values[1, :3, :3] = cv2.Rodrigues(np.asarray((0.0, 0.0, np.pi / 2.0)))[0]
        values[1, :3, 3] = (2.0, 4.0, 6.0)
        output = _interpolate_transform_rows(values, np.asarray((0.0, 0.5, 1.0)))

        np.testing.assert_allclose(output[1, :3, 3], (1.0, 2.0, 3.0), atol=1e-10)
        angle = float(np.linalg.norm(cv2.Rodrigues(output[1, :3, :3])[0]))
        self.assertAlmostEqual(np.pi / 4.0, angle, places=8)

    def test_inserted_hand_preserves_average_bone_lengths_and_palm_proportions(self) -> None:
        left = self.points.copy()
        rotation = cv2.Rodrigues(np.asarray((0.12, -0.08, 0.16)))[0]
        right = (rotation @ (self.points - self.points[0]).T).T + self.points[0] + np.asarray((0.03, -0.02, 0.01))
        midpoint = _intermediate_hand_points(left, right)

        for parent, child in HAND_BONES:
            expected = 0.5 * (
                np.linalg.norm(left[child] - left[parent])
                + np.linalg.norm(right[child] - right[parent])
            )
            self.assertAlmostEqual(expected, float(np.linalg.norm(midpoint[child] - midpoint[parent])), delta=1e-9)
        for first in range(len(PALM_BASE_INDICES)):
            for second in range(first + 1, len(PALM_BASE_INDICES)):
                left_distance = np.linalg.norm(left[PALM_BASE_INDICES[first]] - left[PALM_BASE_INDICES[second]])
                right_distance = np.linalg.norm(right[PALM_BASE_INDICES[first]] - right[PALM_BASE_INDICES[second]])
                actual = np.linalg.norm(midpoint[PALM_BASE_INDICES[first]] - midpoint[PALM_BASE_INDICES[second]])
                self.assertLess(abs(actual - 0.5 * (left_distance + right_distance)) / left_distance, 0.05)

    def test_retimed_video_inserts_a_blended_frame_at_the_same_fps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", {"ALICE_VIDEO_ENCODER": "opencv"}):
            root = Path(temporary)
            source = root / "source.mp4"
            target = root / "retimed.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 12))
            self.assertTrue(writer.isOpened())
            for value in (0, 100, 200):
                writer.write(np.full((12, 16, 3), value, dtype=np.uint8))
            writer.release()
            positions = np.asarray((0.0, 0.5, 1.0, 2.0))

            info = _write_retimed_video(source, target, positions, 3, 10.0, 16, 12)

            self.assertEqual(4, info["frame_count"])
            capture = cv2.VideoCapture(str(target))
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            capture.release()
            self.assertEqual(4, len(frames))
            self.assertAlmostEqual(50.0, float(frames[1].mean()), delta=8.0)

    def test_applied_retimed_video_becomes_the_preferred_projection_media(self) -> None:
        metadata = {
            "source_video": {"file_id": "rgb", "fps": 30.0},
            "retiming": {
                "output_frame_count": 4,
                "source_frame_positions": [0.0, 0.5, 1.0, 2.0],
            },
        }
        applied = {
            "video_path": Path("retimed.mp4"),
            "metadata": metadata,
            "frame_count": 4,
            "application_id": "apply-1",
        }
        with patch("app.projection_correction.applied_projection_source", return_value=applied):
            media, document = preferred_projection_media(
                {},
                {"id": "ep", "fps": 30.0},
                {"file_id": "rgb", "path": "source.mp4", "frame_count": 3, "fps": 30.0},
            )

        self.assertIs(document, metadata)
        self.assertEqual(4, media["frame_count"])
        self.assertEqual([0.0, 0.5, 1.0, 2.0], media["source_frame_positions"])
        self.assertTrue(media["projection_retimed"])

    def test_pending_retimed_video_is_available_for_review_without_becoming_applied_media(self) -> None:
        metadata = {
            "source_video": {"file_id": "rgb", "fps": 30.0},
            "retiming": {
                "output_frame_count": 4,
                "source_frame_positions": [0.0, 0.5, 1.0, 2.0],
            },
        }
        staged = {
            "video_path": Path("pending-retimed.mp4"),
            "metadata": metadata,
            "frame_count": 4,
            "review_status": "pending",
            "applied": False,
        }
        source_media = {"file_id": "rgb", "path": "source.mp4", "frame_count": 3, "fps": 30.0}
        with (
            patch("app.projection_correction.review_projection_source", return_value=staged),
            patch("app.projection_correction.applied_projection_source", return_value=None),
        ):
            review_media, review_document = preferred_projection_review_media({}, {"id": "ep", "fps": 30.0}, source_media)
            processing_media, processing_document = preferred_projection_media({}, {"id": "ep", "fps": 30.0}, source_media)

        self.assertIs(review_document, metadata)
        self.assertEqual("pending-retimed.mp4", review_media["path"])
        self.assertEqual("pending", review_media["projection_review_status"])
        self.assertTrue(review_media["projection_review"])
        self.assertEqual(source_media, processing_media)
        self.assertIsNone(processing_document)

    def test_applied_snapshot_lookup_uses_reviewed_hdf5(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = Path(temporary)
            snapshot = sidecar / "changes" / "applications" / "apply-1" / "artifacts"
            snapshot.mkdir(parents=True)
            metadata_path = snapshot / "ep.projection.alice"
            hdf5_path = snapshot / "ep.projection.hdf5"
            hdf5_path.write_bytes(b"hdf5")
            metadata_path.write_text(json.dumps({
                "schema": PROJECTION_CORRECTION_SCHEMA,
                "source_transform": {"relative_path": "1.hdf5"},
                "summary": {"frame_count": 300},
            }), encoding="utf-8")
            current_path = sidecar / "changes" / "current.alice"
            current_path.write_text(json.dumps({
                "entries": {
                    "projection_correction:ep": {
                        "application_id": "apply-1",
                        "change_id": "change-1",
                        "artifacts": [
                            {"snapshot_path": hdf5_path.relative_to(sidecar).as_posix()},
                            {"snapshot_path": metadata_path.relative_to(sidecar).as_posix(), "schema": PROJECTION_CORRECTION_SCHEMA},
                        ],
                    }
                }
            }), encoding="utf-8")

            result = applied_projection_source({"sidecar_path": str(sidecar)}, {"id": "ep"})
            self.assertEqual(hdf5_path.resolve(), result["path"])
            self.assertEqual("1.hdf5", result["source_relative_path"])
            self.assertEqual(300, result["frame_count"])

    def test_joint_curation_and_export_prefer_applied_snapshot(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            sidecar = root / ".alicePD"
            root.mkdir()
            sidecar.mkdir()
            original = root / "1.hdf5"
            corrected = sidecar / "corrected.hdf5"
            names = (*side_hand_joint_names("left"), *side_hand_joint_names("right"), "camera")
            for path, offset in ((original, 0.0), (corrected, 0.1)):
                with h5py.File(path, "w") as handle:
                    transforms = handle.create_group("transforms")
                    for name in names:
                        values = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 4, axis=0)
                        values[:, 0, 3] = offset
                        transforms.create_dataset(name, data=values)
            manifest = {
                "id": "projection-test",
                "root_path": str(root),
                "sidecar_path": str(sidecar),
                "files": [{"relative_path": "1.hdf5", "episode_id": "ep", "extension": ".hdf5"}],
                "schema_profile": {"inventory": {"files": []}, "understanding": {"streams": []}},
            }
            episode = {"id": "ep", "frame_count": 4}
            applied = {
                "path": corrected,
                "source_relative_path": "1.hdf5",
                "application_id": "apply-1",
                "frame_count": 4,
            }
            with patch("app.joint_overlay.review_projection_source", return_value={**applied, "review_status": "applied", "applied": True}):
                _CANDIDATE_CACHE.clear()
                self.assertEqual("projection_correction", _candidate_sources(manifest, episode)[0]["source"])
            with patch("app.projection_correction.applied_projection_source", return_value=applied):
                export_path, relative, count = _find_transform_source(manifest, episode, output_format="lerobot")
                self.assertEqual(corrected.resolve(), export_path)
                self.assertEqual("1.hdf5", relative)
                self.assertEqual(4, count)
            with patch("app.projection_correction.applied_projection_source", return_value=applied):
                candidates = _signal_candidates(manifest, episode)
                self.assertTrue(candidates)
                self.assertTrue(all(item.get("absolute_path") == str(corrected) for item in candidates if item["kind"] == "joint"))
                bundle = _load_signal_bundle(manifest, episode, {}, frame_count=4)
                self.assertIsNotNone(bundle["joint"])
                self.assertTrue(all(item.get("source") == "applied_projection_correction" for item in bundle["bindings"]))

    def test_public_request_contract_accepts_projection_correction(self) -> None:
        request = BatchAnalysisRequest(
            operation="projection_correction",
            episode_ids=["ep"],
            media_file_ids={"ep": "rgb"},
            sample_fps=1.0,
            max_gap_seconds=0.75,
            adjustment_rate=0.72,
            adjustment_mode="dynamic",
            wrist_point_source="model",
            hand_pose_backend="alicepose",
            dynamic_low_confidence=0.25,
            dynamic_mid_confidence=0.70,
            dynamic_low_multiplier=0.2,
            dynamic_mid_multiplier=0.8,
            dynamic_high_multiplier=1.4,
        )
        self.assertEqual("projection_correction", request.operation)
        self.assertEqual(0.72, request.adjustment_rate)
        self.assertEqual("dynamic", request.adjustment_mode)
        self.assertEqual("model", request.wrist_point_source)
        self.assertEqual("alicepose", request.hand_pose_backend)
        self.assertEqual(0.25, request.dynamic_low_confidence)
        self.assertEqual(1.4, request.dynamic_high_multiplier)
        model = HandPoseModelConfig()
        self.assertEqual("hand_pose", model.slot)
        self.assertEqual("mediapipe", model.kind)
        alicepose = HandPoseModelConfig(kind="alicepose", model_path="Alicepose-21k-v1.pt")
        self.assertEqual("alicepose", alicepose.kind)


if __name__ == "__main__":
    unittest.main()
