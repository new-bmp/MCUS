from __future__ import annotations

import unittest

from app.dataset_modes import dataset_mode


class DatasetModeTests(unittest.TestCase):
    def test_egodex_uses_embedded_camera_contract(self) -> None:
        mode = dataset_mode({"format_family": "egodex"})
        self.assertEqual("egodex", mode["family"])
        self.assertEqual("egodex_embedded_camera_v1", mode["hand_visibility_backend"])
        self.assertFalse(mode["requires_external_hand_camera_extrinsics"])

    def test_nexus_does_not_inherit_egodex_projection(self) -> None:
        mode = dataset_mode({"format_family": "nexus_multimodal"})
        self.assertEqual("nexus_multimodal", mode["family"])
        self.assertEqual("nexus_calibrated_tracking_v1", mode["hand_visibility_backend"])
        self.assertEqual("nexus_dexweaveg1_20_to_mano21", mode["joint_adapter"])

    def test_conflicting_family_and_strategy_is_conservative(self) -> None:
        mode = dataset_mode({
            "format_family": "egodex",
            "format_map": {"processing_strategy": {"id": "nexus_sensor_fusion_v1"}},
        })
        self.assertTrue(mode["conflict"])
        self.assertEqual("unknown", mode["family"])
        self.assertIsNone(mode["hand_visibility_backend"])

    def test_bare_node_shape_does_not_choose_a_mode(self) -> None:
        mode = dataset_mode({"files": [{"shape": [12, 20, 3]}]})
        self.assertEqual("unknown", mode["family"])
        self.assertEqual("disabled_until_confirmed", mode["camera_projection_mode"])


if __name__ == "__main__":
    unittest.main()
