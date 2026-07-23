from __future__ import annotations

import unittest

from app.models import OPEN_VOCAB_HAND_CLASSES, _open_vocab_proximity_classes


class YoloePromptTests(unittest.TestCase):
    def test_default_prompts_include_hand_and_robot_hand(self) -> None:
        self.assertIn("hand", OPEN_VOCAB_HAND_CLASSES)
        self.assertIn("robot hand", OPEN_VOCAB_HAND_CLASSES)

    def test_object_terms_cannot_remove_or_duplicate_default_hand_prompts(self) -> None:
        prompts = _open_vocab_proximity_classes(["apple", "Hand", "robot hand", "apple"])

        self.assertEqual(prompts[0:3], ["apple", "hand", "robot hand"])
        self.assertEqual(sum(item.casefold() == "hand" for item in prompts), 1)
        self.assertEqual(sum(item.casefold() == "robot hand" for item in prompts), 1)
        self.assertIn("robot gripper", prompts)


if __name__ == "__main__":
    unittest.main()
