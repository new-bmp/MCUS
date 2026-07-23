from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py

from app.behavior_annotator import load_behavior_ontology


class BehaviorOntologyTests(unittest.TestCase):
    def tearDown(self) -> None:
        load_behavior_ontology.cache_clear()

    def test_missing_external_directory_uses_builtin_ontology(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "part1-moved"
            ontology = load_behavior_ontology(str(missing))

        self.assertEqual("builtin", ontology["source"])
        self.assertEqual("directory_unavailable", ontology["fallback_reason"])
        self.assertGreaterEqual(ontology["category_count"], 10)
        self.assertIn("place", {item["label"] for item in ontology["categories"]})

    def test_empty_external_directory_uses_builtin_ontology(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ontology = load_behavior_ontology(temporary)

        self.assertEqual("builtin", ontology["source"])
        self.assertEqual("no_readable_hdf5_annotations", ontology["fallback_reason"])

    def test_readable_external_directory_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            category = Path(temporary) / "custom-place"
            category.mkdir()
            with h5py.File(category / "0.hdf5", "w") as handle:
                handle.attrs["task"] = "custom placement"
                handle.attrs["llm_description"] = "Place the peach in the bowl."
                handle.attrs["llm_verbs"] = ["place"]
                handle.attrs["llm_objects"] = ["peach", "bowl"]
            ontology = load_behavior_ontology(temporary)

        self.assertEqual("external_hdf5", ontology["source"])
        self.assertEqual(1, ontology["category_count"])
        self.assertEqual("custom-place", ontology["categories"][0]["label"])


if __name__ == "__main__":
    unittest.main()
