import tempfile
import unittest
from pathlib import Path

import dsv41_native_compare as compare


def capture(top=7, other=8, *, phase="prompt"):
    return {
        "schema": compare.SCHEMA,
        "checkpoint": {"config_sha256": "c", "index_sha256": "i"},
        "runtime": {"backend": "reference", "hardware": "fixture", "cuda": "fixture", "libraries": {}},
        "settings": {"world": 4, "context": 64, "greedy": True, "drafts": False, "engram": False},
        "cases": [{
            "name": "tiny",
            "phase": phase,
            "prompt_tokens": [1, 2],
            "forced_tokens": [3],
            "rows": [{"position": 1, "forced_token": 3, "candidates": [
                {"token": top, "logit": 4.0}, {"token": other, "logit": 3.25}
            ]}],
        }],
    }


class CompareTests(unittest.TestCase):
    def test_equal_top_records_margins_and_writes_atomically(self):
        result = compare.compare(capture(), capture())
        self.assertTrue(result["pass"])
        self.assertEqual(result["summary"]["prompt"], {"rows": 1, "top_equal": 1, "top_different": 0})
        self.assertEqual(result["rows"][0]["reference_margin"], 0.75)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "receipt.json"
            compare.write_atomic(path, result)
            self.assertIn('"pass": true', path.read_text())

    def test_different_top_is_retained_not_hidden_by_tolerance(self):
        reference = capture()
        native = capture(8, 7)
        result = compare.compare(reference, native)
        self.assertFalse(result["pass"])
        self.assertEqual(result["summary"]["prompt"]["top_different"], 1)
        self.assertEqual(result["rows"][0]["native_gap_for_reference_top"], 0.75)

    def test_missing_cross_top_and_setting_drift_are_refused(self):
        native = capture(9, 10)
        with self.assertRaisesRegex(compare.CompareError, "both engines"):
            compare.compare(capture(), native)
        native = capture()
        native["settings"]["world"] = 2
        with self.assertRaisesRegex(compare.CompareError, "world"):
            compare.compare(capture(), native)

    def test_unknown_or_changed_phase_is_refused(self):
        with self.assertRaisesRegex(compare.CompareError, "phase"):
            compare.compare(capture(phase="other"), capture(phase="other"))
        with self.assertRaisesRegex(compare.CompareError, "phase"):
            compare.compare(capture(phase="prompt"), capture(phase="decode"))


if __name__ == "__main__":
    unittest.main()
