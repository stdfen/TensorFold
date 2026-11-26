import math
import unittest

import dsv41_native_fixture as fixture


class NativeFixtureTeacherTest(unittest.TestCase):
    def test_official_float_tables(self):
        self.assertEqual(fixture.e4m3(0x38), 1.0)
        self.assertEqual(fixture.e4m3(0x40), 2.0)
        self.assertEqual(fixture.e4m3(0x7E), 448.0)
        self.assertTrue(math.isnan(fixture.e4m3(0x7F)))
        self.assertEqual(fixture.e2m1(0x7), 6.0)
        self.assertEqual(fixture.e2m1(0xF), -6.0)
        self.assertEqual(fixture.ue8m0(127), 1.0)
        self.assertEqual(fixture.ue8m0(128), 2.0)

    def test_bf16_round_to_nearest_even(self):
        self.assertEqual(fixture.to_bf16(1.0), 1.0)
        self.assertEqual(fixture.to_bf16(1.00390625), 1.0)
        self.assertEqual(fixture.to_bf16(1.01171875), 1.015625)

    def test_cuda_header_keeps_all_projection_kinds(self):
        report = {"cases": [
            {"name": "b", "kind": "bf16", "n": 1, "k": 1, "weight": [0, 0], "scale": [], "input": [1.0], "expected": [0.0]},
            {"name": "8", "kind": "mxfp8", "n": 1, "k": 1, "weight": [0], "scale": [127], "input": [1.0], "expected": [0.0]},
            {"name": "4", "kind": "mxfp4", "n": 1, "k": 2, "weight": [0], "scale": [127], "input": [1.0, 1.0], "expected": [0.0]},
            {"name": "o", "kind": "wo_a", "n": 1, "k": 1, "weight": [0], "scale": [127], "input": [1.0], "expected": [0.0]},
        ]}
        header = fixture.cuda_header(report)
        self.assertIn("tf_dsv41_official_case_count", header)
        self.assertIn('{"o", 3, 1, 1,', header)


if __name__ == "__main__":
    unittest.main()
