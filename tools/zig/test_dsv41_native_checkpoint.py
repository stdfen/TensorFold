import json
import struct
import tempfile
import unittest
from pathlib import Path

import dsv41_native_checkpoint as native


def safetensors(entries):
    offset = 0
    header = {}
    for name, dtype, shape in entries:
        size = native.DTYPE_BYTES[dtype]
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(raw)) + raw + bytes(offset)


class NativeCheckpointTest(unittest.TestCase):
    def test_parse_and_tensorfold_expert_splits(self):
        raw = safetensors([
            ("layers.6.ffn.experts.0.w1.weight", "I8", [64, 32]),
            ("layers.6.ffn.experts.0.w1.scale", "F8_E8M0", [64, 2]),
            ("layers.6.ffn.experts.0.w2.weight", "I8", [32, 64]),
            ("layers.6.ffn.experts.0.w2.scale", "F8_E8M0", [32, 4]),
        ])
        tensors = native.parse_header("model.safetensors", raw[:8 + struct.unpack_from("<Q", raw)[0]])
        w1 = native.plan_tensor(tensors["layers.6.ffn.experts.0.w1.weight"], 4)
        w2 = native.plan_tensor(tensors["layers.6.ffn.experts.0.w2.weight"], 4)
        self.assertEqual((w1.split, w1.axis, w1.rank_bytes, w1.read_spans), ("col", 0, 512, 1))
        self.assertEqual((w2.split, w2.axis, w2.rank_bytes, w2.read_spans), ("row", 1, 512, 32))
        self.assertEqual(native.validate_pairs(tensors), [])

    def test_local_checkpoint_and_text_only_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = [
                ("embed.weight", "BF16", [8, 4]),
                ("vision.patch_embed.proj.weight", "BF16", [4, 4]),
                ("mtp.0.norm.weight", "BF16", [4]),
            ]
            blob = safetensors(entries)
            (root / "model-00001-of-00001.safetensors").write_bytes(blob)
            (root / "config.json").write_text(json.dumps({
                "model_type": "deepseek_v41",
                "quantization_config": {
                    "quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
                    "scale_fmt": "ue8m0", "expert_dtype": "fp4",
                },
            }))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 104},
                "weight_map": {name: "model-00001-of-00001.safetensors" for name, _, _ in entries},
            }))
            report = native.summarize(*native.load_checkpoint(native.Source(temp, "main")), 4)
            self.assertTrue(report["validation"]["ok"])
            self.assertEqual(report["full_bytes"]["scope"]["excluded_multimodal"], 32)
            self.assertEqual(report["full_bytes"]["scope"]["optional_draft"], 8)
            self.assertEqual(report["per_rank_bytes"]["total"], 16)
            candidate = {
                "checkpoint": {
                    key: report["checkpoint"][key]
                    for key in ("tensors", "shards", "header_payload_bytes")
                },
                "world": report["world"],
                "tensors": [dict(item) for item in report["tensors"]],
            }
            self.assertEqual(native.compare_native_plan(report, candidate), [])
            candidate["tensors"][0]["rank_bytes"] += 1
            failures = native.compare_native_plan(report, candidate)
            self.assertEqual(len(failures), 1)
            self.assertIn("rank_bytes", failures[0])


if __name__ == "__main__":
    unittest.main()
