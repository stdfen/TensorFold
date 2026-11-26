#!/usr/bin/env python3
"""Extract tiny, digest-keyed arithmetic fixtures from official DeepSeek-V4.1 safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import dsv41_native_checkpoint as checkpoint


def e4m3(code: int) -> float:
    if code & 0x7F == 0x7F:
        return math.nan
    sign = -1.0 if code & 0x80 else 1.0
    exponent, mantissa = (code >> 3) & 0xF, code & 7
    return sign * (mantissa / 8.0 * 2.0**-6 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1(code: int) -> float:
    return (-1.0 if code & 8 else 1.0) * E2M1[code & 7]


def ue8m0(code: int) -> float:
    if code == 255:
        raise checkpoint.CheckpointError("fixture encountered reserved UE8M0 code 255")
    return 2.0 ** (code - 127)


def bf16(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def to_bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return bf16(rounded & 0xFFFF)


class Reader:
    def __init__(self, source: checkpoint.Source, tensors: dict[str, checkpoint.Tensor]):
        self.source, self.tensors, self.headers = source, tensors, {}

    def data_start(self, shard: str) -> int:
        if shard not in self.headers:
            raw = self.source.read(shard, (0, 7))
            if len(raw) != 8:
                raise checkpoint.CheckpointError(f"short safetensors prefix: {shard}")
            self.headers[shard] = 8 + struct.unpack("<Q", raw)[0]
        return self.headers[shard]

    def bytes(self, name: str, offset: int, length: int) -> bytes:
        tensor = self.tensors[name]
        if offset < 0 or length < 0 or offset + length > tensor.nbytes:
            raise checkpoint.CheckpointError(f"slice outside {name}")
        start = self.data_start(tensor.shard) + tensor.begin + offset
        raw = self.source.read(tensor.shard, (start, start + length - 1))
        if len(raw) != length:
            raise checkpoint.CheckpointError(f"short tensor slice: {name}")
        return raw

    def rows(self, name: str, rows: int, columns: int) -> bytes:
        tensor = self.tensors[name]
        item = checkpoint.DTYPE_BYTES[tensor.dtype]
        stride = tensor.shape[1] * item
        return b"".join(self.bytes(name, row * stride, columns * item) for row in range(rows))


def vector() -> list[float]:
    return [((i * 17) % 29 - 14) / 16.0 for i in range(32)]


def digest(weight: bytes, scale: bytes, x: list[float]) -> str:
    h = hashlib.sha256(weight + scale)
    for value in x:
        h.update(struct.pack("<d", value))
    return h.hexdigest()


def fixture(model: str) -> dict:
    source = checkpoint.Source(model, "main")
    config, tensors, index = checkpoint.load_checkpoint(source)
    reader, x = Reader(source, tensors), vector()
    cases = []

    def add(name: str, kind: str, weight: bytes, scale: bytes, values: list[list[float]], *, bf16_weight=False):
        expected = []
        for row in values:
            if bf16_weight:
                row = [to_bf16(value) for value in row]
            expected.append(sum(value * input_value for value, input_value in zip(row, x)))
        cases.append({
            "name": name, "kind": kind, "n": len(values), "k": len(x),
            "weight": list(weight), "scale": list(scale), "input": x,
            "expected": expected, "payload_sha256": digest(weight, scale, x),
        })

    name = "embed.weight"
    raw = reader.rows(name, 2, 32)
    rows = [[bf16(struct.unpack_from("<H", raw, 2 * (r * 32 + c))[0]) for c in range(32)] for r in range(2)]
    add(name, "bf16", raw, b"", rows)

    name = "layers.0.attn.wq_a"
    weight = reader.rows(name + ".weight", 2, 32)
    scale = reader.bytes(name + ".scale", 0, 1)
    rows = [[e4m3(weight[r * 32 + c]) * ue8m0(scale[0]) for c in range(32)] for r in range(2)]
    add(name, "mxfp8", weight, scale, rows)

    name = "layers.0.ffn.experts.0.w1"
    weight = reader.rows(name + ".weight", 2, 16)
    scale_tensor = tensors[name + ".scale"]
    scale = b"".join(reader.bytes(name + ".scale", r * scale_tensor.shape[1], 1) for r in range(2))
    rows = []
    for r in range(2):
        codes = [(weight[r * 16 + c // 2] & 0xF) if c % 2 == 0 else (weight[r * 16 + c // 2] >> 4) for c in range(32)]
        rows.append([e2m1(code) * ue8m0(scale[r]) for code in codes])
    add(name, "mxfp4", weight, scale, rows)

    name = "layers.0.attn.wo_a"
    weight = reader.rows(name + ".weight", 2, 32)
    scale = reader.bytes(name + ".scale", 0, 1)
    rows = [[e4m3(weight[r * 32 + c]) * ue8m0(scale[0]) for c in range(32)] for r in range(2)]
    add(name, "wo_a", weight, scale, rows, bf16_weight=True)

    config_bytes = source.read("config.json")
    index_bytes = source.read("model.safetensors.index.json")
    return {
        "schema": "tensorfold.deepseek_v41.native_reference.v1",
        "checkpoint": {
            "model_type": config.get("model_type"),
            "tensors": len(tensors), "shards": len(set(index["weight_map"].values())),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        },
        "cases": cases,
    }


def cuda_header(report: dict) -> str:
    """Emit a dependency-free C++ header consumed by the CUDA device fixture."""
    lines = [
        "// Generated by tools/zig/dsv41_native_fixture.py; do not edit.",
        "#pragma once",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "struct TfDsv41OfficialCase { const char* name; int kind; size_t n, k; const uint8_t* weight; size_t weight_len; const uint8_t* scale; size_t scale_len; const float* input; const float* expected; };",
    ]
    kinds = {"bf16": 0, "mxfp8": 1, "mxfp4": 2, "wo_a": 3}
    for i, case in enumerate(report["cases"]):
        weight = ", ".join(f"0x{x:02x}" for x in case["weight"])
        lines.append(f"static const uint8_t tf_dsv41_weight_{i}[] = {{{weight}}};")
        if case["scale"]:
            scale = ", ".join(f"0x{x:02x}" for x in case["scale"])
            lines.append(f"static const uint8_t tf_dsv41_scale_{i}[] = {{{scale}}};")
        values = ", ".join(f"{float(x).hex()}f" for x in case["input"])
        expected = ", ".join(f"{float(x).hex()}f" for x in case["expected"])
        lines.append(f"static const float tf_dsv41_input_{i}[] = {{{values}}};")
        lines.append(f"static const float tf_dsv41_expected_{i}[] = {{{expected}}};")
    lines.append("static const TfDsv41OfficialCase tf_dsv41_official_cases[] = {")
    for i, case in enumerate(report["cases"]):
        scale = f"tf_dsv41_scale_{i}" if case["scale"] else "nullptr"
        lines.append(
            f'  {{"{case["name"]}", {kinds[case["kind"]]}, {case["n"]}, {case["k"]}, '
            f"tf_dsv41_weight_{i}, sizeof(tf_dsv41_weight_{i}), {scale}, {len(case['scale'])}, "
            f"tf_dsv41_input_{i}, tf_dsv41_expected_{i}}},"
        )
    lines.append("};")
    lines.append("static const size_t tf_dsv41_official_case_count = sizeof(tf_dsv41_official_cases) / sizeof(tf_dsv41_official_cases[0]);")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--cuda-header", help="also write a C++ header for native_mixed_official_test.cu")
    args = parser.parse_args()
    report = fixture(args.model)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.cuda_header:
        Path(args.cuda_header).write_text(cuda_header(report), encoding="utf-8")
    print(f"wrote {len(report['cases'])} cases to {args.output}")
    for case in report["cases"]:
        print(case["kind"], case["name"], case["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
