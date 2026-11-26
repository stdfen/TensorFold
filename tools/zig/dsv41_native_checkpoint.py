#!/usr/bin/env python3
"""Inspect DeepSeek-V4.1 official safetensors and plan TensorFold TP slices."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

MAX_HEADER_BYTES = 100_000_000
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
}
EXPERT_RE = re.compile(r"^(?:layers|mtp)\.\d+\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tensor:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class SlicePlan:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    scope: str
    residency: str
    encoding: str
    split: str
    axis: int | None
    rank_bytes: int
    read_spans: int
    contiguous: bool


class Source:
    def __init__(self, model: str, revision: str):
        self.local = Path(model)
        self.remote = model.startswith(("http://", "https://"))
        if self.remote:
            base = model.rstrip("/")
            if "huggingface.co" in base and "/resolve/" not in base:
                base += "/resolve/" + urllib.parse.quote(revision, safe="")
            self.base = base

    def read(self, name: str, byte_range: tuple[int, int] | None = None) -> bytes:
        if not self.remote:
            path = self.local / name
            try:
                with path.open("rb") as handle:
                    if byte_range is None:
                        return handle.read()
                    start, end = byte_range
                    handle.seek(start)
                    return handle.read(end - start + 1)
            except OSError as exc:
                raise CheckpointError(f"cannot read {path}: {exc}") from exc
        url = self.base + "/" + urllib.parse.quote(name)
        request = urllib.request.Request(url)
        if byte_range is not None:
            request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
        except OSError as exc:
            raise CheckpointError(f"cannot read {url}: {exc}") from exc
        if byte_range is not None and len(data) != byte_range[1] - byte_range[0] + 1:
            raise CheckpointError(f"server ignored or truncated range for {name}")
        return data


def read_json(source: Source, name: str) -> dict[str, Any]:
    try:
        value = json.loads(source.read(name))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"bad JSON in {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"{name} must contain a JSON object")
    return value


def parse_header(shard: str, raw: bytes) -> dict[str, Tensor]:
    if len(raw) < 8:
        raise CheckpointError(f"short safetensors prefix in {shard}")
    size = struct.unpack_from("<Q", raw)[0]
    if size > MAX_HEADER_BYTES or len(raw) != size + 8:
        raise CheckpointError(f"bad safetensors header length in {shard}: {size}")
    try:
        root = json.loads(raw[8:])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"bad safetensors JSON in {shard}: {exc}") from exc
    out: dict[str, Tensor] = {}
    for name, value in root.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise CheckpointError(f"bad tensor entry {name} in {shard}")
        dtype, shape, offsets = value.get("dtype"), value.get("shape"), value.get("data_offsets")
        valid = dtype in DTYPE_BYTES and isinstance(shape, list) and isinstance(offsets, list) and len(offsets) == 2
        if not valid:
            raise CheckpointError(f"bad tensor metadata {name} in {shard}")
        if any(not isinstance(x, int) or x < 0 for x in shape + offsets) or offsets[1] < offsets[0]:
            raise CheckpointError(f"bad tensor geometry {name} in {shard}")
        elements = math.prod(shape)
        if elements * DTYPE_BYTES[dtype] != offsets[1] - offsets[0]:
            raise CheckpointError(f"byte count does not match dtype/shape for {name}")
        out[name] = Tensor(name, shard, dtype, tuple(shape), offsets[0], offsets[1])
    return out


def load_checkpoint(source: Source) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Any]]:
    config = read_json(source, "config.json")
    index = read_json(source, "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointError("index has no weight_map")
    expected = set(weight_map)
    shards = sorted(set(weight_map.values()))

    def load_shard(shard: str) -> tuple[str, dict[str, Tensor]]:
        prefix = source.read(shard, (0, 7))
        size = struct.unpack_from("<Q", prefix)[0]
        if size > MAX_HEADER_BYTES:
            raise CheckpointError(f"oversized safetensors header in {shard}: {size}")
        return shard, parse_header(shard, source.read(shard, (0, size + 7)))

    workers = min(8, len(shards)) if source.remote else 1
    tensors: dict[str, Tensor] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for shard, header in executor.map(load_shard, shards):
            duplicate = tensors.keys() & header.keys()
            if duplicate:
                raise CheckpointError(f"duplicate tensors across shards: {sorted(duplicate)[:3]}")
            tensors.update(header)
    missing, extra = expected - tensors.keys(), tensors.keys() - expected
    misplaced = [name for name, tensor in tensors.items() if weight_map.get(name) != tensor.shard]
    if missing or extra or misplaced:
        raise CheckpointError(
            f"index/header mismatch: missing={len(missing)} extra={len(extra)} misplaced={len(misplaced)}"
        )
    return config, tensors, index


def scope(name: str) -> str:
    if name.startswith(("vision.", "aligner.")) or name in {"image_start", "image_end", "image_newline"}:
        return "excluded_multimodal"
    if name.startswith("mtp."):
        return "optional_draft"
    return "text"


def residency(name: str) -> str:
    if EXPERT_RE.match(name):
        return "cold_expert"
    if ".engram.embed." in name:
        return "cold_engram"
    return "resident"


def encoding(tensor: Tensor) -> str:
    if EXPERT_RE.match(tensor.name) and tensor.name.endswith(".weight"):
        return "mxfp4_e2m1_packed"
    if tensor.dtype == "F8_E4M3":
        return "mxfp8_e4m3"
    if tensor.dtype == "F8_E8M0":
        return "ue8m0_scale"
    if tensor.dtype == "BF16":
        return "bf16"
    return tensor.dtype.lower()


def split_rule(name: str) -> tuple[str, int | None]:
    match = EXPERT_RE.match(name)
    if match:
        return ("row" if match.group(2) == "w2" else "col", 1 if match.group(2) == "w2" else 0)
    if name in {"embed.weight", "head.weight"} or ".engram.embed." in name:
        return "rows", 0
    col = (".attn.wq_b.", ".attn.wo_a.", ".ffn.shared_experts.w1.", ".ffn.shared_experts.w3.")
    row = (".attn.wo_b.", ".ffn.shared_experts.w2.")
    if any(x in name for x in col):
        return "col", 0
    if any(x in name for x in row):
        return "row", 1
    return "replicated", None


def plan_tensor(tensor: Tensor, world: int) -> SlicePlan:
    tensor_scope = scope(tensor.name)
    tensor_residency = residency(tensor.name)
    split, axis = split_rule(tensor.name)
    rank_bytes, spans = tensor.nbytes, 1
    if tensor_scope != "text":
        rank_bytes = 0
    elif axis is not None:
        extent = tensor.shape[axis]
        if split == "rows" and ".engram.embed." in tensor.name:
            part = math.ceil(extent / world)
            rank_bytes = tensor.nbytes // extent * part
        else:
            if extent % world:
                raise CheckpointError(f"{tensor.name}: axis {axis} extent {extent} is not divisible by world {world}")
            rank_bytes = tensor.nbytes // world
        if axis == 1 and len(tensor.shape) == 2:
            spans = tensor.shape[0]
    return SlicePlan(
        tensor.name, tensor.shard, tensor.dtype, tensor.shape, tensor_scope, tensor_residency,
        encoding(tensor), split, axis, rank_bytes, spans, spans == 1,
    )


def validate_config(config: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if config.get("model_type") != "deepseek_v41":
        failures.append("model_type must be deepseek_v41")
    quant = config.get("quantization_config", {})
    expected = {
        "quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0", "expert_dtype": "fp4",
    }
    for key, value in expected.items():
        if quant.get(key) != value:
            failures.append(f"quantization_config.{key} must be {value!r}")
    return failures


def validate_pairs(tensors: dict[str, Tensor]) -> list[str]:
    failures: list[str] = []
    for tensor in tensors.values():
        if not tensor.name.endswith(".weight") or tensor.dtype not in {"I8", "F8_E4M3"}:
            continue
        scale_name = tensor.name[:-6] + "scale"
        scale = tensors.get(scale_name)
        if scale is None or scale.dtype != "F8_E8M0":
            failures.append(f"{tensor.name}: missing F8_E8M0 scale")
            continue
        if len(tensor.shape) != 2 or len(scale.shape) != 2:
            failures.append(f"{tensor.name}: weight and scale must be rank 2")
            continue
        logical_k = tensor.shape[1] * (2 if EXPERT_RE.match(tensor.name) else 1)
        expected = (
            (tensor.shape[0], math.ceil(logical_k / 32))
            if EXPERT_RE.match(tensor.name) or ".engram.embed." in tensor.name
            else (math.ceil(tensor.shape[0] / 32), math.ceil(logical_k / 32))
        )
        if scale.shape != expected:
            failures.append(f"{tensor.name}: scale {scale.shape}, expected {expected}")
    return failures


def summarize(config: dict[str, Any], tensors: dict[str, Tensor], index: dict[str, Any], world: int) -> dict[str, Any]:
    plans = [plan_tensor(tensor, world) for tensor in tensors.values()]
    config_failures = validate_config(config)
    pair_failures = validate_pairs(tensors)
    bytes_by_scope: Counter[str] = Counter()
    bytes_by_residency: Counter[str] = Counter()
    rank_bytes_by_residency: Counter[str] = Counter()
    bytes_by_encoding: Counter[str] = Counter()
    spans_by_split: Counter[str] = Counter()
    tensors_by_scope: Counter[str] = Counter()
    tensors_by_encoding: Counter[str] = Counter()
    shards: defaultdict[str, int] = defaultdict(int)
    for tensor, item in zip(tensors.values(), plans):
        bytes_by_scope[item.scope] += tensor.nbytes
        bytes_by_residency[item.residency] += tensor.nbytes
        rank_bytes_by_residency[item.residency] += item.rank_bytes
        bytes_by_encoding[item.encoding] += tensor.nbytes
        spans_by_split[item.split] += item.read_spans
        tensors_by_scope[item.scope] += 1
        tensors_by_encoding[item.encoding] += 1
        shards[item.shard] += tensor.nbytes
    indexed_total = index.get("metadata", {}).get("total_size")
    header_total = sum(t.nbytes for t in tensors.values())
    total_failure = [] if indexed_total in (None, header_total) else [
        f"index total_size={indexed_total}, headers={header_total}"
    ]
    return {
        "checkpoint": {
            "model_type": config.get("model_type"),
            "tensors": len(tensors),
            "shards": len(shards),
            "header_payload_bytes": header_total, "index_total_size": indexed_total,
        },
        "world": world,
        "tensor_counts": {"scope": dict(tensors_by_scope), "encoding": dict(tensors_by_encoding)},
        "full_bytes": {
            "scope": dict(bytes_by_scope), "residency": dict(bytes_by_residency),
            "encoding": dict(bytes_by_encoding),
        },
        "per_rank_bytes": {"residency": dict(rank_bytes_by_residency), "total": sum(p.rank_bytes for p in plans)},
        "read_spans_per_rank": dict(spans_by_split),
        "validation": {
            "ok": not config_failures and not pair_failures and not total_failure,
            "config_failures": config_failures, "pair_failures": pair_failures, "size_failures": total_failure,
        },
        "tensors": [asdict(item) for item in sorted(plans, key=lambda x: x.name)],
    }


def human_bytes(value: int) -> str:
    return f"{value / 2**30:.3f} GiB"


def print_summary(report: dict[str, Any], include_tensors: bool) -> None:
    checkpoint = report["checkpoint"]
    print(f"checkpoint: {checkpoint['model_type']} tensors={checkpoint['tensors']} shards={checkpoint['shards']}")
    print(f"payload: {human_bytes(checkpoint['header_payload_bytes'])} world={report['world']}")
    print("scope:")
    for key, value in sorted(report["full_bytes"]["scope"].items()):
        print(f"  {key:20s} {human_bytes(value)}")
    print("per-rank residency:")
    for key, value in sorted(report["per_rank_bytes"]["residency"].items()):
        print(f"  {key:20s} {human_bytes(value)}")
    print(f"  {'total':20s} {human_bytes(report['per_rank_bytes']['total'])}")
    print("read spans per rank:")
    for key, value in sorted(report["read_spans_per_rank"].items()):
        print(f"  {key:20s} {value}")
    validation = report["validation"]
    print("validation:", "PASS" if validation["ok"] else "FAIL")
    for kind in ("config_failures", "pair_failures", "size_failures"):
        for failure in validation[kind]:
            print(f"  {kind}: {failure}")
    if include_tensors:
        for tensor in report["tensors"]:
            print(json.dumps(tensor, separators=(",", ":")))


PLAN_FIELDS = (
    "name", "shard", "dtype", "shape", "scope", "residency", "encoding",
    "split", "axis", "rank_bytes", "read_spans", "contiguous",
)


def compare_native_plan(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Compare the P0 report with the Zig stream without depending on map order."""
    failures: list[str] = []
    for key in ("tensors", "shards", "header_payload_bytes"):
        expected = reference.get("checkpoint", {}).get(key)
        actual = candidate.get("checkpoint", {}).get(key)
        if expected != actual:
            failures.append(f"checkpoint.{key}: P0={expected!r} native={actual!r}")
    if reference.get("world") != candidate.get("world"):
        failures.append(f"world: P0={reference.get('world')!r} native={candidate.get('world')!r}")

    expected_tensors = reference.get("tensors", [])
    actual_tensors = candidate.get("tensors", [])
    if len(expected_tensors) != len(actual_tensors):
        failures.append(f"tensor count: P0={len(expected_tensors)} native={len(actual_tensors)}")
    expected_by_name = {item.get("name"): item for item in expected_tensors}
    actual_by_name = {item.get("name"): item for item in actual_tensors}
    missing = sorted(expected_by_name.keys() - actual_by_name.keys())
    extra = sorted(actual_by_name.keys() - expected_by_name.keys())
    if missing:
        failures.append(f"native missing {len(missing)} tensors; first={missing[:3]}")
    if extra:
        failures.append(f"native has {len(extra)} extra tensors; first={extra[:3]}")
    for name in sorted(expected_by_name.keys() & actual_by_name.keys()):
        expected, actual = expected_by_name[name], actual_by_name[name]
        for field in PLAN_FIELDS:
            expected_value, actual_value = expected.get(field), actual.get(field)
            if field == "shape":
                expected_value, actual_value = list(expected_value), list(actual_value)
            if expected_value != actual_value:
                failures.append(
                    f"{name}.{field}: P0={expected_value!r} native={actual_value!r}"
                )
                if len(failures) >= 50:
                    failures.append("comparison stopped after 50 mismatches")
                    return failures
    return failures


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect-native", "plan-native"))
    parser.add_argument("model", help="local checkpoint directory or Hugging Face repository URL")
    parser.add_argument("--world", type=int, default=1, help="tensor-parallel world size")
    parser.add_argument("--revision", default="main", help="remote Hugging Face revision")
    parser.add_argument("--json", dest="json_path", help="write the full machine-readable report")
    parser.add_argument("--compare-native", help="compare all tensor plans with tf-dsv41 --json output")
    parser.add_argument("--tensors", action="store_true", help="print one JSON line per tensor")
    args = parser.parse_args(list(argv))
    if args.world < 1:
        parser.error("--world must be positive")
    if args.command == "inspect-native":
        args.world = 1
    return args


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    try:
        report = summarize(*load_checkpoint(Source(args.model, args.revision)), args.world)
    except CheckpointError as exc:
        print(f"FAIL dsv41 {args.command}: {exc}", file=sys.stderr)
        return 1
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(report, args.tensors)
    comparison_failures: list[str] = []
    if args.compare_native:
        try:
            candidate = json.loads(Path(args.compare_native).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FAIL native comparison: {exc}", file=sys.stderr)
            return 1
        comparison_failures = compare_native_plan(report, candidate)
        print("native comparison:", "PASS" if not comparison_failures else "FAIL")
        for failure in comparison_failures:
            print(f"  {failure}")
    return 0 if report["validation"]["ok"] and not comparison_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
