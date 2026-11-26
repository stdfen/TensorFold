#!/usr/bin/env python3
"""Compare DeepSeek-V4.1 teacher-forced TP4 captures and write a strict correctness receipt."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

SCHEMA = "tensorfold.deepseek_v41.teacher_forced.v1"
RECEIPT_SCHEMA = "tensorfold.deepseek_v41.teacher_forced.compare.v1"


class CompareError(ValueError):
    pass


def load(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise CompareError(f"{path}: expected schema {SCHEMA}")
    for key in ("checkpoint", "runtime", "settings", "cases"):
        if key not in data:
            raise CompareError(f"{path}: missing {key}")
    return data


def _top(row: dict, where: str) -> tuple[int, float, float, dict[int, float]]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise CompareError(f"{where}: at least two candidates are required")
    values: dict[int, float] = {}
    for item in candidates:
        token, logit = item.get("token"), item.get("logit")
        if not isinstance(token, int) or not isinstance(logit, (int, float)):
            raise CompareError(f"{where}: malformed candidate")
        if token in values:
            raise CompareError(f"{where}: duplicate candidate token {token}")
        values[token] = float(logit)
    ranked = sorted(values.items(), key=lambda x: (-x[1], x[0]))
    return ranked[0][0], ranked[0][1], ranked[0][1] - ranked[1][1], values


def compare(reference: dict, native: dict) -> dict:
    if reference["checkpoint"] != native["checkpoint"]:
        raise CompareError("checkpoint identity differs")
    for field in ("world", "context", "greedy", "drafts", "engram"):
        if reference["settings"].get(field) != native["settings"].get(field):
            raise CompareError(f"setting differs: {field}")
    ref_cases = {case["name"]: case for case in reference["cases"]}
    got_cases = {case["name"]: case for case in native["cases"]}
    if len(ref_cases) != len(reference["cases"]) or len(got_cases) != len(native["cases"]):
        raise CompareError("duplicate case name")
    if ref_cases.keys() != got_cases.keys():
        raise CompareError("case set differs")

    rows_out: list[dict] = []
    phases: Counter[str] = Counter()
    phase_equal: Counter[str] = Counter()
    token_arrays_equal = True
    for name in ref_cases:
        ref, got = ref_cases[name], got_cases[name]
        if ref.get("phase") not in ("prompt", "decode") or got.get("phase") != ref.get("phase"):
            raise CompareError(f"{name}: phase must match and be prompt or decode")
        for field in ("phase", "prompt_tokens", "forced_tokens"):
            if ref.get(field) != got.get(field):
                if field.endswith("tokens"):
                    token_arrays_equal = False
                else:
                    raise CompareError(f"{name}: {field} differs")
        rr, gr = ref.get("rows"), got.get("rows")
        if not isinstance(rr, list) or not isinstance(gr, list) or len(rr) != len(gr):
            raise CompareError(f"{name}: row count differs")
        phase = ref["phase"]
        if phase not in ("prompt", "decode"):
            raise CompareError(f"{name}: phase must be prompt or decode")
        for index, (a, b) in enumerate(zip(rr, gr)):
            if a.get("position") != b.get("position") or a.get("forced_token") != b.get("forced_token"):
                raise CompareError(f"{name} row {index}: position/forced token differs")
            rt, rl, rm, rv = _top(a, f"reference {name} row {index}")
            nt, nl, nm, nv = _top(b, f"native {name} row {index}")
            if rt not in nv or nt not in rv:
                raise CompareError(f"{name} row {index}: captures must retain both engines' top tokens")
            equal = rt == nt
            phases[phase] += 1
            phase_equal[phase] += int(equal)
            rows_out.append({
                "case": name,
                "phase": phase,
                "position": a["position"],
                "forced_token": a["forced_token"],
                "reference_top": rt,
                "native_top": nt,
                "top_equal": equal,
                "reference_margin": rm,
                "native_margin": nm,
                "native_gap_for_reference_top": nl - nv[rt],
                "reference_gap_for_native_top": rl - rv[nt],
            })

    summary = {
        phase: {"rows": phases[phase], "top_equal": phase_equal[phase], "top_different": phases[phase] - phase_equal[phase]}
        for phase in ("prompt", "decode")
    }
    passed = token_arrays_equal and all(x["top_different"] == 0 for x in summary.values())
    return {
        "schema": RECEIPT_SCHEMA,
        "checkpoint": reference["checkpoint"],
        "settings": reference["settings"],
        "reference_runtime": reference["runtime"],
        "native_runtime": native["runtime"],
        "token_arrays_equal": token_arrays_equal,
        "summary": summary,
        "rows": rows_out,
        "pass": passed,
    }


def write_atomic(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("native")
    parser.add_argument("output")
    args = parser.parse_args()
    result = compare(load(args.reference), load(args.native))
    write_atomic(args.output, result)
    for phase, stats in result["summary"].items():
        print(f"{phase}: {stats['top_equal']}/{stats['rows']} top tokens equal")
    print("PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
