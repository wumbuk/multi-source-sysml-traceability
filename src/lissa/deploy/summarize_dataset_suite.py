#!/usr/bin/env python3
"""Combine completed repeated-run LiSSA summaries for multiple datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument("--datasets", required=True, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mismatch-output", required=True, type=Path)
    parser.add_argument("--round-count", type=int, choices=(2, 3), default=3)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required result does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    summaries: dict[str, Any] = {}
    mismatches: dict[str, Any] = {}
    methods: set[str] = set()
    for dataset in args.datasets:
        dataset_root = args.suite_root / dataset
        summary = read_json(dataset_root / "metrics.json")
        mismatch = read_json(dataset_root / "mismatch.json")
        if summary.get("round_count") != args.round_count or mismatch.get("round_count") != args.round_count:
            raise ValueError(f"{dataset} does not contain exactly {args.round_count} completed rounds")
        methods.add(str(summary.get("method", "LiSSA")))
        summaries[dataset] = summary
        mismatches[dataset] = mismatch

    if len(methods) != 1:
        raise ValueError(f"Datasets use inconsistent methods: {sorted(methods)}")
    method = methods.pop()

    combined = {
        "method": method,
        "model": "Qwen3.5-27B",
        "rounds_per_dataset": args.round_count,
        "datasets": summaries,
    }
    combined_mismatches = {
        "method": method,
        "rounds_per_dataset": args.round_count,
        "datasets": mismatches,
    }
    args.output.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.mismatch_output.write_text(
        json.dumps(combined_mismatches, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved multi-dataset summary: {args.output.resolve()}")
    print(f"Saved multi-dataset mismatch index: {args.mismatch_output.resolve()}")


if __name__ == "__main__":
    main()
