#!/usr/bin/env python3
"""Aggregate isolated rounds of the typed LiSSA experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


BINARY_FIELDS = ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy")
TYPE_FIELDS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
CONDITIONAL_FIELDS = ("evaluated", "correct", "accuracy", "unknown_type_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--round-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mismatch-output", required=True, type=Path)
    return parser.parse_args()


def aggregate(rounds: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        values = [float(item[field]) for item in rounds]
        result[field] = {
            "mean": statistics.fmean(values),
            "population_std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.round_count not in (2, 3):
        raise ValueError("Typed LiSSA protocol requires two or three rounds")

    binary_rounds: list[dict[str, Any]] = []
    type_rounds: list[dict[str, Any]] = []
    conditional_rounds: list[dict[str, Any]] = []
    mismatch_rounds: list[dict[str, Any]] = []
    for number in range(1, args.round_count + 1):
        round_root = args.run_root / f"round_{number}"
        metrics_path = round_root / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("meta", {}).get("not_original_lissa") is not True:
            raise ValueError(f"Round is not marked as the typed LiSSA variant: {metrics_path}")
        binary_rounds.append({"round": number, **payload["rq1_dataset_scope"]})
        type_rounds.append(
            {"round": number, **payload["rq2_relation_type"]["end_to_end_three_class"]}
        )
        conditional_rounds.append(
            {
                "round": number,
                **payload["rq2_relation_type"]["conditional_on_correct_existence"],
            }
        )
        mismatch_path = round_root / "mismatch.json"
        mismatch_rounds.append(
            {
                "round": number,
                "mismatch_file": str(mismatch_path.resolve()),
                **json.loads(mismatch_path.read_text(encoding="utf-8")),
            }
        )

    result = {
        "method": "LiSSA typed-relation variant",
        "not_original_lissa": True,
        "round_count": args.round_count,
        "rq1_dataset_scope": {
            "rounds": binary_rounds,
            "aggregate": aggregate(binary_rounds, BINARY_FIELDS),
        },
        "rq2_relation_type": {
            "end_to_end_three_class": {
                "rounds": type_rounds,
                "aggregate": aggregate(type_rounds, TYPE_FIELDS),
            },
            "conditional_on_correct_existence": {
                "rounds": conditional_rounds,
                "aggregate": aggregate(conditional_rounds, CONDITIONAL_FIELDS),
            },
        },
    }
    mismatch_result = {
        "method": "LiSSA typed-relation variant",
        "round_count": args.round_count,
        "rounds": mismatch_rounds,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.mismatch_output.write_text(
        json.dumps(mismatch_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved typed {args.round_count}-round summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()
