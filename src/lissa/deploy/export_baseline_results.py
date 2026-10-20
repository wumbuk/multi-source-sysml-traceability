#!/usr/bin/env python3
"""Package LiSSA's original metrics and mismatches as JSON without changing evaluation logic."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


RESULT_PATTERNS = {
    "tp": r"^\* True Positives:\s*(\d+)\s*$",
    "fp": r"^\* False Positives:\s*(\d+)\s*$",
    "fn": r"^\* False Negatives:\s*(\d+)\s*$",
    "precision": r"^\* Precision:\s*([0-9.eE+-]+)\s*$",
    "recall": r"^\* Recall:\s*([0-9.eE+-]+)\s*$",
    "f1": r"^\* F1:\s*([0-9.eE+-]+)\s*$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LiSSA's existing Markdown metrics and FP/FN links as JSON."
    )
    parser.add_argument("--results-md", required=True, type=Path)
    parser.add_argument("--trace-links", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--config-working-dir",
        required=True,
        type=Path,
        help="Working directory used when LiSSA resolved relative config paths.",
    )
    parser.add_argument("--metrics-out", required=True, type=Path)
    parser.add_argument("--mismatch-out", required=True, type=Path)
    return parser.parse_args()


def parse_original_metrics(path: Path) -> dict[str, int | float]:
    text = path.read_text(encoding="utf-8")
    metrics: dict[str, int | float] = {}
    for name, pattern in RESULT_PATTERNS.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing {name!r} in LiSSA results file: {path}")
        metrics[name] = int(match.group(1)) if name in {"tp", "fp", "fn"} else float(match.group(1))
    return metrics


def encode_identifier_part(value: str) -> str:
    return (
        value.strip()
        .replace("%", "%25")
        .replace("|", "%7C")
        .replace(",", "%2C")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def identifier(side: str, row: dict[str, str]) -> str:
    return "|".join(
        [
            side,
            encode_identifier_part(row[f"type{side}"]),
            encode_identifier_part(row[f"path{side}"]),
            encode_identifier_part(row[f"Name{side}"]),
        ]
    )


def load_gold_standard(
    config_path: Path, config_working_dir: Path
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], dict[str, str]], Path, int]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gold_config = config["gold_standard_configuration"]
    if str(gold_config.get("format", "trace_links")).lower() != "pair_csv":
        raise ValueError("The deployment exporter currently requires pair_csv gold standard format")

    source_side = str(gold_config.get("source_side", "B")).strip().upper()
    target_side = str(gold_config.get("target_side", "A")).strip().upper()
    relation_column = str(gold_config.get("relation_column", "Relation"))
    positive_prefix = str(gold_config.get("positive_relation_prefix", "1"))
    if source_side == target_side or source_side not in {"A", "B"} or target_side not in {"A", "B"}:
        raise ValueError("source_side and target_side must be different values from A/B")

    configured_path = Path(gold_config["path"])
    gold_path = configured_path if configured_path.is_absolute() else config_working_dir / configured_path
    gold_path = gold_path.resolve()

    positives: set[tuple[str, str]] = set()
    row_by_link: dict[tuple[str, str], dict[str, str]] = {}
    source_ids: set[str] = set()
    target_ids: set[str] = set()
    with gold_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            link = (identifier(source_side, row), identifier(target_side, row))
            row_by_link.setdefault(link, dict(row))
            source_ids.add(link[0])
            target_ids.add(link[1])
            if row[relation_column].strip().startswith(positive_prefix):
                positives.add(link)
    return positives, row_by_link, gold_path, len(source_ids) * len(target_ids)


def safe_ratio(numerator: int, denominator: int, *, empty_value: float = 0.0) -> float:
    return numerator / denominator if denominator else empty_value


def dataset_scope_metrics(
    predictions: set[tuple[str, str]],
    positives: set[tuple[str, str]],
    labeled_pairs: set[tuple[str, str]],
) -> dict[str, int | float | str]:
    """Calculate a closed confusion matrix over explicitly labeled CSV pairs only."""
    predictions_in_dataset = predictions & labeled_pairs
    true_positives = predictions_in_dataset & positives
    false_positives = predictions_in_dataset - positives
    false_negatives = positives - predictions_in_dataset
    negatives = labeled_pairs - positives
    true_negatives = negatives - predictions_in_dataset
    predictions_outside_dataset = predictions - labeled_pairs

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    tn = len(true_negatives)
    precision = safe_ratio(tp, tp + fp, empty_value=1.0)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    accuracy = safe_ratio(tp + tn, len(labeled_pairs))

    return {
        "evaluation_scope": "explicitly labeled CSV pairs",
        "labeled_pairs": len(labeled_pairs),
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fp_outside_dataset": len(predictions_outside_dataset),
    }


def load_predictions(path: Path) -> set[tuple[str, str]]:
    predictions: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"Invalid trace-link row in {path}: {row!r}")
            predictions.add((row[0], row[1]))
    return predictions


def mismatch_record(
    link: tuple[str, str], kind: str, row_by_link: dict[tuple[str, str], dict[str, str]]
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": kind,
        "source_id": link[0],
        "target_id": link[1],
    }
    row = row_by_link.get(link)
    if row is not None:
        record["csv_row"] = row
    else:
        record["csv_row"] = None
        record["note"] = "This source-target combination is absent from the paired CSV."
    return record


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    original_metrics = parse_original_metrics(args.results_md.resolve())
    gold, row_by_link, gold_path, candidate_universe = load_gold_standard(
        args.config.resolve(), args.config_working_dir.resolve()
    )
    predictions = load_predictions(args.trace_links.resolve())

    true_positives = predictions & gold
    false_positives = predictions - gold
    false_negatives = gold - predictions
    labeled_pairs = set(row_by_link)
    scoped_metrics = dataset_scope_metrics(predictions, gold, labeled_pairs)
    recomputed_counts = {
        "tp": len(true_positives),
        "fp": len(false_positives),
        "fn": len(false_negatives),
    }
    for name, value in recomputed_counts.items():
        if original_metrics[name] != value:
            raise ValueError(
                f"Mismatch between original LiSSA {name}={original_metrics[name]} and link sets ({value})"
            )

    metrics_payload = {
        "meta": {
            "method": "LiSSA",
            "task": "binary trace-link recovery",
            "results_markdown": str(args.results_md.resolve()),
            "prediction_file": str(args.trace_links.resolve()),
            "gold_standard": str(gold_path),
            "closed_world_candidate_universe": candidate_universe,
            "labeled_csv_pairs": len(labeled_pairs),
        },
        "rq1": {
            **original_metrics,
            "tn": None,
            "accuracy": None,
            "note": "Values are copied from LiSSA's original Statistics output; LiSSA reports no TN/accuracy.",
        },
        "rq1_dataset_scope": scoped_metrics,
        "rq2": {
            "status": "not_applicable",
            "metrics": None,
            "reason": "Original LiSSA predicts binary trace-link existence and does not output Refine/Satisfy relation types.",
        },
    }
    mismatch_payload = {
        "counts": {
            "false_positives": len(false_positives),
            "false_positives_inside_dataset": int(scoped_metrics["fp"]),
            "false_positives_outside_dataset": int(scoped_metrics["fp_outside_dataset"]),
            "false_negatives": len(false_negatives),
        },
        "false_positives": [
            mismatch_record(link, "FP", row_by_link) for link in sorted(false_positives)
        ],
        "false_negatives": [
            mismatch_record(link, "FN", row_by_link) for link in sorted(false_negatives)
        ],
    }

    write_json(args.metrics_out.resolve(), metrics_payload)
    write_json(args.mismatch_out.resolve(), mismatch_payload)

    print("RQ1: Link Existence Prediction (LiSSA original metrics)")
    print(json.dumps(metrics_payload["rq1"], ensure_ascii=False, indent=2))
    print("RQ1: Explicitly labeled CSV-pair metrics (post-hoc evaluation only)")
    print(json.dumps(metrics_payload["rq1_dataset_scope"], ensure_ascii=False, indent=2))
    print("RQ2: not applicable - original LiSSA does not predict relation types")
    print(f"Saved metrics: {args.metrics_out.resolve()}")
    print(f"Saved mismatches: {args.mismatch_out.resolve()}")


if __name__ == "__main__":
    main()
