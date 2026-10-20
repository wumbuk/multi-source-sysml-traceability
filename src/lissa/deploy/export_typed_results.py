#!/usr/bin/env python3
"""Export binary and Refine/Satisfy metrics for the typed LiSSA variant."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from export_baseline_results import (
    dataset_scope_metrics,
    identifier,
    load_predictions,
    mismatch_record,
    parse_original_metrics,
    safe_ratio,
    write_json,
)


GOLD_CLASSES = ("Refine", "Satisfy", "None")
PREDICTED_CLASSES = ("Refine", "Satisfy", "None", "Unknown")
DECISION_MARKER = "TYPED_TRACE_DECISION\t"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-md", required=True, type=Path)
    parser.add_argument("--trace-links", required=True, type=Path)
    parser.add_argument("--round-log", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--config-working-dir", required=True, type=Path)
    parser.add_argument("--metrics-out", required=True, type=Path)
    parser.add_argument("--mismatch-out", required=True, type=Path)
    parser.add_argument("--predictions-out", required=True, type=Path)
    return parser.parse_args()


def normalize_gold_label(value: str, positive_prefix: str) -> str:
    label = value.strip()
    if not label.startswith(positive_prefix):
        return "None"
    suffix = label[len(positive_prefix) :].strip().lstrip("/").strip().lower()
    if suffix == "refine":
        return "Refine"
    if suffix == "satisfy":
        return "Satisfy"
    raise ValueError(f"Unsupported positive relation label: {value!r}")


def load_labeled_pairs(
    config_path: Path, config_working_dir: Path
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], str], Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gold_config = config["gold_standard_configuration"]
    if str(gold_config.get("format", "trace_links")).lower() != "pair_csv":
        raise ValueError("Typed evaluation requires a pair_csv gold standard")

    source_side = str(gold_config.get("source_side", "B")).strip().upper()
    target_side = str(gold_config.get("target_side", "A")).strip().upper()
    relation_column = str(gold_config.get("relation_column", "Relation"))
    positive_prefix = str(gold_config.get("positive_relation_prefix", "1"))
    configured_path = Path(gold_config["path"])
    gold_path = configured_path if configured_path.is_absolute() else config_working_dir / configured_path
    gold_path = gold_path.resolve()

    rows: dict[tuple[str, str], dict[str, str]] = {}
    labels: dict[tuple[str, str], str] = {}
    with gold_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            link = (identifier(source_side, row), identifier(target_side, row))
            if link in rows:
                raise ValueError(f"Duplicate labeled source-target pair in {gold_path}: {link}")
            rows[link] = dict(row)
            labels[link] = normalize_gold_label(row[relation_column], positive_prefix)
    if not rows:
        raise ValueError(f"No labeled rows in {gold_path}")
    return rows, labels, gold_path


def parse_typed_decisions(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        marker_index = line.find(DECISION_MARKER)
        if marker_index < 0:
            continue
        fields = line[marker_index + len(DECISION_MARKER) :].split("\t")
        if len(fields) != 4:
            raise ValueError(f"Malformed typed decision at {path}:{line_number}: {fields!r}")
        source_id, target_id, existence, relation_type = fields
        if existence not in {"yes", "no"}:
            raise ValueError(f"Invalid existence decision at {path}:{line_number}: {existence!r}")
        if relation_type not in PREDICTED_CLASSES:
            raise ValueError(f"Invalid relation type at {path}:{line_number}: {relation_type!r}")
        if existence == "no" and relation_type != "None":
            raise ValueError(f"Negative decision must use relation type None at {path}:{line_number}")
        link = (source_id, target_id)
        decision = {"exists": existence == "yes", "relation_type": relation_type}
        previous = decisions.get(link)
        if previous is not None and previous != decision:
            raise ValueError(f"Conflicting typed decisions for {link}: {previous} vs {decision}")
        decisions[link] = decision
    if not decisions:
        raise ValueError(f"No {DECISION_MARKER.strip()} records found in {path}")
    return decisions


def class_metrics(
    actual_by_link: dict[tuple[str, str], str], predicted_by_link: dict[tuple[str, str], str]
) -> dict[str, Any]:
    confusion = {
        actual: {predicted: 0 for predicted in PREDICTED_CLASSES} for actual in GOLD_CLASSES
    }
    for link, actual in actual_by_link.items():
        confusion[actual][predicted_by_link[link]] += 1

    per_class: dict[str, Any] = {}
    for label in GOLD_CLASSES:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in GOLD_CLASSES if other != label)
        fn = sum(confusion[label][other] for other in PREDICTED_CLASSES if other != label)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        per_class[label] = {
            "support": sum(confusion[label].values()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": safe_ratio(2 * precision * recall, precision + recall),
        }

    total = len(actual_by_link)
    correct = sum(confusion[label][label] for label in GOLD_CLASSES)
    macro_precision = sum(per_class[label]["precision"] for label in GOLD_CLASSES) / len(GOLD_CLASSES)
    macro_recall = sum(per_class[label]["recall"] for label in GOLD_CLASSES) / len(GOLD_CLASSES)
    macro_f1 = sum(per_class[label]["f1"] for label in GOLD_CLASSES) / len(GOLD_CLASSES)
    weighted_f1 = safe_ratio(
        sum(per_class[label]["f1"] * per_class[label]["support"] for label in GOLD_CLASSES), total
    )
    return {
        "evaluation_scope": "all explicitly labeled CSV pairs",
        "labels": list(GOLD_CLASSES),
        "predicted_labels": list(PREDICTED_CLASSES),
        "total": total,
        "correct": correct,
        "accuracy": safe_ratio(correct, total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def main() -> None:
    args = parse_args()
    original_metrics = parse_original_metrics(args.results_md.resolve())
    row_by_link, gold_labels, gold_path = load_labeled_pairs(
        args.config.resolve(), args.config_working_dir.resolve()
    )
    labeled_pairs = set(row_by_link)
    positives = {link for link, label in gold_labels.items() if label != "None"}
    predictions = load_predictions(args.trace_links.resolve())
    decisions = parse_typed_decisions(args.round_log.resolve())
    positive_decisions = {link for link, decision in decisions.items() if decision["exists"]}
    if positive_decisions != predictions:
        missing = sorted(predictions - positive_decisions)
        extra = sorted(positive_decisions - predictions)
        raise ValueError(
            "Typed-decision positives do not match LiSSA trace links: "
            f"missing_decisions={missing[:5]}, extra_decisions={extra[:5]}"
        )

    binary_metrics = dataset_scope_metrics(predictions, positives, labeled_pairs)
    recomputed = {
        "tp": len(predictions & positives),
        "fp": len(predictions - positives),
        "fn": len(positives - predictions),
    }
    for field, value in recomputed.items():
        if original_metrics[field] != value:
            raise ValueError(
                f"Original LiSSA {field}={original_metrics[field]} differs from recomputed {value}"
            )

    predicted_labels: dict[tuple[str, str], str] = {}
    for link in labeled_pairs:
        decision = decisions.get(link)
        predicted_labels[link] = (
            decision["relation_type"] if decision is not None and decision["exists"] else "None"
        )
    end_to_end = class_metrics(gold_labels, predicted_labels)

    binary_true_positives = predictions & positives & labeled_pairs
    correct_positive_types = {
        link for link in binary_true_positives if predicted_labels[link] == gold_labels[link]
    }
    conditional = {
        "scope": "gold-positive CSV pairs whose existence was correctly predicted",
        "evaluated": len(binary_true_positives),
        "correct": len(correct_positive_types),
        "accuracy": safe_ratio(len(correct_positive_types), len(binary_true_positives)),
        "unknown_type_count": sum(
            predicted_labels[link] == "Unknown" for link in binary_true_positives
        ),
    }

    prediction_records = []
    for link, decision in sorted(decisions.items()):
        prediction_records.append(
            {
                "source_id": link[0],
                "target_id": link[1],
                "exists": decision["exists"],
                "relation_type": decision["relation_type"],
                "in_labeled_csv": link in labeled_pairs,
                "gold_label": gold_labels.get(link),
                "correct_end_to_end": link in labeled_pairs
                and predicted_labels[link] == gold_labels[link],
            }
        )

    type_errors = []
    for link in sorted(labeled_pairs):
        if predicted_labels[link] != gold_labels[link]:
            type_errors.append(
                {
                    "source_id": link[0],
                    "target_id": link[1],
                    "gold_label": gold_labels[link],
                    "predicted_label": predicted_labels[link],
                    "csv_row": row_by_link[link],
                }
            )

    metrics_payload = {
        "meta": {
            "method": "LiSSA typed-relation variant",
            "task": "hierarchical trace existence followed by Refine/Satisfy classification",
            "not_original_lissa": True,
            "gold_standard": str(gold_path),
            "labeled_csv_pairs": len(labeled_pairs),
            "candidate_decisions": len(decisions),
            "candidate_decisions_inside_csv": len(set(decisions) & labeled_pairs),
            "candidate_decisions_outside_csv": len(set(decisions) - labeled_pairs),
        },
        "rq1_original_lissa_scope": original_metrics,
        "rq1_dataset_scope": binary_metrics,
        "rq2_relation_type": {
            "end_to_end_three_class": end_to_end,
            "conditional_on_correct_existence": conditional,
        },
    }
    mismatch_payload = {
        "binary": {
            "false_positives": [
                mismatch_record(link, "FP", row_by_link) for link in sorted(predictions - positives)
            ],
            "false_negatives": [
                mismatch_record(link, "FN", row_by_link) for link in sorted(positives - predictions)
            ],
        },
        "relation_type_errors": type_errors,
    }

    write_json(args.metrics_out.resolve(), metrics_payload)
    write_json(args.mismatch_out.resolve(), mismatch_payload)
    write_json(args.predictions_out.resolve(), prediction_records)
    print("RQ1 dataset-scope binary metrics:")
    print(json.dumps(binary_metrics, ensure_ascii=False, indent=2))
    print("RQ2 end-to-end relation-type metrics:")
    print(json.dumps(end_to_end, ensure_ascii=False, indent=2))
    print("RQ2 conditional relation-type accuracy:")
    print(json.dumps(conditional, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
