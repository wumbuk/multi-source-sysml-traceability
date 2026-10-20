"""Run independent Oracle relation-type inference on one prepared dataset."""

import argparse
import json
import csv
from pathlib import Path

from oracle_client import call_oracle_llm
from oracle_dataset import relation_type
from oracle_prompt import build_prompt, parse_relation_type


LABELS = ("Refine", "Satisfy")
EXPECTED = {"Air_Condition": 45, "Architecting_Spacecraft": 12, "Extract_Requirement_Values": 28}


def read_rows(path, project):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if reader.fieldnames is None or "SourceSequence" not in reader.fieldnames:
            raise ValueError("Oracle input must contain SourceSequence")
    if len(rows) != EXPECTED[project]:
        raise ValueError(f"{project}: expected {EXPECTED[project]} rows, found {len(rows)}")
    for row in rows:
        relation_type(row["Relation"])
    return rows


def score(records):
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    failures = {label: 0 for label in LABELS}
    errors = []
    for record in records:
        gold, pred = record["gold_relation_type"], record["predicted_relation_type"]
        if pred is None:
            failures[gold] += 1
            errors.append(record)
        else:
            matrix[gold][pred] += 1
            if gold != pred:
                errors.append(record)
    total = len(records)
    correct = sum(matrix[label][label] for label in LABELS)
    f1 = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        # Parse/API failures count as false negatives for their gold class.
        fn = sum(
            r["gold_relation_type"] == label and r["predicted_relation_type"] != label
            for r in records
        )
        f1[label] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "task": "oracle_relation_type_only",
        "labels": list(LABELS),
        "total": total,
        "valid_predictions": total - sum(failures.values()),
        "failed_predictions": sum(failures.values()),
        "type_accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1.values()) / len(LABELS),
        "per_class_f1": f1,
        "confusion_matrix": {
            "rows": list(LABELS), "columns": list(LABELS),
            "values": [[matrix[g][p] for p in LABELS] for g in LABELS],
            "excluded_failures_by_gold": failures,
        },
    }, errors


def run(rows, output_dir, client_call=call_oracle_llm, fail_fast=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "oracle_predictions.json"
    records = []
    for index, row in enumerate(rows, start=1):
        prompt = build_prompt(row)
        raw = None
        error = None
        try:
            raw = client_call(prompt)
            predicted = parse_relation_type(raw)
        except Exception as exc:
            predicted = None
            error = f"{type(exc).__name__}: {exc}"
        records.append({
            "sample_index": index,
            "SourceSequence": row["SourceSequence"],
            "gold_relation_type": relation_type(row["Relation"]),
            "predicted_relation_type": predicted,
            "status": "ok" if error is None else "failed",
            "error": error,
            "raw_response": raw,
        })
        # Checkpoint every sample so an interrupted Slurm job retains evidence.
        predictions_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{index}/{len(rows)} SourceSequence={row['SourceSequence']} status={records[-1]['status']}", flush=True)
        if fail_fast and error is not None:
            break
    metrics, errors = score(records)
    (output_dir / "oracle_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "oracle_mismatches.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "oracle_confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gold\\pred", *LABELS])
        for label, counts in zip(LABELS, metrics["confusion_matrix"]["values"]):
            writer.writerow([label, *counts])
    return metrics


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, choices=tuple(EXPECTED))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Validate and inspect one prompt without contacting a model")
    parser.add_argument("--one-sample", action="store_true", help="Call the model for only the first validated sample")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first API or parsing failure")
    args = parser.parse_args()
    input_path = args.input or root / "oracle" / "datasets" / f"{args.project}.csv"
    rows = read_rows(input_path, args.project)
    if args.preflight:
        prompt = build_prompt(rows[0])
        print(json.dumps({"project": args.project, "sample": 1, "prompt": prompt}, ensure_ascii=False, indent=2))
        return
    if args.one_sample:
        rows = rows[:1]
    metrics = run(rows, args.output_dir, fail_fast=args.fail_fast)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if metrics["failed_predictions"]:
        raise SystemExit(
            f"Oracle run is incomplete: {metrics['failed_predictions']} of "
            f"{metrics['total']} predictions failed; inspect oracle_predictions.json"
        )


if __name__ == "__main__":
    main()
