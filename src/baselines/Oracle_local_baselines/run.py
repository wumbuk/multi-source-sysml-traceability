"""Run the existing Embedding and Supervised baselines once on 85 Oracle pairs."""

import csv
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment_settings import SETTINGS
sys.path.insert(0, str(ROOT / "Llm_baseline"))
from oracle_dataset import PROJECT_COUNTS, build, read_csv, relation_type  # noqa: E402


PROJECTS = tuple(PROJECT_COUNTS)
LABELS = ("Refine", "Satisfy")
MINILM = Path(SETTINGS["embedding"]["model_path"])
ROBERTA = Path(SETTINGS["supervised"]["model_path"])


def write_inference_inputs(gold_dir, inference_dir):
    """Remove gold Relation before either baseline reads the query pairs."""
    inference_dir.mkdir(parents=True, exist_ok=False)
    for project in PROJECTS:
        fields, rows = read_csv(gold_dir / f"{project}.csv")
        if len(rows) != sum(PROJECT_COUNTS[project].values()):
            raise ValueError(f"{project}: Oracle row count changed")
        input_fields = [field for field in fields if field != "Relation"]
        if "SourceSequence" not in input_fields:
            raise ValueError(f"{project}: SourceSequence is missing")
        with (inference_dir / f"{project}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=input_fields)
            writer.writeheader()
            writer.writerows(
                {field: row[field] for field in input_fields} for row in rows
            )


def score(matrix):
    total = sum(matrix[gold][pred] for gold in LABELS for pred in LABELS)
    correct = sum(matrix[label][label] for label in LABELS)
    f1 = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[gold][label] for gold in LABELS if gold != label)
        fn = sum(matrix[label][pred] for pred in LABELS if pred != label)
        f1[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "task": "oracle_relation_type_only",
        "labels": list(LABELS),
        "total": total,
        "valid_predictions": total,
        "failed_predictions": 0,
        "type_accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1.values()) / len(LABELS),
        "per_class_f1": f1,
        "confusion_matrix": {
            "rows": list(LABELS),
            "columns": list(LABELS),
            "values": [[matrix[gold][pred] for pred in LABELS] for gold in LABELS],
        },
    }


def evaluate(prediction_path, gold_path, method, output_dir):
    _, gold_rows = read_csv(gold_path)
    _, predictions = read_csv(prediction_path)
    if len(predictions) != len(gold_rows):
        raise ValueError(
            f"{method}: expected {len(gold_rows)} predictions, got {len(predictions)}"
        )
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    mismatches = []
    pred_field = "best_ref_relation_type" if method == "embedding" else "oracle_pred_label"
    for index, (gold_row, pred_row) in enumerate(zip(gold_rows, predictions), start=1):
        sequence = gold_row["SourceSequence"]
        if pred_row.get("SourceSequence") != sequence:
            raise ValueError(f"{method}: row {index} SourceSequence/order mismatch")
        gold = relation_type(gold_row["Relation"])
        pred_value = pred_row.get(pred_field, "").strip()
        predicted = pred_value.split("/", 1)[-1] if pred_value.startswith("1/") else pred_value
        if predicted not in LABELS:
            raise ValueError(
                f"{method}: row {index} has invalid Oracle type {pred_value!r}"
            )
        matrix[gold][predicted] += 1
        if gold != predicted:
            mismatches.append({
                "SourceSequence": sequence,
                "NameA": gold_row["NameA"],
                "NameB": gold_row["NameB"],
                "gold_relation_type": gold,
                "predicted_relation_type": predicted,
            })
    metrics = score(matrix)
    (output_dir / "oracle_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "oracle_mismatches.json").write_text(
        json.dumps(mismatches, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "oracle_confusion_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["gold\\pred", *LABELS])
        for label in LABELS:
            writer.writerow([label, *(matrix[label][pred] for pred in LABELS)])
    return metrics


def execute(command, cwd, log_path):
    print(f"[RUN] {' '.join(str(part) for part in command)}", flush=True)
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MPLBACKEND": "Agg",
                "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1", "PYTHONIOENCODING": "utf-8"})
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('embedding', 'supervised', 'both'), default='both')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=SETTINGS['supervised']['epochs'],
                        help='Supervised training epochs (configured default: 12)')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    methods = ('embedding', 'supervised') if args.method == 'both' else (args.method,)
    for method in methods:
        if not (MINILM if method == 'embedding' else ROBERTA).is_dir():
            raise FileNotFoundError(f'Local model missing: {method}')
    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "Oracle_local_baselines" / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    gold_dir = run_dir / "validated_gold"
    summary = build(ROOT / "Data" / "Oracle_Positive_85.csv", ROOT / "Data", gold_dir)
    if summary["total"] != 85 or summary["Refine"] != 64 or summary["Satisfy"] != 21:
        raise ValueError("Oracle source counts changed")
    inference_dir = run_dir / "inference_inputs"
    write_inference_inputs(gold_dir, inference_dir)
    print(f"Validated 85 Oracle rows; run ID: {run_id}", flush=True)

    all_results = {}
    for method in methods:
        method_results = {}
        combined = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
        for project in PROJECTS:
            kg = ROOT / "Data" / project / "KG.csv"
            input_path = inference_dir / f"{project}.csv"
            gold_path = gold_dir / f"{project}.csv"
            if not kg.is_file():
                raise FileNotFoundError(kg)
            if method == "embedding":
                cwd = ROOT / "Embedding_matched_experiment"
                output_dir = cwd / "oracle_outputs" / run_id / project
                prediction_path = output_dir / "match_results.csv"
                command = [
                    sys.executable, "main.py", "--kg-samples", str(kg),
                    "--query-pairs", str(input_path), "--output", str(prediction_path),
                    "--model-name", str(MINILM), "--threshold", str(SETTINGS['embedding']['threshold']),
                    "--seed", str(args.seed), "--device", "cuda",
                ]
            else:
                cwd = ROOT / "Supervised_experiment"
                output_dir = cwd / "oracle_outputs" / run_id / project
                prediction_path = output_dir / "judge_predictions.csv"
                command = [
                    sys.executable, "quanliang_main.py", "--train-path", str(kg),
                    "--test-path", str(input_path), "--output-dir", str(output_dir),
                    "--model-path", str(ROBERTA), "--epochs", str(args.epochs), "--seed", str(args.seed),
                    "--oracle-positive-only",
                ]
            output_dir.mkdir(parents=True, exist_ok=False)
            (output_dir / 'run_config.json').write_text(json.dumps({
                'method': method, 'command': command, 'seed': args.seed,
                'epochs': args.epochs if method == 'supervised' else None,
                'protocol': 'Oracle configuration; epochs explicit, other settings unchanged',
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"[START] {method} / {project}", flush=True)
            execute(command, cwd, output_dir / "run.log")
            metrics = evaluate(prediction_path, gold_path, method, output_dir)
            method_results[project] = {"output_dir": str(output_dir), "metrics": metrics}
            for gold, values in zip(LABELS, metrics["confusion_matrix"]["values"]):
                for pred, count in zip(LABELS, values):
                    combined[gold][pred] += count
            print(
                f"[DONE] {method} / {project}: "
                f"{metrics['type_accuracy']:.4f} ({metrics['total']} rows)", flush=True
            )
        method_results["combined"] = score(combined)
        all_results[method] = method_results
        method_root = output_dir.parent
        (method_root / 'oracle_metrics_all_85.json').write_text(
            json.dumps(method_results['combined'], ensure_ascii=False, indent=2), encoding='utf-8')
        (method_root / 'summary.json').write_text(
            json.dumps(method_results, ensure_ascii=False, indent=2), encoding='utf-8')
        (run_dir / f"{method}_combined_metrics.json").write_text(
            json.dumps(method_results["combined"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (run_dir / "summary.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"All Oracle local baselines completed: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
