#!/usr/bin/env python3
"""Prepare real Oracle inputs and score three fresh type-only runs per dataset."""

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

DATASETS = {
    "Air_Condition": ("air-condition-typed-qwen-config.json", 358, {"Refine": 40, "Satisfy": 5}),
    "Extract_Requirement_Values": ("extract-value-typed-qwen-config.json", 429, {"Refine": 14, "Satisfy": 14}),
    "Architecting_Spacecraft": ("space-craft-typed-qwen-config.json", 153, {"Refine": 10, "Satisfy": 2}),
}
LABELS = ("Refine", "Satisfy")
PROTOCOL = "oracle_known_existence_type_only"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def identifier(side, row):
    def encode(value):
        return (value.strip().replace("%", "%25").replace("|", "%7C")
                .replace(",", "%2C").replace("\r", "%0D").replace("\n", "%0A"))
    return "|".join([side, *(encode(row[key + side]) for key in ("type", "path", "Name"))])


def prepare(project, suite):
    if suite.exists():
        raise FileExistsError(f"Refusing to overwrite suite: {suite}")
    prepared = {}
    for dataset, (config_file, expected_rows, supports) in DATASETS.items():
        csv_path = project / "Data" / dataset / "Judge.csv"
        config_path = project / "reproduction" / config_file
        config = read(config_path)
        gold_config = config["gold_standard_configuration"]
        if (gold_config["format"] != "pair_csv" or gold_config["source_side"] != "B"
                or gold_config["target_side"] != "A"
                or not gold_config["path"].endswith(f"Data/{dataset}/Judge.csv")):
            raise ValueError(f"Unexpected gold configuration: {config_path}")
        for side in ("source", "target"):
            if config[f"{side}_preprocessor"]["name"] != "artifact":
                raise ValueError("Expected artifact preprocessing")
            if config[f"{side}_artifact_provider"]["name"] != "pair_csv":
                raise ValueError("Expected pair_csv artifact provider")
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected_rows:
            raise ValueError(f"Unexpected CSV row count: {dataset}: {len(rows)}")
        gold, seen = [], set()
        counts = dict.fromkeys(LABELS, 0)
        for index, row in enumerate(rows, 1):
            key = (identifier("B", row), identifier("A", row))
            if key in seen:
                raise ValueError(f"Duplicate CSV pair: {dataset}, record {index}")
            seen.add(key)
            label = row[gold_config.get("relation_column", "Relation")].strip()
            if label == "0":
                continue
            if label not in ("1/Refine", "1/Satisfy"):
                raise ValueError(f"Unsupported gold label: {label}")
            relation = label.split("/", 1)[1]
            counts[relation] += 1
            gold.append({"source_id": key[0], "target_id": key[1],
                         "csv_record_number": index, "gold_type": relation})
        if counts != supports:
            raise ValueError(f"Unexpected positive labels: {dataset}: {counts}")
        prepared[dataset] = (csv_path, config_path, gold)

    inputs = suite / "_inputs"
    inputs.mkdir(parents=True)
    files = []
    for name in ("OracleTypeExperiment.java", "oracle_type_prompt.json", "oracle_experiment.py"):
        destination = inputs / name
        shutil.copy2(project / "deploy" / name, destination)
        files.append(destination)
    for dataset, (csv_path, config_path, gold) in prepared.items():
        folder = inputs / dataset
        folder.mkdir()
        shutil.copy2(csv_path, folder / "Judge.csv")
        shutil.copy2(config_path, folder / "config.json")
        write(folder / "gold.json", gold)
        write(folder / "cases.json", [{"source_id": r["source_id"], "target_id": r["target_id"]} for r in gold])
        files.extend(folder / name for name in ("Judge.csv", "config.json", "gold.json", "cases.json"))
    # Metadata is separate from the metrics JSON.
    write(inputs / "manifest.json", {"protocol": PROTOCOL, "rounds_per_dataset": 3,
          "files": {str(path.relative_to(suite).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in files}})
    print(f"Prepared 45 + 28 + 12 Oracle cases, three fresh runs each: {suite}")


def verify_inputs(suite):
    manifest = read(suite / "_inputs/manifest.json")
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Not a real Oracle type-only suite")
    for relative, expected in manifest["files"].items():
        if hashlib.sha256((suite / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input snapshot was modified: {relative}")


def score(records):
    matrix = {label: dict.fromkeys(LABELS, 0) for label in LABELS}
    for r in records:
        matrix[r["gold_type"]][r["predicted_type"]] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        support = sum(matrix[label].values())
        fn = support - tp
        per_class[label] = {"support": support, "tp": tp, "fp": fp, "fn": fn,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    correct = sum(matrix[label][label] for label in LABELS)
    return {"positive_count": len(records), "correct_type_count": correct,
            "type_accuracy_oracle": correct / len(records),
            "macro_f1_oracle": statistics.fmean(x["f1"] for x in per_class.values()),
            "per_class": per_class, "confusion_matrix": matrix}


def evaluated_records(suite, dataset, round_number):
    folder = suite / dataset / f"round_{round_number}"
    info = read(folder / "run_info.json")
    if (info["protocol"] != PROTOCOL or info["dataset"] != dataset or info["round"] != round_number
            or info["thinking"] is not True or info["cache_enabled"] is not False
            or info["model"] != "Qwen3.5-27B" or info["client_max_tokens"] is not None):
        raise ValueError(f"Wrong inference protocol in {folder}")
    gold = read(suite / "_inputs" / dataset / "gold.json")
    if info["expected_predictions"] != len(gold):
        raise ValueError(f"Wrong expected positive count in {folder}")
    predictions = read(folder / "predictions.json")
    by_pair = {}
    for row in predictions:
        key = (row["source_id"], row["target_id"])
        if (key in by_pair or row["predicted_type"] not in LABELS or row["dataset"] != dataset
                or row["round"] != round_number):
            raise ValueError(f"Invalid or duplicate Oracle prediction in {folder}")
        by_pair[key] = row
    if set(by_pair) != {(r["source_id"], r["target_id"]) for r in gold}:
        raise ValueError(f"Incomplete or extra predictions in {folder}; no metrics will be produced")
    return [{"dataset": dataset, "round": round_number, **r,
             "predicted_type": by_pair[(r["source_id"], r["target_id"])]["predicted_type"]} for r in gold]


def evaluate(suite, dataset, round_number):
    verify_inputs(suite)
    records = evaluated_records(suite, dataset, round_number)
    metrics = {"dataset": dataset, "round": round_number, **score(records)}
    folder = suite / dataset / f"round_{round_number}"
    if any((folder / name).exists() for name in ("metrics.json", "mismatch.json")):
        raise FileExistsError(f"Metrics already exist: {folder}")
    write(folder / "metrics.json", metrics)
    write(folder / "mismatch.json", [r for r in records if r["predicted_type"] != r["gold_type"]])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def aggregate(rounds):
    return {"round_count": 3, "positive_count_per_round": rounds[0]["positive_count"], "rounds": rounds,
            "aggregate": {key: {"mean": statistics.fmean(r[key] for r in rounds),
                                "population_std": statistics.pstdev(r[key] for r in rounds)}
                          for key in ("type_accuracy_oracle", "macro_f1_oracle")}}


def summarize(suite, selected):
    verify_inputs(suite)
    names = [selected] if selected else list(DATASETS)
    datasets = {}
    combined = {number: [] for number in (1, 2, 3)}
    for name in names:
        rounds = []
        for number in (1, 2, 3):
            records = evaluated_records(suite, name, number)
            computed = {"dataset": name, "round": number, **score(records)}
            if read(suite / name / f"round_{number}/metrics.json") != computed:
                raise ValueError(f"Saved metrics disagree with predictions: {name}, round {number}")
            rounds.append(computed)
            combined[number].extend(records)
        datasets[name] = aggregate(rounds)
    if selected:
        write(suite / selected / "metrics.json", datasets[selected])
        write(suite / selected / "mismatch.json", [r for rows in combined.values() for r in rows
                                                   if r["gold_type"] != r["predicted_type"]])
        return
    overall = aggregate([{"round": number, **score(rows)} for number, rows in combined.items()])
    write(suite / "metrics.json", {"datasets": datasets, "overall": overall})
    write(suite / "mismatch.json", [r for rows in combined.values() for r in rows
                                    if r["gold_type"] != r["predicted_type"]])
    rows = []
    for name, value in [*datasets.items(), ("Overall", overall)]:
        rows.append({"dataset": name, "positive_count_per_round": value["positive_count_per_round"],
                     "round_count": 3, **{f"{key}_{stat}": v for key, values in value["aggregate"].items()
                                            for stat, v in values.items()}})
    with (suite / "metrics.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--suite", type=Path, required=True)
    for command in ("verify", "evaluate", "summarize"):
        p = sub.add_parser(command)
        p.add_argument("--suite", type=Path, required=True)
        if command != "verify":
            p.add_argument("--dataset", choices=DATASETS, required=command == "evaluate")
        if command == "evaluate":
            p.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.project.resolve(), args.suite.resolve())
    elif args.command == "verify":
        verify_inputs(args.suite)
    elif args.command == "evaluate":
        evaluate(args.suite, args.dataset, args.round)
    else:
        summarize(args.suite, args.dataset)


if __name__ == "__main__":
    main()
