"""Validate Oracle_Positive_85.csv and materialize Judge-backed Oracle inputs."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_COUNTS = {
    "Air_Condition": {"Refine": 40, "Satisfy": 5},
    "Architecting_Spacecraft": {"Refine": 10, "Satisfy": 2},
    "Extract_Requirement_Values": {"Refine": 14, "Satisfy": 14},
}
MODEL_FIELDS = (
    "NameA", "typeA", "pathA", "propertyA",
    "NameB", "typeB", "propertyB", "contextB",
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        return list(reader.fieldnames), list(reader)


def relation_type(value):
    label = str(value).strip()
    if label not in {"1/Refine", "1/Satisfy"}:
        raise ValueError(f"Oracle row has invalid positive Relation: {value!r}")
    return label.split("/", 1)[1]


def prepare(source, data_root):
    oracle_fields, source_rows = read_csv(source)
    required = {"Project", "SourceSequence", "Relation", *MODEL_FIELDS}
    if not required.issubset(oracle_fields):
        raise ValueError(f"Oracle CSV missing columns: {sorted(required - set(oracle_fields))}")
    if len(source_rows) != 85:
        raise ValueError(f"Expected 85 Oracle rows, found {len(source_rows)}")

    found = Counter()
    staged = {}
    for project, expected in PROJECT_COUNTS.items():
        judge_fields, judge_rows = read_csv(data_root / project / "Judge.csv")
        comparison_fields = [name for name in judge_fields if name in oracle_fields]
        if set(comparison_fields) != set(judge_fields):
            raise ValueError(f"{project}: Oracle CSV missing Judge columns")
        if not required.intersection(MODEL_FIELDS).issubset(judge_fields):
            raise ValueError(f"{project}: Judge CSV missing model input columns")
        staged[project] = (judge_fields, judge_rows, comparison_fields, [], set())

    for source_index, row in enumerate(source_rows, start=2):
        project = row["Project"].strip()
        if project not in staged:
            raise ValueError(f"Oracle CSV line {source_index}: unknown Project {project!r}")
        judge_fields, judge_rows, fields, selected, seen = staged[project]
        try:
            sequence = int(row["SourceSequence"].strip())
        except ValueError as exc:
            raise ValueError(f"Oracle CSV line {source_index}: invalid SourceSequence") from exc
        # SourceSequence is the original Judge.csv physical line number:
        # header is line 1, first sample is line 2.
        if not 2 <= sequence <= len(judge_rows) + 1:
            raise ValueError(f"{project}: SourceSequence {sequence} outside Judge.csv")
        if sequence in seen:
            raise ValueError(f"{project}: duplicate SourceSequence {sequence}")
        seen.add(sequence)
        judge_row = judge_rows[sequence - 2]
        for field in fields:
            if row[field].strip() != judge_row[field].strip():
                raise ValueError(
                    f"{project} SourceSequence {sequence}: {field} differs from Judge.csv"
                )
        kind = relation_type(row["Relation"])
        if relation_type(judge_row["Relation"]) != kind:
            raise ValueError(f"{project} SourceSequence {sequence}: gold label differs")
        found[(project, kind)] += 1
        # Keep the source file's sample order and the full, unmodified Judge row.
        selected.append({**judge_row, "SourceSequence": str(sequence)})

    for project, expected in PROJECT_COUNTS.items():
        for kind, count in expected.items():
            if found[(project, kind)] != count:
                raise ValueError(f"{project}/{kind}: expected {count}, found {found[(project, kind)]}")
    if sum(found.values()) != 85 or sum(found[(p, "Refine")] for p in staged) != 64 or sum(found[(p, "Satisfy")] for p in staged) != 21:
        raise ValueError("Oracle totals must be 85, Refine=64, Satisfy=21")
    return staged, found


def build(source, data_root, output_dir):
    staged, found = prepare(source, data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"total": 85, "Refine": 64, "Satisfy": 21, "projects": {}}
    for project, (judge_fields, _, _, rows, _) in staged.items():
        target = output_dir / f"{project}.csv"
        with target.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*judge_fields, "SourceSequence"])
            writer.writeheader()
            writer.writerows(rows)
        summary["projects"][project] = {
            "total": len(rows),
            "Refine": found[(project, "Refine")],
            "Satisfy": found[(project, "Satisfy")],
        }
    (output_dir / "validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "Data" / "Oracle_Positive_85.csv")
    parser.add_argument("--data-root", type=Path, default=root / "Data")
    parser.add_argument("--output-dir", type=Path, default=root / "Llm_baseline" / "oracle" / "datasets")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.data_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
