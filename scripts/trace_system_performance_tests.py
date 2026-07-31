from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pickle
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MINIMUMS = {
    "xmi_supported_size_mb": 8.0,
    "xmi_parse_success_rate": 0.72,
    "xmi_parse_seconds": 38.0,
    "candidate_field_completeness": 0.72,
    "kg_cache_load_seconds": 13.0,
    "kg_first_build_seconds": 150.0,
    "trace_context_seconds": 38.0,
    "llm_end_to_end_seconds": 150.0,
    "existence_accuracy": 0.64,
    "relation_type_accuracy": 0.56,
    "review_relation_count": 800,
}

TARGETS = {
    "xmi_supported_size_mb": 10.0,
    "xmi_parse_success_rate": 0.90,
    "xmi_parse_seconds": 30.0,
    "candidate_field_completeness": 0.90,
    "kg_cache_load_seconds": 10.0,
    "kg_first_build_seconds": 120.0,
    "trace_context_seconds": 30.0,
    "llm_end_to_end_seconds": 120.0,
    "existence_accuracy": 0.80,
    "relation_type_accuracy": 0.70,
    "review_relation_count": 1000,
}

REQUIRED_CANDIDATE_FIELDS = [
    "NameA",
    "pathA",
    "contextA",
    "propertyA",
    "typeA",
    "NameB",
    "pathB",
    "contextB",
    "propertyB",
    "typeB",
]

TRACE_CONTEXT_TIMING_KEYS = [
    "graph_context_extraction",
    "case_based_retrieval",
    "ontology_reasoning",
    "information_construction_overhead",
]


@dataclass
class MetricResult:
    name: str
    value: float | int | None
    minimum: float | int | None
    target: float | int | None
    direction: str
    status: str
    note: str = ""


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def judge_metric(
    name: str,
    value: float | int | None,
    direction: str,
    note: str = "",
) -> MetricResult:
    minimum = MINIMUMS.get(name)
    target = TARGETS.get(name)
    if value is None:
        return MetricResult(name, None, minimum, target, direction, "SKIP", note)

    if direction == "higher":
        if minimum is not None and value < minimum:
            status = "FAIL"
        elif target is not None and value < target:
            status = "PASS_MINIMUM"
        else:
            status = "PASS_TARGET"
    elif direction == "lower":
        if minimum is not None and value > minimum:
            status = "FAIL"
        elif target is not None and value > target:
            status = "PASS_MINIMUM"
        else:
            status = "PASS_TARGET"
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return MetricResult(name, value, minimum, target, direction, status, note)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_records_json(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("records", "results", "predictions", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def test_xmi_parsing(project_root: Path, xmi_dir: Path | None, package_path: list[str] | None) -> list[MetricResult]:
    if not xmi_dir:
        return [
            judge_metric("xmi_supported_size_mb", None, "higher", "No --xmi-dir provided."),
            judge_metric("xmi_parse_success_rate", None, "higher", "No --xmi-dir provided."),
            judge_metric("xmi_parse_seconds", None, "lower", "No --xmi-dir provided."),
        ]
    xmi_files = sorted([*xmi_dir.glob("*.xmi"), *xmi_dir.glob("*.xml")])
    if not xmi_files:
        return [
            judge_metric("xmi_supported_size_mb", None, "higher", f"No XMI/XML files found in {xmi_dir}."),
            judge_metric("xmi_parse_success_rate", None, "higher", f"No XMI/XML files found in {xmi_dir}."),
            judge_metric("xmi_parse_seconds", None, "lower", f"No XMI/XML files found in {xmi_dir}."),
        ]

    sys.path.insert(0, str(project_root))
    extractor_module = importlib.import_module("Port_Information.information_extract")
    extractor = extractor_module.XMIInformationExtractor()

    success_count = 0
    parsed_sizes_mb: list[float] = []
    parse_seconds: list[float] = []
    errors: list[str] = []
    for path in xmi_files:
        started = time.perf_counter()
        try:
            extractor.parse(path, package_path=package_path)
            elapsed = time.perf_counter() - started
            success_count += 1
            parsed_sizes_mb.append(path.stat().st_size / (1024 * 1024))
            parse_seconds.append(elapsed)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    max_file_mb = max((p.stat().st_size / (1024 * 1024) for p in xmi_files), default=0.0)
    max_success_mb = max(parsed_sizes_mb, default=None)
    success_rate = success_count / len(xmi_files)
    p80_seconds = percentile(parse_seconds, 80)

    size_note = f"test_files={len(xmi_files)}, max_file_mb={max_file_mb:.2f}"
    if max_file_mb < MINIMUMS["xmi_supported_size_mb"]:
        size_note += "; corpus does not include an 8MB-class XMI file, so size support is not fully proven"
    if errors:
        size_note += "; errors=" + " | ".join(errors[:3])

    return [
        judge_metric("xmi_supported_size_mb", max_success_mb, "higher", size_note),
        judge_metric("xmi_parse_success_rate", success_rate, "higher", f"parsed={success_count}/{len(xmi_files)}"),
        judge_metric("xmi_parse_seconds", p80_seconds, "lower", "80th percentile parse time."),
    ]


def test_candidate_csv(candidate_csv: Path | None) -> list[MetricResult]:
    if not candidate_csv:
        return [judge_metric("candidate_field_completeness", None, "higher", "No --candidate-csv provided.")]
    with candidate_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [judge_metric("candidate_field_completeness", 0.0, "higher", "Candidate CSV has no rows.")]

    present_fields = [field for field in REQUIRED_CANDIDATE_FIELDS if field in rows[0]]
    total_cells = len(REQUIRED_CANDIDATE_FIELDS) * len(rows)
    filled = 0
    for row in rows:
        for field in REQUIRED_CANDIDATE_FIELDS:
            value = row.get(field, "")
            if str(value).strip():
                filled += 1
    completeness = filled / total_cells if total_cells else 0.0
    note = f"rows={len(rows)}, present_required_fields={len(present_fields)}/{len(REQUIRED_CANDIDATE_FIELDS)}"
    return [judge_metric("candidate_field_completeness", completeness, "higher", note)]


def test_kg(project_root: Path, kg_csv: Path | None, kg_cache_path: Path | None) -> list[MetricResult]:
    results: list[MetricResult] = []
    if kg_cache_path:
        started = time.perf_counter()
        try:
            if kg_cache_path.suffix.lower() in {".pkl", ".pickle"}:
                with kg_cache_path.open("rb") as f:
                    pickle.load(f)
            elif kg_cache_path.suffix.lower() == ".graphml":
                import networkx as nx

                nx.read_graphml(kg_cache_path)
            else:
                read_json(kg_cache_path)
            elapsed = time.perf_counter() - started
            results.append(judge_metric("kg_cache_load_seconds", elapsed, "lower", str(kg_cache_path)))
        except Exception as exc:
            results.append(judge_metric("kg_cache_load_seconds", None, "lower", f"cache load failed: {exc}"))
    else:
        results.append(judge_metric("kg_cache_load_seconds", None, "lower", "No --kg-cache-path provided."))

    if kg_csv:
        sys.path.insert(0, str(project_root))
        started = time.perf_counter()
        try:
            construct_module = importlib.import_module("kg_retrieve.construct_KG")
            graph, *_ = construct_module.construct_kg_init(str(kg_csv))
            elapsed = time.perf_counter() - started
            note = f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}, source={kg_csv}"
            results.append(judge_metric("kg_first_build_seconds", elapsed, "lower", note))
        except Exception as exc:
            results.append(judge_metric("kg_first_build_seconds", None, "lower", f"kg build failed: {exc}"))
    else:
        results.append(judge_metric("kg_first_build_seconds", None, "lower", "No --kg-csv provided."))

    return results


def iter_timing_samples(timing_json: Path) -> list[dict[str, Any]]:
    data = read_json(timing_json)
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        return [x for x in data["samples"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def test_timing_json(timing_json: Path | None, percentile_value: float) -> list[MetricResult]:
    if not timing_json:
        return [
            judge_metric("trace_context_seconds", None, "lower", "No --timing-json provided."),
            judge_metric("llm_end_to_end_seconds", None, "lower", "No --timing-json provided."),
        ]
    samples = iter_timing_samples(timing_json)
    context_seconds: list[float] = []
    total_seconds: list[float] = []
    for sample in samples:
        timing = sample.get("timing_ms", sample)
        if not isinstance(timing, dict):
            continue
        context_ms = sum(float(timing.get(key, 0.0) or 0.0) for key in TRACE_CONTEXT_TIMING_KEYS)
        if context_ms:
            context_seconds.append(context_ms / 1000.0)
        total_ms = timing.get("online_total", timing.get("total", timing.get("end_to_end")))
        if total_ms is not None:
            total_seconds.append(float(total_ms) / 1000.0)

    context_value = percentile(context_seconds, percentile_value)
    total_value = percentile(total_seconds, percentile_value)
    return [
        judge_metric(
            "trace_context_seconds",
            context_value,
            "lower",
            f"{percentile_value:g}th percentile, samples={len(context_seconds)}",
        ),
        judge_metric(
            "llm_end_to_end_seconds",
            total_value,
            "lower",
            f"{percentile_value:g}th percentile, samples={len(total_seconds)}",
        ),
    ]


def normalize_relation(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in {"", "0", "none", "null", "nan", "false", "no relation", "no_relation"}:
        return ""
    aliases = {
        "refineto": "refine",
        "refine_to": "refine",
        "refine-to": "refine",
        "satisfyto": "satisfy",
        "satisfy_to": "satisfy",
        "satisfy-to": "satisfy",
        "derive_reqt": "derive",
        "derive-reqt": "derive",
    }
    return aliases.get(text.replace(" ", ""), text)


def find_prediction_field(row: dict[str, Any], candidates: list[str], contains: list[str]) -> Any:
    for key in candidates:
        if key in row:
            return row[key]
    for key, value in row.items():
        if all(token in key for token in contains):
            return value
    return None


def test_accuracy(metrics_json: Path | None, predictions_json: Path | None, golden_csv: Path | None) -> list[MetricResult]:
    if metrics_json:
        data = read_json(metrics_json)
        existence_accuracy = None
        relation_type_accuracy = None
        if isinstance(data, dict):
            existence_accuracy = data.get("rq1", {}).get("accuracy")
            rq2 = data.get("rq2", {})
            if isinstance(rq2, dict):
                for key in ("pipeline_conditional", "oracle"):
                    section = rq2.get(key)
                    if isinstance(section, dict) and section.get("type_accuracy") is not None:
                        relation_type_accuracy = section["type_accuracy"]
                        break
        return [
            judge_metric("existence_accuracy", existence_accuracy, "higher", str(metrics_json)),
            judge_metric("relation_type_accuracy", relation_type_accuracy, "higher", str(metrics_json)),
        ]

    if not predictions_json or not golden_csv:
        return [
            judge_metric("existence_accuracy", None, "higher", "Provide --metrics-json or both --predictions-json and --golden-csv."),
            judge_metric("relation_type_accuracy", None, "higher", "Provide --metrics-json or both --predictions-json and --golden-csv."),
        ]

    predictions = load_records_json(predictions_json)
    with golden_csv.open("r", encoding="utf-8-sig", newline="") as f:
        gold_rows = list(csv.DictReader(f))
    n = min(len(predictions), len(gold_rows))
    if n == 0:
        return [
            judge_metric("existence_accuracy", 0.0, "higher", "No comparable records."),
            judge_metric("relation_type_accuracy", 0.0, "higher", "No comparable records."),
        ]

    exist_correct = 0
    type_correct = 0
    type_total = 0
    for pred, gold in zip(predictions[:n], gold_rows[:n]):
        gold_relation = normalize_relation(gold.get("Relation"))
        pred_exists = find_prediction_field(pred, ["exists", "是否存在追溯关系"], ["是否", "存在"])
        pred_relation = find_prediction_field(pred, ["selected_relation", "relation", "选择的关系"], ["关系"])
        if pred_exists is None:
            pred_relation_norm = normalize_relation(pred_relation)
            pred_exists_bool = bool(pred_relation_norm)
        else:
            pred_exists_bool = bool(pred_exists) if isinstance(pred_exists, bool) else str(pred_exists).strip().lower() in {"true", "1", "yes"}
            pred_relation_norm = normalize_relation(pred_relation)
        gold_exists = bool(gold_relation)
        if pred_exists_bool == gold_exists:
            exist_correct += 1
        if gold_exists:
            type_total += 1
            if pred_relation_norm == gold_relation:
                type_correct += 1

    return [
        judge_metric("existence_accuracy", exist_correct / n, "higher", f"compared={n}"),
        judge_metric("relation_type_accuracy", type_correct / type_total if type_total else None, "higher", f"positive_gold={type_total}"),
    ]


def test_review_results(review_results_json: Path | None) -> list[MetricResult]:
    if not review_results_json:
        return [judge_metric("review_relation_count", None, "higher", "No --review-results-json provided.")]
    records = load_records_json(review_results_json)
    status_fields = 0
    for row in records:
        if "review_status" in row or "status" in row:
            status_fields += 1
    note = f"records={len(records)}, records_with_review_status={status_fields}"
    return [judge_metric("review_relation_count", len(records), "higher", note)]


def print_summary(results: list[MetricResult]) -> None:
    print("\nTrace system performance metrics")
    print("=" * 88)
    print(f"{'metric':34} {'value':>12} {'minimum':>12} {'target':>12} {'status':>14}")
    print("-" * 88)
    for r in results:
        value = "SKIP" if r.value is None else f"{r.value:.4g}" if isinstance(r.value, float) else str(r.value)
        minimum = "" if r.minimum is None else str(r.minimum)
        target = "" if r.target is None else str(r.target)
        print(f"{r.name:34} {value:>12} {minimum:>12} {target:>12} {r.status:>14}")
        if r.note:
            print(f"  note: {r.note}")
    print("=" * 88)


def parse_package_path(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split("/") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate trace_system minimum and target performance metrics."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--xmi-dir", type=Path)
    parser.add_argument("--xmi-package-path", type=str)
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--kg-csv", type=Path)
    parser.add_argument("--kg-cache-path", type=Path)
    parser.add_argument("--timing-json", type=Path)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--predictions-json", type=Path)
    parser.add_argument("--golden-csv", type=Path)
    parser.add_argument("--review-results-json", type=Path)
    parser.add_argument("--percentile", type=float, default=80.0)
    parser.add_argument("--report-json", type=Path, default=Path("output/pdf/performance_metrics_report.json"))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    results: list[MetricResult] = []
    results.extend(test_xmi_parsing(project_root, args.xmi_dir, parse_package_path(args.xmi_package_path)))
    results.extend(test_candidate_csv(args.candidate_csv))
    results.extend(test_kg(project_root, args.kg_csv, args.kg_cache_path))
    results.extend(test_timing_json(args.timing_json, args.percentile))
    results.extend(test_accuracy(args.metrics_json, args.predictions_json, args.golden_csv))
    results.extend(test_review_results(args.review_results_json))

    print_summary(results)
    report = {
        "minimums": MINIMUMS,
        "targets": TARGETS,
        "results": [asdict(r) for r in results],
        "summary": {
            "fail": sum(1 for r in results if r.status == "FAIL"),
            "pass_minimum": sum(1 for r in results if r.status == "PASS_MINIMUM"),
            "pass_target": sum(1 for r in results if r.status == "PASS_TARGET"),
            "skip": sum(1 for r in results if r.status == "SKIP"),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to: {args.report_json}")
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
