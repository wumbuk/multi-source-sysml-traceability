# Embedding matching result analysis tool.
# Reads match_results.csv, computes RQ1/RQ2 metrics, and exports mismatch samples
# in the same structure as the LLM baseline mismatch JSON.

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from sklearn.metrics import accuracy_score, classification_report, f1_score
except ImportError as exc:
    accuracy_score = None
    classification_report = None
    f1_score = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


RESULTS = Path(__file__).resolve().parent.parent / "results" / "Embedding_baseline"
DEFAULT_MATCH_CSV = RESULTS / "match_results.csv"
DEFAULT_OUTPUT_METRICS = RESULTS / "match_results_metrics.json"
DEFAULT_OUTPUT_MISMATCH = RESULTS / "match_results_mismatch.json"

NO_RELATION = "__NO_RELATION__"
NO_PREDICTION = "__NO_PREDICTION__"

CSV_REQUIRED_FIELDS = [
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
    "Relation",
    "pred_has_relation",
    "pred_relation_type",
]

NO_RELATION_TOKENS = {
    "",
    "0",
    "null",
    "none",
    "nan",
    "n/a",
    "na",
    "no relation",
    "no_relation",
    "no-relation",
    "unrelated",
    "false",
    "无",
    "无关系",
    "没有关系",
    "不存在",
}

RELATION_ALIASES = {
    "refine": "Refine",
    "refineto": "Refine",
    "refine to": "Refine",
    "refine_to": "Refine",
    "refine-to": "Refine",
    "satisfy": "Satisfy",
    "satisfyto": "Satisfy",
    "satisfy to": "Satisfy",
    "satisfy_to": "Satisfy",
    "satisfy-to": "Satisfy",
    "derivereqt": "Derive",
    "derive reqt": "Derive",
    "derive_reqt": "Derive",
    "derive-reqt": "Derive",
    "derive": "Derive",
}

TRUE_BOOL_TOKENS = {
    "true",
    "yes",
    "y",
    "1",
    "是",
    "有",
    "有关系",
    "存在",
    "存在关系",
    "存在追溯关系",
}

FALSE_BOOL_TOKENS = {
    "false",
    "no",
    "n",
    "0",
    "",
    "null",
    "none",
    "否",
    "无",
    "无关系",
    "没有",
    "没有关系",
    "不存在",
    "不存在关系",
    "不存在追溯关系",
}


def load_csv(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)
    return rows, fieldnames


def validate_csv_fields(fieldnames: Sequence[str]) -> None:
    missing_fields = [field for field in CSV_REQUIRED_FIELDS if field not in fieldnames]
    if missing_fields:
        raise KeyError(f"CSV 缺少字段: {missing_fields}; 当前可用字段: {list(fieldnames)}")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0

    text = str(value).strip().lower()
    normalized_text = re.sub(r"[\s_\-]+", " ", text).strip()

    if text in TRUE_BOOL_TOKENS or normalized_text in TRUE_BOOL_TOKENS:
        return True
    if text in FALSE_BOOL_TOKENS or normalized_text in FALSE_BOOL_TOKENS:
        return False
    if "不存在" in text or "无关系" in text or "没有关系" in text:
        return False
    if "存在" in text or "有关系" in text:
        return True
    return False


def normalize_relation(rel: Any) -> str:
    if rel is None:
        return NO_RELATION

    text = str(rel).strip()
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    if lowered in NO_RELATION_TOKENS:
        return NO_RELATION

    match = re.match(r"^\s*\d+\s*/\s*(.+?)\s*$", text)
    if match:
        text = match.group(1).strip()
        lowered = re.sub(r"\s+", " ", text.lower()).strip()
        if lowered in NO_RELATION_TOKENS:
            return NO_RELATION

    alias_key = lowered
    compact_key = re.sub(r"[\s_\-]+", "", alias_key)
    if alias_key in RELATION_ALIASES:
        return RELATION_ALIASES[alias_key]
    if compact_key in RELATION_ALIASES:
        return RELATION_ALIASES[compact_key]
    return text


def is_no_relation(rel: str) -> bool:
    return rel == NO_RELATION


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def strip_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def build_samples(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        seq = i + 2
        pred_has_relation = parse_bool(row.get("pred_has_relation"))
        pred_relation = normalize_relation(row.get("pred_relation_type"))
        oracle_relation = normalize_relation(row.get("best_ref_relation_type"))
        gold_relation = normalize_relation(row.get("Relation"))

        if not pred_has_relation:
            pred_relation = NO_RELATION

        sample = {
            "seq": seq,
            "row": row,
            "pred_has_relation": pred_has_relation and not is_no_relation(pred_relation),
            "gold_has_relation": not is_no_relation(gold_relation),
            "pred_relation": pred_relation,
            "oracle_relation": oracle_relation,
            "gold_relation": gold_relation,
            "pred_relation_raw": row.get("pred_relation_type"),
            "gold_relation_raw": row.get("Relation"),
        }
        samples.append(sample)

    return samples


def compute_rq1_metrics(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0

    for sample in samples:
        pred_has_relation = sample["pred_has_relation"]
        gold_has_relation = sample["gold_has_relation"]

        if pred_has_relation and gold_has_relation:
            tp += 1
        elif pred_has_relation and not gold_has_relation:
            fp += 1
        elif not pred_has_relation and not gold_has_relation:
            tn += 1
        elif not pred_has_relation and gold_has_relation:
            fn += 1

    total = tp + fp + tn + fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
    }


def unique_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered_values: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered_values.append(value)
    return ordered_values


def build_rq2_eval_samples(
    samples: Sequence[Dict[str, Any]],
    rq2_setting: str,
    pipeline_eval: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    eval_samples: List[Dict[str, Any]] = []
    coverage = {
        "gold_positive_total": sum(1 for sample in samples if sample["gold_has_relation"]),
        "predicted_positive_total": sum(1 for sample in samples if sample["pred_has_relation"]),
        "evaluated_positive_total": 0,
        "missed_gold_positive_total": sum(
            1
            for sample in samples
            if sample["gold_has_relation"] and not sample["pred_has_relation"]
        ),
        "false_positive_link_total": sum(
            1
            for sample in samples
            if sample["pred_has_relation"] and not sample["gold_has_relation"]
        ),
    }

    for sample in samples:
        gold_has_relation = sample["gold_has_relation"]
        pred_has_relation = sample["pred_has_relation"]

        if rq2_setting == "oracle":
            if not gold_has_relation:
                continue
            rq2_pred_relation = sample["oracle_relation"]
            if is_no_relation(rq2_pred_relation):
                raise ValueError(
                    "Oracle evaluation requires 'best_ref_relation_type' "
                    "for every gold-positive sample."
                )
        elif rq2_setting == "pipeline" and pipeline_eval == "conditional":
            if not (pred_has_relation and gold_has_relation):
                continue
            rq2_pred_relation = sample["pred_relation"]
        elif rq2_setting == "pipeline" and pipeline_eval == "end2end":
            if not gold_has_relation:
                continue
            rq2_pred_relation = sample["pred_relation"] if pred_has_relation else NO_PREDICTION
        else:
            raise ValueError(
                f"不支持的 RQ2 设置: rq2_setting={rq2_setting}, pipeline_eval={pipeline_eval}"
            )

        eval_sample = dict(sample)
        eval_sample["rq2_pred_relation"] = rq2_pred_relation
        eval_samples.append(eval_sample)

    coverage["evaluated_positive_total"] = len(eval_samples)
    return eval_samples, coverage


def manual_classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    total_support = 0
    macro_precision = macro_recall = macro_f1 = 0.0
    weighted_precision = weighted_recall = weighted_f1 = 0.0
    micro_tp = micro_fp = micro_fn = 0

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
        total_support += support
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    label_count = len(labels)
    micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
    report["micro avg"] = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1-score": safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall),
        "support": total_support,
    }
    report["macro avg"] = {
        "precision": safe_div(macro_precision, label_count),
        "recall": safe_div(macro_recall, label_count),
        "f1-score": safe_div(macro_f1, label_count),
        "support": total_support,
    }
    report["weighted avg"] = {
        "precision": safe_div(weighted_precision, total_support),
        "recall": safe_div(weighted_recall, total_support),
        "f1-score": safe_div(weighted_f1, total_support),
        "support": total_support,
    }
    report["accuracy"] = safe_div(sum(1 for t, p in zip(y_true, y_pred) if t == p), len(y_true))
    return report


def compute_rq2_metrics(
    eval_samples: Sequence[Dict[str, Any]],
    rq2_setting: str,
    pipeline_eval: str,
) -> Dict[str, Any]:
    y_true = [sample["gold_relation"] for sample in eval_samples]
    y_pred = [sample["rq2_pred_relation"] for sample in eval_samples]
    relation_types = unique_keep_order(y_true)
    evaluated_total = len(eval_samples)
    correct_type_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    if evaluated_total == 0 or not relation_types:
        macro_f1 = 0.0
        type_accuracy = 0.0
        report: Dict[str, Any] = {}
    elif f1_score is not None and accuracy_score is not None and classification_report is not None:
        macro_f1 = float(
            f1_score(
                y_true,
                y_pred,
                labels=relation_types,
                average="macro",
                zero_division=0,
            )
        )
        type_accuracy = float(accuracy_score(y_true, y_pred))
        report = classification_report(
            y_true,
            y_pred,
            labels=relation_types,
            output_dict=True,
            zero_division=0,
        )
    else:
        report = manual_classification_report(y_true, y_pred, relation_types)
        macro_f1 = float(report["macro avg"]["f1-score"])
        type_accuracy = safe_div(correct_type_predictions, evaluated_total)

    return {
        "setting": rq2_setting,
        "pipeline_eval": pipeline_eval if rq2_setting == "pipeline" else None,
        "macro_f1": macro_f1,
        "type_accuracy": type_accuracy,
        "evaluated_total": evaluated_total,
        "relation_types": relation_types,
        "classification_report": report,
        "correct_type_predictions": correct_type_predictions,
    }


def make_mismatch_sample(sample: Dict[str, Any], mismatch_type: str) -> Dict[str, Any]:
    row = sample["row"]
    pred_choice: Optional[str]
    if sample["pred_has_relation"]:
        pred_choice = sample["pred_relation"]
    else:
        pred_choice = None

    return {
        "序号": sample["seq"],
        "错误类型": mismatch_type,
        "CSV原始数据": {
            "NameA": strip_value(row.get("NameA")),
            "pathA": strip_value(row.get("pathA")),
            "contextA": strip_value(row.get("contextA")),
            "propertyA": strip_value(row.get("propertyA")),
            "typeA": strip_value(row.get("typeA")),
            "NameB": strip_value(row.get("NameB")),
            "pathB": strip_value(row.get("pathB")),
            "contextB": strip_value(row.get("contextB")),
            "propertyB": strip_value(row.get("propertyB")),
            "typeB": strip_value(row.get("typeB")),
            "Relation_gold": sample["gold_relation_raw"],
        },
        "JSON预测结果": {
            "是否存在追溯关系": sample["pred_has_relation"],
            "选择的关系": pred_choice,
            "推理过程": "",
            "决定关系的原因": "",
            "置信度": strip_value(row.get("max_similarity")),
        },
    }


def collect_mismatches(
    samples: Sequence[Dict[str, Any]],
    include_rq2_no_prediction: bool,
) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []

    for sample in samples:
        pred_has_relation = sample["pred_has_relation"]
        gold_has_relation = sample["gold_has_relation"]
        pred_relation = sample["pred_relation"]
        gold_relation = sample["gold_relation"]

        if pred_has_relation and not gold_has_relation:
            mismatches.append(make_mismatch_sample(sample, "FP"))
        elif not pred_has_relation and gold_has_relation:
            mismatches.append(make_mismatch_sample(sample, "FN"))
        elif pred_has_relation and gold_has_relation and pred_relation != gold_relation:
            mismatches.append(make_mismatch_sample(sample, "Type_Mismatch"))

        if include_rq2_no_prediction and gold_has_relation and not pred_has_relation:
            mismatches.append(make_mismatch_sample(sample, "RQ2_No_Prediction"))

    return mismatches


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def build_metrics_output(
    args: argparse.Namespace,
    row_count: int,
    rq1_metrics: Optional[Dict[str, Any]],
    rq2_metrics_by_setting: Optional[Dict[str, Dict[str, Any]]],
    coverage_by_setting: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    paper_ready: Dict[str, Any] = {}
    if rq1_metrics is not None:
        paper_ready["rq1"] = {
            "precision": format_metric(rq1_metrics["precision"]),
            "recall": format_metric(rq1_metrics["recall"]),
            "f1": format_metric(rq1_metrics["f1"]),
            "accuracy": format_metric(rq1_metrics["accuracy"]),
        }
    if rq2_metrics_by_setting is not None:
        paper_ready["rq2"] = {
            name: {
                "macro_f1": format_metric(metrics["macro_f1"]),
                "type_accuracy": format_metric(metrics["type_accuracy"]),
            }
            for name, metrics in rq2_metrics_by_setting.items()
        }

    return {
        "meta": {
            "match_csv": str(args.match_csv),
            "mode": args.mode,
            "rq2_setting": args.rq2_setting,
            "pipeline_eval": args.pipeline_eval if args.rq2_setting == "pipeline" else None,
            "csv_rows": row_count,
        },
        "rq1": rq1_metrics,
        "rq2": rq2_metrics_by_setting,
        "coverage": coverage_by_setting,
        "paper_ready": paper_ready,
    }


def print_file_info(rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    print(f"CSV rows: {len(rows)}")
    print(f"CSV columns: {len(fieldnames)}")
    print(f"Required fields OK: 是")
    print("\n=== 前3条对齐验证 ===")
    for i, row in enumerate(rows[:3]):
        print(
            f"序号 {i + 2}: "
            f"pred_has_relation={row.get('pred_has_relation')}, "
            f"pred_relation_type={row.get('pred_relation_type')} | "
            f"NameA={strip_value(row.get('NameA'))}, "
            f"NameB={strip_value(row.get('NameB'))}, "
            f"Relation={row.get('Relation')}"
        )


def print_rq1_metrics(rq1_metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print("RQ1: Link Existence Prediction（二分类）")
    print("=" * 50)
    print(f"TP (预测有, 实际有): {rq1_metrics['tp']}")
    print(f"FP (预测有, 实际无): {rq1_metrics['fp']}")
    print(f"TN (预测无, 实际无): {rq1_metrics['tn']}")
    print(f"FN (预测无, 实际有): {rq1_metrics['fn']}")
    print()
    print(
        f"Accuracy:  {rq1_metrics['accuracy']:.4f} "
        f"({rq1_metrics['tp'] + rq1_metrics['tn']}/{rq1_metrics['total']})"
    )
    print(f"Precision: {rq1_metrics['precision']:.4f}")
    print(f"Recall:    {rq1_metrics['recall']:.4f}")
    print(f"F1:        {rq1_metrics['f1']:.4f}")


def print_classification_report(report: Dict[str, Any], relation_types: Sequence[str]) -> None:
    print("\nclassification report:")
    if not report:
        print("N/A（evaluated_total 为 0）")
        return

    print(f"{'label':<16}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>10}")
    for label in relation_types:
        stats = report.get(label, {})
        print(
            f"{label:<16}"
            f"{float(stats.get('precision', 0.0)):>12.4f}"
            f"{float(stats.get('recall', 0.0)):>12.4f}"
            f"{float(stats.get('f1-score', 0.0)):>12.4f}"
            f"{int(stats.get('support', 0)):>10}"
        )

    for avg_name in ("micro avg", "macro avg", "weighted avg"):
        if avg_name not in report:
            continue
        stats = report[avg_name]
        print(
            f"{avg_name:<16}"
            f"{float(stats.get('precision', 0.0)):>12.4f}"
            f"{float(stats.get('recall', 0.0)):>12.4f}"
            f"{float(stats.get('f1-score', 0.0)):>12.4f}"
            f"{int(stats.get('support', 0)):>10}"
        )


def print_rq2_metrics(rq2_metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print("RQ2: Relation Type Prediction")
    print("=" * 50)
    print(f"rq2_setting: {rq2_metrics['setting']}")
    if rq2_metrics["setting"] == "pipeline":
        print(f"pipeline_eval: {rq2_metrics['pipeline_eval']}")
    print(f"Macro-F1:       {rq2_metrics['macro_f1']:.4f}")
    print(f"Type Accuracy:  {rq2_metrics['type_accuracy']:.4f}")
    print(f"evaluated_total: {rq2_metrics['evaluated_total']}")
    print(f"relation_types: {rq2_metrics['relation_types']}")
    print_classification_report(rq2_metrics["classification_report"], rq2_metrics["relation_types"])


def print_coverage(coverage: Dict[str, int]) -> None:
    print("\n=== Pipeline coverage ===")
    print(f"gold_positive_total: {coverage['gold_positive_total']}")
    print(f"predicted_positive_total: {coverage['predicted_positive_total']}")
    print(f"evaluated_positive_total: {coverage['evaluated_positive_total']}")
    print(f"missed_gold_positive_total: {coverage['missed_gold_positive_total']}")
    print(f"false_positive_link_total: {coverage['false_positive_link_total']}")


def print_mismatch_stats(mismatches: Sequence[Dict[str, Any]], max_print_errors: int) -> None:
    counts = Counter(sample["错误类型"] for sample in mismatches)

    print("\n" + "=" * 50)
    print("错误样本统计")
    print("=" * 50)
    print(f"FP 数量: {counts.get('FP', 0)}")
    print(f"FN 数量: {counts.get('FN', 0)}")
    print(f"Type_Mismatch 数量: {counts.get('Type_Mismatch', 0)}")
    print(f"RQ2_No_Prediction 数量: {counts.get('RQ2_No_Prediction', 0)}")

    if max_print_errors <= 0 or not mismatches:
        return

    print(f"\n=== 前 {min(max_print_errors, len(mismatches))} 条错误样本 ===")
    for sample in mismatches[:max_print_errors]:
        csv_data = sample["CSV原始数据"]
        json_pred = sample["JSON预测结果"]
        print(
            f"序号{sample['序号']} | {sample['错误类型']} | "
            f"JSON预测={json_pred['选择的关系']} | "
            f"CSV实际={csv_data['Relation_gold']} | "
            f"NameA={csv_data['NameA']} | NameB={csv_data['NameB']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze embedding match_results.csv for RQ1/RQ2 metrics."
    )
    parser.add_argument("--match_csv", type=Path, default=DEFAULT_MATCH_CSV, help="match_results.csv 文件路径")
    parser.add_argument(
        "--mode",
        choices=["rq1", "rq2", "both"],
        default="both",
        help="计算 RQ1、RQ2 或二者，默认 both",
    )
    parser.add_argument(
        "--rq2_setting",
        choices=["oracle", "pipeline"],
        default="oracle",
        help="RQ2 评测设置，默认 oracle",
    )
    parser.add_argument(
        "--pipeline_eval",
        choices=["conditional", "end2end"],
        default="conditional",
        help="pipeline RQ2 评测方式，默认 conditional",
    )
    parser.add_argument("--output_metrics", type=Path, default=DEFAULT_OUTPUT_METRICS, help="保存指标结果 JSON")
    parser.add_argument("--output_mismatch", type=Path, default=DEFAULT_OUTPUT_MISMATCH, help="保存错误样本 JSON")
    parser.add_argument(
        "--max_print_errors",
        type=int,
        default=15,
        help="终端最多打印多少条错误样本，默认 15",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, fieldnames = load_csv(args.match_csv)
    validate_csv_fields(fieldnames)
    samples = build_samples(rows)

    print_file_info(rows, fieldnames)

    do_rq1 = args.mode in ("rq1", "both")
    do_rq2 = args.mode in ("rq2", "both")

    rq1_metrics = compute_rq1_metrics(samples)
    rq2_eval_samples, coverage = build_rq2_eval_samples(
        samples,
        rq2_setting=args.rq2_setting,
        pipeline_eval=args.pipeline_eval,
    )
    rq2_metrics = compute_rq2_metrics(
        rq2_eval_samples,
        rq2_setting=args.rq2_setting,
        pipeline_eval=args.pipeline_eval,
    )

    if SKLEARN_IMPORT_ERROR is not None and do_rq2:
        print("\nWarning: 当前环境未安装 sklearn，RQ2 使用脚本内置实现计算；安装 sklearn 后会使用 sklearn.metrics。")

    if do_rq1:
        print_rq1_metrics(rq1_metrics)

    if do_rq2:
        print_rq2_metrics(rq2_metrics)
        if args.rq2_setting == "pipeline":
            print_coverage(coverage)

    oracle_eval_samples, oracle_coverage = build_rq2_eval_samples(
        samples,
        rq2_setting="oracle",
        pipeline_eval="conditional",
    )
    pipeline_conditional_eval_samples, pipeline_conditional_coverage = build_rq2_eval_samples(
        samples,
        rq2_setting="pipeline",
        pipeline_eval="conditional",
    )
    pipeline_end2end_eval_samples, pipeline_end2end_coverage = build_rq2_eval_samples(
        samples,
        rq2_setting="pipeline",
        pipeline_eval="end2end",
    )
    rq2_metrics_by_setting = {
        "oracle": compute_rq2_metrics(
            oracle_eval_samples,
            rq2_setting="oracle",
            pipeline_eval="conditional",
        ),
        "pipeline_end2end": compute_rq2_metrics(
            pipeline_end2end_eval_samples,
            rq2_setting="pipeline",
            pipeline_eval="end2end",
        ),
        "pipeline_conditional": compute_rq2_metrics(
            pipeline_conditional_eval_samples,
            rq2_setting="pipeline",
            pipeline_eval="conditional",
        ),
    }
    coverage_by_setting = {
        "oracle": oracle_coverage,
        "pipeline_end2end": pipeline_end2end_coverage,
        "pipeline_conditional": pipeline_conditional_coverage,
    }

    include_rq2_no_prediction = (
        do_rq2 and args.rq2_setting == "pipeline" and args.pipeline_eval == "end2end"
    )
    mismatches = collect_mismatches(
        samples,
        include_rq2_no_prediction=include_rq2_no_prediction,
    )
    print_mismatch_stats(mismatches, args.max_print_errors)

    metrics_output = build_metrics_output(
        args=args,
        row_count=len(rows),
        rq1_metrics=rq1_metrics if do_rq1 else None,
        rq2_metrics_by_setting=rq2_metrics_by_setting if do_rq2 else None,
        coverage_by_setting=coverage_by_setting if do_rq2 else {},
    )
    save_json(args.output_metrics, metrics_output)
    save_json(args.output_mismatch, mismatches)
    print(f"\n指标结果已保存: {args.output_metrics}")
    print(f"错误样本已保存: {args.output_mismatch}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"错误: {exc}") from exc
