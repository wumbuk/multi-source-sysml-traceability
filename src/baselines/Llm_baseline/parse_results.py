# LLM 基线预测结果解析与评估工具
# 功能：
# 1) 读取 LLM 预测 JSON + 真实标签 CSV
# 2) 支持全量输出和 increment 输出；优先按 JSON 中的“序号”和 CSV 行号对齐
# 3) 计算 RQ1 二分类指标（TP/FP/TN/FN、Accuracy、Precision、Recall、F1）
# 4) 计算 RQ2 关系类型指标（Oracle / Pipeline conditional / Pipeline end-to-end）
# 5) 输出不相符样本到 JSON 文件

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


# ============ 默认路径配置 ============
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "results" / "Llm_baseline"
JSON_PATH = OUTPUT_DIR / "llm_predictions.json"
CSV_PATH = ROOT / "Data" / "Air_Condition" / "Judge.csv"

NO_RELATION = "__NO_RELATION__"
NO_PREDICTION = "__NO_PREDICTION__"

JSON_REQUIRED_FIELDS = [
    "是否存在追溯关系",
    "选择的关系",
]

CSV_REQUIRED_FIELDS = [
    "Relation",
    "NameA",
    "NameB",
    "pathA",
    "pathB",
    "contextA",
    "contextB",
    "propertyA",
    "propertyB",
    "typeA",
    "typeB",
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


def load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON 顶层结构应为 list，当前为: {type(data).__name__}")
    return data


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


def normalize_relation(rel: Any, source: str) -> str:
    if rel is None:
        return NO_RELATION

    text = str(rel).strip()
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    if lowered in NO_RELATION_TOKENS:
        return NO_RELATION

    if source == "csv":
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


def normalize_json_rel(rel: Any) -> str:
    return normalize_relation(rel, source="json")


def normalize_csv_rel(rel: Any) -> str:
    return normalize_relation(rel, source="csv")


def is_no_relation(rel: str) -> bool:
    return rel == NO_RELATION


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def strip_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def validate_json_fields(json_data: Sequence[Dict[str, Any]]) -> None:
    for i, record in enumerate(json_data):
        if not isinstance(record, dict):
            raise ValueError(f"JSON 第 {i + 1} 条不是 object，当前类型: {type(record).__name__}")

        missing_fields = [field for field in JSON_REQUIRED_FIELDS if field not in record]
        if missing_fields:
            raise KeyError(
                f"JSON 第 {i + 1} 条缺少字段: {missing_fields}; "
                f"当前可用字段: {list(record.keys())}"
            )


def validate_csv_fields(fieldnames: Sequence[str]) -> None:
    missing_fields = [field for field in CSV_REQUIRED_FIELDS if field not in fieldnames]
    if missing_fields:
        raise KeyError(f"CSV 缺少字段: {missing_fields}; 当前可用字段: {list(fieldnames)}")


def parse_seq(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def make_default_prediction(seq: int) -> Dict[str, Any]:
    return {
        "序号": seq,
        "是否存在追溯关系": False,
        "选择的关系": "null",
        "推理过程": "",
        "决定关系的原因": "",
        "置信度": 0.0,
    }


def build_prediction_map(json_data: Sequence[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], int]:
    prediction_map: Dict[int, Dict[str, Any]] = {}
    duplicate_count = 0

    for index, record in enumerate(json_data):
        seq = parse_seq(record.get("序号"))
        if seq is None:
            # 没有“序号”时，兼容旧逻辑：JSON 第 1 条对应 CSV 数据第 1 行，即序号 2。
            seq = index + 2

        if seq in prediction_map:
            duplicate_count += 1
        prediction_map[seq] = record

    return prediction_map, duplicate_count


def build_samples(
    json_data: Sequence[Dict[str, Any]],
    csv_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    prediction_map, duplicate_count = build_prediction_map(json_data)
    samples: List[Dict[str, Any]] = []
    missing_prediction_total = 0

    for i, c in enumerate(csv_rows):
        seq = i + 2
        has_prediction_record = seq in prediction_map
        j = prediction_map.get(seq, make_default_prediction(seq))
        if not has_prediction_record:
            missing_prediction_total += 1

        json_exists_raw = j["是否存在追溯关系"]
        pred_relation_raw = j["选择的关系"]
        gold_relation_raw = c["Relation"]

        normalized_pred_relation = normalize_json_rel(pred_relation_raw)
        normalized_gold_relation = normalize_csv_rel(gold_relation_raw)
        json_exists_bool = parse_bool(json_exists_raw)

        pred_has_relation = json_exists_bool and not is_no_relation(normalized_pred_relation)
        gold_has_relation = not is_no_relation(normalized_gold_relation)

        samples.append(
            {
                "seq": seq,
                "json": j,
                "csv": c,
                "has_prediction_record": has_prediction_record,
                "json_exists_raw": json_exists_raw,
                "json_exists_bool": json_exists_bool,
                "pred_relation_raw": pred_relation_raw,
                "gold_relation_raw": gold_relation_raw,
                "pred_relation": normalized_pred_relation,
                "gold_relation": normalized_gold_relation,
                "pred_has_relation": pred_has_relation,
                "gold_has_relation": gold_has_relation,
            }
        )

    alignment_info = {
        "prediction_records": len(json_data),
        "csv_rows": len(csv_rows),
        "missing_prediction_total": missing_prediction_total,
        "duplicate_seq_total": duplicate_count,
    }
    return samples, alignment_info


def attach_oracle_predictions(
    samples: Sequence[Dict[str, Any]],
    oracle_json_data: Sequence[Dict[str, Any]],
) -> None:
    oracle_prediction_map, _ = build_prediction_map(oracle_json_data)
    for sample in samples:
        oracle_record = oracle_prediction_map.get(sample["seq"])
        oracle_relation = (
            normalize_json_rel(oracle_record.get("选择的关系"))
            if oracle_record is not None
            else NO_RELATION
        )
        sample["oracle_relation"] = oracle_relation


def has_complete_oracle_predictions(samples: Sequence[Dict[str, Any]]) -> bool:
    return all(
        not is_no_relation(sample.get("oracle_relation", NO_RELATION))
        for sample in samples
        if sample["gold_has_relation"]
    )


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
    setting: str,
    pipeline_eval: Optional[str] = None,
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

        if setting == "oracle":
            if not gold_has_relation:
                continue
            rq2_pred_relation = sample.get("oracle_relation", NO_RELATION)
            if is_no_relation(rq2_pred_relation):
                raise ValueError(
                    "Oracle evaluation requires a relation-type prediction "
                    "for every gold-positive sample."
                )
        elif setting == "pipeline" and pipeline_eval == "conditional":
            if not (pred_has_relation and gold_has_relation):
                continue
            rq2_pred_relation = sample["pred_relation"]
        elif setting == "pipeline" and pipeline_eval == "end2end":
            if not gold_has_relation:
                continue
            rq2_pred_relation = sample["pred_relation"] if pred_has_relation else NO_PREDICTION
        else:
            raise ValueError(f"不支持的 RQ2 设置: setting={setting}, pipeline_eval={pipeline_eval}")

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
    setting: str,
    pipeline_eval: Optional[str],
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
        "setting": setting,
        "pipeline_eval": pipeline_eval if setting == "pipeline" else None,
        "macro_f1": macro_f1,
        "type_accuracy": type_accuracy,
        "evaluated_total": evaluated_total,
        "relation_types": relation_types,
        "classification_report": report,
        "correct_type_predictions": correct_type_predictions,
    }


def compute_all_rq2(samples: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    rq2_results: Dict[str, Any] = {}
    coverage_results: Dict[str, Dict[str, int]] = {}

    if has_complete_oracle_predictions(samples):
        oracle_samples, oracle_coverage = build_rq2_eval_samples(samples, "oracle")
        rq2_results["oracle"] = compute_rq2_metrics(oracle_samples, "oracle", None)
        coverage_results["oracle"] = oracle_coverage

    conditional_samples, conditional_coverage = build_rq2_eval_samples(
        samples,
        "pipeline",
        "conditional",
    )
    rq2_results["pipeline_conditional"] = compute_rq2_metrics(
        conditional_samples,
        "pipeline",
        "conditional",
    )
    coverage_results["pipeline_conditional"] = conditional_coverage

    end2end_samples, end2end_coverage = build_rq2_eval_samples(
        samples,
        "pipeline",
        "end2end",
    )
    rq2_results["pipeline_end2end"] = compute_rq2_metrics(
        end2end_samples,
        "pipeline",
        "end2end",
    )
    coverage_results["pipeline_end2end"] = end2end_coverage

    return rq2_results, coverage_results


def make_mismatch_sample(sample: Dict[str, Any], mismatch_type: str) -> Dict[str, Any]:
    c = sample["csv"]
    j = sample["json"]

    return {
        "序号": sample["seq"],
        "错误类型": mismatch_type,
        "CSV原始数据": {
            "NameA": strip_value(c["NameA"]),
            "pathA": strip_value(c["pathA"]),
            "contextA": strip_value(c.get("contextA", "")),
            "propertyA": strip_value(c["propertyA"]),
            "typeA": strip_value(c["typeA"]),
            "NameB": strip_value(c["NameB"]),
            "pathB": strip_value(c["pathB"]),
            "contextB": strip_value(c["contextB"]),
            "propertyB": strip_value(c["propertyB"]),
            "typeB": strip_value(c["typeB"]),
            "Relation_gold": sample["gold_relation_raw"],
        },
        "JSON预测结果": {
            "是否存在追溯关系": j["是否存在追溯关系"],
            "选择的关系": j["选择的关系"],
            "推理过程": j.get("推理过程", ""),
            "决定关系的原因": j.get("决定关系的原因", ""),
            "置信度": j.get("置信度", 0.0),
        },
    }


def collect_mismatches(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mismatched_output: List[Dict[str, Any]] = []

    for sample in samples:
        pred_has_relation = sample["pred_has_relation"]
        gold_has_relation = sample["gold_has_relation"]
        pred_relation = sample["pred_relation"]
        gold_relation = sample["gold_relation"]

        mismatch_type = None
        if pred_has_relation and not gold_has_relation:
            mismatch_type = "FP"
        elif not pred_has_relation and gold_has_relation:
            mismatch_type = "FN"
        elif pred_has_relation and gold_has_relation and pred_relation != gold_relation:
            mismatch_type = "Type_Mismatch"

        if mismatch_type:
            mismatched_output.append(make_mismatch_sample(sample, mismatch_type))

    return mismatched_output


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def build_metrics_output(
    args: argparse.Namespace,
    alignment_info: Dict[str, int],
    rq1_metrics: Dict[str, Any],
    rq2_results: Dict[str, Any],
    coverage_results: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    return {
        "meta": {
            "gold_csv": str(args.gold_csv),
            "pred_json": str(args.pred_json),
            "alignment": alignment_info,
        },
        "rq1": rq1_metrics,
        "rq2": rq2_results,
        "coverage": coverage_results,
        "paper_ready": {
            "rq1": {
                "precision": format_metric(rq1_metrics["precision"]),
                "recall": format_metric(rq1_metrics["recall"]),
                "f1": format_metric(rq1_metrics["f1"]),
                "accuracy": format_metric(rq1_metrics["accuracy"]),
            },
            "rq2": {
                name: {
                    "macro_f1": format_metric(metrics["macro_f1"]),
                    "type_accuracy": format_metric(metrics["type_accuracy"]),
                }
                for name, metrics in rq2_results.items()
            },
        },
    }


def print_file_info(
    json_data: Sequence[Dict[str, Any]],
    csv_rows: Sequence[Dict[str, Any]],
    alignment_info: Dict[str, int],
) -> None:
    print(f"JSON records: {len(json_data)}")
    print(f"CSV rows (excl header): {len(csv_rows)}")
    print(f"是否长度一致: {'是' if len(json_data) == len(csv_rows) else '否'}")
    print("对齐方式: 按 JSON 字段“序号”和 CSV 行号对齐")
    print(f"缺失预测记录数（按预测无关系处理）: {alignment_info['missing_prediction_total']}")
    if alignment_info["duplicate_seq_total"] > 0:
        print(f"Warning: JSON 中存在重复序号，已使用最后一次出现的记录。重复数: {alignment_info['duplicate_seq_total']}")
    if len(json_data) != len(csv_rows):
        print("Warning: JSON 和 CSV 行数不同；这通常表示增量输出，脚本已按缺失预测=无关系处理。")


def print_alignment_preview(samples: Sequence[Dict[str, Any]]) -> None:
    print("\n=== 前3条对齐验证 ===")
    for sample in samples[:3]:
        c = sample["csv"]
        print(
            f'序号 {sample["seq"]}: '
            f'JSON 是否存在={sample["json_exists_raw"]}, '
            f'选择的关系={sample["pred_relation_raw"]}, '
            f'是否有预测记录={sample["has_prediction_record"]} | '
            f'CSV NameA={strip_value(c["NameA"])}, '
            f'NameB={strip_value(c["NameB"])}, '
            f'Relation={sample["gold_relation_raw"]}'
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


def print_rq2_metrics(name: str, rq2_metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print(f"RQ2: Relation Type Prediction - {name}")
    print("=" * 50)
    print(f"rq2_setting: {rq2_metrics['setting']}")
    if rq2_metrics["setting"] == "pipeline":
        print(f"pipeline_eval: {rq2_metrics['pipeline_eval']}")
    print(f"Macro-F1:       {rq2_metrics['macro_f1']:.4f}")
    print(f"Type Accuracy:  {rq2_metrics['type_accuracy']:.4f}")
    print(f"evaluated_total: {rq2_metrics['evaluated_total']}")
    print(f"relation_types: {rq2_metrics['relation_types']}")
    print_classification_report(rq2_metrics["classification_report"], rq2_metrics["relation_types"])


def print_coverage(name: str, coverage: Dict[str, int]) -> None:
    print(f"\n=== Coverage - {name} ===")
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
    parser = argparse.ArgumentParser(description="Analyze baseline LLM traceability prediction results.")
    parser.add_argument("--gold_csv", type=Path, default=CSV_PATH, help="gold CSV 文件路径")
    parser.add_argument("--pred_json", type=Path, default=JSON_PATH, help="模型预测 JSON 文件路径")
    parser.add_argument(
        "--oracle_pred_json",
        type=Path,
        help=(
            "Optional Oracle JSON generated with gold links supplied. "
            "Every gold-positive sample must contain a type in '选择的关系'."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--output_metrics", type=Path, help="可选，保存指标结果 JSON")
    parser.add_argument("--output_mismatch", type=Path, help="可选，保存错误样本 JSON")
    parser.add_argument(
        "--max_print_errors",
        type=int,
        default=15,
        help="终端最多打印多少条错误样本，默认 15",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    json_data = load_json(args.pred_json)
    csv_rows, csv_fieldnames = load_csv(args.gold_csv)
    validate_json_fields(json_data)
    validate_csv_fields(csv_fieldnames)

    samples, alignment_info = build_samples(json_data, csv_rows)
    if args.oracle_pred_json is not None:
        oracle_json_data = load_json(args.oracle_pred_json)
        validate_json_fields(oracle_json_data)
        attach_oracle_predictions(samples, oracle_json_data)
    print_file_info(json_data, csv_rows, alignment_info)
    print_alignment_preview(samples)

    rq1_metrics = compute_rq1_metrics(samples)
    rq2_results, coverage_results = compute_all_rq2(samples)
    if "oracle" not in rq2_results:
        print(
            "\nWarning: skipped Oracle metrics. Supply --oracle_pred_json "
            "with a forced relation-type prediction for every gold-positive sample."
        )

    if SKLEARN_IMPORT_ERROR is not None:
        print("\nWarning: 当前环境未安装 sklearn，RQ2 使用脚本内置实现计算；安装 sklearn 后会使用 sklearn.metrics。")

    print_rq1_metrics(rq1_metrics)
    for name, metrics in rq2_results.items():
        print_rq2_metrics(name, metrics)
        if name.startswith("pipeline"):
            print_coverage(name, coverage_results[name])

    mismatches = collect_mismatches(samples)
    print_mismatch_stats(mismatches, args.max_print_errors)

    output_mismatch = args.output_mismatch or (args.output_dir / "llm_predictions_mismatch.json")
    save_json(output_mismatch, mismatches)
    print(f"\n错误样本已保存: {output_mismatch}")

    if args.output_metrics:
        metrics_output = build_metrics_output(
            args=args,
            alignment_info=alignment_info,
            rq1_metrics=rq1_metrics,
            rq2_results=rq2_results,
            coverage_results=coverage_results,
        )
        save_json(args.output_metrics, metrics_output)
        print(f"指标结果已保存: {args.output_metrics}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"错误: {exc}") from exc
