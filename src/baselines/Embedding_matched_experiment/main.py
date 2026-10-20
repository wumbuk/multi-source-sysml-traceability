import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_settings import SETTINGS
LOCAL_MODEL_PATH = SETTINGS['embedding']['model_path']


DEFAULT_FIELDS = [
	"NameA",
	"typeA",
	"propertyA",
	"NameB",
	"contextB",
	"typeB",
]

#运行参数配置
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Embedding matching experiment for element-requirement node pairs."
	)
	parser.add_argument(
		"--kg-samples",
		type=str,
		required=True,
		help="Path to KG training samples (csv/json/jsonl).",
	)
	parser.add_argument(
		"--query-pairs",
		type=str,
		required=True,
		help="Path to query node pairs to judge (csv/json/jsonl).",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="match_results.csv",
		help="Output file path for prediction results.",
	)
	parser.add_argument(
		"--model-name",
		type=str,
		default=LOCAL_MODEL_PATH,
		help="Local SentenceTransformer model path (offline).",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=SETTINGS['embedding']['threshold'],
		help="Similarity threshold for relation judgment.",
	)
	parser.add_argument(
		"--label-col",
		type=str,
		default="Relation",
		help="Label column in KG samples. Positive samples are used as reference vectors.",
	)
	parser.add_argument(
		"--id-col",
		type=str,
		default="pair_id",
		help="Optional id column to track rows.",
	)
	parser.add_argument(
		"--fields",
		type=str,
		nargs="+",
		default=DEFAULT_FIELDS,
		help="Input fields to build pair sentence.",
	)
	parser.add_argument(
		"--metrics-plot",
		type=str,
		default="",
		help="Output image path for metrics plot. If empty, save as <output_stem>_metrics.png.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducible experiment setup.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda",
		choices=["cuda", "cpu", "auto"],
		help="Device used by SentenceTransformer. Use cuda for reproducible GPU runs.",
	)
	return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
	os.environ["PYTHONHASHSEED"] = str(seed)
	os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	if torch.cuda.is_available():
		torch.backends.cuda.matmul.allow_tf32 = False
		torch.backends.cudnn.allow_tf32 = False
	if hasattr(torch, "use_deterministic_algorithms"):
		try:
			torch.use_deterministic_algorithms(True, warn_only=True)
		except TypeError:
			torch.use_deterministic_algorithms(True)


#加载本地 SentenceTransformer 模型
def resolve_device(device: str) -> str:
	device_name = str(device).strip().lower()
	if device_name == "auto":
		return "cuda" if torch.cuda.is_available() else "cpu"
	if device_name == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("--device cuda was requested, but CUDA is not available.")
	if device_name not in {"cuda", "cpu"}:
		raise ValueError(f"Unsupported device: {device}. Use cuda, cpu, or auto.")
	return device_name


# Load local SentenceTransformer model
def load_embedding_model(model_path: str, device: str) -> SentenceTransformer:
	resolved_path = Path(model_path)
	if not resolved_path.exists():
		raise FileNotFoundError(
			f"Local model path not found: {model_path}. Please set --model-name to a valid local path."
		)
	resolved_device = resolve_device(device)
	print(f"Using embedding device: {resolved_device}")
	# Force local/offline loading to avoid remote downloads from Hugging Face.
	model = SentenceTransformer(str(resolved_path), device=resolved_device, local_files_only=True)
	model.max_seq_length = SETTINGS['embedding']['max_length']
	model.eval()
	return model


#????????? csv/json/jsonl ??
def load_table(path: str) -> pd.DataFrame:
	file_path = Path(path)
	if not file_path.exists():
		raise FileNotFoundError(f"Input file does not exist: {file_path}")

	suffix = file_path.suffix.lower()
	if suffix == ".csv":
		return pd.read_csv(file_path)
	if suffix == ".json":
		data = json.loads(file_path.read_text(encoding="utf-8"))
		if isinstance(data, dict):
			data = data.get("data", data)
		return pd.DataFrame(data)
	if suffix == ".jsonl":
		records = []
		with file_path.open("r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if line:
					records.append(json.loads(line))
		return pd.DataFrame(records)
	raise ValueError(f"Unsupported file type: {suffix}. Use csv/json/jsonl.")

#文本规范化函数，处理 None、NaN 和多余空格
def normalize_text(value: object) -> str:
	if value is None:
		return ""
	if isinstance(value, float) and pd.isna(value):
		return ""
	text = str(value).strip()
	return " ".join(text.split())

#构建句对输入文本，格式示例：
# "关键信息 元素名称: xxx; 元素类型: xxx; 元素属性: xxx [SEP] 需求名称: xxx; 需求描述: xxx; 需求类型: xxx"
def build_pair_sentence(row: pd.Series) -> str:
	# 按你当前 CSV 的列定义构造句对: A 侧信息 [SEP] B 侧需求信息
	element_name = normalize_text(row.get("NameA"))
	element_type = normalize_text(row.get("typeA"))
	element_attr = normalize_text(row.get("propertyA"))
	req_name = normalize_text(row.get("NameB"))
	req_desc = normalize_text(row.get("contextB"))
	req_type = normalize_text(row.get("typeB"))

	left = (
		f"关键信息 元素名称: {element_name}; 元素类型: {element_type}; 元素属性: {element_attr}"
	)
	right = f"需求名称: {req_name}; 需求描述: {req_desc}; 需求类型: {req_type}"
	return f"{left} [SEP] {right}"

#验证输入表格是否包含构建句对所需的字段，如果缺失则抛出异常提示用户检查输入数据格式
def validate_fields(df: pd.DataFrame, fields: Sequence[str], file_name: str) -> None:
	missing = [c for c in fields if c not in df.columns]
	if missing:
		raise ValueError(
			f"{file_name} missing required columns: {missing}. Available: {list(df.columns)}"
		)

#正样本判定函数，支持数值型和文本型标签的多种表达方式
def is_positive_label(label: object) -> bool:
	# 正样本判定规则:
	# 1) 数值型标签: 1 视为正样本
	# 2) 文本型标签: 以 "1" 开头（如 1/Refine）或在白名单中的值视为正样本
	if isinstance(label, (int, float)):
		return int(label) == 1
	label_text = normalize_text(label).lower()
	if label_text.startswith("1"):
		return True
	return label_text in {
		"1",
		"true",
		"yes",
		"y",
		"positive",
		"pos",
		"related",
		"match",
		"有关系",
		"是",
	}

#从原始标签中提取关系类型，负样本统一记为“无关系”
def extract_relation_type(label: object) -> str:
	if not is_positive_label(label):
		return "无关系"
	label_text = normalize_text(label)
	if "/" in label_text:
		_, relation_type = label_text.split("/", 1)
		relation_type = normalize_text(relation_type)
		if relation_type:
			return relation_type
	return "有关系"

#编码文本列表为向量，使用 SentenceTransformer 模型，返回归一化的张量形式向量
def encode_texts(
	model: SentenceTransformer,
	texts: Sequence[str],
	batch_size: int = 64,
):
	return model.encode(
		texts,
		batch_size=batch_size,
		convert_to_tensor=True,
		normalize_embeddings=True,
		show_progress_bar=True,
	)

#核心匹配函数，计算查询对与正样本参考向量的相似度，并根据阈值判断是否有关系
def run_matching(
	kg_df: pd.DataFrame,
	query_df: pd.DataFrame,
	model: SentenceTransformer,
	threshold: float,
	label_col: str,
	id_col: str,
) -> pd.DataFrame:
	if label_col not in kg_df.columns:
		raise ValueError(
			f"KG samples missing label column '{label_col}', cannot collect positive vectors."
		)

	kg_df = kg_df.copy()
	query_df = query_df.copy()

	kg_df["pair_sentence"] = kg_df.apply(build_pair_sentence, axis=1)
	query_df["pair_sentence"] = query_df.apply(build_pair_sentence, axis=1)

	# 只收集 KG 样本中的正样本作为参考向量库（你的实验核心设定）
	positive_mask = kg_df[label_col].apply(is_positive_label)
	positive_df = kg_df[positive_mask].reset_index(drop=True)
	if positive_df.empty:
		raise ValueError("No positive samples found in KG samples.")

	ref_texts = positive_df["pair_sentence"].tolist()
	query_texts = query_df["pair_sentence"].tolist()

	print(f"KG total samples: {len(kg_df)}")
	print(f"Collected positive reference samples: {len(positive_df)}")
	print(f"Query pairs to judge: {len(query_df)}")

	ref_embeddings = encode_texts(model, ref_texts)
	query_embeddings = encode_texts(model, query_texts)

	sim_matrix = util.cos_sim(query_embeddings, ref_embeddings)
	max_scores, best_indices = sim_matrix.max(dim=1)

	query_df["max_similarity"] = max_scores.cpu().numpy()
	query_df["best_ref_index"] = best_indices.cpu().numpy()
	# 当最大相似度 >= 阈值时，判定为“有关系”
	query_df["pred_has_relation"] = (query_df["max_similarity"] >= threshold).astype(int)
	query_df["best_ref_label"] = query_df["best_ref_index"].map(
		positive_df[label_col].to_dict()
	)
	query_df["best_ref_relation_type"] = query_df["best_ref_label"].apply(extract_relation_type)
	# 多分类预测：有关系时，关系类型跟随余弦相似度最高的参考样本；否则判为“无关系”
	query_df["pred_relation_type"] = query_df.apply(
		lambda row: row["best_ref_relation_type"] if row["pred_has_relation"] == 1 else "无关系",
		axis=1,
	)
	if label_col in query_df.columns:
		query_df["true_relation_type"] = query_df[label_col].apply(extract_relation_type)
		query_df["relation_type_correct"] = (
			query_df["pred_relation_type"] == query_df["true_relation_type"]
		).astype(int)

	if id_col in positive_df.columns:
		query_df["best_ref_pair_id"] = query_df["best_ref_index"].map(
			positive_df[id_col].to_dict()
		)
	else:
		query_df["best_ref_pair_id"] = query_df["best_ref_index"]

	return query_df


def evaluate_if_possible(df: pd.DataFrame, label_col: str) -> Optional[Dict[str, float]]:
	if label_col not in df.columns:
		return None
	y_true = df[label_col].apply(is_positive_label).astype(int)
	y_pred = df["pred_has_relation"].astype(int)

	tp = int(((y_true == 1) & (y_pred == 1)).sum())
	fp = int(((y_true == 0) & (y_pred == 1)).sum())
	fn = int(((y_true == 1) & (y_pred == 0)).sum())
	tn = int(((y_true == 0) & (y_pred == 0)).sum())

	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = (
		2 * precision * recall / (precision + recall)
		if (precision + recall) > 0
		else 0.0
	)
	acc = (tp + tn) / len(df) if len(df) > 0 else 0.0
	true_relation_type = df[label_col].apply(extract_relation_type)
	pred_relation_type = (
		df["pred_relation_type"]
		if "pred_relation_type" in df.columns
		else y_pred.map({1: "有关系", 0: "无关系"})
	)
	# 关系类型准确率仅在“有关系判定正确”的样本（TP）上统计
	# 即：真实有关系且预测也为有关系，再比较关系类型是否一致
	tp_mask = (y_true == 1) & (y_pred == 1)
	relation_type_total = int(tp_mask.sum())
	if int(tp_mask.sum()) > 0:
		relation_type_correct = int(
			(pred_relation_type[tp_mask] == true_relation_type[tp_mask]).sum()
		)
		relation_type_accuracy = relation_type_correct / relation_type_total
	else:
		relation_type_correct = 0
		relation_type_accuracy = 0.0

	return {
		"tp": tp,
		"fp": fp,
		"fn": fn,
		"tn": tn,
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"accuracy": acc,
		"relation_type_accuracy": relation_type_accuracy,
		"relation_type_total": relation_type_total,
		"relation_type_correct": relation_type_correct,
		"total": int(len(df)),
	}


def print_evaluation_summary(metrics: Dict[str, float]) -> None:
	tp = int(metrics["tp"])
	fp = int(metrics["fp"])
	tn = int(metrics["tn"])
	fn = int(metrics["fn"])
	total = int(metrics.get("total", tp + fp + tn + fn))
	correct = tp + tn
	relation_type_total = int(metrics.get("relation_type_total", tp))
	relation_type_correct = int(metrics.get("relation_type_correct", 0))

	print("=" * 50)
	print("二分类混淆矩阵（只看有无关系）")
	print(f"TP (预测有, 实际有): {tp}")
	print(f"FP (预测有, 实际无): {fp}")
	print(f"TN (预测无, 实际无): {tn}")
	print(f"FN (预测无, 实际有): {fn}")
	print(f"Accuracy: {metrics['accuracy']:.4f} ({correct}/{total})")
	print(f"Precision: {metrics['precision']:.4f}")
	print(f"Recall: {metrics['recall']:.4f}")
	print(f"F1: {metrics['f1']:.4f}")
	print("=" * 50)
	print("关系类型判断准确性")
	print(f"双方都有关系的样本数: {relation_type_total}")
	print(f"关系类型判断正确数: {relation_type_correct}")
	print(
		f"关系类型准确率: {metrics['relation_type_accuracy']:.4f} "
		f"({relation_type_correct}/{relation_type_total})"
	)


def plot_metrics(metrics: Dict[str, float], plot_path: Path) -> None:
	metric_names = ["precision", "recall", "f1", "accuracy"]
	if "relation_type_accuracy" in metrics:
		metric_names.append("relation_type_accuracy")
	metric_values = [float(metrics[name]) for name in metric_names]

	count_names = ["tp", "fp", "tn", "fn"]
	count_values = [float(metrics[name]) for name in count_names]

	fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

	axes[0].bar(
		metric_names,
		metric_values,
		color=["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"],
	)
	axes[0].set_ylim(0.0, 1.0)
	axes[0].set_title("Core Metrics")
	axes[0].set_ylabel("Score")
	for idx, value in enumerate(metric_values):
		axes[0].text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

	axes[1].bar(count_names, count_values, color=["#2CA02C", "#D62728", "#FF7F0E", "#1F77B4"])
	axes[1].set_title("Confusion Matrix Counts")
	axes[1].set_ylabel("Count")
	for idx, value in enumerate(count_values):
		axes[1].text(idx, value, f"{int(value)}", ha="center", va="bottom", fontsize=9)

	fig.tight_layout()
	plot_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(plot_path, dpi=150)
	plt.close(fig)


def main() -> None:
	args = parse_args()
	set_reproducible_seed(args.seed)
	print(f"Using random seed: {args.seed}")

	kg_df = load_table(args.kg_samples)
	query_df = load_table(args.query_pairs)

	validate_fields(kg_df, args.fields, "kg_samples")
	validate_fields(query_df, args.fields, "query_pairs")

	model = load_embedding_model(args.model_name, args.device)
	result_df = run_matching(
		kg_df=kg_df,
		query_df=query_df,
		model=model,
		threshold=args.threshold,
		label_col=args.label_col,
		id_col=args.id_col,
	)

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
	print(f"Saved prediction results to: {output_path}")

	metrics = evaluate_if_possible(result_df, args.label_col)
	if metrics is not None:
		print_evaluation_summary(metrics)
		plot_path = (
			Path(args.metrics_plot)
			if args.metrics_plot
			else output_path.with_name(f"{output_path.stem}_metrics.png")
		)
		plot_metrics(metrics, plot_path)
		print(f"Saved metrics plot to: {plot_path}")
	else:
		print("No label column in query pairs, skipped evaluation.")


if __name__ == "__main__":
    #运行命令
	#判断数据路径：D:\Baseline_Experiment\Data\Judge.csv
	#KG样本路径：D:\Baseline_Experiment\Data\Model_Requirement.csv
    # python main.py --kg-samples 你的KG样本.csv --query-pairs 你的待判断样本.csv --output D:\Baseline_Experiment\Embedding_matched_experiment\match_results.csv --threshold 0.60
	main()
