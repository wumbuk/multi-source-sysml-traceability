import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
	AutoModel,
	AutoTokenizer,
	DataCollatorWithPadding,
	Trainer,
	TrainingArguments,
	EarlyStoppingCallback,
	set_seed,
)


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_settings import ROOT, SETTINGS
DEFAULT_TRAIN_PATH = str(ROOT / 'Data' / SETTINGS['dataset'] / 'KG.csv')
DEFAULT_TEST_PATH = str(ROOT / 'Data' / SETTINGS['dataset'] / 'Judge.csv')
DEFAULT_OUTPUT_DIR = str(ROOT / 'results' / 'supervised_baseline' / SETTINGS['dataset'])
DEFAULT_MODEL_PATH = SETTINGS['supervised']['model_path']
LABEL_COL = "Relation"
INPUT_FIELDS = ["NameA", "typeA", "propertyA", "NameB", "contextB", "typeB"]
PAIR_KEY_FIELDS = ["NameA", "pathA", "typeA", "propertyA", "NameB", "contextB", "typeB"]


class PairDataset(Dataset):
	def __init__(self, encodings: Dict[str, List[int]], labels: Optional[List[int]] = None):
		self.encodings = encodings
		self.labels = labels

	def __len__(self) -> int:
		return len(self.encodings["input_ids"])

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
		if self.labels is not None:
			item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
		return item


class FullFineTuningSoftmaxClassifier(nn.Module):
	def __init__(
		self,
		backbone: AutoModel,
		num_labels: int,
		class_weights: Optional[torch.Tensor] = None,
	):
		super().__init__()
		# 全量微调：基模型参数参与训练更新。
		self.backbone = backbone
		self.num_labels = num_labels
		hidden_size = int(backbone.config.hidden_size)
		# 新增一层 softmax 多分类头。
		self.softmax_head = nn.Linear(hidden_size, num_labels)
		self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		labels: Optional[torch.Tensor] = None,
	):
		backbone_outputs = self.backbone(
			input_ids=input_ids,
			attention_mask=attention_mask,
			return_dict=True,
		)
		# 取 [CLS] 向量送入 softmax 分类头。
		pooled = backbone_outputs.last_hidden_state[:, 0]
		logits = self.softmax_head(pooled)
		outputs = {"logits": logits}
		if labels is not None:
			outputs["loss"] = self.loss_fn(logits, labels)
		return outputs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Supervised multi-class traceability baseline.")
	parser.add_argument("--train-path", type=str, default=DEFAULT_TRAIN_PATH)
	parser.add_argument("--test-path", type=str, default=DEFAULT_TEST_PATH)
	parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
	parser.add_argument("--max-length", type=int, default=256)
	parser.add_argument("--epochs", type=int, default=SETTINGS['supervised']['epochs'])
	parser.add_argument("--lr", type=float, default=2e-5)
	parser.add_argument(
		"--lr-scheduler-type",
		type=str,
		default="cosine",
		choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
		help="Learning rate scheduler type for dynamic LR.",
	)
	parser.add_argument(
		"--warmup-ratio",
		type=float,
		default=0.1,
		help="Warmup ratio in [0, 1] for dynamic LR schedule.",
	)
	parser.add_argument("--train-batch-size", type=int, default=16)
	parser.add_argument("--eval-batch-size", type=int, default=32)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--oracle-positive-only",
		action="store_true",
		help="Also predict Refine/Satisfy conditional on an existing relation, without using test labels.",
	)
	parser.add_argument("--val-ratio", type=float, default=0.1)
	parser.add_argument(
		"--class-weight-mode",
		type=str,
		default="sqrt_balanced",
		choices=["none", "balanced", "sqrt_balanced"],
		help="Class weighting used by CrossEntropyLoss.",
	)
	parser.add_argument(
		"--early-stopping-patience",
		type=int,
		default=0,
		help="Stop after this many evaluations without relation-F1 improvement; disabled by default.",
	)
	return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
	os.environ["PYTHONHASHSEED"] = str(seed)
	os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	set_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	if hasattr(torch, "use_deterministic_algorithms"):
		try:
			torch.use_deterministic_algorithms(True, warn_only=True)
		except TypeError:
			torch.use_deterministic_algorithms(True)


def load_csv(path: str) -> pd.DataFrame:
	file_path = Path(path)
	if not file_path.exists():
		raise FileNotFoundError(f"File not found: {file_path}")
	try:
		return pd.read_csv(file_path, encoding="utf-8-sig")
	except UnicodeDecodeError:
		return pd.read_csv(file_path, encoding="gbk")


def normalize_text(value: object) -> str:
	if value is None:
		return ""
	if isinstance(value, float) and pd.isna(value):
		return ""
	return " ".join(str(value).strip().split())


def build_input_text(row: pd.Series) -> str:
	req_text = (
		f"NameB: {normalize_text(row.get('NameB'))}; "
		f"contextB: {normalize_text(row.get('contextB'))}; "
		f"typeB: {normalize_text(row.get('typeB'))}"
	)
	model_text = (
		f"NameA: {normalize_text(row.get('NameA'))}; "
		f"pathA: {normalize_text(row.get('pathA'))}; "
		f"typeA: {normalize_text(row.get('typeA'))}; "
		f"propertyA: {normalize_text(row.get('propertyA'))}"
	)
	return f"[REQ] {req_text} [MODEL] {model_text}"


def validate_columns(df: pd.DataFrame, data_name: str) -> None:
	required = list(dict.fromkeys(INPUT_FIELDS + PAIR_KEY_FIELDS))
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"{data_name} missing required columns: {missing}")


def build_pair_keys(df: pd.DataFrame) -> pd.Series:
	return df[PAIR_KEY_FIELDS].apply(
		lambda row: "||".join([normalize_text(x) for x in row.tolist()]),
		axis=1,
	)


def build_label_mapping(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
	labels = [normalize_text(x) for x in train_df[LABEL_COL].tolist()]
	if any(x == "" for x in labels):
		raise ValueError("Train labels contain empty values in Relation column.")
	unique_labels = list(dict.fromkeys(labels))
	label2id = {label: idx for idx, label in enumerate(unique_labels)}
	id2label = {idx: label for label, idx in label2id.items()}
	return label2id, id2label


def encode_labels(raw_labels: List[str], label2id: Dict[str, int]) -> List[int]:
	encoded = []
	for x in raw_labels:
		x = normalize_text(x)
		if x not in label2id:
			raise ValueError(f"Found unseen label in eval data: {x}")
		encoded.append(label2id[x])
	return encoded


def has_relation(label: str) -> bool:
	return normalize_text(label).startswith("1")


def extract_relation_type(label: str) -> str:
	label_text = normalize_text(label)
	if not has_relation(label_text):
		return "无关系"
	if "/" in label_text:
		_, relation_type = label_text.split("/", 1)
		relation_type = normalize_text(relation_type)
		if relation_type:
			return relation_type
	return "有关系"


def evaluate_relation_predictions(
	true_raw_labels: List[str],
	pred_raw_labels: List[str],
) -> Dict[str, float]:
	true_has_relation = np.array([has_relation(x) for x in true_raw_labels], dtype=bool)
	pred_has_relation = np.array([has_relation(x) for x in pred_raw_labels], dtype=bool)

	tp = int((true_has_relation & pred_has_relation).sum())
	fp = int((~true_has_relation & pred_has_relation).sum())
	tn = int((~true_has_relation & ~pred_has_relation).sum())
	fn = int((true_has_relation & ~pred_has_relation).sum())
	total = len(true_raw_labels)

	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
	accuracy = (tp + tn) / total if total > 0 else 0.0

	tp_mask = true_has_relation & pred_has_relation
	relation_type_total = int(tp_mask.sum())
	if relation_type_total > 0:
		true_relation_types = [extract_relation_type(x) for x in true_raw_labels]
		pred_relation_types = [extract_relation_type(x) for x in pred_raw_labels]
		relation_type_correct = int(
			sum(
				pred_relation_types[i] == true_relation_types[i]
				for i in range(len(true_relation_types))
				if tp_mask[i]
			)
		)
		relation_type_accuracy = relation_type_correct / relation_type_total
	else:
		relation_type_correct = 0
		relation_type_accuracy = 0.0

	return {
		"tp": tp,
		"fp": fp,
		"tn": tn,
		"fn": fn,
		"accuracy": float(accuracy),
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
		"relation_type_accuracy": float(relation_type_accuracy),
		"relation_type_total": relation_type_total,
		"relation_type_correct": relation_type_correct,
		"total": total,
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


def build_compute_metrics(no_relation_id: int):
	def compute_metrics(eval_pred):
		logits, labels = eval_pred
		preds = np.argmax(logits, axis=1)
		precision, recall, f1, _ = precision_recall_fscore_support(
			labels, preds, average="macro", zero_division=0
		)
		acc = accuracy_score(labels, preds)

		true_has_relation = labels != no_relation_id
		pred_has_relation = preds != no_relation_id
		relation_precision, relation_recall, relation_f1, _ = precision_recall_fscore_support(
			true_has_relation,
			pred_has_relation,
			average="binary",
			zero_division=0,
		)
		return {
			"precision": float(precision),
			"recall": float(recall),
			"f1": float(f1),
			"accuracy": float(acc),
			"relation_precision": float(relation_precision),
			"relation_recall": float(relation_recall),
			"relation_f1": float(relation_f1),
		}

	return compute_metrics


def compute_class_weights(
	labels: List[int],
	num_labels: int,
	mode: str,
) -> Optional[torch.Tensor]:
	if mode == "none":
		return None

	counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
	if np.any(counts == 0):
		raise ValueError(f"Cannot compute class weights because class counts are {counts.tolist()}.")

	weights = len(labels) / (num_labels * counts)
	if mode == "sqrt_balanced":
		weights = np.sqrt(weights)

	# Normalize around 1 so the overall loss scale remains comparable.
	weights = weights / weights.mean()
	return torch.tensor(weights, dtype=torch.float32)


def save_confusion_matrix(
	true_ids: List[int],
	pred_ids: List[int],
	id2label: Dict[int, str],
	plot_path: Path,
) -> None:
	labels = list(sorted(id2label.keys()))
	cm = confusion_matrix(true_ids, pred_ids, labels=labels)

	fig, ax = plt.subplots(figsize=(8, 6))
	im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
	fig.colorbar(im, ax=ax)
	ax.set_title("Confusion Matrix")
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_xticks(range(len(labels)))
	ax.set_yticks(range(len(labels)))
	ax.set_xticklabels([id2label[i] for i in labels], rotation=45, ha="right")
	ax.set_yticklabels([id2label[i] for i in labels])

	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

	fig.tight_layout()
	plot_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(plot_path, dpi=150)
	plt.close(fig)


def save_core_metrics_plot(metrics: Dict[str, float], plot_path: Path) -> None:
	metric_names = ["precision", "recall", "accuracy", "f1", "relation_type_accuracy"]
	metric_values = [float(metrics[name]) for name in metric_names]

	fig, ax = plt.subplots(figsize=(8.5, 4.8))
	bars = ax.bar(
		metric_names,
		metric_values,
		color=["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"],
	)
	ax.set_ylim(0.0, 1.0)
	ax.set_ylabel("Score")
	ax.set_title("Core Evaluation Metrics")
	ax.tick_params(axis="x", rotation=15)

	for bar, value in zip(bars, metric_values):
		ax.text(
			bar.get_x() + bar.get_width() / 2,
			value + 0.02,
			f"{value:.3f}",
			ha="center",
			va="bottom",
			fontsize=9,
		)

	fig.tight_layout()
	plot_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(plot_path, dpi=150)
	plt.close(fig)


def main() -> None:
	args = parse_args()
	if not (0.0 <= args.warmup_ratio <= 1.0):
		raise ValueError(f"--warmup-ratio must be in [0, 1], got {args.warmup_ratio}.")
	if args.early_stopping_patience < 0:
		raise ValueError(
			f"--early-stopping-patience must be >= 0, got {args.early_stopping_patience}."
		)
	set_reproducible_seed(args.seed)
	print(f"Using random seed: {args.seed}")

	train_df = load_csv(args.train_path)
	test_df = load_csv(args.test_path)
	validate_columns(train_df, "train")
	validate_columns(test_df, "test")

	if LABEL_COL not in train_df.columns:
		raise ValueError(f"Train file must contain '{LABEL_COL}' column.")

	train_df = train_df.copy()
	test_df = test_df.copy()
	train_df["input_text"] = train_df.apply(build_input_text, axis=1)
	test_df["input_text"] = test_df.apply(build_input_text, axis=1)

	test_keys = set(build_pair_keys(test_df).tolist())
	train_keys = build_pair_keys(train_df)
	overlap_mask = train_keys.isin(test_keys)
	overlap_count = int(overlap_mask.sum())
	if overlap_count > 0:
		train_df = train_df.loc[~overlap_mask].reset_index(drop=True)
	print(f"Removed overlapped train-test pairs from training set: {overlap_count}")

	if train_df.empty:
		raise ValueError("Training data is empty after removing train-test overlaps.")

	val_df: Optional[pd.DataFrame] = None
	if args.val_ratio > 0:
		raw_labels = train_df[LABEL_COL].astype(str).apply(normalize_text)
		label_counts = raw_labels.value_counts()
		num_classes_before_split = int(label_counts.shape[0])
		if int(label_counts.min()) >= 2:
			val_size = max(1, int(round(len(train_df) * args.val_ratio)))
			if val_size < num_classes_before_split:
				val_size = num_classes_before_split
			if val_size >= len(train_df):
				val_size = len(train_df) - 1
			if val_size > 0 and val_size >= num_classes_before_split:
				train_df, val_df = train_test_split(
					train_df,
					test_size=val_size,
					random_state=args.seed,
					shuffle=True,
					stratify=raw_labels,
				)
				train_df = train_df.reset_index(drop=True)
				val_df = val_df.reset_index(drop=True)
		else:
			print("Skipped validation split because at least one class has fewer than 2 samples.")

	label2id, id2label = build_label_mapping(train_df)
	train_labels = encode_labels(train_df[LABEL_COL].astype(str).tolist(), label2id)
	num_labels = len(label2id)
	if num_labels < 2:
		raise ValueError(
			f"Relation column has only {num_labels} class. At least 2 classes are required for classification."
		)
	if "0" not in label2id:
		raise ValueError("Expected no-relation class label '0' in Relation column.")

	class_weights = compute_class_weights(
		train_labels,
		num_labels=num_labels,
		mode=args.class_weight_mode,
	)
	class_counts = np.bincount(train_labels, minlength=num_labels)
	print(
		"Training class counts: "
		+ ", ".join(f"{id2label[i]}={int(class_counts[i])}" for i in range(num_labels))
	)
	if class_weights is None:
		print("Class weighting disabled.")
	else:
		print(
			f"Class weight mode: {args.class_weight_mode}; weights: "
			+ ", ".join(f"{id2label[i]}={class_weights[i].item():.4f}" for i in range(num_labels))
		)

	model_path = Path(args.model_path)
	if not model_path.exists():
		raise FileNotFoundError(
			f"Model path not found: {model_path}. Please download roberta-base first."
		)

	print(f"Using local model path: {model_path}")
	tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
	# 全量微调：基模型参数不冻结，和 softmax 头一起训练。
	backbone = AutoModel.from_pretrained(str(model_path), local_files_only=True)
	model = FullFineTuningSoftmaxClassifier(
		backbone=backbone,
		num_labels=num_labels,
		class_weights=class_weights,
	)
	trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Trainable parameter count: {trainable_param_count}")
	print(f"Detected {num_labels} relation classes. Using softmax multi-class classification.")

	train_enc = tokenizer(
		train_df["input_text"].tolist(),
		truncation=True,
		max_length=args.max_length,
	)
	train_dataset = PairDataset(train_enc, train_labels)

	has_test_label = LABEL_COL in test_df.columns
	eval_dataset = None
	test_labels: Optional[List[int]] = None
	if val_df is not None:
		val_labels = encode_labels(val_df[LABEL_COL].astype(str).tolist(), label2id)
		val_enc = tokenizer(
			val_df["input_text"].tolist(),
			truncation=True,
			max_length=args.max_length,
		)
		eval_dataset = PairDataset(val_enc, val_labels)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	training_args = TrainingArguments(
		output_dir=str(output_dir / "checkpoints"),
		optim="adamw_torch",
		learning_rate=args.lr,
		lr_scheduler_type=args.lr_scheduler_type,
		warmup_ratio=args.warmup_ratio,
		per_device_train_batch_size=args.train_batch_size,
		per_device_eval_batch_size=args.eval_batch_size,
		num_train_epochs=args.epochs,
		weight_decay=0.01,
		logging_strategy="epoch",
		eval_strategy="epoch" if eval_dataset is not None else "no",
		save_strategy="epoch" if eval_dataset is not None else "no",
		load_best_model_at_end=eval_dataset is not None,
		metric_for_best_model="relation_f1" if eval_dataset is not None else None,
		greater_is_better=True if eval_dataset is not None else None,
		report_to="none",
		seed=args.seed,
	)

	callbacks = []
	if eval_dataset is not None and args.early_stopping_patience > 0:
		callbacks.append(
			EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
		)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		tokenizer=tokenizer,
		data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
		compute_metrics=(
			build_compute_metrics(no_relation_id=label2id["0"])
			if eval_dataset is not None
			else None
		),
		callbacks=callbacks,
	)

	trainer.train()

	test_enc_for_pred = tokenizer(
		test_df["input_text"].tolist(),
		truncation=True,
		max_length=args.max_length,
	)
	test_dataset_for_pred = PairDataset(test_enc_for_pred)
	pred_output = trainer.predict(test_dataset_for_pred)
	pred_ids = np.argmax(pred_output.predictions, axis=1)
	pred_probs = torch.softmax(torch.tensor(pred_output.predictions), dim=1).numpy()
	confidence = pred_probs.max(axis=1)

	test_df["pred_label_id"] = pred_ids
	test_df["pred_label"] = [id2label[int(i)] for i in pred_ids]
	test_df["pred_confidence"] = confidence
	if args.oracle_positive_only:
		positive_labels = ("1/Refine", "1/Satisfy")
		missing = [label for label in positive_labels if label not in label2id]
		if missing:
			raise ValueError(f"Oracle relation types absent from KG training data: {missing}")
		positive_ids = [label2id[label] for label in positive_labels]
		positive_logits = pred_output.predictions[:, positive_ids]
		positive_probs = torch.softmax(torch.tensor(positive_logits), dim=1).numpy()
		positive_choice = np.argmax(positive_logits, axis=1)
		test_df["oracle_pred_label"] = [positive_labels[int(i)] for i in positive_choice]
		test_df["oracle_pred_confidence"] = positive_probs.max(axis=1)

	pred_csv_path = output_dir / "judge_predictions.csv"
	test_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
	print(f"Saved predictions to: {pred_csv_path}")

	label_map_path = output_dir / "label_mapping.json"
	label_map_path.write_text(
		json.dumps({"label2id": label2id, "id2label": id2label}, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	print(f"Saved label mapping to: {label_map_path}")

	if has_test_label:
		test_labels = encode_labels(test_df[LABEL_COL].astype(str).tolist(), label2id)

	if has_test_label and test_labels is not None:
		true_raw_labels = test_df[LABEL_COL].astype(str).tolist()
		pred_raw_labels = test_df["pred_label"].astype(str).tolist()
		metrics = evaluate_relation_predictions(true_raw_labels, pred_raw_labels)
		metrics["num_classes"] = num_labels
		print_evaluation_summary(metrics)
		metrics_path = output_dir / "judge_metrics.json"
		metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
		print(f"Saved metrics to: {metrics_path}")
		metrics_plot_path = output_dir / "judge_metrics_plot.png"
		save_core_metrics_plot(metrics, metrics_plot_path)
		print(f"Saved metrics plot to: {metrics_plot_path}")

		cm_path = output_dir / "judge_confusion_matrix.png"
		save_confusion_matrix(test_labels, pred_ids.tolist(), id2label, cm_path)
		print(f"Saved confusion matrix to: {cm_path}")
	else:
		print("Test file has no Relation column; skipped eval metrics and confusion matrix.")


if __name__ == "__main__":
	main()
