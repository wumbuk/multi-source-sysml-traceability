import argparse
import random
from pathlib import Path

import pandas as pd


LABEL_COL = "Relation"
# 通过样本对唯一键切分，避免同一个样本对同时出现在训练/测试中
KEY_FIELDS = ["NameA", "pathA", "typeA", "propertyA", "NameB", "contextB", "typeB"]
DEFAULT_DATASET_DIR = (
	Path(__file__).resolve().parent
	/ "GenerateContext"
	/ "Air_Condition"
)


def normalize_text(v: object) -> str:
	if v is None:
		return ""
	if isinstance(v, float) and pd.isna(v):
		return ""
	return " ".join(str(v).strip().split())


def build_key(row: pd.Series) -> str:
	return "||".join(normalize_text(row[c]) for c in KEY_FIELDS)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Stratified split by Relation (each class appears in both train and test)."
	)
	parser.add_argument(
		"--input",
		type=str,
		default=str(DEFAULT_DATASET_DIR / "Air_Condition.csv"),
		help="Input CSV file.",
	)
	parser.add_argument(
		"--train-output",
		"--train_output",
		type=str,
		default=str(DEFAULT_DATASET_DIR / "KG.csv"),
		help="Output train CSV path or directory. A directory produces KG.csv.",
	)
	parser.add_argument(
		"--test-output",
		"--test_output",
		type=str,
		default=str(DEFAULT_DATASET_DIR / "Judge.csv"),
		help="Output test CSV path or directory. A directory produces Judge.csv.",
	)
	parser.add_argument(
		"--test-ratio",
		type=float,
		default=0.33,
		help="Test split ratio (0~1).",
	)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def resolve_output_path(output: str | Path, default_filename: str) -> Path:
	output_path = Path(output)
	if output_path.suffix.lower() == ".csv":
		return output_path
	return output_path / default_filename


def split_dataset(
	input_path: str | Path,
	train_output: str | Path,
	test_output: str | Path,
	test_ratio: float = 0.33,
	seed: int = 42,
) -> tuple[Path, Path]:
	input_path = Path(input_path)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")
	if not (0.0 < test_ratio < 1.0):
		raise ValueError("--test-ratio must be between 0 and 1 (exclusive)")

	df = pd.read_csv(input_path)

	missing = [c for c in [LABEL_COL] + KEY_FIELDS if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	df = df.copy()
	df["_label"] = df[LABEL_COL].apply(normalize_text)
	if (df["_label"] == "").any():
		raise ValueError("Relation column contains empty values.")

	# 构造样本对唯一键（不使用正则，仅做 strip/空格归一化）
	df["_key"] = df.apply(build_key, axis=1)

	# 同一 key 出现多个不同 Relation 属于数据冲突
	label_per_key = df.groupby("_key")["_label"].nunique()
	conflict_keys = label_per_key[label_per_key > 1]
	if not conflict_keys.empty:
		example = conflict_keys.head(5).index.tolist()
		raise ValueError(
			"Found keys mapped to multiple Relation labels (data conflict). Examples: "
			+ "; ".join(example)
		)

	# 按 key 去重后，按 Relation 分层抽取 test keys，保证每类在 train/test 都出现
	key_label_df = df[["_key", "_label"]].drop_duplicates().reset_index(drop=True)

	rng = random.Random(seed)
	test_keys: set[str] = set()
	for label, sub in key_label_df.groupby("_label"):
		keys = sub["_key"].tolist()
		if len(keys) < 2:
			raise ValueError(
				f"Label '{label}' has only {len(keys)} unique keys; cannot ensure it appears in both train and test."
			)
		rng.shuffle(keys)
		n_test = int(round(len(keys) * test_ratio))
		n_test = max(1, n_test)
		n_test = min(n_test, len(keys) - 1)
		test_keys.update(keys[:n_test])

	df_test = df[df["_key"].isin(test_keys)].drop(columns=["_key", "_label"])
	df_train = df[~df["_key"].isin(test_keys)].drop(columns=["_key", "_label"])

	train_labels = set(df_train[LABEL_COL].apply(normalize_text).unique().tolist())
	test_labels = set(df_test[LABEL_COL].apply(normalize_text).unique().tolist())
	if train_labels != test_labels:
		raise RuntimeError(
			"Split failed to preserve all labels in both sets. "
			f"train_only={sorted(train_labels - test_labels)}, test_only={sorted(test_labels - train_labels)}"
		)

	train_out = resolve_output_path(train_output, "KG.csv")
	test_out = resolve_output_path(test_output, "Judge.csv")
	train_out.parent.mkdir(parents=True, exist_ok=True)
	test_out.parent.mkdir(parents=True, exist_ok=True)

	# utf-8-sig 方便 Excel 打开
	df_train.to_csv(train_out, index=False, encoding="utf-8-sig")
	df_test.to_csv(test_out, index=False, encoding="utf-8-sig")

	print("数据划分完成：按 Relation 分层 + 按样本对 key 切分（避免泄露）")
	print(f"训练集: {train_out} rows={len(df_train)}")
	print(f"测试集: {test_out} rows={len(df_test)}")
	print(f"Relation 类别数: {len(train_labels)}")
	return train_out, test_out


def main() -> None:
	args = parse_args()
	split_dataset(
		input_path=args.input,
		train_output=args.train_output,
		test_output=args.test_output,
		test_ratio=args.test_ratio,
		seed=args.seed,
	)


if __name__ == "__main__":
	main()
