"""Run the 85 positive Oracle samples once with the hosted DeepSeek API."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from config import api_key as configured_api_key
from oracle_dataset import PROJECT_COUNTS, prepare
from oracle_run import run, score


def load_env_file(path):
    if not path.is_file():
        if (os.getenv("LLM_API_KEY") or configured_api_key) not in (None, "", "EMPTY", "replace_with_your_api_key"):
            return
        raise FileNotFoundError(f"No API configuration file or configured key: {path}")
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry in {path}")
        name, value = (part.strip() for part in line.split("=", 1))
        if name in {"LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_REQUEST_TIMEOUT_SECONDS"}:
            os.environ.setdefault(name, value.strip('"\''))
    if os.getenv("LLM_API_KEY") in (None, "", "EMPTY", "replace_with_your_api_key"):
        raise ValueError("LLM_API_KEY is missing")


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path,
                        default=project_root / ".env")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    load_env_file(args.env_file)
    if os.getenv("LLM_MODEL", "deepseek-v4-pro").lower() != "deepseek-v4-pro":
        raise ValueError("This launcher requires LLM_MODEL=deepseek-v4-pro")
    os.environ["LLM_MODEL"] = "deepseek-v4-pro"
    os.environ.setdefault("LLM_BASE_URL", "https://api.deepseek.com")

    source = project_root / "Data" / "Oracle_Positive_85.csv"
    staged, _ = prepare(source, project_root / "Data")
    output_root = args.output_root or (
        project_root / "Llm_baseline" / "oracle" / "outputs" / "deepseek-v4-pro"
        / f"run_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}"
    )
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    print(f"Oracle output: {output_root}", flush=True)
    all_records = []
    for project in PROJECT_COUNTS:
        rows = staged[project][3]
        print(f"Starting {project}: {len(rows)} samples", flush=True)
        metrics = run(rows, output_root / project, fail_fast=True)
        all_records.extend(json.loads(
            (output_root / project / "oracle_predictions.json").read_text(encoding="utf-8")
        ))
        if metrics["failed_predictions"] or metrics["total"] != len(rows):
            raise RuntimeError(f"{project} stopped at a failed prediction; inspect {output_root / project}")
    combined, _ = score(all_records)
    if combined["total"] != 85:
        raise AssertionError("Combined Oracle run must contain exactly 85 predictions")
    (output_root / "oracle_metrics_all_85.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(combined, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
