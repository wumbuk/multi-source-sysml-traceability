"""Centralized, environment-driven configuration for the experiment pipeline.

Do not store API keys or machine-specific absolute paths in this file. Override
the defaults with environment variables documented in ``.env.example``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _env_path(name: str, default: Path) -> Path:
    """Return an expanded path from the environment or a project-relative default."""
    raw_value = os.getenv(name)
    if not raw_value:
        return default.expanduser().absolute()
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.absolute()


def _env_optional_path(name: str, default: Path) -> Optional[Path]:
    """Return None when an explicitly configured path is empty."""
    raw_value = os.getenv(name)
    if raw_value is not None:
        raw_value = raw_value.strip()
        if not raw_value:
            return None
        path = Path(raw_value).expanduser()
    else:
        path = default.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.absolute()


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    return int(raw_value) if raw_value else default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    return float(raw_value) if raw_value else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable {name} must be one of "
        "1/0, true/false, yes/no, or on/off."
    )


def _env_str(name: str, default: str) -> str:
    raw_value = os.getenv(name)
    return raw_value.strip() if raw_value and raw_value.strip() else default


# ---------------------------------------------------------------------------
# Dataset and output paths
# ---------------------------------------------------------------------------

DATASET_DIR = _env_path(
    "TRACE_DATASET_DIR",
    PROJECT_ROOT / "Data" / "GenerateContext" / "Air_Condition",
)
SEMANTIC_DATASET_PATH = _env_path(
    "TRACE_SEMANTIC_DATASET_PATH",
    DATASET_DIR / "Air_Condition.csv",
)
SOURCE_DATASET_PATH = _env_path(
    "TRACE_SOURCE_DATASET_PATH",
    SEMANTIC_DATASET_PATH,
)
KG_CSV_PATH = _env_path("TRACE_KG_CSV_PATH", DATASET_DIR / "KG.csv")
JUDGE_CSV_PATH = _env_path("TRACE_JUDGE_CSV_PATH", DATASET_DIR / "Judge.csv")
PORT_INFORMATION_PATH: Optional[Path] = _env_optional_path(
    "TRACE_PORT_INFORMATION_PATH",
    PROJECT_ROOT
    / "Data"
    / "Port_Information"
    / "关于端口信息流动的XMI.txt",
)

OUTPUT_RESULT_PATH = _env_path(
    "TRACE_OUTPUT_RESULT_PATH",
    PROJECT_ROOT / "result" / "runs" / "latest" / "predictions.json",
)
SFT_SAMPLE_PATH = _env_path(
    "TRACE_SFT_SAMPLE_PATH",
    PROJECT_ROOT / "result" / "sft_samples" / "llm_samples.jsonl",
)

# Preprocessing is opt-in so a normal local run cannot overwrite a dataset.
RUN_SEMANTIC_ENRICHMENT = _env_bool(
    "TRACE_RUN_SEMANTIC_ENRICHMENT",
    False,
)
RUN_DATA_SPLIT = _env_bool("TRACE_RUN_DATA_SPLIT", False)
DATA_SPLIT_TEST_RATIO = _env_float("TRACE_DATA_SPLIT_TEST_RATIO", 0.33)
DATA_SPLIT_SEED = _env_int("TRACE_DATA_SPLIT_SEED", 42)


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

# Full-method runs include interaction context; false forces is_port=False.
USE_PORT_CONTEXT = _env_bool("TRACE_USE_PORT_CONTEXT", True)

# Accepts either a Hugging Face model ID or a local model directory.
EMBEDDING_MODEL_NAME_OR_PATH = _env_str(
    "TRACE_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Examples: "cuda", "cuda:0", "cpu". None lets sentence-transformers decide.
EMBEDDING_DEVICE: Optional[str] = (
    os.getenv("TRACE_EMBEDDING_DEVICE", "").strip() or None
)


# ---------------------------------------------------------------------------
# Main LLM service (OpenAI-compatible API, including vLLM)
# ---------------------------------------------------------------------------

# Local vLLM deployments usually ignore the key but the OpenAI client requires
# a non-empty value. Set TRACE_LLM_API_KEY for authenticated remote services.
LLM_API_KEY = _env_str("TRACE_LLM_API_KEY", "EMPTY")

_DEFAULT_LLM_BASE_URL = _env_str(
    "TRACE_LLM_BASE_URL",
    "http://localhost:8080/v1",
)
_DEFAULT_LLM_MODEL = _env_str("TRACE_LLM_MODEL", "Qwen/Qwen3.5-27B")

SINGLE_STAGE_BASE_URL = _env_str(
    "TRACE_SINGLE_STAGE_BASE_URL",
    _DEFAULT_LLM_BASE_URL,
)
SINGLE_STAGE_MODEL_NAME = _env_str(
    "TRACE_SINGLE_STAGE_MODEL",
    _DEFAULT_LLM_MODEL,
)
SINGLE_STAGE_API_KEY = _env_str(
    "TRACE_SINGLE_STAGE_API_KEY",
    LLM_API_KEY,
)

TWO_STAGE_BASE_URL = _env_str(
    "TRACE_TWO_STAGE_BASE_URL",
    _DEFAULT_LLM_BASE_URL,
)
TWO_STAGE_MODEL_NAME = _env_str(
    "TRACE_TWO_STAGE_MODEL",
    _DEFAULT_LLM_MODEL,
)
TWO_STAGE_API_KEY = _env_str(
    "TRACE_TWO_STAGE_API_KEY",
    LLM_API_KEY,
)
TWO_STAGE_TIMEOUT_SECONDS = _env_int("TRACE_TWO_STAGE_TIMEOUT_SECONDS", 120)


# ---------------------------------------------------------------------------
# Semantic enrichment API
# ---------------------------------------------------------------------------

# No default API key is provided intentionally. Configure it only through the
# environment before enabling semantic enrichment.
SEMANTIC_ENRICHMENT_API_KEY = os.getenv("TRACE_ENRICHMENT_API_KEY", "")
SEMANTIC_ENRICHMENT_BASE_URL = _env_str(
    "TRACE_ENRICHMENT_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
SEMANTIC_ENRICHMENT_MODEL = _env_str(
    "TRACE_ENRICHMENT_MODEL",
    "deepseek-v4-pro",
)


# ---------------------------------------------------------------------------
# Java ontology reasoner
# ---------------------------------------------------------------------------

REASONER_PROJECT_DIR = (
    PROJECT_ROOT / "protege_reasoner" / "kg-reasoner"
).absolute()
REASONER_JAR_PATH = _env_path(
    "TRACE_REASONER_JAR_PATH",
    REASONER_PROJECT_DIR
    / "build"
    / "libs"
    / "kg-reasoner-1.0-SNAPSHOT.jar",
)
REASONER_WORK_DIR = _env_path(
    "TRACE_REASONER_WORK_DIR",
    REASONER_PROJECT_DIR,
)
ONTOLOGY_TTL_PATH = _env_path(
    "TRACE_ONTOLOGY_PATH",
    REASONER_PROJECT_DIR / "ontology" / "KG.ttl",
)
