# Multi-Source Context-Enhanced Traceability Prediction for SysML

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](../../LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

English | [中文](README_zh-CN.md)

Run `src/ours/main.py` to predict traceability links and relation types between requirements and SysML model elements.

## Repository Layout

```text
.
|-- Ablation/                       # Ablation settings and guide
|-- main.py                         # End-to-end experiment entry point
|-- config.py                       # Centralized environment-driven settings
|-- construct_json.py               # LLM context construction
|-- call_llm/                       # Single-stage and two-stage LLM calls
|-- add_information_to_graph/       # Predicted relation conversion
|-- Generate_data/                  # Sample generation utility
|-- lib/                            # Graph visualization assets
|-- enlarge_function/               # Semantic enrichment
|-- kg_retrieve/                    # KG construction and context retrieval
|-- Port_Information/               # XMI port/interaction extraction
|-- protege_reasoner/               # Java/SWRL ontology reasoner
|-- Data/                           # Dataset and split utilities
`-- result/                         # Runtime outputs
```

Input datasets are in `Data/<dataset>`. Experiment predictions are in
[`results/ours`](../../results/ours).

## Starting the pipeline

Run the following commands from `src/ours` in Windows PowerShell.
The Python pipeline calls a separately running LLM service; it does not launch
the model server.

### 1. Python, PyTorch and Java

Use Python 3.12 and JDK 17. Start from the repository root:

```powershell
cd src/ours
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-runtime.txt
python -m pip check
java -version
```

For GPU embedding, install the appropriate PyTorch build using the
[official selector](https://pytorch.org/get-started/locally/) before installing
`requirements-runtime.txt`. 

Java must be on `PATH`. The build targets Java 8 bytecode. Build the JAR with
a JDK compatible with Gradle 8.6 (for example JDK 17), running `.\gradlew.bat jar`
inside `protege_reasoner/kg-reasoner` (`./gradlew jar` on Linux/macOS).
The wrapper uses Gradle 8.6; see its
[Java compatibility matrix](https://docs.gradle.org/8.6/userguide/compatibility.html).

### 2. Configure embedding and model access

For initial setup only (do not overwrite an existing `.env`):

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env`:

```dotenv
TRACE_DATASET_DIR=Data/Air_Condition
TRACE_SOURCE_DATASET_PATH=Data/Air_Condition/Air_Condition.csv
TRACE_SEMANTIC_DATASET_PATH=Data/Air_Condition/Air_Condition.csv
TRACE_KG_CSV_PATH=Data/Air_Condition/KG.csv
TRACE_JUDGE_CSV_PATH=Data/Air_Condition/Judge.csv
TRACE_PORT_INFORMATION_PATH=Data/Port_Information/关于端口信息流动的XMI.txt
TRACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TRACE_EMBEDDING_DEVICE=cpu
TRACE_LLM_BASE_URL=http://localhost:8080/v1
TRACE_LLM_MODEL=Qwen/Qwen3.5-27B
TRACE_LLM_API_KEY=EMPTY
TRACE_USE_PORT_CONTEXT=true
TRACE_ABLATION_VARIANT=
TRACE_ORACLE_TYPE_ONLY=false
TRACE_RUN_SEMANTIC_ENRICHMENT=false
TRACE_RUN_DATA_SPLIT=false
```

Use `sentence-transformers/all-MiniLM-L6-v2` to download the embedding model
on first use, or set `TRACE_EMBEDDING_MODEL` to an existing local model directory.
Set `TRACE_EMBEDDING_DEVICE=cuda` for GPU embedding.

Relative paths resolve from this code directory. Shell environment variables
take precedence over `.env`. The single-space values in the PowerShell examples
explicitly override any `.env` selection; the program strips surrounding whitespace. Empty `TRACE_SINGLE_STAGE_*` / `TRACE_TWO_STAGE_*`
settings inherit the common `TRACE_LLM_*` values; nonempty overrides take priority.

### 3. Verify the independently running LLM service

Replace the example API URL, model ID and key with actual service values.
The service must support model listing, chat completions, JSON output and the
thinking parameters used by this code. 

```powershell
python -c "from openai import OpenAI; from config import TWO_STAGE_BASE_URL,TWO_STAGE_API_KEY; c=OpenAI(base_url=TWO_STAGE_BASE_URL,api_key=TWO_STAGE_API_KEY,timeout=10,max_retries=0); print([m.id for m in c.models.list().data])"
```

Connection refused means the server is not listening or the URL is wrong.
Authentication failures require correcting the key. Model-list success checks
connectivity only; a real prediction run also checks chat compatibility.

### 4. Run one sample, then the full dataset

The configuration above selects `KG.csv` and `Judge.csv` under
`Data/Air_Condition`. Set these paths explicitly in `.env` after copying the
template. The port parser accepts annotated three-section text and complete XML.
For Air Condition, generate the port input from the supplied XMI model:

```powershell
python Port_Information/information_extract.py --xmi "Port_Information/Data/Port_Information/VehicleCCU Problem.xml" --output "Data/Port_Information/关于端口信息流动的XMI.txt"
```

For other datasets, set the corresponding XMI or extracted port input.
See the [port extraction guide](Port_Information/README.md).
Full-method runs enable interaction context by default and supply it only for port samples.

```powershell
$env:TRACE_ABLATION_VARIANT=" "
$env:TRACE_USE_PORT_CONTEXT="true"
$env:TRACE_ORACLE_TYPE_ONLY="false"
$env:TRACE_RERUN_SEQUENCES="2"
$env:TRACE_OUTPUT_RESULT_PATH="result/runs/smoke/predictions.json"
$env:TRACE_SFT_SAMPLE_PATH="result/runs/smoke/llm_samples.jsonl"
python main.py
```

Sample IDs start at 2, matching the original CSV-row convention. After a
successful smoke run, clear the selection and choose a separate output folder:

```powershell
$env:TRACE_RERUN_SEQUENCES=" "
$env:TRACE_OUTPUT_RESULT_PATH="result/runs/full/predictions.json"
$env:TRACE_SFT_SAMPLE_PATH="result/runs/full/llm_samples.jsonl"
python main.py
```

Outputs are created automatically:

- `predictions.json`: predictions with original sample IDs.
- `predictions_timing.json`: preprocessing and per-sample inference timings.
- `llm_samples.jsonl`: request/response records at the configured path.

The default Judge dataset has 358 rows; a completed full run should contain
358 predictions. Progress is saved after each pair. Abnormal termination also
backs up completed results to `temporary_results/<timestamp>/`; this does not
mean the run completed and does not automatically resume it.
Reusing a prediction path overwrites earlier predictions; JSONL records append.
No fake predictions are generated when a real API call fails.

Check the prediction count:

```powershell
python -c "import json; from pathlib import Path; p=Path('result/runs/full/predictions.json'); print('Predictions:', len(json.loads(p.read_text(encoding='utf-8'))))"
```

To change datasets, update `TRACE_DATASET_DIR`, `TRACE_KG_CSV_PATH`,
`TRACE_JUDGE_CSV_PATH` and the corresponding port input. Explicit CSV paths in
`.env` do not follow a dataset-directory change automatically. Preprocessing
is opt-in through `TRACE_RUN_SEMANTIC_ENRICHMENT` and `TRACE_RUN_DATA_SPLIT`.

## Ablation experiments

See [Ablation/README.md](Ablation/README.md) for exact replacement locations and
PowerShell commands. The original snippets are preserved:

- [Case_Retrieval.txt](Ablation/Case_Retrieval.txt)
- [Hierarchical_Path.txt](Ablation/Hierarchical_Path.txt)
- [Ontology_Constraint.txt](Ablation/Ontology_Constraint.txt)
- [Semantic_Enrichment.txt](Ablation/Semantic_Enrichment.txt)
- [Structural_Context.txt](Ablation/Structural_Context.txt)

Replace only the final `output_dict` assignment in `construct_json.py`, or set
`TRACE_ABLATION_VARIANT` to one of these five names without editing code.
Leave it empty for the full dictionary and when using a manual replacement.
The automatic and manual methods must not be applied together.

For Interaction Context ablation, set `TRACE_USE_PORT_CONTEXT=false` and leave
`TRACE_ABLATION_VARIANT` empty. Every sample then uses `is_port=False`; the
dictionary remains unchanged and the port input is not parsed.

Change one factor at a time and use independent output folders. These are
input-field ablations; upstream retrieval/ontology computations still run.
Paths/descriptions inside cases and subsystem nodes are retained exactly as
specified by the supplied snippets.
