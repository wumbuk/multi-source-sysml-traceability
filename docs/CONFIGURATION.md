# Configuration Reference

The project reads local settings from environment variables. When
`python-dotenv` is installed, variables in the project-root `.env` file are
loaded automatically. Environment variables already defined by the operating
system take precedence.

## Local setup

```powershell
Copy-Item .env.example .env
```

The real `.env` is ignored by Git. Do not place credentials in `config.py`,
scripts, notebooks, command examples, or experiment JSON files.

## Data paths

| Variable | Purpose | Default |
|---|---|---|
| `TRACE_DATASET_DIR` | Dataset directory | `Data/GenerateContext/Air_Condition` |
| `TRACE_SEMANTIC_DATASET_PATH` | Semantically enriched full CSV | `Air_Condition.csv` under the dataset directory |
| `TRACE_KG_CSV_PATH` | Knowledge-graph/training split | `KG.csv` under the dataset directory |
| `TRACE_JUDGE_CSV_PATH` | Evaluation split | `Judge.csv` under the dataset directory |
| `TRACE_PORT_INFORMATION_PATH` | Extracted XMI port information | `Data/Port_Information/关于端口信息流动的XMI.txt` |
| `TRACE_OUTPUT_RESULT_PATH` | Prediction JSON | `result/runs/latest/predictions.json` |
| `TRACE_SFT_SAMPLE_PATH` | Optional raw SFT sample log | `result/sft_samples/llm_samples.jsonl` |

Relative values are resolved from the repository root, so the project can be
moved without editing source code.

## Embedding model

| Variable | Purpose |
|---|---|
| `TRACE_EMBEDDING_MODEL` | Hugging Face model ID or local model directory |
| `TRACE_EMBEDDING_DEVICE` | `cuda`, `cuda:0`, `cpu`, or empty |

The committed default is `sentence-transformers/all-MiniLM-L6-v2`. For exact
reproduction, pin a model revision in the experiment metadata or use a
versioned local snapshot.

## LLM service

| Variable | Purpose |
|---|---|
| `TRACE_LLM_BASE_URL` | Shared OpenAI-compatible endpoint |
| `TRACE_LLM_MODEL` | Shared served model ID |
| `TRACE_LLM_API_KEY` | Shared API key |
| `TRACE_SINGLE_STAGE_BASE_URL` | Optional single-stage endpoint override |
| `TRACE_SINGLE_STAGE_MODEL` | Optional single-stage model override |
| `TRACE_SINGLE_STAGE_API_KEY` | Optional single-stage key override |
| `TRACE_TWO_STAGE_BASE_URL` | Optional two-stage endpoint override |
| `TRACE_TWO_STAGE_MODEL` | Optional two-stage model override |
| `TRACE_TWO_STAGE_API_KEY` | Optional two-stage key override |
| `TRACE_TWO_STAGE_TIMEOUT_SECONDS` | Two-stage request timeout |

Blank per-mode overrides inherit the shared `TRACE_LLM_*` values.

Example local vLLM configuration:

```dotenv
TRACE_LLM_BASE_URL=http://localhost:8080/v1
TRACE_LLM_MODEL=Qwen/Qwen3.5-27B
TRACE_LLM_API_KEY=EMPTY
```

## Semantic enrichment

| Variable | Purpose |
|---|---|
| `TRACE_ENRICHMENT_API_KEY` | Private API key; no committed default |
| `TRACE_ENRICHMENT_BASE_URL` | OpenAI-compatible enrichment endpoint |
| `TRACE_ENRICHMENT_MODEL` | Enrichment model name |

Semantic enrichment fails with a clear error when it is enabled without a key.

## Ontology reasoner

| Variable | Purpose |
|---|---|
| `TRACE_REASONER_JAR_PATH` | Built reasoner fat JAR |
| `TRACE_REASONER_WORK_DIR` | Java process working directory |
| `TRACE_ONTOLOGY_PATH` | Ontology TTL file |

The source ontology is retained in
`protege_reasoner/kg-reasoner/ontology/KG.ttl`; Gradle build outputs are not
committed.
