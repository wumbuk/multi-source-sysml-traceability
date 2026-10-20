# Ablation Experiments

English | [中文](README_zh-CN.md)

The five `.txt` files contain replacement snippets for `output_dict`. Each replaces only the final output dictionary in `construct_inf_json()` in `construct_json.py`. Keep the preceding construction of `judge_cases` and `subsystem_nodes` and the rest of the Python file unchanged.

## Single-Factor Settings

The Chinese strings below are literal input-field names used by the code.

| Variant | Field removed from the input | Content retained |
|---|---|---|
| `Case_Retrieval` | Top-level `可能对追溯关系判断有帮助的案例` (reference cases) | All other context |
| `Hierarchical_Path` | `建模路径` (model path) of the current model and requirement nodes | Paths inside cases and subsystem nodes |
| `Ontology_Constraint` | Top-level `模型元素和需求元素之间可能存在的关系集合` (candidate relations) | All other context |
| `Semantic_Enrichment` | `功能描述` (functional description) of the current model node | Requirement, case, and subsystem-node descriptions |
| `Structural_Context` | Top-level `和模型节点处于同一子系统中的节点信息` (subsystem nodes) | All other context |
| Interaction Context | Set `is_port = False` for every sample | The complete final `output_dict` |

These settings follow the five supplied snippets exactly. They are **LLM input-field ablations**: retrieval, semantic encoding, and ontology reasoning may still execute. Runtime differences therefore cannot be interpreted directly as efficiency gains from removing the corresponding algorithms. Semantic Enrichment ablation does not rebuild the graph from unenriched data.

## Option 1: Environment Variables

`TRACE_ABLATION_VARIANT` accepts the five exact names in the table; leave it blank to use the full dictionary. `TRACE_USE_PORT_CONTEXT=true` is the full-method default and adds interaction context only for port samples. Change one factor at a time and use separate output directories.

Run in PowerShell from `src/ours`:

```powershell
# Full method. Clear the sample selection to process the entire Judge.csv.
$env:TRACE_ABLATION_VARIANT=" "
$env:TRACE_USE_PORT_CONTEXT="true"
$env:TRACE_ORACLE_TYPE_ONLY="false"
$env:TRACE_RERUN_SEQUENCES=" "
$env:TRACE_OUTPUT_RESULT_PATH="result/runs/full/predictions.json"
$env:TRACE_SFT_SAMPLE_PATH="result/runs/full/llm_samples.jsonl"
python main.py
```

For the five field ablations, use the following pattern, shown for Case Retrieval:

```powershell
$env:TRACE_ABLATION_VARIANT="Case_Retrieval"
$env:TRACE_USE_PORT_CONTEXT="true"
$env:TRACE_OUTPUT_RESULT_PATH="result/runs/without_case_retrieval/predictions.json"
$env:TRACE_SFT_SAMPLE_PATH="result/runs/without_case_retrieval/llm_samples.jsonl"
python main.py
```

For the other four variants, change the variant name and output directory.

For Interaction Context ablation:

```powershell
$env:TRACE_ABLATION_VARIANT=" "
$env:TRACE_USE_PORT_CONTEXT="false"
$env:TRACE_OUTPUT_RESULT_PATH="result/runs/without_interaction_context/predictions.json"
$env:TRACE_SFT_SAMPLE_PATH="result/runs/without_interaction_context/llm_samples.jsonl"
python main.py
```

`main.py` then passes `is_port=False` for every sample and skips parsing the port input file. Python uses `False`; `.env` uses `false`. Interaction Context has a separate switch and is not a sixth value of `TRACE_ABLATION_VARIANT`.

Shell environment variables override `.env`. The single-space values above are stripped when read, explicitly clearing any selection in `.env`. Reset the variables or open a new terminal after the experiment so that later full-method runs use the intended settings.

## Option 2: Replace the Final Dictionary Manually

1. Save the complete version of `construct_json.py`.
2. In `construct_inf_json()`, locate `output_dict = {...}` below the comment `构建最终输出字典` (construct the final output dictionary).
3. Replace this assignment with the entire contents of the chosen `.txt` file, preserving function indentation.
4. Keep the final `return json.dumps(...)` and the rest of the function.
5. Leave `TRACE_ABLATION_VARIANT` blank so automatic field removal is not also applied.
6. Keep `TRACE_USE_PORT_CONTEXT=true` for the first five variants and choose a separate output directory.
7. Restore the full dictionary before applying the next variant.

For manual Interaction Context ablation, change
`is_port = is_port_node(model_node) if USE_PORT_CONTEXT else False`
to `is_port = False` before the LLM call in `main.py`. The environment switch above provides the equivalent behavior.

## Running Comparable Experiments

Keep the data split, model version, endpoint, decoding settings, and selected samples fixed. `TRACE_RERUN_SEQUENCES` uses sample IDs starting at 2, following the CSV-row convention; leave it blank for full runs. Keep `TRACE_ORACLE_TYPE_ONLY=false` for ordinary link-existence and relation-type prediction. Enable it only for type prediction on known positive pairs.

Save `predictions.json`, `predictions_timing.json`, and the run configuration for each setting.
