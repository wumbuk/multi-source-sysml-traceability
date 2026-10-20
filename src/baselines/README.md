# Running the Three Baselines

English | [中文](README_zh-CN.md)

Run the commands below from `src/baselines`. Starting at the repository root:

```powershell
cd src/baselines
```

The `python` commands below also work on Linux.

## 1. Prepare the Environment, Models, and Data

Use Python 3.10 or later, preferably in a separate environment. With Conda:

```bash
conda create -n trace_baselines python=3.12 -y
conda activate trace_baselines
```

Install dependencies using the same Python interpreter that will run the experiments:

```bash
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA environment. Start the LLM service separately; this component calls an OpenAI-compatible API. Embedding and Supervised load models from local directories without downloading them automatically. The default Embedding entry point requires CUDA.

Check dependencies and GPU availability:

```bash
python -c "import openai, pandas, sklearn, matplotlib, transformers, sentence_transformers, accelerate, safetensors; import torch; print('Dependencies OK'); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available` must be `True` for the default Embedding entry point. If it is `False`, check the GPU driver and PyTorch CUDA installation. Calling an LLM API does not require a local GPU.

Configure model paths, service access, and datasets in `experiment_config.json`:

- `llm.model` and `llm.base_url`: model ID and endpoint of the LLM service.
- `embedding.model_path`: local all-MiniLM-L6-v2 directory.
- `supervised.model_path`: local RoBERTa-base directory.
- Relative model paths resolve from `src/baselines`; absolute paths are also supported.

The default paths are `models/all-MiniLM-L6-v2` and `models/roberta-base`. Model weights are not included. Place complete model directories at these locations or point the configuration to existing directories. Include model configuration, tokenizer files, and the module configuration required by SentenceTransformer. For Windows paths in JSON, use `/`, for example `D:/huggingface_models/roberta-base`.

```text
Data/
  Air_Condition/
    KG.csv
    Judge.csv
  Architecting_Spacecraft/
    KG.csv
    Judge.csv
  Extract_Requirement_Values/
    KG.csv
    Judge.csv
  Oracle_Positive_85.csv
```

`KG.csv` supplies reference samples for Embedding and training samples for Supervised. `Judge.csv` is the standard experiment test set. `Oracle_Positive_85.csv` is required only for Oracle experiments.

For a remote API, copy `.env.example` to `.env` and set `LLM_API_KEY`. Keep real keys private. Environment variables take precedence over the configuration file.

Create `.env` for the first time in PowerShell:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Replace `LLM_API_KEY=replace_with_your_api_key` with the actual key. For a local vLLM service without authentication, `.env` is optional; the code uses `EMPTY` as a placeholder. When changing models, check `LLM_MODEL` and `LLM_BASE_URL` in `.env` and the current shell: these override the JSON configuration. Remove overrides that are no longer needed.

## 2. Default Parameters

- **Embedding:** all-MiniLM-L6-v2, 384 dimensions, maximum length 256, cosine similarity, threshold 0.95.
- **Supervised:** RoBERTa-base, maximum length 256, training/evaluation batch sizes 16/32, AdamW, learning rate `2e-5`, cosine scheduler, warmup ratio 0.1, weight decay 0.01, cross-entropy with square-root-balanced class weights, and **12 epochs**. Early stopping is disabled by default. Testing uses the checkpoint with the best validation `relation_f1`.
- **Direct-LLM:** temperature 0.0, thinking enabled except for LLaMA-3.1-8B-Instruct, and a 120-second request timeout. DeepSeek uses `thinking.type=enabled`; Qwen/Gemma use `chat_template_kwargs.enable_thinking=true`, which requires server-side template support. No additional `repetition_penalty` is set by default.

## 3. Run Standard Experiments

Each command runs one method on one dataset and evaluates the results:

```bash
python run_experiment.py --method llm --dataset Air_Condition
python run_experiment.py --method embedding --dataset Air_Condition --seed 42
python run_experiment.py --method supervised --dataset Air_Condition --seed 42
```

Replace `--dataset` with `Architecting_Spacecraft` or `Extract_Requirement_Values` for the other datasets.

To change the LLM, edit `llm.model` and `llm.base_url` in `experiment_config.json`. For local models, the model ID must match the served model name. The DeepSeek entry uses model ID `deepseek-v4-pro`, endpoint `https://api.deepseek.com`, and a valid API key.

For three repetitions, run the command three times sequentially. Embedding and Supervised can use seeds 42, 43, and 44. For LLM runs, `--seed` labels the output directory; it is not sent as a sampling parameter. Run experiments one at a time.

To inspect the command without model calls or training:

```bash
python run_experiment.py --method supervised --dataset Air_Condition --dry-run
```

Standard outputs are created under `results/<method>/<dataset>/run_<timestamp>_seed<seed>/`, including configuration, logs, predictions, and metrics. Existing runs are not overwritten.

## 4. Run Oracle Experiments

Oracle experiments predict `Refine` or `Satisfy` for the 85 known positive pairs. First validate the data and code:

```bash
python Llm_baseline/oracle_dataset.py
python Llm_baseline/test_oracle.py
```

### Embedding and Supervised

Each command runs all three datasets sequentially, sharing model paths, thresholds, and training settings with the standard experiments. Supervised defaults to the configured 12 epochs. The default random seed is 42; override it with `--seed`.

```bash
python Oracle_local_baselines/run.py --method embedding --seed 42
python Oracle_local_baselines/run.py --method supervised --seed 42
```

To specify the number of epochs explicitly:

```bash
python Oracle_local_baselines/run.py --method supervised --epochs 12
```

Results are saved under `oracle_outputs/run_<timestamp>/` in the corresponding method directory.

### LLM

Configure the model service and API key, then run each dataset. Use a new `--output-dir` for each run:

```bash
python Llm_baseline/oracle_run.py --project Air_Condition --output-dir results/Llm_baseline/oracle/run_1/Air_Condition
python Llm_baseline/oracle_run.py --project Architecting_Spacecraft --output-dir results/Llm_baseline/oracle/run_1/Architecting_Spacecraft
python Llm_baseline/oracle_run.py --project Extract_Requirement_Values --output-dir results/Llm_baseline/oracle/run_1/Extract_Requirement_Values
```

For DeepSeek, the dedicated entry point processes all 85 pairs sequentially:

```bash
python Llm_baseline/run_deepseek_oracle.py
```

This entry point uses `deepseek-v4-pro`. Do not override it with another model ID or a local API endpoint in the environment. Outputs are created under `Llm_baseline/oracle/outputs/deepseek-v4-pro/run_<timestamp>/`.

Oracle and standard experiments share model-call parameters and retry functions: temperature 0.0, a 120-second timeout, and the same `llm.thinking`, `llm.repetition_penalty`, and retry environment variables. `thinking` defaults to `true`; setting it to `null` omits the parameter in both tasks and uses the server default. LLaMA calls omit the thinking switch.

After the configured retries are exhausted for an API timeout, the request is attempted once more with the same parameters, following the standard experiment behavior. Oracle uses its own task prompt, structured output format, and evaluation. Qwen/Gemma/LLaMA use JSON Schema; DeepSeek uses JSON object mode. Parsing errors are recorded as failures without substituting gold labels. Oracle outputs include type predictions, mismatches, a confusion matrix, Type Accuracy, and Macro-F1.

## 5. Check Inputs and Inspect Outputs

Check that all standard experiment inputs exist:

```bash
python -c "from pathlib import Path; files=[Path('Data')/d/f for d in ('Air_Condition','Architecting_Spacecraft','Extract_Requirement_Values') for f in ('KG.csv','Judge.csv')]; missing=[str(p) for p in files if not p.is_file()]; assert not missing, missing; print('Main experiment inputs OK')"
```

Before running Embedding or Supervised, check the configured model directories:

```bash
python -c "from pathlib import Path; from experiment_settings import SETTINGS; paths=[Path(SETTINGS[m]['model_path']) for m in ('embedding','supervised')]; missing=[str(p) for p in paths if not p.is_dir()]; assert not missing, missing; print('Local model directories OK')"
```

Before running an LLM experiment, confirm that the service is reachable and exposes the configured model ID. This lists models without performing inference:

```bash
python -c "import sys; sys.path.insert(0,'Llm_baseline'); import config; from openai import OpenAI; ids=[m.id for m in OpenAI(base_url=config.base_url,api_key=config.api_key,timeout=120).models.list().data]; assert config.model_name in ids, (config.model_name,ids); print('Configured LLM service OK')"
```

For LLM Oracle, after preparing the data in Section 4, test one real sample using a new output directory:

```bash
python Llm_baseline/oracle_run.py --project Air_Condition --one-sample --fail-fast --output-dir results/Llm_baseline/oracle/preflight_1/Air_Condition
```

Check `oracle_metrics.json` for `total=1`, `valid_predictions=1`, and `failed_predictions=0` before running all samples.

`--dry-run` checks command construction only; it does not validate model files, connectivity, or inference. Standard runs print an `Output:` path and write progress to `run.log` in that directory. A quiet terminal does not necessarily mean execution has stopped. Successful completion prints `Completed:`.

Completed Oracle runs should contain 45, 12, and 28 samples for the respective datasets. For failures, inspect `error` in `oracle_predictions.json`; failed or incomplete runs are not complete experimental results.
