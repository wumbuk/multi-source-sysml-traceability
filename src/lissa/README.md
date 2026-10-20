# LiSSA: Environment Setup and Local Execution

English | [中文](README_zh-CN.md)

## Project Structure

```text
src/lissa/
├── src/                            # Java source and tests
├── Data/
│   ├── Air_Condition/
│   ├── Architecting_Spacecraft/
│   └── Extract_Requirement_Values/
├── reproduction/                   # Dataset configurations
├── embedding_model/all_Min/        # all-MiniLM-L6-v2 model directory
├── deploy/
│   ├── embedding_server.py         # Local embedding endpoint
│   ├── export_typed_results.py     # Prediction and metric export
│   ├── oracle_experiment.py        # Oracle input preparation and evaluation
│   └── OracleTypeExperiment.java   # Oracle type-prediction entry point
├── pom.xml                         # Maven build configuration
├── LICENSE
├── README.md
└── README_zh-CN.md
```

Each dataset's `Judge.csv` contains element pairs and gold labels. Standard experiments retrieve candidates and predict relations. Oracle experiments predict `Refine` or `Satisfy` only for known positive pairs.

## Environment Setup

Requirements: JDK 21, Maven, and Python 3.10 or later. The commands below use Bash and start at the repository root.

### 1. Install Dependencies and Build

```bash
conda create -n lissa_env -c conda-forge python=3.11 openjdk=21 maven -y
conda activate lissa_env
python -m pip install torch sentence-transformers numpy huggingface-hub

cd src/lissa
java -version
javac -version
mvn -version
mvn -DskipTests package
java -jar target/lissa-0.2.0-SNAPSHOT-jar-with-dependencies.jar eval --help
```

### 2. Prepare the Embedding Model

Download the model from `src/lissa`:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', local_dir='embedding_model/all_Min')"
```

If a complete model directory is already available, use its path in `--model-path`. In a separate terminal, activate the environment, enter `src/lissa`, start the CPU embedding endpoint, and keep it running:

```bash
conda activate lissa_env
python deploy/embedding_server.py \
  --model-path embedding_model/all_Min \
  --served-model-name all-MiniLM-L6-v2 \
  --host 127.0.0.1 \
  --port 8001
```

### 3. Configure the LLM

Use an OpenAI-compatible model service with a chat endpoint. You can run the model locally or use an existing endpoint. Environment variables specify the endpoint and model ID.

For local inference with vLLM, install a hardware-compatible version in a separate environment. Example:

```bash
vllm serve /path/to/Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --host 127.0.0.1 \
  --port 8000
```

Replace `/path/to/Qwen3.5-27B` with the actual model directory. See the [vLLM documentation](https://docs.vllm.ai/en/latest/cli/serve/) for installation and serving options.

In the terminal used to run LiSSA:

```bash
export OPENWEBUI_URL=http://127.0.0.1:8001/v1
export OPENWEBUI_API_KEY=EMPTY
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL=Qwen3.5-27B
export LLM_API_KEY=EMPTY
export LLM_ENABLE_THINKING=true
export LLM_MAX_CONCURRENCY=1
unset LLM_MAX_TOKENS
```

`OPENWEBUI_*` are the embedding endpoint variable names used by the code. `LLM_MODEL` must match the model ID exposed by the service. Set the appropriate keys if authentication is required. This example enables Qwen's thinking parameter, which must be supported by the model endpoint.

In Windows PowerShell, use `$env:LLM_MODEL="Qwen3.5-27B"` and the same syntax for other environment variables. The loops and path-variable examples below use Bash.

## Standard Experiments

### 1. Select a Dataset

| Dataset | Configuration file | Judge rows |
|---|---|---:|
| Air_Condition | `air-condition-typed-qwen-config.json` | 358 |
| Architecting_Spacecraft | `space-craft-typed-qwen-config.json` | 153 |
| Extract_Requirement_Values | `extract-value-typed-qwen-config.json` | 429 |

Configurations are under `reproduction`. In each JSON file, configure the candidate count through `target_store.args.max_results`, along with classifier `temperature`, `seed`, and prompts.

### 2. Run and Export Predictions

Run from `src/lissa`. This example performs the first Air_Condition run:

```bash
(
  set -e
  PROJECT_DIR="$(pwd -P)"
  JAR="$PROJECT_DIR/target/lissa-0.2.0-SNAPSHOT-jar-with-dependencies.jar"
  DATASET=Air_Condition
  CONFIG_NAME=air-condition-typed-qwen-config.json
  REPEAT=1
  CONFIG="$PROJECT_DIR/reproduction/$CONFIG_NAME"
  RUN_DIR="$PROJECT_DIR/result/$DATASET/repeat_$REPEAT"

  mkdir -p "$PROJECT_DIR/result/$DATASET"
  mkdir "$RUN_DIR"
  cd "$RUN_DIR"

  java -Dfile.encoding=UTF-8 -jar "$JAR" eval -c "$CONFIG" \
    > experiment.log 2>&1

  reports=(results-*.md)
  traces=(traceLinks-*.csv)
  test "${#reports[@]}" -eq 1
  test "${#traces[@]}" -eq 1
  test -f "${reports[0]}"
  test -f "${traces[0]}"

  python "$PROJECT_DIR/deploy/export_typed_results.py" \
    --results-md "${reports[0]}" \
    --trace-links "${traces[0]}" \
    --round-log experiment.log \
    --config "$CONFIG" \
    --config-working-dir "$RUN_DIR" \
    --metrics-out metrics.json \
    --mismatch-out mismatch.json \
    --predictions-out predictions.json
)
```

The working directory is `result/<dataset>/repeat_<number>`, matching the `../../../Data/...` relative paths in the configuration. To change datasets, update both `DATASET` and `CONFIG_NAME`. To repeat an experiment, change `REPEAT` and use a new output directory.

Outputs appear in `result/Air_Condition/repeat_1`:

- `predictions.json`: relation-existence and type predictions for candidate pairs.
- `metrics.json`: evaluation metrics.
- `mismatch.json`: incorrect predictions.
- `experiment.log`: execution log.
- `results-*.md` and `traceLinks-*.csv`: raw outputs from the Java program.

Retrieval determines the number of candidates, so the number of predictions may differ from the number of Judge rows.

## Oracle Experiments

Oracle uses only the LLM endpoint. Run the following commands from `src/lissa`, keeping the `LLM_*` settings above, including `LLM_ENABLE_THINKING=true` and leaving `LLM_MAX_TOKENS` unset.

### 1. Prepare Inputs and Compile the Entry Point

```bash
PROJECT_DIR="$(pwd -P)"
JAR="$PROJECT_DIR/target/lissa-0.2.0-SNAPSHOT-jar-with-dependencies.jar"
SUITE="$PROJECT_DIR/result/oracle_local"

python deploy/oracle_experiment.py prepare \
  --project "$PROJECT_DIR" --suite "$SUITE"
python deploy/oracle_experiment.py verify --suite "$SUITE"

mkdir -p target/oracle-cli
javac -encoding UTF-8 -cp "$JAR" \
  -d target/oracle-cli deploy/OracleTypeExperiment.java
```

Set `SUITE` to a new output directory. Input preparation extracts 85 positive pairs from the three Judge datasets: 45 from Air_Condition, 12 from Architecting_Spacecraft, and 28 from Extract_Requirement_Values.

### 2. Run and Evaluate

```bash
(
  set -e
  for DATASET in Air_Condition Architecting_Spacecraft Extract_Requirement_Values; do
    for ROUND in 1 2 3; do
      OUTPUT="$SUITE/$DATASET/round_$ROUND"
      java -Dfile.encoding=UTF-8 \
        -cp "$PROJECT_DIR/target/oracle-cli:$JAR" \
        OracleTypeExperiment "$SUITE" "$DATASET" "$ROUND" "$OUTPUT"
      python "$PROJECT_DIR/deploy/oracle_experiment.py" evaluate \
        --suite "$SUITE" --dataset "$DATASET" --round "$ROUND"
    done
  done
  python "$PROJECT_DIR/deploy/oracle_experiment.py" summarize --suite "$SUITE"
)
```

The Oracle evaluator reads runtime outputs from `round_1`, `round_2`, and `round_3`. Each round generates `predictions.json`, `metrics.json`, and `mismatch.json`. After all runs complete, summary metrics are written under `SUITE`.

On Windows, the Java classpath separator is `;`, so use `"$PROJECT_DIR/target/oracle-cli;$JAR"`.
