# LiSSA：环境配置与本地运行

中文 | [English](README.md)

## 项目结构

```text
src/lissa/
├── src/                            # Java 源码与测试
├── Data/
│   ├── Air_Condition/
│   ├── Architecting_Spacecraft/
│   └── Extract_Requirement_Values/
├── reproduction/                   # 各数据集的运行配置
├── embedding_model/all_Min/        # all-MiniLM-L6-v2 模型目录
├── deploy/
│   ├── embedding_server.py         # 本地 embedding 接口
│   ├── export_typed_results.py     # 导出预测与评估结果
│   ├── oracle_experiment.py        # Oracle 输入准备与评估
│   └── OracleTypeExperiment.java   # Oracle 类型预测入口
├── pom.xml                         # Maven 构建配置
├── LICENSE
├── README.md
└── README_zh-CN.md
```

每个数据集的 `Judge.csv` 包含待判断的元素对与真实标签。普通实验执行候选检索和关系预测；Oracle 实验只对真实正例预测 `Refine` 或 `Satisfy`。

## 环境配置

需要 JDK 21、Maven 和 Python 3.10 及以上。以下命令使用 Bash，开始时位于仓库根目录。

### 1. 安装环境并构建

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

### 2. 准备 embedding 模型

在 `src/lissa` 下下载模型：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', local_dir='embedding_model/all_Min')"
```

如果已有完整模型目录，将下方 `--model-path` 改为实际路径。在单独终端中激活环境、进入 `src/lissa`，启动 CPU embedding 接口，并保持终端运行：

```bash
conda activate lissa_env
python deploy/embedding_server.py \
  --model-path embedding_model/all_Min \
  --served-model-name all-MiniLM-L6-v2 \
  --host 127.0.0.1 \
  --port 8001
```

### 3. 配置 LLM

使用提供聊天接口的 OpenAI-compatible 模型服务。模型可以在本机运行，也可以使用已有接口，地址和模型名称由环境变量指定。

若使用 vLLM 在本机加载模型，在独立环境中安装适合硬件的 vLLM，启动示例为：

```bash
vllm serve /path/to/Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --host 127.0.0.1 \
  --port 8000
```

将 `/path/to/Qwen3.5-27B` 替换为实际权重目录。安装和启动参数参见 [vLLM 文档](https://docs.vllm.ai/en/latest/cli/serve/)。

在运行 LiSSA 的终端设置：

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

`OPENWEBUI_*` 是代码中 embedding 接口的变量名。`LLM_MODEL` 应与接口提供的模型 ID 一致；需要认证时填写对应密钥。示例启用 Qwen 的 thinking 参数，模型接口需要支持该参数。

Windows PowerShell 中环境变量写法为 `$env:LLM_MODEL="Qwen3.5-27B"`，其他变量同理。下文的循环和路径变量示例在 Bash 中执行。

## 普通实验

### 1. 选择数据集

| 数据集 | 配置文件 | Judge 行数 |
|---|---|---:|
| Air_Condition | `air-condition-typed-qwen-config.json` | 358 |
| Architecting_Spacecraft | `space-craft-typed-qwen-config.json` | 153 |
| Extract_Requirement_Values | `extract-value-typed-qwen-config.json` | 429 |

配置位于 `reproduction`。可在 JSON 中设置候选数量 `target_store.args.max_results`，以及分类器的 `temperature`、`seed` 和提示词。

### 2. 运行并导出预测

在 `src/lissa` 下执行。示例运行 Air_Condition 的第一次实验：

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

工作目录为 `result/<数据集>/repeat_<编号>`，与配置中 `../../../Data/...` 的相对路径对应。切换数据集时同时修改 `DATASET` 和 `CONFIG_NAME`；重复实验修改 `REPEAT`，并使用新的输出目录。

结果位于 `result/Air_Condition/repeat_1`：

- `predictions.json`：候选元素对的关系存在性和类型预测；
- `metrics.json`：评估指标；
- `mismatch.json`：错例；
- `experiment.log`：运行日志；
- `results-*.md`、`traceLinks-*.csv`：Java 程序生成的原始结果。

候选数量受检索阶段影响，因此 `predictions.json` 的条数可能与 Judge 行数不同。

## Oracle 实验

Oracle 只使用 LLM 接口。以下命令在 `src/lissa` 下执行，沿用前面的 `LLM_*` 环境变量，其中 `LLM_ENABLE_THINKING=true`，`LLM_MAX_TOKENS` 保持未设置。

### 1. 准备输入并编译入口

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

`SUITE` 指向一个新的输出目录。输入准备从三个 Judge 数据集中提取共 85 条正例：Air_Condition 45 条、Architecting_Spacecraft 12 条、Extract_Requirement_Values 28 条。

### 2. 运行与评估

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

Oracle 评估工具按 `round_1`、`round_2`、`round_3` 读取本次运行输出。每轮生成 `predictions.json`、`metrics.json` 和 `mismatch.json`，全部运行完成后在 `SUITE` 下生成汇总指标。

Windows 的 Java classpath 分隔符使用 `;`，对应写法为 `"$PROJECT_DIR/target/oracle-cli;$JAR"`。
