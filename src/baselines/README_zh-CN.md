# 三个基线的运行方法

中文 | [English](README.md)

以下命令均在 `src/baselines` 目录执行。Windows PowerShell 先进入：

```powershell
cd D:\multi-source-sysml-traceability\src/baselines
```

Linux 上进入你实际放置的 `src/baselines` 目录即可；下面的 `python` 命令通用。

## 1. 准备环境、模型与数据

使用 Python 3.10 或以上，建议独立环境。使用 Conda 时：

```bash
conda create -n trace_baselines python=3.12 -y
conda activate trace_baselines
```

安装依赖，确保安装与运行使用同一个 Python：

```bash
python -m pip install -r requirements.txt
```

根据机器安装适配 CUDA 的 PyTorch。LLM 服务需要事先启动；本目录只负责调用
OpenAI-compatible API。Embedding 和 Supervised 从本地加载模型，不自动下载。
Embedding 的默认运行入口使用 CUDA；没有 CUDA 时不要直接运行该入口。

安装后检查依赖和 GPU：

```bash
python -c "import openai, pandas, sklearn, matplotlib, transformers, sentence_transformers, accelerate, safetensors; import torch; print('Dependencies OK'); print('CUDA available:', torch.cuda.is_available())"
```

运行 Embedding 默认入口时，`CUDA available` 必须为 `True`；否则先修复
GPU 驱动与 PyTorch CUDA 安装。只调用 LLM API 不需要本机 GPU。

在 `experiment_config.json` 中设置模型路径、服务地址和数据集：

- `llm.model`、`llm.base_url`：LLM 服务的模型 ID 和接口地址。
- `embedding.model_path`：本地 all-MiniLM-L6-v2 目录。
- `supervised.model_path`：本地 RoBERTa-base 目录。
- 模型相对路径以 `src/baselines` 为基准，也可以填写绝对路径。

默认路径是 `models/all-MiniLM-L6-v2` 和 `models/roberta-base`，但代码包不包含权重。
可以将完整模型目录放在那里，也可以直接在配置中填写已有模型目录。
不要只放单个权重文件：需包含模型配置、tokenizer，以及 SentenceTransformer
所需的模块配置。JSON 中的 Windows 路径建议使用 `/`，如 `D:/huggingface_models/roberta-base`。

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

`KG.csv` 用于 Embedding 参考样本和 Supervised 训练，`Judge.csv` 用于主实验测试。
只有运行 Oracle 实验时才需要 `Oracle_Positive_85.csv`。

调用远程 API 时，将 `.env.example` 复制为 `.env`，填写 `LLM_API_KEY`。
不要分享包含真实密钥的 `.env`。环境变量优先于配置文件。

PowerShell 中首次创建 `.env`：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

只需将 `LLM_API_KEY=replace_with_your_api_key` 替换为真实密钥。
本地无鉴权的 vLLM 可不创建 `.env`，代码使用 `EMPTY` 占位。
切换模型时，注意 `.env` 或当前终端的 `LLM_MODEL`、`LLM_BASE_URL`
会覆盖 JSON 配置；不需要这些覆盖项时应移除，避免连错服务。

## 2. 默认参数

- Embedding：all-MiniLM-L6-v2，384 维，最大长度 256，余弦相似度，阈值 0.95。
- Supervised：RoBERTa-base，最大长度 256，训练/评估 batch size 16/32，
  AdamW，学习率 `2e-5`，cosine scheduler，warmup ratio 0.1，
  weight decay 0.01，sqrt-balanced 类别权重的交叉熵，**12 epochs**。
  默认不提前停止，训练后使用验证集 `relation_f1` 最佳 checkpoint 进行测试。
- Direct-LLM：temperature 0.0，Thinking enabled（LLaMA-3.1-8B-Instruct 不传该开关），
  请求超时 120 秒。DeepSeek 使用 `thinking.type=enabled`；
  Qwen/Gemma 使用 `chat_template_kwargs.enable_thinking=true`，服务端需支持对应模板。
  `repetition_penalty` 默认不额外设置。

## 3. 运行主实验

每条命令只运行一个方法在一个数据集上的实验，并自动执行评测：

```bash
python run_experiment.py --method llm --dataset Air_Condition
python run_experiment.py --method embedding --dataset Air_Condition --seed 42
python run_experiment.py --method supervised --dataset Air_Condition --seed 42
```

将 `--dataset` 替换为 `Architecting_Spacecraft` 或
`Extract_Requirement_Values` 即可运行另外两个数据集。

切换 LLM 时修改 `experiment_config.json` 中的 `llm.model` 和 `llm.base_url`。
本地模型名称必须与服务端的 served model name 一致；DeepSeek 的模型 ID 为
`deepseek-v4-pro`，接口为 `https://api.deepseek.com`，并需配置有效 API Key。

需要三次重复时，依次运行同一命令三次。Embedding/Supervised 可分别传入
`--seed 42`、`--seed 43`、`--seed 44`。LLM 的 `--seed` 仅标记输出目录，
不会作为采样参数发送。不要同时启动多条实验命令。

仅检查运行命令、不执行模型调用或训练：

```bash
python run_experiment.py --method supervised --dataset Air_Condition --dry-run
```

主实验输出自动创建于
`results/<方法>/<数据集>/run_<时间戳>_seed<种子>/`，
包含运行配置、日志、预测和评测指标，不覆盖已有运行。

## 4. 运行 Oracle 实验

Oracle 只对 85 条真实正例预测 Refine/Satisfy。先验证数据和代码：

```bash
python Llm_baseline/oracle_dataset.py
python Llm_baseline/test_oracle.py
```

### Embedding 与 Supervised

以下命令分别串行运行三个数据集，模型路径、阈值和训练参数与主实验共用。
Supervised 默认读取配置中的 12 epochs，随机种子默认 42，可用 `--seed` 指定：

```bash
python Oracle_local_baselines/run.py --method embedding --seed 42
python Oracle_local_baselines/run.py --method supervised --seed 42
```

需要显式指定轮数时：

```bash
python Oracle_local_baselines/run.py --method supervised --epochs 12
```

结果分别保存到对应方法目录的 `oracle_outputs/run_<时间戳>/`。

### LLM

先配置模型服务和 API Key，再逐个数据集运行。每次使用新的 `--output-dir`：

```bash
python Llm_baseline/oracle_run.py --project Air_Condition --output-dir results/Llm_baseline/oracle/run_1/Air_Condition
python Llm_baseline/oracle_run.py --project Architecting_Spacecraft --output-dir results/Llm_baseline/oracle/run_1/Architecting_Spacecraft
python Llm_baseline/oracle_run.py --project Extract_Requirement_Values --output-dir results/Llm_baseline/oracle/run_1/Extract_Requirement_Values
```

DeepSeek 可用专用入口一次串行完成全部 85 条：

```bash
python Llm_baseline/run_deepseek_oracle.py
```

该入口固定使用 `deepseek-v4-pro`；不要在环境变量中设置其他模型 ID 或本地
API 地址。结果自动保存到 `Llm_baseline/oracle/outputs/deepseek-v4-pro/run_<时间戳>/`。

Oracle 与主实验共用模型调用参数和重试函数：temperature 0.0、120 秒超时，
并读取相同的 `llm.thinking`、`llm.repetition_penalty` 及重试环境变量。
默认 `thinking: true`；设为 `null` 时，主实验和 Oracle 都不传该开关，
使用服务端默认值。LLaMA 不传 Thinking 开关。
API 超时在配置重试耗尽后，按主实验规则以相同参数额外尝试一次。
只有任务提示词、结构化输出格式和评测方式按 Oracle 任务单独处理。
Qwen/Gemma/LLaMA 使用 JSON Schema，DeepSeek 使用 JSON object；
解析失败会记录为失败，不用真实标签填补。Oracle 输出类型预测、错例、
混淆矩阵、Type Accuracy 与 Macro-F1。

## 5. 正式运行前检查与查看结果

检查输入文件是否齐全：

```bash
python -c "from pathlib import Path; files=[Path('Data')/d/f for d in ('Air_Condition','Architecting_Spacecraft','Extract_Requirement_Values') for f in ('KG.csv','Judge.csv')]; missing=[str(p) for p in files if not p.is_file()]; assert not missing, missing; print('Main experiment inputs OK')"
```

运行 Embedding/Supervised 前，检查配置中的模型路径：

```bash
python -c "from pathlib import Path; from experiment_settings import SETTINGS; paths=[Path(SETTINGS[m]['model_path']) for m in ('embedding','supervised')]; missing=[str(p) for p in paths if not p.is_dir()]; assert not missing, missing; print('Local model directories OK')"
```

运行 LLM 前，确认服务已启动、网络可达且模型 ID 匹配（只查询模型列表，不做推理）：

```bash
python -c "import sys; sys.path.insert(0,'Llm_baseline'); import config; from openai import OpenAI; ids=[m.id for m in OpenAI(base_url=config.base_url,api_key=config.api_key,timeout=120).models.list().data]; assert config.model_name in ids, (config.model_name,ids); print('Configured LLM service OK')"
```

若做 LLM Oracle，在完成第 4 节数据准备后，先做一条真实样本预检。使用新目录名：

```bash
python Llm_baseline/oracle_run.py --project Air_Condition --one-sample --fail-fast --output-dir results/Llm_baseline/oracle/preflight_1/Air_Condition
```

查看预检的 `oracle_metrics.json`，确认 `total=1`、`valid_predictions=1`、
`failed_predictions=0`，再运行全部样本。

`--dry-run` 只验证命令构造，不验证模型文件、网络或真实推理成功。
主实验启动后会打印 `Output:` 路径；运行过程写入该目录的 `run.log`，
终端暂时没有新输出不表示停止。正常完成会打印 `Completed:`。
Oracle 正常完成应包含各数据集完整样本数 45、12、28；失败时检查
`oracle_predictions.json` 中的 `error`，不能把失败或未完成运行当作正式结果。
