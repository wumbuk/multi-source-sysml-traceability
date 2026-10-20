# 面向 SysML 的多源上下文增强追溯关系预测

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](../../LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

中文 | [English](README.md)

运行入口为 `src/ours/main.py`，预测需求与 SysML 模型元素之间的追溯关系及类型。

## 目录结构

```text
.
|-- Ablation/                       # 消融设置与运行说明
|-- main.py                         # 端到端实验入口
|-- config.py                       # 环境变量驱动的集中配置
|-- construct_json.py               # LLM 上下文构造
|-- call_llm/                       # 单阶段和两阶段 LLM 调用
|-- add_information_to_graph/       # 预测关系转换
|-- Generate_data/                  # 样本生成工具
|-- lib/                            # 图可视化资源
|-- enlarge_function/               # 语义丰富
|-- kg_retrieve/                    # 知识图谱构建与上下文检索
|-- Port_Information/               # XMI 端口和交互信息抽取
|-- protege_reasoner/               # Java/SWRL 本体推理器
|-- Data/                           # 数据集与划分工具
`-- result/                         # 运行输出
```

输入数据位于 `Data/<dataset>`，实验预测位于 [`results/ours`](../../results/ours)。

## 环境配置

以下命令在 Windows PowerShell 中执行，工作目录为 `src/ours`。
Python 主程序会调用独立运行的 LLM 服务，不会自动启动大模型服务器。

### 1. Python 与 Java

使用 Python 3.12 和 JDK 17。从仓库根目录创建 Python 环境：

```powershell
cd src/ours
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

CPU embedding 环境可按以下顺序安装：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-runtime.txt
python -m pip check
```

GPU 环境先按 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)
选择适合本机的 PyTorch，再安装 `requirements-runtime.txt`。

Java 必须在 `PATH` 中，先执行：

```powershell
python --version
java -version
```

构建目标为 Java 8 字节码；构建时使用兼容 Gradle 8.6 的 JDK，例如 JDK 17。
首次使用需要构建，在 `protege_reasoner\kg-reasoner` 中执行 `.\gradlew.bat jar`。
Wrapper 使用 Gradle 8.6；构建环境参见
[Gradle 兼容性说明](https://docs.gradle.org/8.6/userguide/compatibility.html)。
仅运行已有 JAR 不需要重新下载 Gradle 依赖。

### 2. 创建配置并准备 embedding 模型

首次配置时执行：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

编辑 `.env`，示例：

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

embedding 模型为 `sentence-transformers/all-MiniLM-L6-v2`。按模型 ID 加载时，首次需联网下载；离线运行时将 `TRACE_EMBEDDING_MODEL` 设置为本地模型目录，例如 `D:/models/all-MiniLM-L6-v2`。使用 GPU 时设置 `TRACE_EMBEDDING_DEVICE=cuda`。

`.env` 的相对路径以本目录为基准；系统和终端环境变量优先于 `.env`。
下列 PowerShell 示例用一个空格清空选择，程序会去掉两端空白；
这能覆盖 `.env` 中的旧值，避免部分 PowerShell 版本将空字符串解释为删除环境变量。
`TRACE_SINGLE_STAGE_*`、`TRACE_TWO_STAGE_*` 留空时继承 `TRACE_LLM_*`；
已有非空覆盖配置时，修改通用变量不会覆盖它们。

### 3. 确认 LLM 服务已运行

上面的 `localhost:8080` 只是默认配置，必须有独立的模型服务在该地址监听。
使用远程服务时，把地址改成实际服务器地址（通常以 `/v1` 结尾），
模型名应与服务返回的 ID 一致，密钥填写服务实际要求的值。
服务需要兼容本代码的模型列表接口、聊天接口、JSON 输出和 thinking 参数。

先测试实际使用的两阶段接口：

```powershell
python -c "from openai import OpenAI; from config import TWO_STAGE_BASE_URL,TWO_STAGE_API_KEY; c=OpenAI(base_url=TWO_STAGE_BASE_URL,api_key=TWO_STAGE_API_KEY,timeout=10,max_retries=0); print([m.id for m in c.models.list().data])"
```

只有列表中有可用模型才继续。连接被拒绝表示服务未启动或地址错误；
认证错误表示密钥配置不正确。此检查不生成预测，也不证明聊天请求一定成功。

## 运行

### 1. 准备数据并验证单个样本

上述配置使用 `Data/Air_Condition/KG.csv` 和 `Judge.csv`。
复制配置模板后，需要在 `.env` 中显式设置上述数据路径。
解析器支持带注释三段式片段，以及完整 XML 文档。
Air Condition 可使用项目中的 XMI 模型生成端口输入：

```powershell
python Port_Information/information_extract.py --xmi "Port_Information/Data/Port_Information/VehicleCCU Problem.xml" --output "Data/Port_Information/关于端口信息流动的XMI.txt"
```

其他数据集需要设置对应的 XMI 或已提取的端口输入，详见 [端口提取说明](Port_Information/README_zh-CN.md)。
完整方法默认启用端口上下文，仅端口类型样本会将其传给 LLM。

首次用单个原始样本编号验证，结果放到独立目录：

```powershell
$env:TRACE_ABLATION_VARIANT=" "
$env:TRACE_USE_PORT_CONTEXT="true"
$env:TRACE_ORACLE_TYPE_ONLY="false"
$env:TRACE_RERUN_SEQUENCES="2"
$env:TRACE_OUTPUT_RESULT_PATH="result/runs/smoke/predictions.json"
$env:TRACE_SFT_SAMPLE_PATH="result/runs/smoke/llm_samples.jsonl"
python main.py
```

编号从 2 开始，对应 CSV 的原始行号。

### 2. 全量运行与检查结果

单样本成功后，清空样本选择并更换输出目录：

```powershell
$env:TRACE_RERUN_SEQUENCES=" "
$env:TRACE_OUTPUT_RESULT_PATH="result/runs/full/predictions.json"
$env:TRACE_SFT_SAMPLE_PATH="result/runs/full/llm_samples.jsonl"
python main.py
```

输出目录自动创建：

- `predictions.json`：预测列表，包含原始“序号”、关系存在性、类型及解释等字段。
- `predictions_timing.json`：离线耗时和逐样本在线耗时。
- `llm_samples.jsonl`：按上述配置保存两阶段请求记录。

默认 Air Condition 的 `Judge.csv` 有 358 行，全量正常完成应输出 358 条预测。
每处理一个样本会保存进度。异常退出时，已完成内容还会备份到
`temporary_results/<时间戳>/`；这不代表剩余样本已经完成，也不是自动续跑功能。
重复使用同一个预测输出路径会覆盖此前结果；JSONL 请求记录会追加。

检查预测数量：

```powershell
python -c "import json; from pathlib import Path; p=Path('result/runs/full/predictions.json'); print('Predictions:', len(json.loads(p.read_text(encoding='utf-8'))))"
```

切换数据集时同时更新 `.env` 中的 `TRACE_DATASET_DIR`、
`TRACE_KG_CSV_PATH`、`TRACE_JUDGE_CSV_PATH`，以及对应的端口输入路径；
`.env.example` 中显式填写的 CSV 路径不会随数据集目录自动改变。
仅在需要重新做语义丰富或数据划分时开启对应 `TRACE_RUN_*` 开关，
并配置语义丰富服务和输入/输出路径。

## 消融实验

消融配置见 [Ablation](Ablation/README_zh-CN.md)：

| 设置 | 对应片段 / 操作 |
| --- | --- |
| Case Retrieval | [Case_Retrieval.txt](Ablation/Case_Retrieval.txt) |
| Hierarchical Path | [Hierarchical_Path.txt](Ablation/Hierarchical_Path.txt) |
| Ontology Constraint | [Ontology_Constraint.txt](Ablation/Ontology_Constraint.txt) |
| Semantic Enrichment | [Semantic_Enrichment.txt](Ablation/Semantic_Enrichment.txt) |
| Structural Context | [Structural_Context.txt](Ablation/Structural_Context.txt) |
| Interaction Context | 设置 `TRACE_USE_PORT_CONTEXT=false`，使 `is_port` 恒为 `False` |

五份片段只替换 `construct_json.py` 中最终的 `output_dict` 赋值。
也可不编辑代码，直接设置 `TRACE_ABLATION_VARIANT=Case_Retrieval` 等五个精确名称；
留空恢复完整字典。手动替换时必须留空此变量，避免重复删字段。
Interaction Context 消融不替换字典，保持 `TRACE_ABLATION_VARIANT` 为空即可。

每次只改变一个因素，保留相同数据、模型和样本，并使用独立的结果目录。
这些字段消融不跳过检索或本体计算；层次路径和语义丰富的删除范围严格遵循片段，
不会额外清除案例或子系统节点中的相似字段。
完整 PowerShell 命令及替换位置见 [消融操作说明](Ablation/README_zh-CN.md)。
