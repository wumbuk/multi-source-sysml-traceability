# 面向 SysML 的多源上下文增强追溯关系预测

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

[English](README.md)

本仓库包含一个面向 SysML 模型元素与需求元素之间追溯关系预测的研究原型及实验材料。整体框架融合语义丰富、知识图谱上下文、候选关系检索、端口交互信息、Ontology/SWRL 约束和大语言模型推理。

## 方法组成

1. **语义丰富**：为描述不足的模型元素生成更完整的功能语义。
2. **层次路径上下文**：利用元素 `Path` 中的模型层级信息。
3. **结构上下文**：从 SysML 知识图谱中获取邻接节点、子系统和结构依赖。
4. **交互上下文**：从 XMI 中抽取端口、连接器和信息流证据。
5. **候选关系检索**：检索与当前元素相似的节点所关联的候选边，并根据两端匹配得分排序。
6. **本体约束**：利用 SWRL 规则限定合法的候选关系空间。
7. **LLM 推理**：完成关系存在性和关系类型预测，并输出结构化 JSON。

当前基准数据主要评测 `Refine` 和 `Satisfy`，其他关系类型可通过别名映射和本体规则继续扩展。

## 目录结构

```text
.
|-- main.py                         # 端到端实验入口
|-- config.py                       # 环境变量驱动的集中配置
|-- construct_json.py               # LLM 上下文构造
|-- call_llm/                       # 单阶段和两阶段 LLM 调用
|-- enlarge_function/               # 语义丰富
|-- kg_retrieve/                    # 知识图谱构建与上下文检索
|-- Port_Information/               # XMI 端口和交互信息抽取
|-- protege_reasoner/               # Java/SWRL 本体推理器
|-- Data/                           # 数据集与划分工具
|-- result_analysis_tools/          # RQ1/RQ2 指标计算
|-- result/                         # 保留的实验指标和结果
|-- scripts/                        # 辅助评测和文档脚本
`-- docs/                           # 配置和使用文档
```

## 安装

建议使用 Python 3.12。`requirements.txt` 中的固定版本来自 Python 3.12.13
的 `rag_5060` Conda 环境：

```powershell
git clone https://github.com/wumbuk/multi-source-sysml-traceability.git
cd multi-source-sysml-traceability
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

当前环境使用 CUDA 12.8 nightly 版 PyTorch。如果目标机器没有兼容的
NVIDIA CUDA 环境，应将其替换为对应的稳定版或 CPU 版 PyTorch。

构建 Java 推理器：

```powershell
cd protege_reasoner\kg-reasoner
.\gradlew.bat jar
cd ..\..
```

## 脱敏配置

模型路径、API 地址和密钥均由 [config.py](config.py) 集中读取。首先创建本地配置：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中配置：

- `TRACE_EMBEDDING_MODEL`：向量模型 ID 或本地目录；
- `TRACE_EMBEDDING_DEVICE`：`cuda`、`cuda:0`、`cpu`，留空则自动选择；
- `TRACE_LLM_BASE_URL`：OpenAI 兼容接口地址，例如本地 vLLM；
- `TRACE_LLM_MODEL`：服务端模型名称；
- `TRACE_LLM_API_KEY`：API 密钥，本地 vLLM 通常可使用 `EMPTY`；
- `TRACE_ENRICHMENT_API_KEY`：语义丰富服务的私有密钥；
- `TRACE_REASONER_JAR_PATH`：Java 推理器 JAR 路径。

`.env.example` 只包含占位配置，真实 `.env` 已被 Git 忽略。完整说明见 [docs/CONFIGURATION.md](docs/CONFIGURATION.md)。

## 运行

默认配置使用 Air Condition 数据集：

```powershell
python main.py
```

主流程读取 `KG.csv` 和 `Judge.csv`，构建图上下文、执行本体推理、调用 LLM，并将预测结果写入 `TRACE_OUTPUT_RESULT_PATH`。

## 指标计算

```powershell
python result_analysis_tools\analyze_stats.py `
  --gold_csv Data\GenerateContext\Air_Condition\Judge.csv `
  --pred_json result\runs\latest\predictions.json `
  --mode both `
  --rq2_setting both `
  --pipeline_eval conditional `
  --output_metrics result\runs\latest\metrics.json `
  --output_mismatch result\runs\latest\mismatch.json
```

工具输出 RQ1 的 TP、FP、TN、FN、Accuracy、Precision、Recall 和 F1，以及 RQ2 的 Oracle/Pipeline Macro-F1 和 Type Accuracy。

## 上传注意事项

- 不提交 API 密钥、私有数据、模型权重和本地绝对路径。
- 不提交 Java 构建目录、临时结果及大体积原始生成日志。
- 每次论文实验应记录模型精确版本、vLLM/Python 依赖版本、GPU/驱动、解码参数和随机种子。
- 如果密钥曾经进入历史文件，应在服务商侧撤销并重新生成，仅删除源码中的明文并不能使旧密钥失效。

## 许可证

本项目采用 [GNU Affero General Public License v3.0](LICENSE)（`AGPL-3.0`）
开源许可证。
