# 消融实验 / Ablation experiments

中文 | [English](README.md)

这里的五个 `.txt` 文件是原始 `output_dict` 替换片段，内容保持不变。
它们替换的是 `construct_json.py` 中 `construct_inf_json()` 的最终输出字典，
不是整个 Python 文件，也不替换前面的 `judge_cases`、`subsystem_nodes` 构造代码。

## 单因素设置

| 实验 / Variant | 从输入中移除的字段 | 保留内容 |
| --- | --- | --- |
| `Case_Retrieval` | 顶层“可能对追溯关系判断有帮助的案例” | 其他上下文 |
| `Hierarchical_Path` | 当前模型节点和需求节点的“建模路径” | 案例和子系统节点内部的路径仍保留 |
| `Ontology_Constraint` | 顶层“模型元素和需求元素之间可能存在的关系集合” | 其他上下文 |
| `Semantic_Enrichment` | 当前模型节点的“功能描述” | 需求描述、案例和子系统节点描述仍保留 |
| `Structural_Context` | 顶层“和模型节点处于同一子系统中的节点信息” | 其他上下文 |
| Interaction Context | 所有样本均令 `is_port = False` | 最终 `output_dict` 保持完整 |

这些设置严格按所提供的五份片段实现。它们是 **LLM 输入字段消融**：
检索、语义编码和本体推理的计算仍可能执行，不能将耗时差异直接解释为
移除相应算法后的效率提升。Semantic Enrichment 设置也不是重新用未丰富数据构图。

## 方法一：通过环境变量切换（无需编辑代码）

`TRACE_ABLATION_VARIANT` 支持表中的五个精确名称；留空为完整字典。
`TRACE_USE_PORT_CONTEXT=true` 为完整方法的默认值，端口样本才附加交互上下文。
每次只设置一个消融因素，输出写到独立目录，避免覆盖其他实验。

在 `src/ours` 下的 PowerShell 中运行：

```powershell
# 完整方法。清空选择后处理整个 Judge.csv。
$env:TRACE_ABLATION_VARIANT=" "
$env:TRACE_USE_PORT_CONTEXT="true"
$env:TRACE_ORACLE_TYPE_ONLY="false"
$env:TRACE_RERUN_SEQUENCES=" "
$env:TRACE_OUTPUT_RESULT_PATH="result/runs/full/predictions.json"
$env:TRACE_SFT_SAMPLE_PATH="result/runs/full/llm_samples.jsonl"
python main.py
```

五种字段消融以 Case Retrieval 为例：

```powershell
$env:TRACE_ABLATION_VARIANT="Case_Retrieval"
$env:TRACE_USE_PORT_CONTEXT="true"
$env:TRACE_OUTPUT_RESULT_PATH="result/runs/without_case_retrieval/predictions.json"
$env:TRACE_SFT_SAMPLE_PATH="result/runs/without_case_retrieval/llm_samples.jsonl"
python main.py
```

其他四种只需替换变量值，并使用新的输出目录。

Interaction Context 消融：

```powershell
$env:TRACE_ABLATION_VARIANT=" "
$env:TRACE_USE_PORT_CONTEXT="false"
$env:TRACE_OUTPUT_RESULT_PATH="result/runs/without_interaction_context/predictions.json"
$env:TRACE_SFT_SAMPLE_PATH="result/runs/without_interaction_context/llm_samples.jsonl"
python main.py
```

此时 `main.py` 对每个样本都传入 `is_port=False`，并跳过端口输入文件解析。
Python 布尔值拼写为 `False`，不是 `false`；`.env` 中使用 `false`。
Interaction Context 通过自己的开关控制，不是 `TRACE_ABLATION_VARIANT` 的第六个值。

Shell 环境变量优先于 `.env`。示例中的一个空格会在读取时被去除，
用于显式覆盖 `.env` 中可能存在的旧选择。结束实验后重置变量或打开新终端，
避免后续完整方法实验沿用前一次消融设置。

## 方法二：手动替换最终字典

1. 保存完整版本的 `construct_json.py`。
2. 在 `construct_inf_json()` 中定位“构建最终输出字典”下面的 `output_dict = {...}`。
3. 用对应 `.txt` 的全部内容替换这一个赋值语句，保持函数内缩进。
4. 保留最后的 `return json.dumps(...)`，不要覆盖整个函数。
5. 将 `TRACE_ABLATION_VARIANT` 留空，避免与自动删字段逻辑重复执行。
6. 前五种消融保持 `TRACE_USE_PORT_CONTEXT=true`，并单独设置输出目录。
7. 下一个实验前恢复完整字典，再替换下一个片段。

手动做 Interaction Context 消融时，在 `main.py` 的 LLM 调用前将
`is_port = is_port_node(model_node) if USE_PORT_CONTEXT else False`
改为 `is_port = False` 即可；更方便的等价方法是上述环境变量开关。

## 对照实验记录

保持数据划分、模型版本、模型接口、解码设置及样本选择相同。
`TRACE_RERUN_SEQUENCES` 的编号从 2 开始，沿用 CSV 行号约定；正式全量实验留空。
不要开启 `TRACE_ORACLE_TYPE_ONLY`，除非正在专门运行已知有关系样本的类型预测实验。
每组保存 `predictions.json`、`predictions_timing.json` 及本次配置。
