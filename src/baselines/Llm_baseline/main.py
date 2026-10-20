# LLM基线实验主程序：
# 1) 读取测试样本并构造提示词调用LLM
# 2) 解析LLM输出并转换为统一标签
# 3) 统计TP/FP/FN/TN与核心评估指标并落盘
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from openai import OpenAI

from config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST_PATH,
    LLM_PREDICTIONS_FILENAME,
    api_key,
    base_url,
    model_name,
    request_extra_body,
)


RETRYABLE_OPENAI_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
MAX_API_RETRIES = int(os.getenv("LLM_MAX_API_RETRIES", "0"))
RETRY_BASE_DELAY_SECONDS = float(os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "2.0"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))


def create_completion_with_retry(client, messages, thinking_type: str = "", *,
                                 model=None, response_format=None):
    # Both Direct-LLM and Oracle use these exact inference/retry settings.
    # thinking_type is retained for older callers; config controls Thinking.
    selected_model = model or model_name
    extra_body = request_extra_body(selected_model)
    request = {
        "model": selected_model,
        "messages": messages,
        "timeout": REQUEST_TIMEOUT_SECONDS,
        "temperature": 0.0,
    }
    if extra_body:
        request["extra_body"] = extra_body
    if response_format is not None:
        request["response_format"] = response_format
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            return client.chat.completions.create(**request)
        except RETRYABLE_OPENAI_ERRORS as exc:
            if attempt >= MAX_API_RETRIES:
                raise
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
            print(
                f"OpenAI request failed ({type(exc).__name__}). "
                f"Retrying in {delay:.0f}s "
                f"[{attempt + 1}/{MAX_API_RETRIES}]..."
            )
            time.sleep(delay)


def call_llm_api(input_text):
    # Send the prompt to the configured model and return its text response.
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        max_retries=0,
    )
    messages = [{"role": "user", "content": input_text}]
    try:
        try:
            response = create_completion_with_retry(client, messages, "enabled")
        except APITimeoutError:
            print(
                "Request timed out after all retries. "
                "Retrying once with the same model settings..."
            )
            response = create_completion_with_retry(client, messages, "disabled")

        content = response.choices[0].message.content
        return content.strip() if content else None
    except NotFoundError:
        print(f"Model {model_name} not found. Please check the model name and try again.")
        return None
    except RETRYABLE_OPENAI_ERRORS as exc:
        print(
            f"OpenAI request failed after {MAX_API_RETRIES} retries: "
            f"{type(exc).__name__}: {exc}"
        )
        return None

def normalize_text(value) -> str:
    # 模块2：文本标准化模块
    # 职责：统一处理None/NaN/空白，避免后续比较出错。
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


NO_RELATION_TOKENS = {
    "",
    "0",
    "null",
    "none",
    "nan",
    "n/a",
    "na",
    "false",
    "no relation",
    "no_relation",
    "no-relation",
    "unrelated",
    "does not exist",
    "not exist",
    "not exists",
    "no traceability relationship",
    "no traceability relation",
    "\u65e0\u5173\u7cfb",
    "\u4e0d\u5b58\u5728",
    "\u4e0d\u5b58\u5728\u5173\u7cfb",
    "\u6ca1\u6709\u5173\u7cfb",
}


def has_relation(label: str) -> bool:
    # Binary gold-label rule: 0/empty/no-relation tokens are negative; relation labels are positive.
    text = normalize_text(label)
    return text.lower() not in NO_RELATION_TOKENS


def has_valid_pred_relation(relation_type) -> bool:
    # A positive prediction must provide a concrete relation type; true + null is treated as no relation.
    text = normalize_text(relation_type)
    return text.lower() not in NO_RELATION_TOKENS


def normalize_relation_type(type_text: str) -> str:
    # 模块4：关系类型归一化模块
    # 职责：将模型输出中的关系类型统一到 Refine/Satisfy 等规范写法。
    text = normalize_text(type_text)
    low = text.lower()
    if "refine" in low:
        return "Refine"
    if "satisfy" in low:
        return "Satisfy"
    return text


def extract_relation_type(label: str) -> str:
    # 模块5：从真实标签提取关系类型模块
    # 例如：1/Refine -> Refine；非正样本 -> 无关系。
    label_text = normalize_text(label)
    if not has_relation(label_text):
        return "无关系"
    if "/" in label_text:
        _, relation_type = label_text.split("/", 1)
        relation_type = normalize_relation_type(relation_type)
        if relation_type:
            return relation_type
    return "有关系"


def parse_llm_response(response_text: str):
    # 模块6：LLM输出解析模块
    # 职责：兼容JSON/非JSON输出，抽取 是否存在追溯关系、选择的关系、推理过程、决定关系的原因、置信度。
    text = normalize_text(response_text)
    default_result = {
        "是否存在追溯关系": False,
        "选择的关系": None,
        "推理过程": "",
        "决定关系的原因": "",
        "置信度": 0.0,
    }
    if not text:
        return default_result

    json_match = re.search(r"\{[\s\S]*\}", text)
    candidate = json_match.group(0) if json_match else text

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # 无法解析JSON时，尝试从文本中推断
        relation_low = text.lower()
        if "存在" in text and "不" not in text:
            default_result["是否存在追溯关系"] = True
            default_result["选择的关系"] = normalize_relation_type(text)
        default_result["推理过程"] = text
        return default_result

    # 解析新格式字段：是否存在追溯关系 / 选择的关系 / 推理过程 / 决定关系的原因 / 置信度
    exists = parsed.get("是否存在追溯关系")
    if isinstance(exists, bool):
        pred_has_relation = exists
    elif isinstance(exists, str):
        pred_has_relation = exists.lower() in {"true", "存在", "yes", "1"}
    else:
        # 兼容旧格式：relation 字段
        relation_text = normalize_text(parsed.get("relation", ""))
        relation_low = relation_text.lower()
        if ("不存在" in relation_text) or ("不" in relation_text and "存在" in relation_text):
            pred_has_relation = False
        elif "存在" in relation_text or relation_low in {"true", "yes", "1"}:
            pred_has_relation = True
        else:
            pred_has_relation = False

    # 选择的关系
    rel_type = parsed.get("选择的关系") or parsed.get("type") or parsed.get("relation_type")
    if rel_type is None or normalize_text(str(rel_type)).lower() in {"null", "none", ""}:
        relation_type = None
    else:
        relation_type = normalize_relation_type(str(rel_type))

    # 推理过程
    reasoning = normalize_text(str(parsed.get("推理过程", parsed.get("reasoning", ""))))

    # 决定关系的原因
    reason = normalize_text(str(parsed.get("决定关系的原因", parsed.get("reason", ""))))

    # 置信度
    confidence_raw = parsed.get("置信度", parsed.get("confidence", 0.0))
    try:
        confidence = float(confidence_raw)
    except (ValueError, TypeError):
        confidence = 0.0

    if not pred_has_relation:
        relation_type = None

    return {
        "是否存在追溯关系": pred_has_relation,
        "选择的关系": relation_type,
        "推理过程": reasoning,
        "决定关系的原因": reason,
        "置信度": confidence,
    }


def compute_experiment_metrics(df: pd.DataFrame):
    # 模块7：实验指标统计模块
    # 计算：TP/FP/FN/TN、Accuracy、Precision、Recall、F1、Relation_Type_accuracy。
    tp = int(((df["true_has_relation"] == 1) & (df["pred_has_relation"] == 1)).sum())
    fp = int(((df["true_has_relation"] == 0) & (df["pred_has_relation"] == 1)).sum())
    fn = int(((df["true_has_relation"] == 1) & (df["pred_has_relation"] == 0)).sum())
    tn = int(((df["true_has_relation"] == 0) & (df["pred_has_relation"] == 0)).sum())

    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    tp_mask = (df["true_has_relation"] == 1) & (df["pred_has_relation"] == 1)
    if int(tp_mask.sum()) > 0:
        relation_type_accuracy = float(
            (df.loc[tp_mask, "pred_relation_type"] == df.loc[tp_mask, "true_relation_type"]).mean()
        )
    else:
        relation_type_accuracy = 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Relation_Type_accuracy": float(relation_type_accuracy),
    }


def main():
    # 模块8：实验主流程模块
    # 步骤：读数据 -> 构造prompt并调用LLM -> 解析预测 -> 统计指标 -> 保存结果。
    data = pd.read_csv(DEFAULT_TEST_PATH).copy()

    true_has_relation_list = []
    pred_has_relation_list = []
    true_relation_type_list = []
    pred_relation_type_list = []
    raw_response_list = []

    llm_predictions = []

    for index, row in data.iterrows():
        # 逐样本推理：构造提示词并调用LLM。
        # prompt = f"""你是一名SysML建模工程师，请根据以下信息判断需求和SysML模型元素之间是否存在追溯关系，如果存在是哪一种
        # 元素名称：{row['NameA']}
        # 元素类型：{row['typeA']}
        # 元素建模路径：{row['pathA']}
        # 元素属性：{row['propertyA']}
        # 元素名称：{row['NameB']}
        # 元素类型：{row['typeB']}
        # 元素属性：{row['propertyB']}
        # 需求描述：{row['contextB']}
        # 请判断元素A和元素B之间是否存在追溯关系。
        # 目前考虑的追溯关系：Refine,Satisfy

        # 注意事项：
        # 1.不要强行匹配标签中的关系，如果根据输入信息无法判断存在关系，请输出"不存在"。
        # 2.如果存在关系，请判断是Refine还是Satisfy，输出对应的关系类型。
        # 3.请给出你的推理过程和决定关系的原因，以及你的置信度（0-1之间的小数）。
        # 4.用中文回答

        # 输出JSON格式:
        # {{
        #     "是否存在追溯关系": true或false,
        #     "选择的关系": "Refine"或"Satisfy"或null,
        #     "推理过程": "...",
        #     "决定关系的原因": "...",
        #     "置信度": 0.95
        # }}
        # """

        prompt = f"""You are a SysML modeling engineer. Based on the following information, determine whether a traceability relationship exists between the requirement and the SysML model element, and if so, which type.

        Element Name: {row['NameA']}
        Element Type: {row['typeA']}
        Element Modeling Path: {row['pathA']}
        Element Properties: {row['propertyA']}
        Element Name: {row['NameB']}
        Element Type: {row['typeB']}
        Element Properties: {row['propertyB']}
        Requirement Description: {row['contextB']}

        Determine whether a traceability relationship exists between Element A and Element B.
        Currently considered traceability relationships: Refine, Satisfy

        Important Notes:
        1. Do not force a match with the labels; if you cannot determine a relationship based on the provided information, output "does not exist".
        2. If a relationship exists, determine whether it is Refine or Satisfy, and output the corresponding relationship type.
        3. Provide your reasoning process, the reason for your decision, and your confidence level (a decimal between 0 and 1).
        4. Answer in English.

        Think step by step:
        1. First, analyze the Element A properties and what it represents in SysML.
        2. Then, analyze Element B (the requirement) and its context.
        3. Compare them to determine if one refines or satisfies the other.
        4. Finally, state your conclusion.

        Output JSON format:
        {{
            "是否存在追溯关系": true or false,
            "选择的关系": "Refine" or "Satisfy" or null,
            "推理过程": "...",
            "决定关系的原因": "...",
            "置信度": 0.95
        }}
        """ 
        llm_response = call_llm_api(prompt)
        print(f"LLM response for index {index}: {llm_response}")

        true_label = normalize_text(row.get("Relation"))
        # 真实标签标准化：同时得到二分类标签和关系类型标签。
        true_has_relation = 1 if has_relation(true_label) else 0
        true_relation_type = extract_relation_type(true_label)

        # 预测结果标准化：将LLM原始输出映射到统一标签空间。
        pred_result = parse_llm_response(llm_response)
        pred_has_relation_bool = pred_result["\u662f\u5426\u5b58\u5728\u8ffd\u6eaf\u5173\u7cfb"]
        pred_relation_type = (
            pred_result["\u9009\u62e9\u7684\u5173\u7cfb"]
            if pred_result["\u9009\u62e9\u7684\u5173\u7cfb"]
            else "\u65e0\u5173\u7cfb"
        )
        pred_has_relation = 1 if pred_has_relation_bool and has_valid_pred_relation(pred_relation_type) else 0

        true_has_relation_list.append(true_has_relation)
        pred_has_relation_list.append(pred_has_relation)
        true_relation_type_list.append(true_relation_type)
        pred_relation_type_list.append(pred_relation_type)
        raw_response_list.append(normalize_text(llm_response))

        # 收集LLM预测的完整JSON记录
        llm_predictions.append({
            "是否存在追溯关系": pred_result["是否存在追溯关系"],
            "选择的关系": pred_result["选择的关系"],
            "推理过程": pred_result["推理过程"],
            "决定关系的原因": pred_result["决定关系的原因"],
            "置信度": pred_result["置信度"],
        })

    data["true_has_relation"] = true_has_relation_list
    data["pred_has_relation"] = pred_has_relation_list
    data["true_relation_type"] = true_relation_type_list
    data["pred_relation_type"] = pred_relation_type_list
    data["llm_raw_response"] = raw_response_list

    data["confusion_flag"] = data.apply(
        # 为每条样本标注混淆矩阵类别，便于误差分析。
        lambda x: "TP" if x["true_has_relation"] == 1 and x["pred_has_relation"] == 1
        else ("FP" if x["true_has_relation"] == 0 and x["pred_has_relation"] == 1
              else ("FN" if x["true_has_relation"] == 1 and x["pred_has_relation"] == 0 else "TN")),
        axis=1,
    )

    metrics = compute_experiment_metrics(data)

    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_csv_path = output_dir / "llm_judge_predictions.csv"
    metrics_json_path = output_dir / "llm_judge_metrics.json"

    # 输出明细与汇总结果，支持复现实验与后续分析。
    data.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
    metrics_json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存LLM预测的完整JSON记录（供后续解析工具使用）
    llm_json_path = output_dir / LLM_PREDICTIONS_FILENAME
    llm_json_path.write_text(json.dumps(llm_predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved predictions to: {result_csv_path}")
    print(f"Saved metrics to: {metrics_json_path}")
    print(f"Saved LLM predictions JSON to: {llm_json_path}")
    print("Experiment metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
