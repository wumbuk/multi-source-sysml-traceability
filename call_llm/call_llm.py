import json
import re
import time
from datetime import datetime
from openai import OpenAI  # 假设你本地有 qwen 的 OpenAI 接口
from openai import APITimeoutError
from openai import NotFoundError
from config import (
    SFT_SAMPLE_PATH,
    SINGLE_STAGE_API_KEY,
    SINGLE_STAGE_BASE_URL,
    SINGLE_STAGE_MODEL_NAME,
    TWO_STAGE_API_KEY,
    TWO_STAGE_BASE_URL,
    TWO_STAGE_MODEL_NAME,
    TWO_STAGE_TIMEOUT_SECONDS,
)


def _append_sft_sample(record):
    try:
        SFT_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **record,
        }
        with SFT_SAMPLE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[SFT sample save failed] {e}")


def _strip_markdown_fence(text: str) -> str:
    """
    去除 ```json / ``` 包裹，避免影响 JSON 解析。
    """
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    return text.rstrip("`").strip()


def _escape_unescaped_quotes_in_strings(s: str) -> str:
    """
    修复字符串内部未转义的双引号：
    当处于 JSON 字符串内，遇到一个双引号且其后不是合法字符串终止符，
    视为内容引号并自动转义为 \"。
    """
    out = []
    in_string = False
    escaped = False
    n = len(s)
    i = 0

    while i < n:
        ch = s[i]

        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue

        if ch == "\\":
            out.append(ch)
            escaped = True
            i += 1
            continue

        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
                i += 1
                continue

            # 在字符串中遇到双引号：判断它是不是字符串结束符
            j = i + 1
            while j < n and s[j].isspace():
                j += 1

            # 若后继字符不是 JSON 字符串结束后的合法分隔符，则按内容引号处理
            if j < n and s[j] not in [",", "}", "]", ":"]:
                out.append(r'\"')
                i += 1
                continue

            in_string = False
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _escape_control_chars_in_strings(s: str) -> str:
    """
    修复 JSON 字符串中的原始控制字符。
    LLM 常会在字符串里直接输出真实换行，这在 JSON 中是非法的。
    """
    out = []
    in_string = False
    escaped = False

    for ch in s:
        if escaped:
            out.append(ch)
            escaped = False
            continue

        if ch == "\\":
            out.append(ch)
            escaped = True
            continue

        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue

        if in_string:
            code = ord(ch)
            # JSON 字符串中不允许出现 0x00-0x1F 的原始控制字符
            if code < 0x20:
                if ch == "\n":
                    out.append(r"\n")
                elif ch == "\r":
                    out.append(r"\r")
                elif ch == "\t":
                    out.append(r"\t")
                elif ch == "\b":
                    out.append(r"\b")
                elif ch == "\f":
                    out.append(r"\f")
                else:
                    out.append(f"\\u{code:04x}")
                continue

        out.append(ch)

    return "".join(out)


def _repair_common_json_issues(text: str) -> str:
    """
    修复常见的 LLM JSON 格式问题。
    """
    fixed = _strip_markdown_fence(text)
    # 去除对象/数组结束前的尾逗号
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    # 修复字符串中的原始换行/回车/制表符
    fixed = _escape_control_chars_in_strings(fixed)
    # 修复字符串内部裸双引号
    fixed = _escape_unescaped_quotes_in_strings(fixed)
    # 再做一遍控制字符修复，避免引号修复后字符串边界变化带来的遗漏
    fixed = _escape_control_chars_in_strings(fixed)
    return fixed


def _parse_bool(value_text: str):
    val = value_text.strip().lower()
    if val == "true":
        return True
    if val == "false":
        return False
    return None


def _parse_loose_json_fields(text: str):
    """
    当标准 JSON 解析失败时，尝试按字段名做宽松提取。
    目标是避免下游拿到 None。
    """
    if not text:
        return None

    cleaned = _strip_markdown_fence(text)
    block = _extract_first_json_like_block(cleaned) or cleaned

    bool_match = re.search(r'"是否存在追溯关系"\s*:\s*(true|false)', block, flags=re.IGNORECASE)
    relation_null_match = re.search(r'"选择的关系"\s*:\s*null', block, flags=re.IGNORECASE)
    relation_str_match = re.search(r'"选择的关系"\s*:\s*"([\s\S]*?)"', block)
    reason_match = re.search(r'"决定关系的原因"\s*:\s*"([\s\S]*?)"', block)
    process_match = re.search(r'"推理过程"\s*:\s*"([\s\S]*?)"\s*,\s*"决定关系的原因"', block)
    conf_match = re.search(r'"置信度"\s*:\s*([0-9]+(?:\.[0-9]+)?)', block)

    exists = _parse_bool(bool_match.group(1)) if bool_match else False
    relation = None
    if relation_str_match:
        relation = relation_str_match.group(1)
    elif relation_null_match:
        relation = None

    process = process_match.group(1) if process_match else ""
    reason = reason_match.group(1) if reason_match else ""
    confidence = float(conf_match.group(1)) if conf_match else 0.0

    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0

    return {
        "是否存在追溯关系": bool(exists),
        "选择的关系": relation,
        "推理过程": process,
        "决定关系的原因": reason,
        "置信度": confidence,
    }


def _normalize_answer_shape(answer):
    """
    统一输出结构，确保下游字段可用。
    """
    if not isinstance(answer, dict):
        return {
            "是否存在追溯关系": False,
            "选择的关系": None,
            "推理过程": "",
            "决定关系的原因": "解析失败，使用默认结果",
            "置信度": 0.0,
        }

    normalized = {
        "是否存在追溯关系": bool(answer.get("是否存在追溯关系", False)),
        "选择的关系": answer.get("选择的关系", None),
        "推理过程": str(answer.get("推理过程", "")),
        "决定关系的原因": str(answer.get("决定关系的原因", "")),
        "置信度": answer.get("置信度", 0.0),
    }

    try:
        normalized["置信度"] = float(normalized["置信度"])
    except (TypeError, ValueError):
        normalized["置信度"] = 0.0

    if normalized["置信度"] < 0:
        normalized["置信度"] = 0.0
    if normalized["置信度"] > 1:
        normalized["置信度"] = 1.0

    if normalized["选择的关系"] is not None:
        normalized["选择的关系"] = str(normalized["选择的关系"])

    return normalized


def _extract_first_json_like_block(text: str):
    """
    提取文本中的首个 JSON 风格块（对象或数组），支持不完全合法 JSON 的先提取后修复。
    """
    if not text:
        return None

    start = None
    for idx, ch in enumerate(text):
        if ch in "[{":
            start = idx
            break

    if start is None:
        return None

    stack = []
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if escaped:
            escaped = False
            continue

        if ch == "\\":
            if in_string:
                escaped = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                continue
            left = stack.pop()
            if (left == "{" and ch != "}") or (left == "[" and ch != "]"):
                return None
            if not stack:
                return text[start:i + 1]

    return None


def _try_parse_json_with_repair(text: str):
    """
    尝试解析 JSON，失败时做一次轻量修复后再解析。
    """
    if not text:
        return None

    candidate = _strip_markdown_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    candidate = _repair_common_json_issues(candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 最后尝试：从候选文本中提取首个 JSON 块后再次修复解析
    block = _extract_first_json_like_block(candidate)
    if block:
        repaired_block = _repair_common_json_issues(block)
        try:
            return json.loads(repaired_block)
        except json.JSONDecodeError:
            return None

    return None


def _remaining_timeout_seconds(start_time, timeout_seconds):
    if timeout_seconds is None:
        return None

    remaining = timeout_seconds - (time.monotonic() - start_time)
    if remaining <= 0:
        raise TimeoutError(f"两阶段调用超过时间限制: {timeout_seconds} 秒")
    return remaining

#单阶段法调用，防止vLLM的thinking模式破坏严格引导的JSON输出
def _call_qwen_single_stage(information_json,is_port,port_information):
    client = OpenAI(
        base_url=SINGLE_STAGE_BASE_URL,
        api_key=SINGLE_STAGE_API_KEY,
    )
    if is_port:
        dynamic_port_information = f"""
    ========================
    【系统内各端口之间的信息流向】
    ========================
    {port_information}
    """
    else:
        dynamic_port_information = ""

    prompt = f"""
    你是一个系统工程与模型驱动工程（MBSE）专家，专注于SysML模型元素与需求之间的语义追溯关系分析。

    你的任务是判断“模型节点”和“需求节点”之间是否存在traceability relationship，并在存在时，从给定候选关系集合中选择最合适的关系类型。

    ========================
    【输入信息】
    ========================
    {information_json}

    {dynamic_port_information}

    ========================
    【任务说明】
    ========================
    本任务是基于SysML语义的追溯关系判定问题，而不是简单的文本匹配或关键词匹配。

    你需要综合以下信息进行判断：
    - 模型节点信息（功能、属性、结构）
    - 需求节点信息（功能约束或系统要求）
    - 同子系统节点信息（用于理解系统级功能语义背景）
    - 相似案例（用于学习历史映射模式）


    ========================
    【推理步骤要求】
    ========================
    请按照以下逻辑逐层推理：

    (1) 模型语义解析
        - 提取模型节点的核心功能语义（不是名称）
        - 判断该节点在子系统中的作用（性能指标 / 功能实现 / 控制 / 结构属性等）
        - 判断该节点属于：
            - 功能执行节点
            - 功能输入节点
            - 功能输出节点
            - 用户交互节点
            - 状态感知节点
            - 控制触发节点
            - 中间传输节点
            - 普通内部通信节点

    (2) 子系统语义背景建模
        - 基于同子系统节点信息，抽象该子系统整体功能
        - 注意：子系统仅提供上下文，不决定最终关系

    (3) 需求语义解析
        - 提取需求节点的约束类型（性能 / 功能 / 限制 / 行为 / 用户交互）
        - 提取需求真正关注的目标

    (4) 语义映射判断（核心）
        判断模型节点是否对需求节点存在以下任一作用：

        - 约束（constraint）
        - 细化（refinement）
        - 实现/满足（satisfaction）
        - 验证支撑（verification）
        - 派生关系（derivation）
        - 弱关联（trace）

        ⚠️ 注意：
        - 必须基于“模型节点 → 需求节点”的语义作用方向判断
        - 子系统一致性 ≠ 必然存在追溯关系
        - 名称相似不构成关系

    (5) 功能贡献层级分析（非常重要）

        不要仅判断模型节点是否与需求“相关”。

        需要进一步判断：

        该模型节点在系统功能实现过程中，
        是否对需求目标形成“关键语义贡献”。

        请区分以下两类节点：

        ① 普通中间传输节点
           特点：
           - 仅负责内部数据转发
           - 仅负责内部通信
           - 不决定功能是否发生
           - 去掉后系统功能仍可能成立
           - 与需求只有间接关联

           → 通常不构成traceability relationship

        ② 功能关键贡献节点
           特点：
           - 是功能执行的重要入口
           - 是用户意图进入系统的重要接口
           - 是功能触发条件
           - 是系统状态感知前提
           - 是核心功能执行步骤
           - 是最终功能输出接口
           - 缺少该节点后，需求目标无法有效实现

           → 可以构成traceability relationship

    (6) 案例参考机制
        - 优先参考高置信案例中的结构映射模式
        - 案例仅用于辅助关系类型判断
        - 不允许机械匹配案例

    (7) 关系约束
        - 最终选择的关系必须严格来自“模型元素和需求元素之间可能存在的关系集合”
        - 不允许生成未定义关系类型

    ========================
    【端口类型补充规则】
    ========================
    仅当模型节点属于 Port 类型（如 Control_port / Information_port / Physical_port）时，才使用以下规则。

    1. port_information 的作用

        port_information 用于帮助理解：

        - 端口实际传输的内容
        - 信息/控制/物理流方向
        - 端口所属系统
        - 端口在系统功能链路中的位置

        ⚠️ 注意：
        port_information 只是辅助理解端口语义，
        不能直接作为存在追溯关系的依据。

    2. 端口名称可能重复

        必须结合以下信息唯一确定端口：

        - 端口名称
        - 所属系统
        - Path路径

        不允许仅根据端口名判断。

    3. 必须分析完整功能链路，而不是单条信息流

        不要只看某一个flow。

        应尝试分析：

        输入什么
        ↓
        系统如何处理
        ↓
        输出什么
        ↓
        最终实现什么功能

        即：

        功能链路（functional chain）
        信息链路（information flow chain）
        控制链路（control chain）

    4. 分析端口在链路中的语义角色

        例如：

        - 用户输入接口
        - 功能输入接口
        - 功能输出接口
        - 控制触发接口
        - 状态反馈接口
        - 感知数据接口
        - 中间路由接口
        - 内部控制转发接口

    5. 区分“用户控制”与“系统内部控制”

        如果某端口只是接收系统内部模块处理后的控制信号，
        那么它可能只是内部执行接口。

        这种情况下：

        - 它可能与最终功能有关
        - 但不一定直接对应用户交互需求

        因此不能仅因为它参与控制链路，
        就直接判定存在traceability relationship。

    6. 判断端口是否属于功能关键环节

        如果端口属于以下情况之一：

        - 用户意图进入系统的关键入口
        - 功能执行的重要触发接口
        - 系统感知外部状态的重要入口
        - 核心功能输入/输出接口
        - 完整功能链路中的关键功能节点

        则该端口可能与需求存在traceability relationship。

        但如果只是：

        - 普通内部转发
        - 一般内部通信
        - 非关键数据路由

        则通常不应判定为存在关系。

    ========================
    【决策规则】
    ========================
    - 若模型节点对需求节点存在明确语义作用（约束/实现/细化/验证/派生）→ 判定存在追溯关系
    - 若仅存在弱语义关联或同域关联但无明确功能贡献 → 判定不存在追溯关系
    - 不允许强行匹配关系

    ========================
    【输出要求】
    ========================
    输出信息格式必须严格按照下面的JSON格式，属性名称也要保持一致：
    {{
        "是否存在追溯关系": true / false,
        "选择的关系": "关系类型 或 null",
        "推理过程": "分步骤说明：模型语义 + 子系统语义 + 需求语义 + 功能贡献分析 + 映射判断 + 案例参考",
        "决定关系的原因": "一句话总结（≤50字）",
        "置信度": 0~1之间的小数
    }}

    ========================
    【关键约束】
    ========================
    - 必须区分：子系统语义 ≠ 追溯关系依据
    - 必须以“模型→需求语义作用”为核心判断标准
    - 必须避免仅基于名称或同系统归属进行判断
    - 必须分析模型节点在功能链路中的真实作用
    - 不允许把所有内部控制端口都判定为存在关系
    - 也不允许只把最终输出端口视为存在关系
    - 必须判断节点是否对需求形成关键功能贡献
    - JSON字段必须严格合法
    - 推理应保持简洁但完整
    - 请先逐步推导，将完整的分析步骤、依据和逻辑链条直接写入 "推理过程" 字段中（建议使用 1. 2. 3. 分点）
    - 最终严格输出符合 JSON Schema 的对象，不要包含任何额外字符
    """

    json_schema = {
        "name": "traceability_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "是否存在追溯关系": {"type": "boolean"},
                "选择的关系": {"type": "string"},
                "推理过程": {"type": "string"},
                "决定关系的原因": {"type": "string"},
                "置信度": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": [
                "是否存在追溯关系", "选择的关系",
                "推理过程", "决定关系的原因", "置信度"
            ],
            "additionalProperties": False
        }
    }

    preferred_model = SINGLE_STAGE_MODEL_NAME
    model_id = resolve_model_id(client, preferred_model)

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            },
            extra_body={
                # vLLM 专属：关闭 thinking 避免破坏 strict 引导解码
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
    except NotFoundError:
        fallback_model = resolve_model_id(client, preferred_model=None)
        response = client.chat.completions.create(
            model=fallback_model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": True
                }
            },
        )

    content = normalize_message_content(response.choices[0].message.content)
    answer = parse_llm_output(content)

    if answer is None:
        answer = extract_first_json(content)

    if answer is None:
        answer = _parse_loose_json_fields(content)

    answer = _normalize_answer_shape(answer)
    _append_sft_sample({
        "mode": "single_stage",
        "model": model_id,
        "is_port": is_port,
        "prompt": prompt,
        "raw_output": content,
        "parsed_answer": answer,
    })

    return answer


def resolve_model_id(client, preferred_model=None):
    """
    返回可用模型 ID：优先使用 preferred_model，不可用时回退到服务端首个模型。
    """
    models_resp = client.models.list()
    model_ids = [m.id for m in getattr(models_resp, "data", []) if getattr(m, "id", None)]

    if preferred_model and preferred_model in model_ids:
        return preferred_model

    if preferred_model and model_ids:
        print(f"[模型回退] 配置模型不可用: {preferred_model}，改用: {model_ids[0]}")
        return model_ids[0]

    if model_ids:
        return model_ids[0]

    raise ValueError(
        f"未从模型服务获取到可用模型，请检查 {SINGLE_STAGE_BASE_URL}/models"
    )

def normalize_message_content(content):
    """
    兼容 OpenAI SDK 中 content 可能为 string 或 content-part 列表的情况。
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    return str(content)

def parse_llm_output(llm_output: str):
    """
    从带 <think> 标签的 LLM 输出中提取最终 JSON，并返回标准字典。
    解析逻辑：
    1. 找到 </think> 之后的内容
    2. 去掉多余空格
    3. 尝试直接解析 JSON
    """
    if not llm_output:
        return None

    if not isinstance(llm_output, str):
        llm_output = str(llm_output)

    # 1️⃣ 思考模式关闭时通常直接返回 JSON，先尝试容错解析全内容
    raw_text = llm_output.strip()
    parsed = _try_parse_json_with_repair(raw_text)
    if parsed is not None:
        return parsed

    # 2️⃣ 若存在 think 标签，再提取尾部内容
    if "</think>" in raw_text:
        json_str = raw_text.split("</think>")[-1].strip()
    else:
        json_str = raw_text

    # 3️⃣ 尝试容错解析 JSON
    parsed = _try_parse_json_with_repair(json_str)
    if parsed is not None:
        return parsed

    # 4️⃣ 最终失败时打印信息
    try:
        json.loads(json_str)
    except json.JSONDecodeError as e:
        print("[解析失败]", e)
        print("内容预览:", json_str[:200])
        return None

    return None


def extract_first_json(text: str):
    """
    从任意文本中提取首个 JSON 对象/数组并解析。
    """
    if not text:
        return None

    if not isinstance(text, str):
        text = str(text)

    # 先尝试标准方式
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            continue

    # 标准失败时，提取首个 JSON 块并进行修复解析
    block = _extract_first_json_like_block(text)
    parsed = _try_parse_json_with_repair(block) if block else None
    if parsed is not None:
        return parsed

    return None




#============================================
# 两阶段法调用，保持模型推理深度和输出格式的稳定性
#=============================================

def call_qwen_two_stage(
    information_json,
    is_port,
    port_information,
    stage2_retries=2,
):
    """
    【两阶段调用】Stage1: 开启 thinking 获取推理 → Stage2: 关闭 thinking + strict JSON 输出
    完全复用你原有的解析函数、prompt 模板、vLLM 调用逻辑
    """
    two_stage_start_time = time.monotonic()
    client = OpenAI(
        base_url=TWO_STAGE_BASE_URL,
        api_key=TWO_STAGE_API_KEY,
    )

    # ========== 构造动态上下文 ==========
    if is_port:
        dynamic_port_information = f"""
    ========================
    【系统内各端口之间的信息流向】
    ========================
    {port_information}
    """
    else:
        dynamic_port_information = ""

    stage1_prompt =  f"""
    你是一个系统工程与模型驱动工程（MBSE）专家，专注于SysML模型元素与需求之间的语义追溯关系分析。

    你的任务是：
    判断“模型节点”和“需求节点”之间是否存在 traceability relationship，
    并且只能从给定候选关系集合中选择合法关系类型。

    ========================
    【输入信息】
    ========================
    {information_json}
    {dynamic_port_information}


    ========================
    【最高优先级约束】
    ========================

    你最终输出的关系：

    ⚠️ 只能来自：
    “模型元素和需求元素之间可能存在的关系集合”

    禁止：
    - 自造关系
    - 输出不存在的关系名
    - 输出模糊关系
    - 输出类似：
    "相关"
    "有关"
    "connected"
    "associated"

    如果无法严格映射到候选关系集合：

    → 必须判定：
        是否存在追溯关系 = false
        选择的关系 = null

    ⚠️ null 是唯一合法的“无关系”表示
    ⚠️ 禁止使用字符串 "null"
    ⚠️ 禁止使用 "None"
    ⚠️ 禁止输出其他词代替

    ========================
    【核心分析对象（必须重点关注）】
    ========================

    你必须重点分析：

    【模型节点信息】
    - 名称（如果是缩写，可以尝试还原）
    - 信息描述（重点分析）
    - 建模路径
    - 元素属性
    - 元素类型

    【需求节点信息】
    - 名称
    - 信息描述（重点分析）
    - 建模路径
    - 元素属性
    - 元素类型

    ⚠️ 注意：

    最终关系判断，
    必须主要建立在：

    “模型节点自身语义”
    与
    “需求节点真正关注的目标”

    之间的语义作用关系上。

    禁止：

    ❌ 仅根据：
    - 名称相似
    - 出现在同一子系统
    - 出现在同一功能链
    - 存在信息流
    - 存在控制流

    就强行建立关系。

    ========================
    【任务本质】
    ========================

    本任务是：

    “语义功能贡献分析”

    不是：

    ❌ 文本匹配
    ❌ 关键词匹配
    ❌ 信息流连接匹配

    你必须判断：

    模型节点是否真正对需求目标形成：

    - 直接实现
    - 功能细化
    - 约束
    - 派生
    - 验证
    - 明确语义支撑

    而不是：

    “只是在系统运行过程中参与了某一步”。

    ========================
    【关系定义（关键）】
    ========================

    【Refine（细化）】

    Refine 表示：

    模型元素对需求进行：

    - 功能展开
    - 子步骤分解
    - 过程细化
    - 行为细化

    即：

    需求 → 被进一步解释得更具体。

    ⚠️ Refine 的核心：

    模型元素描述的是：
    “如何进一步展开这个需求”。

    而不是：

    “为了系统运行顺便参与”。

    --------------------------------

    【Satisfy（满足）】

    Satisfy 表示：

    模型元素直接承担需求目标的实现。

    即：

    删除该模型元素后：

    → 需求目标将无法成立。

    通常属于：

    - 核心执行功能
    - 最终功能承担者
    - 直接实现需求能力

    在判断存在关系的情况下，在做关系类型选择时：
    1.如果元素的建模路径包含1 Problem Domain，那更倾向于判断为Refine关系
    2.如果元素的建模路径包含2 Solution Domain，那更倾向于判断为Satisfy关系

    但是如果候选关系集合中没有相应的关系，那就只能判定为不存在追溯关系。
    ========================================
    【关键限制：禁止“功能链路扩散式推理”】
    ========================================

    ⚠️ 非常重要：

    不能因为：

    一个节点位于某个功能链路中，
    就自动与链路上的所有需求建立关系。

    必须分析：

    “该需求真正关注的核心目标是什么”。

    --------------------------------

    【错误示例（严格禁止）】

    需求：
    "Heating & Cooling"

    真正关注：
    - 是否具备加热能力
    - 是否具备制冷能力

    而：

    "Read Desired Temperature"

    只是：
    - 参数读取步骤
    - 温度设定输入

    它并不决定：
    系统是否具有 Heating/Cooling 能力。

    因此：

    ❌ 不应建立 refine/satisfy 关系。

    --------------------------------

    再例如：

    需求：
    "Manual Control"

    真正关注：
    - 用户能否启动/停止系统
    - 用户是否具备控制权

    而：

    Processing → Cooling 的内部控制端口：

    只是：
    - 系统内部控制传递

    并不对应：
    “用户控制能力”本身。

    因此：

    ❌ 不应建立关系。

    ========================================
    【功能贡献判定（核心规则）】
    ========================================

    必须区分：

    --------------------------------
    ① 真正需求贡献节点
    --------------------------------

    只有当模型节点：

    ✔ 真正决定需求目标成立

    ✔ 是需求目标不可替代的组成部分

    ✔ 直接承担：
    - 实现
    - 细化
    - 约束
    - 验证

    时：

    才允许建立关系。

    --------------------------------
    ② 链路参与节点（禁止误判）
    --------------------------------

    以下情况：

    即使参与系统运行，
    通常也不构成追溯关系：

    - 参数读取
    - 普通数据传递
    - 内部控制转发
    - 中间路由
    - 一般状态同步
    - 普通通信
    - 内部协调

    ⚠️ 特别注意：

    “参与功能链”
    ≠
    “对需求有语义贡献”

    ========================
    【端口类型补充规则】
    ========================
    仅当模型节点属于 Port 类型（如 Control_port / Information_port / Physical_port）时，才使用以下规则。

    1. port_information 的作用

        port_information 用于帮助理解：

        - 端口实际传输的内容
        - 信息/控制/物理流方向
        - 端口所属系统
        - 端口在系统功能链路中的位置

        ⚠️ 注意：
        port_information 只是辅助理解端口语义，
        不能直接作为存在追溯关系的依据。

    2. 端口名称可能重复

        必须结合以下信息唯一确定端口：

        - 端口名称
        - 所属系统
        - Path路径

        不允许仅根据端口名判断。

    3. 必须分析完整功能链路，而不是单条信息流

        不要只看某一个flow。

        应尝试分析：

        输入什么
        ↓
        系统如何处理
        ↓
        输出什么
        ↓
        最终实现什么功能

        即：

        功能链路（functional chain）
        信息链路（information flow chain）
        控制链路（control chain）

    4. 分析端口在链路中的语义角色

        例如：

        - 用户输入接口
        - 功能输入接口
        - 功能输出接口
        - 控制触发接口
        - 状态反馈接口
        - 感知数据接口
        - 中间路由接口
        - 内部控制转发接口

    5. 区分“用户控制”与“系统内部控制”

        如果某端口只是接收系统内部模块处理后的控制信号，
        那么它可能只是内部执行接口。

        这种情况下：

        - 它可能与最终功能有关
        - 但不一定直接对应用户交互需求

        因此不能仅因为它参与控制链路，
        就直接判定存在traceability relationship。

    6. 判断端口是否属于功能关键环节

        如果端口属于以下情况之一：

        - 用户意图进入系统的关键入口
        - 功能执行的重要触发接口
        - 系统感知外部状态的重要入口
        - 核心功能输入/输出接口
        - 完整功能链路中的关键功能节点

        则该端口可能与需求存在traceability relationship。

        但如果只是：

        - 普通内部转发
        - 一般内部通信
        - 非关键数据路由

        则通常不应判定为存在关系。

    ========================================
    【功能链路分析规则】
    ========================================

    允许分析：

    输入
    → 处理
    → 控制
    → 执行
    → 输出
    → 最终效果

    但：

    ⚠️ 功能链路只能用于：

    “理解系统语义”

    不能直接作为关系成立依据。

    最终仍必须回到：

    模型节点本身，
    是否真正服务于：

    需求核心目标。

    ========================================
    【案例参考规则】
    ========================================

    案例只能辅助。

    必须同时满足：

    ✔ 模型元素类型一致
    ✔ 需求类型一致

    否则：

    ❌ 禁止参考。

    禁止机械套用案例。

    ========================================
    【最终决策规则】
    ========================================

    仅当满足以下条件之一：

    ✔ 模型节点直接实现需求
    ✔ 模型节点真正细化需求
    ✔ 模型节点是需求不可替代关键部分

    才允许建立关系。

    否则：

    → 必须输出：
        无追溯关系
        选择的关系 = null

    ========================================
    【输出要求 - Stage1】
    ========================================

    请使用：

    <think>
    ...
    </think>

    输出完整推理。

    推理必须包含：

    1. 模型节点核心语义
    2. 需求真正关注的目标
    3. 功能贡献分析
    4. 是否只是链路参与节点
    5. 是否属于真正语义贡献
    6. 最终关系判定依据

    然后输出最终结论（非JSON）：

    若存在关系：
    - 是否存在关系
    - 关系类型

    若不存在关系：
    - 明确写：
    无追溯关系
    - 并说明：
    选择的关系 = null
    """

    preferred_model = TWO_STAGE_MODEL_NAME
    model_id = resolve_model_id(client, preferred_model)

    # ===== Stage 1: 获取推理内容 =====
    try:
        stage1_response = client.chat.completions.create(
            # model="deepseek-v4-pro",
            model=model_id,
            messages=[{"role": "user", "content": stage1_prompt}],
            # max_tokens=stage1_max_tokens,
            timeout=_remaining_timeout_seconds(two_stage_start_time, TWO_STAGE_TIMEOUT_SECONDS),
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},  # ✅ 开启 thinking
                # "repetition_penalty": 1.15,
                # "thinking": {"type": "enabled"}
            },
            temperature=0.0,
            # top_p=0.75,
        )
        stage1_raw = normalize_message_content(stage1_response.choices[0].message.content)
        # print("[Stage 1 原始输出]", stage1_raw)  # 打印原始输出前500字符预览
        # 提取 </think> 之后的纯文本推理（兼容 thinking 标签）
        if "</think>" in stage1_raw:
            reasoning_text = stage1_raw.split("</think>")[-1].strip()
        else:
            # 兜底：若无标签，取最后 80% 内容作为推理摘要
            reasoning_text = stage1_raw[-int(len(stage1_raw)*0.8):].strip()

    except (TimeoutError, APITimeoutError) as e:
        print(f"[两阶段超时] {e}，降级为单阶段调用")
        return _call_qwen_single_stage(information_json, is_port, port_information)
    except Exception as e:
        print(f"[Stage 1 失败] {e}，降级为单阶段调用")
        # 降级：直接走原有单阶段逻辑
        return _call_qwen_single_stage(information_json, is_port, port_information)


    print("[Stage 1 推理内容]", reasoning_text[:500])  # 打印推理内容前500字符预览

    # ========== Stage 2: 强制 JSON 输出 ==========
    json_schema = {
        "name": "traceability_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "是否存在追溯关系": {"type": "boolean"},
                "选择的关系": {"type": "string"},
                "推理过程": {"type": "string"},
                "决定关系的原因": {"type": "string"},
                "置信度": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": [
                "是否存在追溯关系", "选择的关系",
                "推理过程", "决定关系的原因", "置信度"
            ],
            "additionalProperties": False
        }
    }

    # Stage 2 Prompt：注入 Stage 1 的推理结果，要求直接收敛为 JSON
    stage2_prompt = f"""
    你已完成推理，现在只做“结果收敛”。

    【推理内容】
    {reasoning_text}

    ========================
    【强制规则】
    ========================

    1. 不允许重新推理
    2. 不允许新增信息
    3. 关系类型必须：

    ✔ 在候选集合中
    否则：
    → 自动改为 null

    4. 若为无关系：
    必须：
    - "是否存在追溯关系": false
    - "选择的关系": null

    ❌ 禁止输出：
    - "无"
    - "None"
    - "Other"
    - 任意字符串

    只能是：
    → null（JSON）

    --------------------------------

    【Refine / Satisfy 最终校验】

    若为：
    - Refine（细化）：
    模型元素是需求的下位展开，提供了功能分解或过程细化，但不是最终实现者。通常在Problem Domain中对需求分解时使用。

    - Satisfy（满足）：
    模型元素直接实现了需求目标，是最终功能承担者。通常在Solution Domain中对需求实现时使用。
    --------------------------------

    【输出格式要求】

    - 合法 JSON
    - 不要 ```json
    - 不要多余字段
    - 置信度 ∈ [0,1]

    ========================
    【输出 Schema】
    ========================

    {{
        "是否存在追溯关系": true/false,
        "选择的关系": "候选关系 或 null",
        "推理过程": "1. xxx 2. xxx 3. xxx（≤300字）",
        "决定关系的原因": "一句话（≤50字）",
        "置信度": 0.85
    }}
    """


    # ===== Stage 2: 带重试的结构化输出 =====
    last_error = None
    last_stage2_raw = None
    for attempt in range(stage2_retries + 1):  # +1 表示首次尝试
        try:
            stage2_response = client.chat.completions.create(
                # model="deepseek-v4-pro",
                model=model_id,
                messages=[{"role": "user", "content": stage2_prompt}],
                # max_tokens=stage2_max_tokens,
                # timeout=_remaining_timeout_seconds(two_stage_start_time, TWO_STAGE_TIMEOUT_SECONDS),
                response_format={
                    'type': 'json_object',
                    "json_schema": json_schema
                },
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},  # ✅ 关闭 thinking
                    # vLLM < 0.7.0 备用方案：
                    # "guided_json": json_schema["schema"]
                    # "thinking": {"type": "disabled"}
                }
            )

            content = normalize_message_content(stage2_response.choices[0].message.content)
            last_stage2_raw = content

            # 复用你原有的完整解析链路（保持兼容性）
            answer = parse_llm_output(content)
            if answer is None:
                answer = extract_first_json(content)
            if answer is None:
                answer = _parse_loose_json_fields(content)

            if answer is not None:
                # 标准化 + 置信度裁剪
                normalized_answer = _normalize_answer_shape(answer)
                _append_sft_sample({
                    "mode": "two_stage",
                    "model": model_id,
                    "is_port": is_port,
                    "stage1_prompt": stage1_prompt,
                    "stage1_raw_output": stage1_raw,
                    "stage2_prompt": stage2_prompt,
                    "stage2_raw_output": content,
                    "parsed_answer": normalized_answer,
                    "parse_status": "stage2_success",
                })
                return normalized_answer

            last_error = "解析结果为空"

        except (TimeoutError, APITimeoutError) as e:
            print(f"[两阶段超时] {e}，降级为单阶段调用")
            return _call_qwen_single_stage(information_json, is_port, port_information)
        except Exception as e:
            last_error = str(e)
            if attempt < stage2_retries:
                print(f"[Stage 2 重试 {attempt+1}/{stage2_retries}] {e}")
                continue
            break

    # ===== 最终兜底：宽松解析 Stage 1 原始输出 =====
    # print(f"[Stage 2 失败] {last_error}，尝试从 Stage 1 输出中宽松提取")
    fallback_answer = _parse_loose_json_fields(stage1_raw)
    if fallback_answer:
        normalized_answer = _normalize_answer_shape(fallback_answer)
        _append_sft_sample({
            "mode": "two_stage",
            "model": model_id,
            "is_port": is_port,
            "stage1_prompt": stage1_prompt,
            "stage1_raw_output": stage1_raw,
            "stage2_prompt": stage2_prompt,
            "stage2_raw_output": last_stage2_raw,
            "parsed_answer": normalized_answer,
            "parse_status": "stage1_fallback",
            "last_error": last_error,
        })
        return normalized_answer

    # 全失败时返回安全默认值
    normalized_answer = _normalize_answer_shape(None)
    _append_sft_sample({
        "mode": "two_stage",
        "model": model_id,
        "is_port": is_port,
        "stage1_prompt": stage1_prompt,
        "stage1_raw_output": stage1_raw,
        "stage2_prompt": stage2_prompt,
        "stage2_raw_output": last_stage2_raw,
        "parsed_answer": normalized_answer,
        "parse_status": "failed_default",
        "last_error": last_error,
    })
    return normalized_answer
