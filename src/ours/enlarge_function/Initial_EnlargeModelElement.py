import time

from openai import OpenAI
from config import (
    SEMANTIC_ENRICHMENT_API_KEY,
    SEMANTIC_ENRICHMENT_BASE_URL,
    SEMANTIC_ENRICHMENT_MODEL,
)


def _create_client() -> OpenAI:
    """Create the semantic-enrichment client without embedding credentials."""
    if not SEMANTIC_ENRICHMENT_API_KEY:
        raise RuntimeError(
            "未配置语义丰富 API Key。请设置环境变量 "
            "TRACE_ENRICHMENT_API_KEY 后重试。"
        )
    return OpenAI(
        api_key=SEMANTIC_ENRICHMENT_API_KEY,
        base_url=SEMANTIC_ENRICHMENT_BASE_URL,
    )


def generate_description(name,path,property, element_type):
    """调用配置的 OpenAI 兼容服务生成功能描述。"""
    # 针对类型构造提示
    if "signal" in element_type:
        hint = (
            "Describe the event or message represented by this signal, including the information it "
            "carries and the behavioral or state transitions it is intended to trigger."
        )

    elif "Block" in element_type:
        hint = (
            "Describe the structural definition of this block, including its responsibilities, key "
            "functions, and its role within the larger system architecture."
        )

    elif "valueproperty" in element_type:
        hint = (
            "Describe the specific quantitative attribute represented by this value property, "
            "including its unit, constraints, default values, and its influence on system behavior."
        )

    elif "Action" in element_type:
        hint = (
            "Describe the discrete behavior performed by this action within an activity, including "
            "its input pins, output pins, and the transformation or effect it produces."
        )

    elif "Activity" in element_type:
        hint = (
            "Describe the overall behavior or workflow represented by this activity, including its "
            "goal, major action steps, control flow, and its contribution to system functionality."
        )

    elif "State" in element_type:
        hint = (
            "Describe the system condition or mode represented by this state, including the "
            "behavior exhibited while in this state and the events or conditions that enable "
            "transitions to other states."
        )

    elif "partproperty" in element_type:
        hint = (
            "Describe this internal part of the parent block, including its functional role, "
            "structural contribution, and the fact that it is typed by a Block definition rather "
            "than representing a block itself."
        )

    elif "portproperty" in element_type:
        hint = (
            "Describe the interaction point represented by this port property, including the "
            "kind of flows it exchanges (information, energy, material) and its role in enabling "
            "interfaces between parts."
        )

    else:
        hint = (
            "Describe the core purpose of this model element, its role within the system, and the "
            "key relationships or behaviors it participates in."
        )
    prompt = f"""
    You are a SysML modeling expert. Analyze the function of the model element using the following information:
    - Name (restore meaning from abbreviated words)
    - Property (unit for values/parameters; name for ports)
    - Type (meaning the Element Type in Systems Modeling Language)
    - Modeling path (indicates subsystem location)

    Path: {path}
    Type: {element_type}
    Name: {name}
    Property: {property}

    {hint}
    Output ONLY a single concise English sentence describing the function. No additional text.
    Keep the sentence within 40 words.
    """
    messages = [
        {"role": "system", "content": "你是专属SysML系统工程建模专家，仅处理需求与模型元素追溯关系判断，不输出任何身份介绍、额外解释或无关内容。"},
        {"role": "user", "content": prompt}
    ]
    client = _create_client()
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=SEMANTIC_ENRICHMENT_MODEL,
                temperature=0,
                max_tokens=512,
                messages=messages,
            )
            response_content = response.choices[0].message.content
            response_text = (
                response_content.strip()
                if isinstance(response_content, str)
                else ""
            )
            if not response_text:
                raise ValueError("LLM returned empty content")

            print(response_text)
            return response_text
        except Exception as e:
            print(f"请求失败，重试中... ({attempt + 1}/3)", e)
            time.sleep(2)

    # ======== 全部失败才返回错误 ========
    return "ERROR: Request failed after 3 retries"
