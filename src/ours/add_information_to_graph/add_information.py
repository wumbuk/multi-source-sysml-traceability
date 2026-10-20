import json
import re


def add_edges(model_node,req_node,answer):
    """
    从 LLM 输出构造 edge 字典（仅当存在关系时）
    """
    # 无关系时返回 None
    if answer is None:
        return None
    if not answer.get("是否存在追溯关系", False):
        return None

    relation = answer.get("选择的关系")
    if not relation:
        return None
    edge = {
        "Source": model_node["ID"],
        "Target": req_node["ID"],
        "EType": relation,
        "Reason": answer.get("决定关系的原因",""),
        "Confidence": answer.get("置信度",0.0),
        "Source_type": "LLM"
    }
    return edge
