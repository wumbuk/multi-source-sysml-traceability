"""Oracle-only relation typing. This module never asks whether a link exists."""

import json
import os

from openai import OpenAI

from config import (
    TWO_STAGE_API_KEY,
    TWO_STAGE_BASE_URL,
    TWO_STAGE_MODEL_NAME,
)


RELATION_TYPES = ("Refine", "Satisfy")


def oracle_evidence_instruction(variant):
    if not variant:
        return (
            "Use the element's semantic role, hierarchical path, structural and interaction "
            "context, retrieved cases, and ontology candidates as evidence. Candidates "
            "are hints, not a reason to reject a known link. "
        )
    evidence = {
        "Semantic_Enrichment": "the model element's enriched functional description",
        "Hierarchical_Path": "the current pair's hierarchical paths",
        "Structural_Context": "the same-subsystem node context",
        "Case_Retrieval": "retrieved traceability cases",
        "Ontology_Constraint": "ontology-derived candidate relations",
    }
    if variant not in evidence:
        raise ValueError(f"Unsupported TRACE_ABLATION_VARIANT: {variant}")
    available = [description for component, description in evidence.items() if component != variant]
    return (
        f"Ablation setting: {evidence[variant]} is intentionally omitted. "
        "Do not reconstruct that missing input or treat its absence as evidence against "
        "the confirmed link. Use only the supplied element names, types, properties, "
        "requirement description, and available context, including "
        + "; ".join(available)
        + ". Any supplied ontology candidates are hints, not a reason to reject a known link. "
    )


def _read_type(content):
    if not isinstance(content, str):
        raise ValueError("Oracle response has no text content")
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    response = json.loads(cleaned)
    relation = response.get("relation_type")
    if relation not in RELATION_TYPES:
        raise ValueError(f"Oracle response has invalid relation_type: {relation!r}")
    return relation


def call_qwen_oracle_type(information_json, is_port=False, port_information=""):
    """Classify a known link as Refine or Satisfy, preserving main.py's output schema."""
    context = json.loads(information_json)
    if not isinstance(context, dict):
        raise ValueError("Oracle input must be a JSON object")
    # Gold labels belong only in the evaluation CSV, never in model input.
    for key in ("Relation", "Project", "SourceSequence"):
        if key in context:
            raise ValueError(f"Gold-only field leaked into Oracle context: {key}")

    port_context = port_information if is_port else ""
    variant = os.getenv("TRACE_ABLATION_VARIANT", "").strip()
    evidence_instruction = oracle_evidence_instruction(variant)
    prompt = (
        "You are a SysML requirements-traceability expert. The supplied model-element "
        "and requirement pair is already confirmed to have a direct traceability link. "
        "Do not decide whether a link exists. Determine only whether its type is "
        "Refine or Satisfy. Refine means the element elaborates or decomposes the "
        "requirement; Satisfy means it directly realizes the requirement. "
        + evidence_instruction
        + "Do not infer the type from "
        "name similarity alone. Never output No Relation or null.\n\n"
        f"Context:\n{json.dumps(context, ensure_ascii=False)}\n"
        f"Port information:\n{port_context}\n"
        "Reason about the relation type, then state Refine or Satisfy."
    )
    client = OpenAI(
        base_url=TWO_STAGE_BASE_URL,
        api_key=TWO_STAGE_API_KEY,
        timeout=None,
    )
    first = client.chat.completions.create(
        model=TWO_STAGE_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    reasoning = first.choices[0].message.content or ""
    if not isinstance(reasoning, str):
        raise ValueError("Oracle reasoning response has no text content")
    if "</think>" in reasoning:
        reasoning = reasoning.rsplit("</think>", 1)[-1].strip()

    schema = {
        "name": "oracle_relation_type",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"relation_type": {"type": "string", "enum": list(RELATION_TYPES)}},
            "required": ["relation_type"],
            "additionalProperties": False,
        },
    }
    final_prompt = (
        "The pair is known to have a traceability link. Classify its type only. "
        "Return exactly one JSON object with relation_type = Refine or Satisfy. "
        "No Relation, null and link-existence decisions are forbidden.\n\n"
        + (evidence_instruction + "\n\n" if variant else "")
        + f"Context:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        + (f"Port information:\n{port_context}\n\n" if port_context else "")
        + f"Previous reasoning:\n{reasoning}"
    )
    last_error = None
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=TWO_STAGE_MODEL_NAME,
                messages=[{"role": "user", "content": final_prompt}],
                response_format={"type": "json_schema", "json_schema": schema},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            relation = _read_type(response.choices[0].message.content)
            return {
                "是否存在追溯关系": True,
                "选择的关系": relation,
                "推理过程": reasoning,
                "决定关系的原因": "Oracle relation-type classification",
                "置信度": 0.0,
            }
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError("Oracle type output failed validation after 3 attempts") from last_error
