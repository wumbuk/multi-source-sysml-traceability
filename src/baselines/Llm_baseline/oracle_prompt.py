"""Shared Direct-LLM Oracle prompt and strict two-label response parser."""

import json

from oracle_dataset import MODEL_FIELDS


def build_prompt(row):
    missing = set(MODEL_FIELDS) - set(row)
    if missing:
        raise ValueError(f"Missing Direct-LLM fields: {sorted(missing)}")
    # Explicit whitelist: Relation, Project and SourceSequence cannot enter.
    values = {field: row[field] for field in MODEL_FIELDS}
    return f"""You are a SysML modeling engineer. A direct traceability relationship already exists between this SysML model element and this requirement. Determine only its relation type.

Element Name: {values['NameA']}
Element Type: {values['typeA']}
Element Modeling Path: {values['pathA']}
Element Properties: {values['propertyA']}
Requirement Name: {values['NameB']}
Requirement Type: {values['typeB']}
Requirement Properties: {values['propertyB']}
Requirement Description: {values['contextB']}

Choose exactly one type:
- Refine: the model element decomposes, refines, or concretizes the requirement.
- Satisfy: the model element directly implements or fulfills the requirement's objective.

The existence of a direct relationship is given. Do not judge its existence. Do not output no relation or null. Answer in English with only a JSON object in this exact form: {{"relation_type":"Refine"}} or {{"relation_type":"Satisfy"}}.
"""


def parse_relation_type(response):
    if not isinstance(response, str):
        raise ValueError("Response is missing")
    text = response.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Response is not a JSON object") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"relation_type"}:
        raise ValueError("Response must contain only relation_type")
    if parsed["relation_type"] not in {"Refine", "Satisfy"}:
        raise ValueError("relation_type must be Refine or Satisfy")
    return parsed["relation_type"]
