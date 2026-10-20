"""Oracle-only client with locally validated two-label JSON output."""

import os

from openai import APITimeoutError, OpenAI

from config import api_key, base_url, model_name as configured_model_name
from main import create_completion_with_retry


ORACLE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "oracle_relation_type",
        "schema": {
            "type": "object",
            "properties": {
                "relation_type": {"type": "string", "enum": ["Refine", "Satisfy"]}
            },
            "required": ["relation_type"],
            "additionalProperties": False,
        },
    },
}


def call_oracle_llm(prompt, client=None, model=None):
    """Use main-experiment inference settings; record final failures per row."""
    served_model = model or os.getenv("LLM_MODEL") or configured_model_name
    if client is None:
        client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL") or base_url,
            api_key=os.getenv("LLM_API_KEY") or api_key,
            max_retries=0,
        )
    request = {
        "client": client,
        "model": served_model,
        "messages": [{"role": "user", "content": prompt}],
        # DeepSeek's hosted API supports json_object, while local vLLM supports
        # the stricter json_schema mode. parse_relation_type validates both.
        "response_format": (
            {"type": "json_object"}
            if served_model.lower() == "deepseek-v4-pro"
            else ORACLE_RESPONSE_FORMAT
        ),
    }
    try:
        response = create_completion_with_retry(**request)
    except APITimeoutError:
        print("Request timed out after all retries. "
              "Retrying once with the same model settings...", flush=True)
        response = create_completion_with_retry(**request)
    if not response.choices:
        raise ValueError("Model response has no choices")
    choice = response.choices[0]
    content = choice.message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Empty model response (finish_reason={choice.finish_reason!r})")
    return content.strip()
