import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import config
import main as direct_llm

from oracle_client import ORACLE_RESPONSE_FORMAT, call_oracle_llm
from oracle_dataset import PROJECT_COUNTS, build, read_csv
from oracle_prompt import build_prompt, parse_relation_type
from oracle_run import run, score


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.data_root = Path(__file__).resolve().parent.parent / "Data"
        defaults = patch.dict(config.options, {"thinking": None, "repetition_penalty": None})
        defaults.start()
        self.addCleanup(defaults.stop)

    def test_counts_judge_alignment_and_original_rows(self):
        _, oracle_rows = read_csv(self.data_root / "Oracle_Positive_85.csv")
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            summary = build(self.data_root / "Oracle_Positive_85.csv", self.data_root, dest)
            self.assertEqual((summary["total"], summary["Refine"], summary["Satisfy"]), (85, 64, 21))
            for project, expected in PROJECT_COUNTS.items():
                _, selected = read_csv(dest / f"{project}.csv")
                _, judge = read_csv(self.data_root / project / "Judge.csv")
                self.assertEqual(len(selected), sum(expected.values()))
                self.assertEqual(
                    [r["SourceSequence"] for r in selected],
                    [r["SourceSequence"] for r in oracle_rows if r["Project"] == project],
                )
                for row in selected:
                    source = judge[int(row["SourceSequence"]) - 2]
                    self.assertEqual({k: row[k] for k in source}, source)

    def test_non_whitespace_mismatch_stops_before_writing(self):
        fields, rows = read_csv(self.data_root / "Oracle_Positive_85.csv")
        rows[0]["NameA"] = "different element"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "NameA differs"):
                build(source, self.data_root, output)
            self.assertFalse(output.exists())

    def test_prompt_uses_only_direct_llm_fields(self):
        row = {
            "NameA": "A", "typeA": "Block", "pathA": "[path]", "propertyA": "p",
            "NameB": "B", "typeB": "Requirement", "propertyB": "q", "contextB": "description",
            "Relation": "SECRET_GOLD", "Project": "SECRET_PROJECT", "SourceSequence": "SECRET_SEQUENCE",
        }
        prompt = build_prompt(row)
        for forbidden in ("SECRET_GOLD", "SECRET_PROJECT", "SECRET_SEQUENCE"):
            self.assertNotIn(forbidden, prompt)
        self.assertIn("already exists", prompt)
        self.assertIn('"relation_type"', prompt)

    def test_strict_parser_and_failure_scoring(self):
        self.assertEqual(parse_relation_type('{"relation_type":"Refine"}'), "Refine")
        for raw in (None, "null", '{"relation_type":null}', '{"relation_type":"None"}',
                    '{"relation_type":"Refine","gold":"Satisfy"}', "Refine"):
            with self.assertRaises(ValueError):
                parse_relation_type(raw)
        metrics, errors = score([
            {"gold_relation_type": "Refine", "predicted_relation_type": "Refine"},
            {"gold_relation_type": "Satisfy", "predicted_relation_type": None},
        ])
        self.assertEqual(metrics["confusion_matrix"]["values"], [[1, 0], [0, 0]])
        self.assertEqual(metrics["failed_predictions"], 1)
        self.assertEqual(metrics["type_accuracy"], 0.5)
        self.assertEqual(len(errors), 1)

    def test_oracle_client_constrains_json_and_preserves_server_defaults(self):
        class StubCompletions:
            def __init__(self):
                self.requests = []

            def create(self, **request):
                self.requests.append(request)
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"relation_type":"Refine"}'),
                    finish_reason="stop",
                )])

        completions = StubCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        self.assertEqual(call_oracle_llm("prompt", client=client, model="Qwen3.5-27B"),
                         '{"relation_type":"Refine"}')
        request = completions.requests[0]
        self.assertEqual(request["response_format"], ORACLE_RESPONSE_FORMAT)
        self.assertEqual(request["response_format"]["json_schema"]["schema"]
                         ["properties"]["relation_type"]["enum"], ["Refine", "Satisfy"])
        self.assertNotIn("extra_body", request)
        self.assertNotIn("thinking", request)
        self.assertNotIn("chat_template_kwargs", request)
        self.assertEqual(request["timeout"], 120.0)
        self.assertEqual(len(completions.requests), 1)

        for model in ("Qwen3.5-9B", "gemma4-E4B", "LLaMA3.1-8B"):
            call_oracle_llm("prompt", client=client, model=model)
            for key in ("extra_body", "thinking", "chat_template_kwargs"):
                self.assertNotIn(key, completions.requests[-1])

    def test_oracle_and_direct_requests_share_inference_parameters(self):
        requests = []
        def create(**request):
            requests.append(request)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"relation_type":"Refine"}'),
                finish_reason="stop",
            )])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        for thinking in (None, True, False):
            with patch.dict(config.options, {"thinking": thinking, "repetition_penalty": 1.05}):
                for model in ("Qwen3.5-9B", "Qwen3.5-27B", "gemma4-E4B",
                              "LLaMA3.1-8B", "deepseek-v4-pro"):
                    direct_llm.create_completion_with_retry(
                        client, [{"role": "user", "content": "prompt"}], model=model)
                    call_oracle_llm("prompt", client=client, model=model)
                    oracle_request = dict(requests[-1])
                    oracle_request.pop("response_format")
                    self.assertEqual(requests[-2], oracle_request)
                    body = requests[-1]["extra_body"]
                    if thinking is None or "LLaMA" in model:
                        self.assertEqual(body, {"repetition_penalty": 1.05})
                    elif model.startswith("deepseek"):
                        self.assertEqual(body["thinking"]["type"],
                                         "enabled" if thinking else "disabled")
                    else:
                        self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], thinking)

    def test_oracle_timeout_retries_once_with_unchanged_request(self):
        import httpx
        from openai import APITimeoutError
        requests = []
        def create(**request):
            requests.append(request)
            if len(requests) == 1:
                raise APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"relation_type":"Refine"}'),
                finish_reason="stop",
            )])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(direct_llm, "MAX_API_RETRIES", 0):
            call_oracle_llm("prompt", client=client, model="Qwen3.5-9B")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0], requests[1])

    def test_oracle_client_does_not_hide_empty_or_api_failure(self):
        class StubCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **request):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(choices=[SimpleNamespace(
                        message=SimpleNamespace(content=None), finish_reason="length"
                    )])
                raise TimeoutError("server timed out")

        completions = StubCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with self.assertRaisesRegex(ValueError, "Empty model response.*length"):
            call_oracle_llm("prompt", client=client, model="Qwen3.5-9B")
        with self.assertRaisesRegex(TimeoutError, "server timed out"):
            call_oracle_llm("prompt", client=client, model="Qwen3.5-9B")
        self.assertEqual(completions.calls, 2)

    def test_deepseek_oracle_uses_hosted_api_json_mode(self):
        requests = []

        def create(**request):
            requests.append(request)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"relation_type":"Satisfy"}'),
                finish_reason="stop",
            )])

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        self.assertEqual(
            call_oracle_llm("JSON only", client=client, model="deepseek-v4-pro"),
            '{"relation_type":"Satisfy"}',
        )
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        self.assertNotIn("extra_body", requests[0])
        self.assertEqual(requests[0]["timeout"], 120.0)

    def test_oracle_client_accepts_runtime_deepseek_configuration(self):
        requests = []

        def create(**request):
            requests.append(request)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"relation_type":"Refine"}'),
                finish_reason="stop",
            )])

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        with patch.dict(os.environ, {"LLM_MODEL": "deepseek-v4-pro"}):
            call_oracle_llm("JSON only", client=client)
        self.assertEqual(requests[0]["model"], "deepseek-v4-pro")
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})

    def test_one_sample_inference_with_stub_client(self):
        row = {
            "NameA": "A", "typeA": "Block", "pathA": "[path]", "propertyA": "p",
            "NameB": "B", "typeB": "Requirement", "propertyB": "q", "contextB": "description",
            "Relation": "1/Satisfy", "SourceSequence": "2",
        }
        prompts = []
        def fake_client(prompt):
            prompts.append(prompt)
            return '{"relation_type":"Refine"}'
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            metrics = run([row], target, client_call=fake_client)
            self.assertEqual(metrics["type_accuracy"], 0.0)
            self.assertEqual(metrics["confusion_matrix"]["values"], [[0, 0], [1, 0]])
            self.assertTrue((target / "oracle_predictions.json").is_file())
            self.assertTrue((target / "oracle_mismatches.json").is_file())
            self.assertTrue((target / "oracle_confusion_matrix.csv").is_file())
            self.assertNotIn("1/Satisfy", prompts[0])

    def test_api_failure_is_recorded_without_gold_fallback(self):
        row = {
            "NameA": "A", "typeA": "Block", "pathA": "[path]", "propertyA": "p",
            "NameB": "B", "typeB": "Requirement", "propertyB": "q", "contextB": "description",
            "Relation": "1/Satisfy", "SourceSequence": "2",
        }

        def failing_client(prompt):
            raise TimeoutError("server timed out")

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            metrics = run([row], target, client_call=failing_client)
            records = json.loads(
                (target / "oracle_predictions.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["failed_predictions"], 1)
            self.assertEqual(records[0]["status"], "failed")
            self.assertIn("TimeoutError: server timed out", records[0]["error"])
            self.assertIsNone(records[0]["predicted_relation_type"])

    def test_fail_fast_stops_before_the_next_api_request(self):
        row = {
            "NameA": "A", "typeA": "Block", "pathA": "[path]", "propertyA": "p",
            "NameB": "B", "typeB": "Requirement", "propertyB": "q", "contextB": "description",
            "Relation": "1/Refine", "SourceSequence": "2",
        }
        calls = []

        def failing_client(prompt):
            calls.append(prompt)
            raise RuntimeError("API failed")

        with tempfile.TemporaryDirectory() as directory:
            metrics = run([row, row], Path(directory), client_call=failing_client,
                          fail_fast=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(metrics["failed_predictions"], 1)
            self.assertEqual(metrics["total"], 1)


if __name__ == "__main__":
    unittest.main()
