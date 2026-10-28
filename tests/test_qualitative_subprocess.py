import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from src.qualitative import (
    _build_hermes_command,
    _decode_json,
    _extract_json,
    collect_qualitative_candidates,
    generate_qualitative_data,
)


class QualitativeCommandTests(unittest.TestCase):
    def test_fundamental_screen_candidates_merge_with_momentum_candidates(self):
        candidates = collect_qualitative_candidates({
            "actions": [{"symbol": "BOTH", "rsi": 80}],
            "fundamental_screen_candidates": [
                {"symbol": "BOTH", "forward_pe": 18},
                {"symbol": "FUND", "forward_pe": 20},
            ],
        })

        by_symbol = {item["symbol"]: item for item in candidates}
        self.assertEqual(set(by_symbol), {"BOTH", "FUND"})
        self.assertEqual(len(by_symbol["BOTH"]["triggers"]), 2)
        self.assertEqual(by_symbol["FUND"]["strategy_type"], "Fundamental Screen")

    def test_explicit_caller_model_is_passed_to_child(self):
        command = _build_hermes_command(
            "research",
            {
                "HERMES_BIN": "/usr/bin/hermes",
                "QT_QUALITATIVE_PROVIDER": "openai-codex",
                "QT_QUALITATIVE_MODEL": "gpt-5.5",
                "GEMINI_API_KEY": "must-not-be-used-as-routing",
            },
        )

        self.assertEqual(command[0:2], ["/usr/bin/hermes", "chat"])
        self.assertEqual(command[command.index("--provider") + 1], "openai-codex")
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.5")
        self.assertNotIn("--ignore-user-config", command)
        self.assertNotIn("must-not-be-used-as-routing", command)

    def test_default_uses_hermes_profile_instead_of_environment_key_guess(self):
        command = _build_hermes_command("research", {"GEMINI_API_KEY": "not-a-model-setting"})

        self.assertNotIn("--ignore-user-config", command)
        self.assertNotIn("--provider", command)
        self.assertNotIn("--model", command)

    def test_json_decoder_handles_clean_and_fenced_output(self):
        parsed, error = _decode_json('```json\n{"date":"2026-09-29"}\n```')

        self.assertEqual(parsed, {"date": "2026-09-29"})
        self.assertIsNone(error)
        self.assertEqual(_extract_json('{"ok": true}'), {"ok": True})

    def test_invalid_json_has_location_diagnostic_and_fails_closed(self):
        parsed, error = _decode_json('{"analyses": [}\n')

        self.assertIsNone(parsed)
        self.assertIsNotNone(error)
        self.assertIn("line=1", error or "")
        self.assertIn("col=", error or "")
        self.assertIsNone(_extract_json('{"analyses": [}\n'))

    def test_generation_returns_none_and_reports_parse_error(self):
        actions = {
            "date": "2026-09-29",
            "actions": [{"symbol": "TEST", "rsi": 80}],
        }
        output = StringIO()
        with patch(
            "src.qualitative.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout='{"analyses": [}\n', stderr=""),
        ), redirect_stdout(output):
            result = generate_qualitative_data(actions)

        self.assertIsNone(result)
        self.assertIn("JSON 格式錯誤 line=1", output.getvalue())

    def test_generation_rejects_document_class_in_evidence_type(self):
        actions = {
            "date": "2026-09-29",
            "actions": [{"symbol": "TEST", "rsi": 80}],
        }
        payload = {
            "analyses": [{
                "symbol": "TEST",
                "quant_action_override": "confirm",
                "final_recommendation": "待觀察",
                "core_thesis": "thesis",
                "counter_case": "counter case",
                "evidence": [{"type": "company earnings release"}],
                "invalidation_conditions": [],
            }],
        }
        output = StringIO()
        with patch(
            "src.qualitative.subprocess.run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            ),
        ), redirect_stdout(output):
            result = generate_qualitative_data(actions)

        self.assertIsNone(result)
        self.assertIn("evidence.type 必須是", output.getvalue())


if __name__ == "__main__":
    unittest.main()
