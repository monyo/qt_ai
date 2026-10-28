import unittest

from src.notifier import GmailNotifier


class QualitativeRendererTests(unittest.TestCase):
    def test_structured_analysis_fields_render_without_legacy_placeholders(self):
        qualitative_data = {
            "research_packets": [
                {"symbol": "TEST", "triggers": ["ROTATE 賣出"]},
            ],
            "analyses": [{
                "symbol": "TEST",
                "research_archetype": "ROTATE",
                "score": 63,
                "verdict": "🟡 可考慮",
                "core_thesis": "基本面仍具韌性",
                "catalysts": [{
                    "event": "下一季財報",
                    "transmission": "驗證需求",
                    "time_horizon": "下一季",
                    "confirmation_signal": "營收符合指引",
                }],
                "risks": [{
                    "event": "需求下滑",
                    "financial_transmission": "收入受壓",
                    "warning_signal": "公司下修指引",
                }],
                "counter_case": "目前尚無結構性惡化證據",
                "scenarios": {
                    "bear": {"assumptions": "需求轉弱", "result": "估值收縮"},
                },
                "invalidation_conditions": ["公司下修全年指引"],
                "position_guidance": "等待下一季確認，不立即賣出",
                "final_recommendation": "續抱並等待確認",
                "quant_action_override": "confirm_but_wait",
                "evidence": [{
                    "source": "https://investors.example.com/release",
                    "title": "Company earnings release",
                    "published_at": "2026-09-01",
                }],
            }],
        }
        renderer = object.__new__(GmailNotifier)

        html = renderer._format_qualitative_html(qualitative_data)

        for expected in (
            "ROTATE 賣出",
            "基本面仍具韌性",
            "下一季財報",
            "營收符合指引",
            "需求下滑",
            "等待下一季確認，不立即賣出",
            "公司下修全年指引",
            "需求轉弱",
            "Company earnings release",
        ):
            self.assertIn(expected, html)
        self.assertNotIn("催化劑：</strong>—", html)
        self.assertNotIn("風險：</strong>—", html)
        self.assertNotIn("建議：</strong>—", html)

    def test_model_text_is_escaped_in_rendered_html(self):
        qualitative_data = {
            "analyses": [{
                "symbol": "TEST",
                "core_thesis": "<script>alert(1)</script>",
                "catalysts": [],
                "risks": [],
                "invalidation_conditions": [],
                "position_guidance": "等待",
                "evidence": [],
            }],
        }
        renderer = object.__new__(GmailNotifier)

        html = renderer._format_qualitative_html(qualitative_data)

        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
