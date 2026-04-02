"""
test/test_reporter.py  —  Tests for eval/reporter.py

Covers:
  • redact(): replaces SSNs, credit cards, API keys, emails, passwords
  • redact(): leaves clean text unchanged
  • iso_week(): format is YYYY-DD-MM (e.g. '2026-04-04')
  • WeeklyReporter: creates week directory on init
  • WeeklyReporter.write_metrics(): writes valid JSON with expected keys
  • WeeklyReporter.generate_audit_logs():
      - CRITICAL_FAIL always logged (regardless of severity)
      - FAIL with severity >= 7 logged
      - FAIL with severity < 7 NOT logged
      - PASS never logged
  • write_audit_log(): output JSON contains compliance_ref, NIST fields, remediation
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from eval.judges import PASS, FAIL, CRITICAL_FAIL, OVER_REFUSAL
from eval.reporter import WeeklyReporter, redact, iso_week, get_remediation
from eval.metrics import ModelMetrics
from test.conftest import make_scored, make_verdict, make_regex, make_neural


# ── redact() ──────────────────────────────────────────────────────────────────

class TestRedact:
    def test_replaces_ssn(self):
        assert "[SSN_REDACTED]" in redact("SSN: 123-45-6789")
        assert "123-45-6789" not in redact("SSN: 123-45-6789")

    def test_replaces_credit_card_with_spaces(self):
        out = redact("Card: 4111 1111 1111 1111")
        assert "[CC_REDACTED]" in out

    def test_replaces_credit_card_with_dashes(self):
        out = redact("4111-1111-1111-1111")
        assert "[CC_REDACTED]" in out

    def test_replaces_openai_api_key(self):
        out = redact("Key: sk-" + "a" * 40)
        assert "[APIKEY_REDACTED]" in out

    def test_replaces_aws_key(self):
        out = redact("AKIAIOSFODNN7EXAMPLE")
        assert "[AWSKEY_REDACTED]" in out

    def test_replaces_password(self):
        out = redact("password: MyS3cur3Pass!")
        assert "[PASSWORD_REDACTED]" in out

    def test_replaces_email(self):
        out = redact("Contact: user@example.com for details.")
        assert "[EMAIL_REDACTED]" in out
        assert "user@example.com" not in out

    def test_clean_text_unchanged(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert redact(text) == text

    def test_multiple_pii_all_replaced(self):
        text = "SSN 123-45-6789 and email user@test.com"
        out = redact(text)
        assert "123-45-6789" not in out
        assert "user@test.com" not in out
        assert "[SSN_REDACTED]" in out
        assert "[EMAIL_REDACTED]" in out


# ── iso_week() ────────────────────────────────────────────────────────────────

class TestIsoWeek:
    def test_format_is_yyyy_dd_mm(self):
        w = iso_week()
        parts = w.split("-")
        assert len(parts) == 3, f"Expected YYYY-DD-MM, got {w!r}"
        year, day, month = parts
        assert year.isdigit() and len(year) == 4
        assert day.isdigit()   and 1 <= int(day)   <= 31
        assert month.isdigit() and 1 <= int(month) <= 12

    def test_uses_provided_datetime(self):
        dt = datetime(2026, 4, 4, tzinfo=timezone.utc)   # April 4, 2026
        assert iso_week(dt) == "2026-04-04"

    def test_first_day_of_year(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert iso_week(dt) == "2026-01-01"


# ── WeeklyReporter ────────────────────────────────────────────────────────────

class TestWeeklyReporter:
    def test_creates_week_directory(self, tmp_path):
        r = WeeklyReporter(output_dir=str(tmp_path))
        assert r.week_dir.exists()

    def test_creates_audit_subdirectory(self, tmp_path):
        r = WeeklyReporter(output_dir=str(tmp_path))
        assert (r.week_dir / "audit_logs").exists()

    def test_write_metrics_creates_valid_json(self, tmp_path):
        r = WeeklyReporter(output_dir=str(tmp_path))
        m = ModelMetrics(model_id="m1", display_name="M1", week=r.week)
        m.asr = 5.0
        m.for_rate = 2.0
        r.write_metrics([m])
        out = r.week_dir / "metrics.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert "week" in data
        assert "models" in data
        assert data["models"][0]["model_id"] == "m1"

    def test_write_raw_appends_jsonl(self, tmp_path):
        r = WeeklyReporter(output_dir=str(tmp_path))
        entries = [
            {"prompt_id": "p1", "prompt": "hello", "response": "hi",
             "latency_ms": 100, "category": "General", "sample_type": "attack"},
        ]
        r.write_raw("model-a", entries)
        r.write_raw("model-b", entries)
        lines = (r.week_dir / "raw_responses.jsonl").read_text().splitlines()
        assert len(lines) == 2

    def test_write_scored_appends_jsonl(self, tmp_path):
        r = WeeklyReporter(output_dir=str(tmp_path))
        s = make_scored()
        r.write_scored([s, s])
        lines = (r.week_dir / "scored.jsonl").read_text().splitlines()
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert "prompt_id" in row


# ── generate_audit_logs() threshold ──────────────────────────────────────────

class TestAuditLogThreshold:
    """
    Verifies the fixed threshold logic:
      • CRITICAL_FAIL → always logged
      • FAIL + severity >= 7 → logged
      • FAIL + severity < 7  → NOT logged
      • PASS / OVER_REFUSAL  → NOT logged
    """

    def _reporter_with_samples(self, scored_list, tmp_path):
        r = WeeklyReporter(output_dir=str(tmp_path))
        resp_map = {
            s.prompt_id: {"prompt": "test prompt", "response": "test response",
                          "agentic_turns": []}
            for s in scored_list
        }
        r.generate_audit_logs(scored_list, resp_map, "Test Model")
        return r

    def test_critical_fail_always_logged(self, tmp_path):
        s = make_scored(prompt_id="cf1", final=CRITICAL_FAIL, severity=3)
        r = self._reporter_with_samples([s], tmp_path)
        logs = list((r.week_dir / "audit_logs").iterdir())
        assert len(logs) == 1

    def test_fail_high_severity_logged(self, tmp_path):
        s = make_scored(prompt_id="f1", final=FAIL, severity=8)
        r = self._reporter_with_samples([s], tmp_path)
        logs = list((r.week_dir / "audit_logs").iterdir())
        assert len(logs) == 1

    def test_fail_low_severity_not_logged(self, tmp_path):
        s = make_scored(prompt_id="f2", final=FAIL, severity=5)
        r = self._reporter_with_samples([s], tmp_path)
        logs = list((r.week_dir / "audit_logs").iterdir())
        assert len(logs) == 0

    def test_pass_not_logged(self, tmp_path):
        s = make_scored(prompt_id="p1", final=PASS, severity=0)
        r = self._reporter_with_samples([s], tmp_path)
        logs = list((r.week_dir / "audit_logs").iterdir())
        assert len(logs) == 0

    def test_over_refusal_not_logged(self, tmp_path):
        s = make_scored(prompt_id="or1", sample_type="refusal",
                        final=OVER_REFUSAL, severity=0)
        r = self._reporter_with_samples([s], tmp_path)
        logs = list((r.week_dir / "audit_logs").iterdir())
        assert len(logs) == 0

    def test_mixed_batch_correct_count(self, tmp_path):
        samples = [
            make_scored("cf1", final=CRITICAL_FAIL, severity=2),  # logged
            make_scored("f1",  final=FAIL, severity=9),           # logged
            make_scored("f2",  final=FAIL, severity=4),           # NOT logged
            make_scored("p1",  final=PASS, severity=0),           # NOT logged
        ]
        r = self._reporter_with_samples(samples, tmp_path)
        logs = list((r.week_dir / "audit_logs").iterdir())
        assert len(logs) == 2


# ── write_audit_log() content ─────────────────────────────────────────────────

class TestAuditLogContent:
    def test_audit_log_has_required_fields(self, tmp_path):
        r = WeeklyReporter(output_dir=str(tmp_path))
        s = make_scored(prompt_id="cf1", final=CRITICAL_FAIL, severity=9)
        path = r.write_audit_log(s, "bad prompt", "bad response", "Test Model", [])
        data = json.loads(Path(path).read_text())
        for key in ("audit_id", "compliance_ref", "nist_function",
                    "nist_category", "remediation", "chain_of_reasoning",
                    "verdict", "severity"):
            assert key in data, f"Missing key: {key}"

    def test_compliance_ref_is_gsar(self, tmp_path):
        r = WeeklyReporter(output_dir=str(tmp_path))
        s = make_scored(prompt_id="cf1", final=CRITICAL_FAIL, severity=9)
        path = r.write_audit_log(s, "prompt", "response", "Test Model")
        data = json.loads(Path(path).read_text())
        assert "GSAR" in data["compliance_ref"]

    def test_pii_redacted_in_audit_log(self, tmp_path):
        r = WeeklyReporter(output_dir=str(tmp_path))
        s = make_scored(prompt_id="pii1", final=CRITICAL_FAIL, severity=9)
        path = r.write_audit_log(s, "SSN: 123-45-6789", "456-78-9012", "Test Model")
        data = json.loads(Path(path).read_text())
        raw = json.dumps(data)
        assert "123-45-6789" not in raw
        assert "456-78-9012" not in raw
        assert "[SSN_REDACTED]" in raw


# ── get_remediation() ─────────────────────────────────────────────────────────

class TestGetRemediation:
    def test_known_categories_return_specific_guidance(self):
        text = get_remediation("PII Leakage")
        assert len(text) > 10
        assert text != get_remediation("__unknown__")

    def test_unknown_category_returns_fallback(self):
        text = get_remediation("NonExistentCategory")
        assert "Review output policy" in text

    def test_critical_categories_say_critical(self):
        for cat in ("CBRN", "Child Safety"):
            assert "CRITICAL" in get_remediation(cat).upper()
