"""
test/test_metrics.py  —  Tests for eval/metrics.py

Covers:
  • ASR computation (attack failure rate)
  • FOR computation (over-refusal rate on refusal samples)
  • Drift coefficient: positive drift, zero drift, no drift samples
  • Provenance score: full, partial, zero
  • Per-category ASR: populated for attacked categories, zero for untested ones
  • NIST summary: uses correct AI RMF function names, no PROTECT/DETECT
  • ScoredSample.to_dict() serialises all required keys
  • ModelMetrics.to_dict() rounds floats correctly
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from eval.judges import PASS, FAIL, CRITICAL_FAIL, OVER_REFUSAL
from eval.metrics import ScoredSample, ModelMetrics, compute_metrics
from eval.config import RISK_CATEGORIES
from test.conftest import make_scored, make_verdict, make_regex, make_neural


WEEK = "2026-14"
MODEL = "test-model"
DISPLAY = "Test Model"
INVALID_NIST = {"PROTECT", "DETECT", "RESPOND", "RECOVER", "IDENTIFY"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scored_attack(
    pid="a1", cat="Toxicity", final=PASS, severity=0, drift_verdicts=None
) -> ScoredSample:
    s = make_scored(prompt_id=pid, category=cat, sample_type="attack",
                    final=final, severity=severity)
    if drift_verdicts:
        s.drift_verdicts = drift_verdicts
    return s


def _scored_refusal(pid="r1", cat="Toxicity", final=PASS) -> ScoredSample:
    return make_scored(prompt_id=pid, category=cat, sample_type="refusal", final=final)


# ── ASR ───────────────────────────────────────────────────────────────────────

class TestASR:
    def test_zero_asr_all_pass(self):
        scored = [_scored_attack(f"a{i}") for i in range(10)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.asr == 0.0
        assert m.attack_failures == 0

    def test_100_asr_all_fail(self):
        scored = [_scored_attack(f"a{i}", final=FAIL) for i in range(10)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.asr == 100.0

    def test_partial_asr(self):
        scored = (
            [_scored_attack(f"a{i}", final=FAIL) for i in range(4)]
            + [_scored_attack(f"b{i}") for i in range(6)]
        )
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert abs(m.asr - 40.0) < 0.01

    def test_critical_fail_counted_in_asr(self):
        scored = [_scored_attack("cf1", final=CRITICAL_FAIL, severity=9)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.attack_failures == 1
        assert m.critical_failures == 1
        assert m.asr == 100.0

    def test_no_attack_samples_gives_zero_asr(self):
        scored = [_scored_refusal("r1")]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.asr == 0.0
        assert m.total_attack == 0


# ── FOR (False Outright Refusal) ──────────────────────────────────────────────

class TestFOR:
    def test_zero_for_no_over_refusals(self):
        scored = [_scored_refusal(f"r{i}") for i in range(5)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.for_rate == 0.0

    def test_100_for_all_over_refused(self):
        scored = [_scored_refusal(f"r{i}", final=OVER_REFUSAL) for i in range(5)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.for_rate == 100.0

    def test_partial_for(self):
        scored = (
            [_scored_refusal(f"r{i}", final=OVER_REFUSAL) for i in range(2)]
            + [_scored_refusal(f"s{i}") for i in range(8)]
        )
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert abs(m.for_rate - 20.0) < 0.01

    def test_no_refusal_samples_gives_zero_for(self):
        scored = [_scored_attack("a1")]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.for_rate == 0.0
        assert m.total_refusal == 0


# ── Drift Coefficient ─────────────────────────────────────────────────────────

class TestDriftCoefficient:
    def _dv_from_list(self, verdicts: list[str]) -> dict:
        """Create drift_verdicts dict from a list of verdict strings."""
        return {
            i + 1: make_verdict(final=v)
            for i, v in enumerate(verdicts)
        }

    def test_no_drift_samples_gives_zero(self):
        scored = [_scored_attack("a1")]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.drift_coefficient == 0.0

    def test_positive_drift(self):
        """Failure rate increases over turns → positive drift coefficient."""
        # Turn 1: 0% failure, Turn 10: 100% failure → coefficient = (100-0)/(10-1)
        dv = self._dv_from_list(
            [PASS] * 4 + [FAIL] * 3 + [CRITICAL_FAIL] * 3
        )
        scored = [_scored_attack("a1", drift_verdicts=dv)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.drift_coefficient > 0.0

    def test_zero_drift_constant_failures(self):
        """Same failure rate at every turn → zero drift."""
        dv = self._dv_from_list([FAIL] * 5)
        scored = [_scored_attack("a1", drift_verdicts=dv)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.drift_coefficient == 0.0

    def test_negative_drift_model_improves(self):
        """Failure rate decreases over turns → negative drift coefficient."""
        dv = self._dv_from_list([FAIL] * 5 + [PASS] * 5)
        scored = [_scored_attack("a1", drift_verdicts=dv)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.drift_coefficient < 0.0


# ── Provenance Score ──────────────────────────────────────────────────────────

class TestProvenanceScore:
    def _with_reasoning(self, pid, reasoning, parse_error=False) -> ScoredSample:
        s = make_scored(prompt_id=pid)
        s.single_verdict.neural_result.chain_of_reasoning = reasoning
        s.single_verdict.neural_result.parse_error = parse_error
        return s

    def test_full_provenance(self):
        scored = [self._with_reasoning(f"p{i}", "Refused for policy.") for i in range(10)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.provenance_score == 100.0

    def test_zero_provenance_no_reasoning(self):
        scored = [self._with_reasoning(f"p{i}", "") for i in range(5)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert m.provenance_score == 0.0

    def test_parse_errors_excluded_from_provenance(self):
        good = [self._with_reasoning(f"g{i}", "Valid reasoning.") for i in range(7)]
        bad  = [self._with_reasoning(f"b{i}", "text", parse_error=True) for i in range(3)]
        m = compute_metrics(good + bad, MODEL, DISPLAY, WEEK)
        assert abs(m.provenance_score - 70.0) < 0.01


# ── Per-category ASR ──────────────────────────────────────────────────────────

class TestCategoryASR:
    def test_attacked_category_populated(self):
        scored = [
            _scored_attack("a1", cat="Toxicity", final=FAIL),
            _scored_attack("a2", cat="Toxicity", final=PASS),
        ]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert abs(m.category_asr["Toxicity"] - 50.0) < 0.01

    def test_untested_categories_default_zero(self):
        scored = [_scored_attack("a1", cat="Toxicity")]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        for cat in RISK_CATEGORIES:
            assert cat in m.category_asr
        # Untested categories should be 0.0
        assert m.category_asr["CBRN"] == 0.0

    def test_all_22_categories_present_in_output(self):
        scored = [_scored_attack("a1")]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        for cat in RISK_CATEGORIES:
            assert cat in m.category_asr, f"Missing category: {cat}"


# ── NIST Summary ──────────────────────────────────────────────────────────────

class TestNistSummary:
    def test_nist_summary_populated(self):
        scored = [_scored_attack("a1", cat="Toxicity", final=FAIL)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        assert len(m.nist_summary) > 0

    def test_no_csf_functions_in_summary(self):
        """After the fix, no PROTECT or DETECT keys should appear in nist_summary."""
        scored = [_scored_attack(f"a{i}", cat=cat)
                  for i, cat in enumerate(RISK_CATEGORIES)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        for fn in m.nist_summary:
            assert fn not in INVALID_NIST, f"CSF function '{fn}' found in NIST summary"

    def test_nist_summary_has_avg_asr_and_label(self):
        scored = [_scored_attack("a1", cat="Toxicity")]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        for fn, data in m.nist_summary.items():
            assert "avg_asr" in data
            assert "category_label" in data
            assert isinstance(data["avg_asr"], float)


# ── Serialisation ─────────────────────────────────────────────────────────────

class TestSerialisation:
    def test_scored_sample_to_dict_has_required_keys(self):
        s = make_scored()
        d = s.to_dict()
        for key in ("prompt_id", "model_id", "category", "domain",
                    "verdict", "severity", "chain_of_reasoning"):
            assert key in d, f"Missing key: {key}"

    def test_model_metrics_to_dict_rounds_floats(self):
        scored = [_scored_attack(f"a{i}", final=FAIL) for i in range(3)]
        scored += [_scored_attack(f"b{i}") for i in range(7)]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        d = m.to_dict()
        assert isinstance(d["asr"], float)
        assert isinstance(d["for_rate"], float)

    def test_model_metrics_to_dict_has_all_keys(self):
        scored = [_scored_attack("a1")]
        m = compute_metrics(scored, MODEL, DISPLAY, WEEK)
        d = m.to_dict()
        for key in ("model_id", "display_name", "week", "asr", "for_rate",
                    "drift_coefficient", "provenance_score",
                    "category_asr", "nist_summary"):
            assert key in d, f"Missing key: {key}"
