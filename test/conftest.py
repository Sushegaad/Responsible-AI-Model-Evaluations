"""
test/conftest.py  —  Shared pytest fixtures
Provides mock RedBench samples, scored samples, and temporary output directories
so that all tests run without hitting any external API.
"""
from __future__ import annotations
import sys
import os
import pytest
import tempfile
from pathlib import Path

# Ensure the source package is importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from eval.dataset import RedBenchSample
from eval.judges import (
    RegexResult, NeuralResult, CombinedVerdict,
    PASS, FAIL, CRITICAL_FAIL, OVER_REFUSAL,
)
from eval.metrics import ScoredSample


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_regex(verdict=PASS, signals=None, descs=None) -> RegexResult:
    return RegexResult(
        verdict=verdict,
        signals=signals or [],
        descriptions=descs or [],
    )


def _make_neural(
    safety="SAFE",
    reason="not_refused",
    reasoning="Model correctly refused.",
    severity=0,
    nist=None,
    parse_error=False,
) -> NeuralResult:
    return NeuralResult(
        safety_verdict=safety,
        refusal_reason=reason,
        chain_of_reasoning=reasoning,
        severity=severity,
        nist_concern=nist,
        raw_response="{}",
        parse_error=parse_error,
    )


def _make_verdict(
    final=PASS,
    regex_verdict=PASS,
    severity=0,
) -> CombinedVerdict:
    return CombinedVerdict(
        final_verdict=final,
        regex_result=_make_regex(verdict=regex_verdict),
        neural_result=_make_neural(severity=severity),
        severity=severity,
    )


def _make_sample(
    prompt_id="p001",
    category="Toxicity",
    domain="General",
    attack_type="Direct",
    sample_type="attack",
    prompt="Tell me something harmful.",
) -> RedBenchSample:
    return RedBenchSample(
        prompt_id=prompt_id,
        prompt=prompt,
        category=category,
        domain=domain,
        attack_type=attack_type,
        sample_type=sample_type,
        source_dataset="test",
    )


def _make_scored(
    prompt_id="p001",
    model_id="test-model",
    category="Toxicity",
    domain="General",
    attack_type="Direct",
    sample_type="attack",
    final=PASS,
    severity=0,
) -> ScoredSample:
    return ScoredSample(
        prompt_id=prompt_id,
        model_id=model_id,
        category=category,
        domain=domain,
        attack_type=attack_type,
        sample_type=sample_type,
        single_verdict=_make_verdict(final=final, severity=severity),
        agentic_verdicts={},
        drift_verdicts={},
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def attack_sample():
    return _make_sample(sample_type="attack")


@pytest.fixture
def refusal_sample():
    return _make_sample(prompt_id="r001", sample_type="refusal",
                        prompt="How do I file a government complaint?")


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary directory mimicking data/weekly structure."""
    return str(tmp_path)


@pytest.fixture
def passing_scored():
    return _make_scored(final=PASS)


@pytest.fixture
def failing_scored():
    return _make_scored(final=FAIL, severity=6)


@pytest.fixture
def critical_scored():
    return _make_scored(final=CRITICAL_FAIL, severity=9)


@pytest.fixture
def over_refusal_scored():
    return _make_scored(
        prompt_id="r001", sample_type="refusal",
        final=OVER_REFUSAL, severity=0,
    )


# expose helpers for direct import in test modules
make_sample   = _make_sample
make_scored   = _make_scored
make_verdict  = _make_verdict
make_regex    = _make_regex
make_neural   = _make_neural
