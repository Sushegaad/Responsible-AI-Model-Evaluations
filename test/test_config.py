"""
test/test_config.py  —  Tests for eval/config.py

Covers:
  • 22 risk categories present and unique
  • 19 domains present and unique
  • All 22 risk categories have a NIST AI RMF mapping
  • No NIST CSF-only labels (PROTECT / DETECT) remain
  • Valid NIST AI RMF function names only (GOVERN / MAP / MEASURE / MANAGE)
  • Model registry: correct providers, unique IDs, current Claude version strings
  • EvalConfig.from_env reads environment variables correctly
  • get_api_key raises EnvironmentError for missing keys
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from eval.config import (
    RISK_CATEGORIES, DOMAINS, PRIORITY_DOMAINS,
    NIST_MAPPING, MODEL_REGISTRY, EvalConfig, get_api_key,
    JUDGE_MODEL_ID, BASELINE_SYSTEM_PROMPT,
)

VALID_NIST_FUNCTIONS = {"GOVERN", "MAP", "MEASURE", "MANAGE"}
INVALID_CSF_LABELS   = {"PROTECT", "DETECT", "RESPOND", "RECOVER", "IDENTIFY"}


class TestRiskTaxonomy:
    def test_exactly_22_risk_categories(self):
        assert len(RISK_CATEGORIES) == 22

    def test_risk_categories_unique(self):
        assert len(RISK_CATEGORIES) == len(set(RISK_CATEGORIES))

    def test_exactly_19_domains(self):
        assert len(DOMAINS) == 19

    def test_domains_unique(self):
        assert len(DOMAINS) == len(set(DOMAINS))

    def test_priority_domains_are_subset_of_domains(self):
        for d in PRIORITY_DOMAINS:
            assert d in DOMAINS, f"Priority domain '{d}' not in DOMAINS"

    def test_required_priority_domains_present(self):
        required = {"Government & Administrative", "Legal", "Cybersecurity"}
        for d in required:
            assert d in PRIORITY_DOMAINS


class TestNistMapping:
    def test_all_22_categories_mapped(self):
        for cat in RISK_CATEGORIES:
            assert cat in NIST_MAPPING, f"'{cat}' has no NIST mapping"

    def test_no_csf_only_labels_used(self):
        """PROTECT and DETECT belong to NIST CSF, not AI RMF — must not appear."""
        for cat, (fn, _label) in NIST_MAPPING.items():
            assert fn not in INVALID_CSF_LABELS, (
                f"Category '{cat}' uses CSF function '{fn}' — should use an AI RMF function"
            )

    def test_only_valid_nist_ai_rmf_functions(self):
        for cat, (fn, _label) in NIST_MAPPING.items():
            assert fn in VALID_NIST_FUNCTIONS, (
                f"Category '{cat}' has unknown NIST function '{fn}'"
            )

    def test_mapping_values_are_tuples_of_two_strings(self):
        for cat, value in NIST_MAPPING.items():
            assert isinstance(value, tuple) and len(value) == 2, \
                f"NIST_MAPPING['{cat}'] should be a (function, label) tuple"
            fn, label = value
            assert isinstance(fn, str) and isinstance(label, str)

    def test_pii_categories_under_govern(self):
        """PII/Privacy categories should be under GOVERN after the CSF fix."""
        for cat in ("PII Leakage", "Privacy Violation", "Surveillance"):
            fn, _ = NIST_MAPPING[cat]
            assert fn == "GOVERN", f"'{cat}' should be GOVERN, got '{fn}'"

    def test_cyber_categories_under_measure(self):
        """Cybercrime/Malware/Prompt Injection should be MEASURE after the CSF fix."""
        for cat in ("Cybercrime", "Malware", "Prompt Injection"):
            fn, _ = NIST_MAPPING[cat]
            assert fn == "MEASURE", f"'{cat}' should be MEASURE, got '{fn}'"

    def test_critical_categories_under_manage(self):
        for cat in ("Child Safety", "CBRN", "Extremism"):
            fn, _ = NIST_MAPPING[cat]
            assert fn in {"MANAGE", "MEASURE"}, f"'{cat}' should be MANAGE or MEASURE, got '{fn}'"


class TestModelRegistry:
    def test_registry_not_empty(self):
        assert len(MODEL_REGISTRY) > 0

    def test_model_ids_unique(self):
        ids = [m.model_id for m in MODEL_REGISTRY]
        assert len(ids) == len(set(ids)), "Duplicate model IDs in registry"

    def test_all_providers_known(self):
        known = {"anthropic", "openai", "google", "together"}
        for m in MODEL_REGISTRY:
            assert m.provider in known, f"Unknown provider '{m.provider}' for {m.model_id}"

    def test_claude_models_use_current_versions(self):
        """Ensure the outdated 4.5 model strings have been updated to 4.6."""
        for m in MODEL_REGISTRY:
            if m.provider == "anthropic":
                assert "4-5" not in m.api_model_str, (
                    f"{m.model_id} still references deprecated 4.5 API string: {m.api_model_str}"
                )
                assert "4-6" in m.api_model_str or "haiku" in m.api_model_str.lower(), \
                    f"Claude model {m.model_id} does not use a 4.6 string: {m.api_model_str}"

    def test_all_models_have_color(self):
        for m in MODEL_REGISTRY:
            assert m.color.startswith("#") and len(m.color) == 7, \
                f"{m.model_id} has invalid color: {m.color}"

    def test_anthropic_provider_present(self):
        providers = {m.provider for m in MODEL_REGISTRY}
        assert "anthropic" in providers

    def test_judge_model_is_haiku(self):
        assert "haiku" in JUDGE_MODEL_ID.lower()

    def test_baseline_prompt_mentions_gsar(self):
        assert "GSAR" in BASELINE_SYSTEM_PROMPT


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.num_samples_per_model == 50
        assert cfg.attack_ratio == 0.80
        assert cfg.drift_max_turns == 10
        assert cfg.output_dir == "data/weekly"

    def test_from_env_reads_samples(self, monkeypatch):
        monkeypatch.setenv("REDEVAL_NUM_SAMPLES", "42")
        cfg = EvalConfig.from_env()
        assert cfg.num_samples_per_model == 42

    def test_from_env_reads_output_dir(self, monkeypatch):
        monkeypatch.setenv("REDEVAL_OUTPUT_DIR", "/tmp/test-output")
        cfg = EvalConfig.from_env()
        assert cfg.output_dir == "/tmp/test-output"

    def test_from_env_reads_log_level(self, monkeypatch):
        monkeypatch.setenv("REDEVAL_LOG_LEVEL", "DEBUG")
        cfg = EvalConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_rate_limits_have_all_providers(self):
        cfg = EvalConfig()
        for provider in ("anthropic", "openai", "google"):
            assert provider in cfg.rate_limits


class TestGetApiKey:
    def test_returns_key_when_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
        assert get_api_key("anthropic") == "sk-ant-test123"

    def test_raises_environment_error_when_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            get_api_key("anthropic")

    def test_raises_value_error_for_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_api_key("unknown_provider")

    def test_all_known_providers_have_env_var_mapping(self, monkeypatch):
        for provider, env_var in [
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openai",    "OPENAI_API_KEY"),
            ("google",    "GOOGLE_API_KEY"),
        ]:
            monkeypatch.setenv(env_var, "test-key")
            assert get_api_key(provider) == "test-key"
