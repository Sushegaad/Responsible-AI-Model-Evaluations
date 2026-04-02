"""
test/test_dataset.py  —  Tests for eval/dataset.py

Covers:
  • _proportional(): proportional allocation across categories
  • _proportional(): handles edge cases (n=0, n > pool, single category)
  • _stratified(): attack/refusal split, priority domain weighting, reproducibility
  • RedBenchSample dataclass: fields correctly typed
  • load_redbench(): loads from local parquet snapshot, derives sample_type correctly,
    returns correct counts, generates unique prompt_ids
"""
from __future__ import annotations
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from eval.dataset import RedBenchSample, _proportional, _stratified, load_redbench

# Path to the local snapshot (two levels up from test/, then into the snapshot dir)
_SNAPSHOT_DIR = str(Path(__file__).parent.parent / "knoveleng-redbench-April2026")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pool(
    n: int,
    categories: list[str] | None = None,
    domains: list[str] | None = None,
    sample_type: str = "attack",
) -> list[RedBenchSample]:
    cats = categories or ["Toxicity", "Hate Speech", "PII Leakage"]
    doms = domains or ["General"]
    pool = []
    for i in range(n):
        pool.append(RedBenchSample(
            prompt_id=f"p{i:04d}",
            prompt=f"Test prompt {i}",
            category=cats[i % len(cats)],
            domain=doms[i % len(doms)],
            attack_type="Direct",
            sample_type=sample_type,
            source_dataset="synthetic",
        ))
    return pool


# ── RedBenchSample dataclass ──────────────────────────────────────────────────

class TestRedBenchSample:
    def test_fields_present(self):
        s = _make_pool(1)[0]
        assert hasattr(s, "prompt_id")
        assert hasattr(s, "prompt")
        assert hasattr(s, "category")
        assert hasattr(s, "domain")
        assert hasattr(s, "attack_type")
        assert hasattr(s, "sample_type")
        assert hasattr(s, "source_dataset")

    def test_fields_are_strings(self):
        s = _make_pool(1)[0]
        for attr in ("prompt_id", "prompt", "category", "domain",
                     "attack_type", "sample_type", "source_dataset"):
            assert isinstance(getattr(s, attr), str)


# ── _proportional() ───────────────────────────────────────────────────────────

class TestProportional:
    def test_returns_n_samples(self):
        pool = _make_pool(100)
        rng = random.Random(42)
        result = _proportional(pool, 30, rng)
        assert len(result) == 30

    def test_returns_all_when_n_ge_pool(self):
        pool = _make_pool(10)
        rng = random.Random(42)
        result = _proportional(pool, 10, rng)
        assert len(result) == 10

    def test_returns_empty_for_zero_n(self):
        pool = _make_pool(50)
        rng = random.Random(42)
        assert _proportional(pool, 0, rng) == []

    def test_returns_empty_for_empty_pool(self):
        rng = random.Random(42)
        assert _proportional([], 10, rng) == []

    def test_proportional_across_categories(self):
        """Each category should get roughly n * (cat_size / total) samples."""
        # Pool: 60% Toxicity, 40% Hate Speech
        big = _make_pool(60, categories=["Toxicity"])
        small = _make_pool(40, categories=["Hate Speech"])
        pool = big + small
        rng = random.Random(42)
        result = _proportional(pool, 50, rng)
        cats = [s.category for s in result]
        n_tox = cats.count("Toxicity")
        n_hs  = cats.count("Hate Speech")
        # Toxicity should be roughly 30 (±5), Hate Speech roughly 20 (±5)
        assert 20 <= n_tox <= 40, f"Toxicity count {n_tox} out of expected range"
        assert 10 <= n_hs  <= 30, f"Hate Speech count {n_hs} out of expected range"

    def test_no_duplicates(self):
        pool = _make_pool(100)
        rng = random.Random(42)
        result = _proportional(pool, 50, rng)
        ids = [s.prompt_id for s in result]
        assert len(ids) == len(set(ids)), "Duplicate samples in proportional result"

    def test_single_category_pool(self):
        pool = _make_pool(20, categories=["Toxicity"])
        rng = random.Random(42)
        result = _proportional(pool, 10, rng)
        assert len(result) == 10
        assert all(s.category == "Toxicity" for s in result)


# ── _stratified() ─────────────────────────────────────────────────────────────

class TestStratified:
    def test_returns_n_samples(self):
        pool = _make_pool(200)
        rng = random.Random(42)
        result = _stratified(pool, 50, rng, priority_domains=None)
        assert len(result) == 50

    def test_no_duplicates(self):
        pool = _make_pool(200)
        rng = random.Random(42)
        result = _stratified(pool, 100, rng, priority_domains=None)
        ids = [s.prompt_id for s in result]
        assert len(ids) == len(set(ids))

    def test_reproducible_with_same_seed(self):
        pool = _make_pool(200)
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        r1 = _stratified(pool, 50, rng1, priority_domains=None)
        r2 = _stratified(pool, 50, rng2, priority_domains=None)
        assert [s.prompt_id for s in r1] == [s.prompt_id for s in r2]

    def test_different_seeds_different_results(self):
        pool = _make_pool(500)
        r1 = _stratified(pool, 100, random.Random(1), priority_domains=None)
        r2 = _stratified(pool, 100, random.Random(2), priority_domains=None)
        assert [s.prompt_id for s in r1] != [s.prompt_id for s in r2]

    def test_priority_domains_get_40_percent(self):
        """Priority domain samples should make up roughly 40% of the result."""
        prio_doms = ["Government & Administrative", "Legal"]
        other_doms = ["Entertainment", "E-Commerce", "Manufacturing"]
        prio_pool  = _make_pool(300, domains=prio_doms)
        other_pool = _make_pool(300, domains=other_doms)
        pool = prio_pool + other_pool
        rng = random.Random(42)
        result = _stratified(pool, 100, rng, priority_domains=prio_doms)
        n_prio = sum(1 for s in result if s.domain in prio_doms)
        # Expect ~40 ± 10
        assert 25 <= n_prio <= 55, f"Priority domain sample count {n_prio} out of expected ~40"

    def test_empty_pool_returns_empty(self):
        rng = random.Random(42)
        assert _stratified([], 10, rng, priority_domains=None) == []

    def test_zero_n_returns_empty(self):
        pool = _make_pool(50)
        rng = random.Random(42)
        assert _stratified(pool, 0, rng, priority_domains=None) == []

    def test_n_larger_than_pool_returns_all(self):
        pool = _make_pool(10)
        rng = random.Random(42)
        result = _stratified(pool, 100, rng, priority_domains=None)
        assert len(result) == 10


# ── load_redbench() — integration tests against local snapshot ────────────────

class TestLoadRedbench:
    """Integration tests that read from the local parquet snapshot.
    Fast (~0.5 s) — no network calls, no API keys required."""

    def test_returns_requested_number_of_samples(self):
        samples = load_redbench(num_samples=40, seed=42, snapshot_dir=_SNAPSHOT_DIR)
        assert len(samples) == 40

    def test_all_fields_are_non_empty_strings(self):
        samples = load_redbench(num_samples=20, seed=1, snapshot_dir=_SNAPSHOT_DIR)
        for s in samples:
            for attr in ("prompt_id", "prompt", "category", "domain",
                         "attack_type", "sample_type", "source_dataset"):
                val = getattr(s, attr)
                assert isinstance(val, str) and val.strip(), (
                    f"{attr!r} is empty or not a string on sample {s.prompt_id}"
                )

    def test_sample_type_derived_from_category(self):
        """'No Risk' category → refusal; all others → attack."""
        samples = load_redbench(num_samples=100, seed=7, snapshot_dir=_SNAPSHOT_DIR)
        for s in samples:
            if s.category == "No Risk":
                assert s.sample_type == "refusal", (
                    f"Expected refusal for 'No Risk' but got {s.sample_type!r}"
                )
            else:
                assert s.sample_type == "attack", (
                    f"Expected attack for category {s.category!r} but got {s.sample_type!r}"
                )

    def test_attack_ratio_respected(self):
        """Default 80 % attack ratio should produce ~32 attacks in 40 samples."""
        samples = load_redbench(num_samples=40, attack_ratio=0.80, seed=42,
                                snapshot_dir=_SNAPSHOT_DIR)
        n_attack = sum(1 for s in samples if s.sample_type == "attack")
        assert n_attack == 32, f"Expected 32 attack samples, got {n_attack}"

    def test_prompt_ids_are_unique(self):
        samples = load_redbench(num_samples=50, seed=99, snapshot_dir=_SNAPSHOT_DIR)
        ids = [s.prompt_id for s in samples]
        assert len(ids) == len(set(ids)), "Duplicate prompt_ids found"

    def test_reproducible_with_same_seed(self):
        s1 = load_redbench(num_samples=30, seed=123, snapshot_dir=_SNAPSHOT_DIR)
        s2 = load_redbench(num_samples=30, seed=123, snapshot_dir=_SNAPSHOT_DIR)
        assert [s.prompt_id for s in s1] == [s.prompt_id for s in s2]

    def test_different_seeds_give_different_results(self):
        s1 = load_redbench(num_samples=30, seed=1, snapshot_dir=_SNAPSHOT_DIR)
        s2 = load_redbench(num_samples=30, seed=2, snapshot_dir=_SNAPSHOT_DIR)
        assert [s.prompt_id for s in s1] != [s.prompt_id for s in s2]

    def test_missing_snapshot_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No parquet files found"):
            load_redbench(num_samples=10, seed=0, snapshot_dir=str(tmp_path))
