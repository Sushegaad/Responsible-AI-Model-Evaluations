"""
eval/dataset.py
Loads RedBench from HuggingFace (knoveleng/redbench, MIT License) and
returns a reproducible stratified sample for a weekly evaluation run.
"""
from __future__ import annotations
import logging
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RedBenchSample:
    prompt_id:      str
    prompt:         str
    category:       str   # one of 22 risk categories
    domain:         str   # one of 19 domains
    attack_type:    str   # e.g. Jailbreak, Roleplay, Direct
    sample_type:    str   # "attack" | "refusal"
    source_dataset: str   # originating sub-benchmark (e.g. HarmBench)


def load_redbench(
    attack_ratio: float = 0.80,
    num_samples: int = 500,
    seed: Optional[int] = None,
    priority_domains: Optional[list[str]] = None,
) -> list[RedBenchSample]:
    """
    Load and stratify-sample RedBench.

    Stratification guarantees:
      • attack_ratio  of samples are attack prompts
      • remaining     are refusal prompts
      • samples drawn proportionally from all 22 risk categories
      • priority_domains get 40 % weight when specified
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("pip install datasets")

    logger.info("Loading knoveleng/redbench from HuggingFace …")
    try:
        ds = load_dataset("knoveleng/redbench", split="train", trust_remote_code=True)
    except Exception as exc:
        logger.error("Dataset load failed: %s", exc)
        raise

    logger.info("Total RedBench samples: %d", len(ds))
    rng = random.Random(seed)

    all_samples: list[RedBenchSample] = []
    for row in ds:
        prompt = str(row.get("prompt") or row.get("text") or row.get("instruction") or "")
        if not prompt:
            continue
        all_samples.append(RedBenchSample(
            prompt_id      = str(row.get("prompt_id") or row.get("id") or ""),
            prompt         = prompt,
            category       = str(row.get("category")    or row.get("risk_category") or "General"),
            domain         = str(row.get("domain")       or "General"),
            attack_type    = str(row.get("attack_type")  or row.get("attack") or "Direct"),
            sample_type    = str(row.get("type")         or row.get("sample_type") or "attack"),
            source_dataset = str(row.get("source")       or row.get("dataset") or "unknown"),
        ))

    attack_pool  = [s for s in all_samples if s.sample_type == "attack"]
    refusal_pool = [s for s in all_samples if s.sample_type == "refusal"]
    logger.info("Attack pool: %d  |  Refusal pool: %d", len(attack_pool), len(refusal_pool))

    n_attack  = int(num_samples * attack_ratio)
    n_refusal = num_samples - n_attack

    attack_sampled  = _stratified(attack_pool,  n_attack,  rng, priority_domains)
    refusal_sampled = _stratified(refusal_pool, n_refusal, rng, priority_domains)

    combined = attack_sampled + refusal_sampled
    rng.shuffle(combined)

    # de-duplicate by prompt_id
    seen: set[str] = set()
    final: list[RedBenchSample] = []
    for s in combined:
        if s.prompt_id not in seen:
            seen.add(s.prompt_id)
            final.append(s)

    logger.info(
        "Final sample: %d  (%d attack + %d refusal)",
        len(final), len(attack_sampled), len(refusal_sampled),
    )
    return final


# ── internal helpers ─────────────────────────────────────────────────────────

def _stratified(
    pool: list[RedBenchSample],
    n: int,
    rng: random.Random,
    priority_domains: Optional[list[str]],
) -> list[RedBenchSample]:
    if not pool or n <= 0:
        return []
    if n >= len(pool):
        return list(pool)

    if priority_domains:
        prio   = [s for s in pool if s.domain in priority_domains]
        others = [s for s in pool if s.domain not in priority_domains]
        n_prio   = min(int(n * 0.4), len(prio))
        n_others = n - n_prio
        result   = _proportional(prio, n_prio, rng) + _proportional(others, n_others, rng)
    else:
        result = _proportional(pool, n, rng)

    rng.shuffle(result)
    return result[:n]


def _proportional(
    pool: list[RedBenchSample], n: int, rng: random.Random
) -> list[RedBenchSample]:
    if not pool or n <= 0:
        return []
    if n >= len(pool):
        return list(pool)
    by_cat: dict[str, list[RedBenchSample]] = {}
    for s in pool:
        by_cat.setdefault(s.category, []).append(s)
    total = len(pool)
    sampled: list[RedBenchSample] = []
    for items in by_cat.values():
        k = max(1, round(n * len(items) / total))
        sampled.extend(rng.sample(items, min(k, len(items))))
    rng.shuffle(sampled)
    if len(sampled) > n:
        return sampled[:n]
    if len(sampled) < n:
        used = set(id(s) for s in sampled)
        rest = [s for s in pool if id(s) not in used]
        extra = rng.sample(rest, min(n - len(sampled), len(rest)))
        sampled.extend(extra)
    return sampled[:n]
