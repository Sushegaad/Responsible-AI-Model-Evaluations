"""
eval/dataset.py
Loads RedBench from a local snapshot (knoveleng-redbench-April2026/) and
returns a reproducible stratified sample for a weekly evaluation run.

Dataset snapshot: knoveleng/redbench (MIT License), downloaded April 2026.
37 sub-benchmark configs, 29 362 rows total.

Column schema (identical across all 37 configs):
  prompt       – adversarial or benign prompt text
  category     – risk category; "No Risk" marks benign/refusal prompts
  domain       – one of 19 domains (Biology, Education, Finance, …)
  source       – originating sub-benchmark name (e.g. HarmBench)
  task         – always "generation"
  subtask      – always "None" (string)
  language     – ISO language code
  choices      – MC answer choices (not used for generation eval)
  answer       – MC correct answer  (not used for generation eval)
  risk_response / risk_property / domain_response / domain_property – metadata

Sample-type derivation:
  category == "No Risk"  →  sample_type = "refusal"
  any other category     →  sample_type = "attack"
"""
from __future__ import annotations

import glob
import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the local dataset snapshot, relative to the repo root.
# The workflow and pipeline always run from the repo root, so this resolves correctly.
_SNAPSHOT_DIR = os.path.join(
    os.path.dirname(__file__),   # source/eval/
    "..", "..",                  # → repo root
    "knoveleng-redbench-April2026",
)

# Category value that identifies safe / refusal-type prompts
_NO_RISK_LABEL = "No Risk"


@dataclass
class RedBenchSample:
    prompt_id:      str
    prompt:         str
    category:       str   # one of 22 risk categories, or "No Risk"
    domain:         str   # one of 19 domains
    attack_type:    str   # sub-benchmark config name (e.g. DAN, GPTFuzzer)
    sample_type:    str   # "attack" | "refusal"
    source_dataset: str   # originating sub-benchmark (same as attack_type)


def load_redbench(
    attack_ratio: float = 0.80,
    num_samples: int = 50,
    seed: Optional[int] = None,
    priority_domains: Optional[list[str]] = None,
    snapshot_dir: Optional[str] = None,
) -> list[RedBenchSample]:
    """
    Load and stratify-sample RedBench from the local parquet snapshot.

    Stratification guarantees:
      • attack_ratio  of samples are attack prompts
      • remaining     are refusal prompts
      • samples drawn proportionally across all risk categories
      • priority_domains get ~40 % weight when specified
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        raise ImportError("pip install pandas pyarrow")

    data_dir = os.path.abspath(snapshot_dir or _SNAPSHOT_DIR)
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in '{data_dir}'. "
            "Make sure the knoveleng-redbench-April2026/ snapshot is present."
        )

    logger.info("Loading RedBench from local snapshot: %s", data_dir)
    logger.info("Found %d config files", len(parquet_files))

    all_samples: list[RedBenchSample] = []

    for fpath in parquet_files:
        config = os.path.splitext(os.path.basename(fpath))[0]  # e.g. "HarmBench"
        try:
            df = pd.read_parquet(fpath)
        except Exception as exc:
            logger.warning("Skipping '%s': %s", config, exc)
            continue

        for idx, row in enumerate(df.itertuples(index=False)):
            prompt = str(row.prompt).strip()
            if not prompt:
                continue

            category    = str(row.category).strip()
            sample_type = "refusal" if category == _NO_RISK_LABEL else "attack"

            all_samples.append(RedBenchSample(
                prompt_id      = f"{config}_{idx:05d}",
                prompt         = prompt,
                category       = category,
                domain         = str(row.domain).strip(),
                attack_type    = config,
                sample_type    = sample_type,
                source_dataset = config,
            ))

        logger.info("Loaded '%s': %d rows", config, len(df))

    logger.info("Total RedBench samples loaded: %d", len(all_samples))

    rng = random.Random(seed)

    attack_pool  = [s for s in all_samples if s.sample_type == "attack"]
    refusal_pool = [s for s in all_samples if s.sample_type == "refusal"]
    logger.info("Attack pool: %d  |  Refusal pool: %d", len(attack_pool), len(refusal_pool))

    n_attack  = int(num_samples * attack_ratio)
    n_refusal = num_samples - n_attack

    attack_sampled  = _stratified(attack_pool,  n_attack,  rng, priority_domains)
    refusal_sampled = _stratified(refusal_pool, n_refusal, rng, priority_domains)

    combined = attack_sampled + refusal_sampled
    rng.shuffle(combined)

    # de-duplicate by prompt_id (deterministic IDs mean no collisions, but be safe)
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
