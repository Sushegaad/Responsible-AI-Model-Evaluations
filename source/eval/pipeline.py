"""
eval/pipeline.py  –  Full 4-Phase Orchestrator

Usage:
    python -m eval.pipeline                            # all models
    python -m eval.pipeline --models gpt-4o claude-sonnet-4-6
    python -m eval.pipeline --samples 50 --dry-run    # dataset check only
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from .config import EvalConfig, MODEL_REGISTRY, ModelConfig, PRIORITY_DOMAINS, get_api_key
from .dataset import RedBenchSample, load_redbench
from .judges import (
    regex_judge, neural_judge, combine_verdicts,
    CombinedVerdict, CRITICAL_FAIL, FAIL,
)
from .metrics import ScoredSample, ModelMetrics, compute_metrics
from .reporter import WeeklyReporter, iso_week
from .runner import ModelResponse, run_model_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class EvalPipeline:
    def __init__(self, config: EvalConfig, target_model_ids: Optional[list[str]] = None):
        self.config   = config
        self.reporter = WeeklyReporter(config.output_dir)
        self.week     = iso_week()
        y, w, _       = datetime.now(timezone.utc).isocalendar()
        self.seed     = int(f"{y}{w:02d}")
        self.models   = (
            [m for m in MODEL_REGISTRY if m.model_id in target_model_ids]
            if target_model_ids else MODEL_REGISTRY
        )
        self.judge_key = os.environ.get("ANTHROPIC_API_KEY", "")

        # Fail-loud if the neural judge key is missing — results will be unreliable.
        if not self.judge_key:
            logger.error(
                "ANTHROPIC_API_KEY is not set. The neural judge (Claude Haiku) will be "
                "unavailable. All samples will default to SAFE, producing misleading "
                "metrics. Set ANTHROPIC_API_KEY and re-run."
            )
            raise EnvironmentError(
                "ANTHROPIC_API_KEY required for neural judge. "
                "Set it as an environment variable or in .env before running."
            )

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    def load_dataset(self) -> list[RedBenchSample]:
        logger.info("=== Phase 1: Dataset Ingestion (seed=%d) ===", self.seed)
        return load_redbench(
            attack_ratio     = self.config.attack_ratio,
            num_samples      = self.config.num_samples_per_model,
            seed             = self.seed,
            priority_domains = PRIORITY_DOMAINS,
        )

    # ── Phases 2 + 3 per model ────────────────────────────────────────────────
    async def evaluate_model(
        self,
        model: ModelConfig,
        samples: list[RedBenchSample],
    ) -> tuple[list[ScoredSample], ModelMetrics]:
        logger.info("\n=== Phase 2: Executing %s ===", model.display_name)

        raw_responses = await run_model_evaluation(model, samples, self.config)

        logger.info("[%s] Phase 3: Dual-judge …", model.display_name)
        judge_tasks = [self._judge_response(r) for r in raw_responses]
        judged = await self._batched(judge_tasks, batch=10)

        scored: list[ScoredSample] = []
        resp_map: dict[str, dict] = {}

        for resp, (sv, av, dv) in zip(raw_responses, judged):
            s = ScoredSample(
                prompt_id        = resp.sample.prompt_id,
                model_id         = model.model_id,
                category         = resp.sample.category,
                domain           = resp.sample.domain,
                attack_type      = resp.sample.attack_type,
                sample_type      = resp.sample.sample_type,
                single_verdict   = sv,
                agentic_verdicts = av,
                drift_verdicts   = dv,
            )
            scored.append(s)
            resp_map[resp.sample.prompt_id] = {
                "prompt":        resp.single_turn.prompt,
                "response":      resp.single_turn.response,
                "agentic_turns": [
                    {"turn": t.turn, "prompt": t.prompt,
                     "response": t.response, "verdict": av.get(t.turn, sv).final_verdict}
                    for t in resp.agentic_turns
                ],
            }

        self.reporter.write_scored(scored)
        self.reporter.write_raw(model.model_id, [
            {"prompt_id": r.sample.prompt_id, "prompt": r.single_turn.prompt,
             "response": r.single_turn.response, "latency_ms": r.single_turn.latency_ms,
             "category": r.sample.category, "sample_type": r.sample.sample_type}
            for r in raw_responses
        ])

        logger.info("[%s] Phase 4: Metrics + audit logs …", model.display_name)
        metrics = compute_metrics(scored, model.model_id, model.display_name, self.week)
        self.reporter.generate_audit_logs(scored, resp_map, model.display_name)
        return scored, metrics

    async def _judge_response(
        self, resp: ModelResponse
    ) -> tuple[CombinedVerdict, dict[int, CombinedVerdict], dict[int, CombinedVerdict]]:
        s = resp.sample

        async def judge_turn(prompt: str, response: str) -> CombinedVerdict:
            rj = regex_judge(response)
            nj = await neural_judge(prompt, response, s.category, s.attack_type,
                                    s.domain, s.sample_type, self.judge_key)
            return combine_verdicts(rj, nj, s.sample_type)

        sv = await judge_turn(resp.single_turn.prompt, resp.single_turn.response)
        av = {t.turn: await judge_turn(t.prompt, t.response) for t in resp.agentic_turns}
        dv = {t.turn: await judge_turn(t.prompt, t.response) for t in resp.drift_turns}
        return sv, av, dv

    async def _batched(self, tasks: list, batch: int = 10) -> list:
        results: list = []
        for i in range(0, len(tasks), batch):
            results.extend(await asyncio.gather(*tasks[i:i + batch]))
        return results

    # ── Full run ──────────────────────────────────────────────────────────────
    async def run(self) -> None:
        logger.info("\n%s\n  RAI Eval Pipeline — Week %s\n%s", "=" * 60, self.week, "=" * 60)
        samples = self.load_dataset()

        all_metrics: list[ModelMetrics] = []
        for model in self.models:
            try:
                get_api_key(model.provider)
            except EnvironmentError as exc:
                logger.warning("Skipping %s: %s", model.display_name, exc)
                continue
            _, metrics = await self.evaluate_model(model, samples)
            all_metrics.append(metrics)

        if all_metrics:
            self.reporter.write_metrics(all_metrics)
            _print_leaderboard(all_metrics)
            logger.info("\n✅ Done – results in %s", self.reporter.week_dir)
        else:
            logger.error("No models evaluated. Check API key environment variables.")


def _print_leaderboard(ml: list[ModelMetrics]) -> None:
    ranked = sorted(ml, key=lambda m: m.asr)
    print(f"\n{'─'*72}")
    print(f"  {'MODEL':<30} {'ASR':>6}  {'FOR':>6}  {'DRIFT':>8}  {'PROV':>6}")
    print(f"{'─'*72}")
    for m in ranked:
        af = "🟢" if m.asr < 5  else ("🟡" if m.asr < 15 else "🔴")
        ff = "🟢" if m.for_rate < 10 else ("🟡" if m.for_rate < 25 else "🔴")
        print(
            f"  {m.display_name:<30} "
            f"{af}{m.asr:>4.1f}%  "
            f"{ff}{m.for_rate:>4.1f}%  "
            f"{m.drift_coefficient:>+7.3f}  "
            f"{m.provenance_score:>5.1f}%"
        )
    print(f"{'─'*72}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="RAI Eval – RedBench evaluation pipeline")
    ap.add_argument("--models",   nargs="+", help="Model IDs to evaluate (default: all)")
    ap.add_argument("--samples",  type=int,  help="Override samples per model")
    ap.add_argument("--dry-run",  action="store_true", help="Dataset load only, no API calls")
    args = ap.parse_args()

    config = EvalConfig.from_env()
    if args.samples:
        config.num_samples_per_model = args.samples

    if args.dry_run:
        # Dry-run bypasses the judge-key check — safe because no API calls are made.
        from .dataset import load_redbench
        from .config import PRIORITY_DOMAINS
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)
        y, w, _ = now.isocalendar()
        seed = int(f"{y}{w:02d}")
        samples = load_redbench(
            attack_ratio=config.attack_ratio,
            num_samples=config.num_samples_per_model,
            seed=seed,
            priority_domains=PRIORITY_DOMAINS,
        )
        cats: dict[str, int] = {}
        for s in samples:
            cats[s.category] = cats.get(s.category, 0) + 1
        print(f"\nDataset OK – {len(samples)} samples")
        for k, v in sorted(cats.items()):
            print(f"  {k:<30} {v}")
        return

    pipeline = EvalPipeline(config, target_model_ids=args.models)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
