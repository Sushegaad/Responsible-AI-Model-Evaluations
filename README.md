# RAI Eval — Responsible AI Transparency Dashboard

> Weekly automated evaluation of frontier LLMs against the
> [RedBench](https://arxiv.org/abs/2601.03699) adversarial dataset.
> Results published to a public GitHub Pages dashboard every Sunday.

[![Weekly Eval](https://github.com/YOUR_ORG/rai-eval/actions/workflows/weekly_eval.yml/badge.svg)](https://github.com/YOUR_ORG/rai-eval/actions)
[![Dataset: RedBench](https://img.shields.io/badge/Dataset-RedBench-blue)](https://huggingface.co/datasets/knoveleng/redbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Live Dashboard

**[→ https://YOUR_ORG.github.io/rai-eval/](https://YOUR_ORG.github.io/rai-eval/)**

---

## What This Evaluates

| Provider   | Models                                          |
|------------|-------------------------------------------------|
| Anthropic  | Claude Opus 4.5, Claude Sonnet 4.5              |
| OpenAI     | GPT-4o, GPT-4o Mini                             |
| Google     | Gemini 2.0 Flash, Gemini 1.5 Pro                |
| Meta       | Llama 3.3 70B Instruct, Llama 3.1 8B Instruct   |

## Metrics

| Metric               | Definition                                      | Target      |
|----------------------|-------------------------------------------------|-------------|
| **ASR**              | % of attack prompts that bypassed guardrails    | < 5 %       |
| **FOR**              | % of safe prompts incorrectly refused           | < 10 %      |
| **Drift Coefficient**| Safety change per conversation turn             | < 0.5 %/turn|
| **Provenance Score** | % with complete chain-of-reasoning audit trail  | > 95 %      |

## Quick Start

```bash
git clone https://github.com/YOUR_ORG/rai-eval.git
cd rai-eval
pip install -r requirements.txt

cp .env.template .env
# edit .env with your API keys

# Dataset sanity check (no API calls)
python -m eval.pipeline --dry-run

# Lightweight test run (~$1)
python -m eval.pipeline --models claude-sonnet-4-5 gpt-4o-mini --samples 20

# Full weekly run (all 8 models, 500 samples each, ~$29)
python -m eval.pipeline

# Build dashboard data file
python scripts/build_dashboard_data.py
```

## Repository Structure

```
rai-eval/
├── eval/
│   ├── config.py      # Model registry, taxonomy, NIST mapping
│   ├── dataset.py     # RedBench loader + stratified sampler
│   ├── runner.py      # Phase 2: adversarial execution
│   ├── judges.py      # Phase 3: RegexJudge + NeuralJudge
│   ├── metrics.py     # ASR / FOR / Drift / Provenance
│   ├── reporter.py    # JSONL logs + forensic audit trail
│   └── pipeline.py    # CLI orchestrator (entry point)
├── data/weekly/       # Auto-generated; one folder per ISO week
│   └── YYYY-WW/
│       ├── raw_responses.jsonl
│       ├── scored.jsonl
│       ├── metrics.json
│       └── audit_logs/
├── dashboard/         # GitHub Pages site
│   ├── index.html
│   └── data/results.json  (auto-generated)
├── scripts/
│   └── build_dashboard_data.py
└── .github/workflows/weekly_eval.yml
```

## GitHub Actions Setup

1. Fork this repository
2. Add secrets: **Settings → Secrets → Actions**
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
   - `GOOGLE_API_KEY`
   - `TOGETHER_API_KEY`
3. Enable GitHub Pages: **Settings → Pages → Source: GitHub Actions**
4. The workflow runs automatically every Sunday at 00:00 UTC
5. To trigger manually: **Actions → Weekly RAI Evaluation → Run workflow**

## Methodology Summary

| Phase | Name                    | What Happens                                                       |
|-------|-------------------------|--------------------------------------------------------------------|
| 1     | Taxonomy Alignment      | Load & stratify 500 RedBench prompts per model (ISO-week seed)     |
| 2     | Adversarial Execution   | Single-turn → agentic 3-turn wrap → 10-turn drift probe (5 %)      |
| 3     | Dual-Judge Evaluation   | RegexJudge (deterministic) + NeuralJudge (Claude Haiku)            |
| 4     | Compliance Reporting    | NIST AI RMF mapping, GSAR audit logs, dashboard metrics            |

## Cost Estimate

Full 8-model run ≈ **~$29 / week**.
Use `--samples 50` to run a $3 smoke test, or configure `REDEVAL_NUM_SAMPLES=100` for ~$6/week.

## Citation

```bibtex
@misc{dang2026redbench,
  title={RedBench: A Universal Dataset for Comprehensive Red Teaming of Large Language Models},
  author={Quy-Anh Dang and Chris Ngo and Truong-Son Hy},
  year={2026}, eprint={2601.03699}, archivePrefix={arXiv},
  url={https://arxiv.org/abs/2601.03699}
}
```

## License

MIT. RedBench dataset is MIT-licensed — see [knoveleng/redbench](https://huggingface.co/datasets/knoveleng/redbench).

> **Ethics note**: This project uses RedBench prompts as-is for evaluation only.
> It never generates new attack prompts. All audit logs are PII-redacted before git commit.
