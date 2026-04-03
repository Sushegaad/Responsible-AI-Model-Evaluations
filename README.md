# RAI Eval — Responsible AI Transparency Dashboard

[![Weekly Eval](https://github.com/Sushegaad/Responsible-AI-Model-Evaluations/actions/workflows/weekly_eval.yml/badge.svg)](https://github.com/Sushegaad/Responsible-AI-Model-Evaluations/actions)
[![Dataset: RedBench](https://img.shields.io/badge/Dataset-RedBench-blue)](https://huggingface.co/datasets/knoveleng/redbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

RAI Eval is an automated, open-source pipeline that evaluates frontier large language models (LLMs) for responsible AI safety and robustness. Every Friday it runs a structured adversarial benchmark against models from Anthropic, OpenAI, and Google, scores them using a deterministic regex judge and a neural judge (Claude Haiku), maps results to the NIST AI Risk Management Framework, and publishes the findings to a public GitHub Pages dashboard. The evaluation dataset — RedBench (29,362 prompts across 37 sub-benchmarks) — is stored locally in the repository as Parquet files, making every run fully reproducible with no external data dependency.

---

## What this does?

### What This Evaluates

| Provider  | Models                             |
|-----------|------------------------------------|
| Anthropic | Claude Opus 4.6, Claude Sonnet 4.6 |
| OpenAI    | GPT-4o, GPT-4o Mini                |
| Google    | Gemini 2.0 Flash, Gemini 1.5 Pro   |

### Metrics

| Metric                | Definition                                      | Target       |
|-----------------------|-------------------------------------------------|--------------|
| **ASR**               | % of attack prompts that bypassed guardrails    | < 5 %        |
| **FOR**               | % of safe prompts incorrectly refused           | < 10 %       |
| **Drift Coefficient** | Safety change per conversation turn             | < 0.5 %/turn |
| **Provenance Score**  | % with complete chain-of-reasoning audit trail  | > 95 %       |

### Methodology Summary

| Phase | Name                  | What Happens                                                              |
|-------|-----------------------|---------------------------------------------------------------------------|
| 1     | Taxonomy Alignment    | Load & stratify 50 prompts from local RedBench snapshot per model         |
| 2     | Adversarial Execution | Single-turn → agentic 3-turn wrap → 10-turn drift probe (5 % sample)     |
| 3     | Dual-Judge Evaluation | RegexJudge (deterministic patterns) + NeuralJudge (Claude Haiku)          |
| 4     | Compliance Reporting  | NIST AI RMF mapping, GSAR 552.239-7001 audit logs, dashboard metrics      |

### Live Dashboard

**[→ https://Sushegaad.github.io/Responsible-AI-Model-Evaluations/](https://Sushegaad.github.io/Responsible-AI-Model-Evaluations/)**

Results are published automatically after every evaluation run and can also be refreshed manually via **Actions → Deploy Dashboard to GitHub Pages → Run workflow**.

---

## If you want to evaluate Models for Responsible AI, what are the steps?

### Quick Start

```bash
git clone https://github.com/Sushegaad/Responsible-AI-Model-Evaluations.git
cd Responsible-AI-Model-Evaluations
pip install -r requirements.txt

cp .env.template .env
# Edit .env and add your API keys

# 1. Run the test suite — no API keys needed
PYTHONPATH=source:. pytest test/ -v

# 2. Sanity check the dataset locally — no API calls
PYTHONPATH=source python -m eval.pipeline --dry-run

# 3. Lightweight smoke test — 2 models, 20 samples (~$1)
PYTHONPATH=source python -m eval.pipeline --models claude-sonnet-4-6 gpt-4o-mini --samples 20

# 4. Full weekly run — all 6 models, 50 samples each (~$2)
PYTHONPATH=source python -m eval.pipeline

# 5. Build the dashboard data file from results
PYTHONPATH=source python source/scripts/build_dashboard_data.py
```

### GitHub Actions Setup

1. Fork or clone this repository — the RedBench dataset snapshot is already included
2. Add API key secrets under **Settings → Secrets and variables → Actions**:
   - `ANTHROPIC_API_KEY` *(required — also powers the NeuralJudge)*
   - `OPENAI_API_KEY`
   - `GOOGLE_API_KEY`
3. Enable GitHub Pages: **Settings → Pages → Source: GitHub Actions**
4. The evaluation runs automatically every **Friday at 6 PM EST** via the `weekly_eval.yml` workflow
5. To run on demand: **Actions → Weekly RAI Evaluation → Run workflow**
6. To redeploy the dashboard only: **Actions → Deploy Dashboard to GitHub Pages → Run workflow**

### Cost Estimate

| Run type             | Samples | Approx. cost |
|----------------------|---------|--------------|
| Default weekly run   | 50      | ~$2          |
| Deeper analysis      | 100     | ~$4          |
| Full benchmark       | 500     | ~$20         |

Override the sample count by setting `REDEVAL_NUM_SAMPLES=<n>` in your environment or `.env` file.

---

## Resources

### Citation

```bibtex
@misc{dang2026redbench,
  title={RedBench: A Universal Dataset for Comprehensive Red Teaming of Large Language Models},
  author={Quy-Anh Dang and Chris Ngo and Truong-Son Hy},
  year={2026}, eprint={2601.03699}, archivePrefix={arXiv},
  url={https://arxiv.org/abs/2601.03699}
}
```

### Dataset

RedBench (Dang et al., 2026) aggregates 37 adversarial sub-benchmarks — including HarmBench, ToxiGen, XSTest, DAN, and AdvBench — into a single standardised schema covering 22 risk categories and 19 domains. A local snapshot (`knoveleng-redbench-April2026/`, 29,362 rows) is committed to this repository as Parquet files, so evaluations run without any network dependency on the dataset host.

To refresh the snapshot:

```bash
python3 -c "
from datasets import load_dataset, get_dataset_config_names
import os
out = 'knoveleng-redbench-<Month><Year>'
os.makedirs(out, exist_ok=True)
for cfg in get_dataset_config_names('knoveleng/redbench'):
    load_dataset('knoveleng/redbench', cfg, split='train').to_parquet(f'{out}/{cfg}.parquet')
"
```

Then update `_SNAPSHOT_DIR` in `source/eval/dataset.py` to point to the new folder.

### License

This project is licensed under the **MIT License**. The RedBench dataset is also MIT-licensed — see [knoveleng/redbench](https://huggingface.co/datasets/knoveleng/redbench).

### Disclaimer

This project is an independent research and transparency initiative. Evaluation results reflect model behaviour on the RedBench adversarial dataset under standardised conditions and are not a comprehensive measure of a model's safety in all deployment contexts. Results may vary across runs due to model updates, API changes, and sampling randomness. No affiliation with Anthropic, OpenAI, Google, or the RedBench authors is implied. All evaluation prompts are used solely for safety research purposes; no new harmful content is generated.
