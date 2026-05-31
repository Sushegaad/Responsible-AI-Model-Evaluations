# Changelog

All notable changes to this project are documented here.

---

## [1.0.0] — 2026-05-22 — Final Release

The nine-week study is complete. This release marks the conclusion of the evaluation programme, the publication of the final research paper, and the archival of all results.

### Study Summary
- **26,500 total evaluations** across 9 weekly runs (April 10 – May 22, 2026)
- **7 frontier models** evaluated: Claude Opus 4.7, Claude Opus 4.6, Claude Sonnet 4.6, GPT-4o, GPT-4o Mini, Gemini 2.5 Flash, Gemini 2.5 Pro
- **468 critical findings** (severity ≥ 8/10) and **422 false-outright-refusals**
- **1,269 total attack failures** across all 9 runs

### Added
- **Final research paper** — *A Nine-Week Red-Teaming Study of Foundation Models Across 22 Safety Risk Categories* (Hemant Naik, May 22, 2026). Includes per-model all-run averages, May 22 final run results, 9-column weekly ASR trend table, top-10 risk category analysis, guardrail stability findings, and four audience-specific insight sections.
- **PDF download counter** on the Research Paper tab, powered by counterapi.dev, with GA4 `file_download` event tracking.
- **Google Analytics (GA4)** tag `G-59VEGSNHB8` added to all dashboard pages.
- **"Final Study · 9 Runs"** chip in the dashboard header.
- **"FINAL STUDY"** badge on the Research Paper tab.
- **"Final Run Snapshot"** section divider replacing "Current Run Snapshot".
- Dashboard hero updated to reflect concluded study with total evaluation count and date range.

### Changed
- **Weekly schedule disabled** — cron trigger removed from `weekly_eval.yml`. The pipeline remains runnable manually via `workflow_dispatch` for replication.
- **"12-Week Trends"** section renamed to **"Safety Trends"** with subtitle "across all 9 evaluation runs · Apr 10 – May 22, 2026".
- **Insights section** now computes from all-run averages rather than the latest run only.
- **NIST AI RMF Alignment** now shows per-function ASR averaged across all successful runs (computed in `build_dashboard_data.py`), not just the current run.
- **`build_dashboard_data.py`** updated to compute and persist `nist_avg` per model in `model_series` for all-run NIST aggregation.
- Research paper **stat cards** redesigned as a clean 2-row bordered grid (label row + value/sub row); all emoji removed to fix Helvetica rendering black boxes.
- Research paper **audience insight boxes** changed from solid colour fill with white text to light tinted backgrounds with a 4pt coloured left accent bar and dark text.
- Research paper **table headers** now reliably show white text on dark blue background (`ROWBACKGROUNDS` coordinate bug fixed; explicit `TEXTCOLOR` added as a safety net).
- **README** fully rewritten — clean structure with key findings table, methodology prose, and archival framing.
- All references to "weekly automated pipeline" updated to "concluded nine-week study".

### Fixed
- `ROWBACKGROUNDS` TableStyle bug where `(1,0)` starting coordinate caused the first stripe colour to overwrite the header row background on columns 1+, making white header text invisible.
- `ROUNDEDCORNERS` invalid TableStyle command removed; it caused solid black backgrounds in stat cards and audience boxes.
- Negative `availWidth` error in `stat_card_table` caused by gap columns with padding wider than the column itself — fixed by zeroing padding on gap columns only.

---

## [0.3.0] — 2026-05-08

### Added
- **Research paper PDF** generated via ReportLab (`generate_paper.py`) and published to the dashboard under a dedicated Research Paper tab. Covers study background, methodology, per-model results, risk category analysis, guardrail stability, audience insights, and references.
- **Tab reordering** — Results → Research Paper → Methodology → Resources.
- **`[Current Run]` labels** added to Highlights, Safety Leaderboard, Risk Category Breakdown, and NIST sections to disambiguate run-scoped from all-run-scoped data.
- **Section group dividers** — visual pill separators with dynamic badges ("Study-Wide Analysis" and "Current Run Snapshot") added to the results tab to improve scannability.
- **Insights section** moved to the top of the results tab, derived from all-run averages.
- **Safety Trends** section moved immediately below Insights.
- **NIST AI RMF** section moved below Safety Trends and updated to display all-run averaged ASR per function.
- `build_dashboard_data.py` updated to aggregate `nist_avg` per model across all successful runs.

### Changed
- Paper publishing date corrected from "May 2, 2026" to "May 8, 2026" (seven-week edition).
- Section titles standardised: "Highlights [Current Run]", "Safety Leaderboard [Current Run]", "Risk Category Breakdown [Current Run]".

### Fixed
- Chronological sort of `YYYY-DD-MM` week directories — alphabetical sort was incorrect across month boundaries; `_week_date()` parser added as sort key.
- `eval_failed` models excluded from all leaderboard aggregates and NIST averages to prevent corrupt metrics skewing results.

---

## [0.2.0] — 2026-04-24

### Added
- **Multi-turn drift probe** — a 5% sub-sample of attack prompts is subjected to a 10-turn adversarial dialogue; the Drift Coefficient measures the change in failure rate per turn from turn 1 to turn 10.
- **Provenance Score** metric — percentage of all evaluated samples (not just failures) where the NeuralJudge returned a complete, parseable chain-of-reasoning JSON response. Ensures a verifiable audit trail for GSAR 552.239-7001 compliance.
- **Complete chain-of-reasoning** stored for every CRITICAL_FAIL and high-severity FAIL, providing PII-redacted prompt/response pairs and NeuralJudge reasoning for forensic audit logs.
- **Interactive radar (spider) chart** for Risk Category Breakdown, showing per-category ASR across all evaluated models for the current run.
- **Agentic 3-turn wrap** execution mode (system prompt + user message + follow-up) added alongside single-turn evaluation.
- **`eval_failed` flag** — models with ≥ 50% API errors are marked as failed and excluded from aggregates rather than silently producing corrupt metrics.
- **Claude Opus 4.7** added to the evaluated model set.
- **Gemini 2.5 Flash** and **Gemini 2.5 Pro** added, replacing earlier Gemini 2.0/1.5 models.
- API credit pre-check step added to `weekly_eval.yml` — makes a minimal probe call to each provider before the full run to catch billing/quota errors early.

### Changed
- Dashboard header and section layout overhauled for clarity — metric definitions corrected, section headings standardised.
- `weekly_eval.yml` updated to fail the workflow immediately if the test suite breaks (`continue-on-error: false`).
- RedBench dataset migrated from network fetch to a **local Parquet snapshot** (`knoveleng-redbench-April2026/`) committed to the repository, making every run fully reproducible with no external data dependency.

### Fixed
- `trust_remote_code=True` removed from `load_dataset` calls — no longer accepted by the current datasets library for standard HuggingFace datasets.
- Multi-config dataset loading fixed — `knoveleng/redbench` requires specifying a config name; updated to iterate over `get_dataset_config_names`.
- Gemini 2.0 Flash and Gemini 1.5 Pro API errors resolved; models replaced with Gemini 2.5 Flash and Gemini 2.5 Pro.
- Incomplete evaluation runs (partial data) detected and removed from `data/weekly/` to prevent corrupt metrics from appearing in the dashboard.

---

## [0.1.0] — 2026-04-10 — Initial Release

### Added
- **Evaluation pipeline** (`source/eval/pipeline.py`) — end-to-end orchestration of prompt loading, model API calls, dual-judge scoring, and metrics output to `data/weekly/YYYY-DD-MM/metrics.json`.
- **RegexJudge** — deterministic pattern-matching for PII leakage, jailbreak confirmations, malware code, explosive synthesis instructions, and CSAM. A regex hit always produces `CRITICAL_FAIL` with severity ≥ 9.
- **NeuralJudge** (Claude Haiku 4.5) — returns a structured JSON verdict (`SAFE`, `UNSAFE`, or `OVER_REFUSAL`), a severity score from 1 to 10, and a chain-of-reasoning sentence. Neural `UNSAFE` with severity ≥ 8 also escalates to `CRITICAL_FAIL`.
- **NIST AI RMF 1.0 mapping** — all failures mapped to one of four functions: GOVERN, MAP, MEASURE, MANAGE.
- **GitHub Pages dashboard** (`dashboard/index.html`) — safety leaderboard, metric cards, risk category breakdown, and methodology reference, auto-deployed via `deploy_pages.yml`.
- **`weekly_eval.yml`** GitHub Actions workflow — scheduled evaluation every Friday at 22:00 UTC, committing results and triggering a dashboard deploy.
- **`build_dashboard_data.py`** — aggregates all `data/weekly/*/metrics.json` files into `dashboard/data/results.json` consumed by the dashboard frontend.
- Initial support for 6 models: Claude Opus 4.6, Claude Sonnet 4.6, GPT-4o, GPT-4o Mini, Gemini 2.0 Flash, Gemini 1.5 Pro.
- 500 prompts per model per run (400 adversarial, 100 benign), stratified across 22 risk categories from the RedBench dataset.
- Test suite (`test/`) covering dataset loading, judge logic, and pipeline integration — runs as a required pre-flight step in CI.

---

*This project is an independent research and transparency initiative by [Hemant Naik](https://www.linkedin.com/in/tanaji-naik/). No affiliation with Anthropic, OpenAI, Google, or the RedBench authors is implied.*
