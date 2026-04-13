#!/usr/bin/env python3
"""
scripts/build_dashboard_data.py
Aggregates all data/weekly/YYYY-DD-MM/metrics.json files into
dashboard/data/results.json which the GitHub Pages site consumes.

Run from the repository root:
    python scripts/build_dashboard_data.py
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

WEEKLY_DIR     = Path("data/weekly")
DASHBOARD_DATA = Path("dashboard/data")
OUT_FILE       = DASHBOARD_DATA / "results.json"

PROVIDER_MAP = {
    "claude-opus-4-6":   "Anthropic",
    "claude-sonnet-4-6": "Anthropic",
    "gpt-4o":            "OpenAI",
    "gpt-4o-mini":       "OpenAI",
    "gemini-2.5-flash":  "Google",
    "gemini-2.5-pro":    "Google",
}

COLOR_MAP = {
    "claude-opus-4-6":   "#CC785C",
    "claude-sonnet-4-6": "#E8956D",
    "gpt-4o":            "#10A37F",
    "gpt-4o-mini":       "#1ABC9C",
    "gemini-2.5-flash":  "#4285F4",
    "gemini-2.5-pro":    "#34A853",
}


def load_weeks() -> dict[str, dict]:
    weeks: dict[str, dict] = {}
    if not WEEKLY_DIR.exists():
        log.info("No data/weekly directory found — nothing to aggregate.")
        return weeks
    for d in sorted(WEEKLY_DIR.iterdir()):
        mf = d / "metrics.json"
        if mf.exists():
            data = json.loads(mf.read_text())
            weeks[data["week"]] = data
            log.info("  ✓  %s  (%d models)", data["week"], len(data.get("models", [])))
    return weeks


def build(weeks: dict[str, dict]) -> dict:
    sorted_weeks = sorted(weeks)
    latest = sorted_weeks[-1] if sorted_weeks else None

    all_ids: set[str] = set()
    for w in weeks.values():
        for m in w.get("models", []):
            all_ids.add(m["model_id"])

    # Per-model time series
    model_series: list[dict] = []
    for mid in sorted(all_ids):
        asr_t, for_t, drift_t, prov_t = [], [], [], []
        display = mid
        for week in sorted_weeks:
            md = next((m for m in weeks[week].get("models", []) if m["model_id"] == mid), None)
            # Skip failed runs — corrupt metrics would skew trend charts.
            if md and not md.get("eval_failed", False):
                display = md.get("display_name", mid)
                asr_t.append(  {"week": week, "value": md.get("asr",   0)})
                for_t.append(  {"week": week, "value": md.get("for_rate", 0)})
                drift_t.append({"week": week, "value": md.get("drift_coefficient", 0)})
                prov_t.append( {"week": week, "value": md.get("provenance_score",  0)})
        model_series.append({
            "model_id":    mid,
            "display_name": display,
            "provider":    PROVIDER_MAP.get(mid, "Unknown"),
            "color":       COLOR_MAP.get(mid, "#888888"),
            "asr_trend":   asr_t,
            "for_trend":   for_t,
            "drift_trend": drift_t,
            "prov_trend":  prov_t,
        })

    # Leaderboard (current week).
    # Failed models (eval_failed=True) are sorted to the end regardless of ASR,
    # since their metrics are not meaningful.
    leaderboard: list[dict] = []
    if latest:
        def _lb_sort(x: dict) -> tuple:
            failed = x.get("eval_failed", False)
            return (1 if failed else 0, x.get("asr", 999))
        for md in sorted(weeks[latest].get("models", []), key=_lb_sort):
            leaderboard.append({
                **md,
                "provider":   PROVIDER_MAP.get(md["model_id"], "Unknown"),
                "color":      COLOR_MAP.get(md["model_id"], "#888"),
                "eval_failed": md.get("eval_failed", False),
            })

    # Radar data — per-category ASR for latest week (only successful runs)
    radar: dict[str, dict] = {}
    if latest:
        for md in weeks[latest].get("models", []):
            if not md.get("eval_failed", False):
                radar[md["model_id"]] = md.get("category_asr", {})

    total_evals = sum(
        sum(m.get("total_samples", 0) for m in w.get("models", []))
        for w in weeks.values()
    )

    return {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "dataset":           "RedBench · knoveleng/redbench (MIT)",
        "weeks":             sorted_weeks,
        "latest_week":       latest,
        "models":            model_series,
        "leaderboard":       leaderboard,
        "radar_data":        radar,
        "total_evaluations": total_evals,
    }


def main() -> None:
    log.info("Building dashboard data …")
    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)
    weeks = load_weeks()
    result = build(weeks)
    OUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("✅ Written → %s  (%d weeks, %d models)", OUT_FILE, len(weeks), len(result["models"]))


if __name__ == "__main__":
    main()
