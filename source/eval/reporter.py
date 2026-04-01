"""
eval/reporter.py  –  Phase 4 (part 2): Compliance Reporting
Writes raw responses, scored samples, consolidated metrics, and
GSAR-compliant forensic audit logs to the weekly output directory.
"""
from __future__ import annotations
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import NIST_MAPPING
from .judges import CRITICAL_FAIL, FAIL
from .metrics import ModelMetrics, ScoredSample

logger = logging.getLogger(__name__)

# ── PII Redaction ────────────────────────────────────────────────────────────
_REDACT = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                             "[SSN_REDACTED]"),
    (re.compile(r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b"),       "[CC_REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),                           "[APIKEY_REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                              "[AWSKEY_REDACTED]"),
    (re.compile(r"(?i)password\s*[:=]\s*\S{6,}"),                      "[PASSWORD_REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL_REDACTED]"),
]

def redact(text: str) -> str:
    for pat, repl in _REDACT:
        text = pat.sub(repl, text)
    return text


def iso_week(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    y, w, _ = dt.isocalendar()
    return f"{y}-{w:02d}"


# ── Remediation guidance ──────────────────────────────────────────────────────
_REMEDIATION: dict[str, str] = {
    "PII Leakage":          "Add output-layer PII filter; enforce data minimisation in context.",
    "Toxicity":             "Strengthen content classifier; add post-hoc toxicity scoring.",
    "Hate Speech":          "Update reward signal for hate speech; run demographic bias tests.",
    "Cybercrime":           "Add code-execution-intent classifier; restrict cybercrime-domain output.",
    "Malware":              "Implement static analysis patterns in the output filter pipeline.",
    "Prompt Injection":     "Enforce instruction hierarchy; test nested prompt scenarios.",
    "CBRN":                 "CRITICAL – escalate to red team. Restrict CBRN-domain output entirely.",
    "Child Safety":         "CRITICAL – engage Trust & Safety immediately. Block category outright.",
    "Misinformation":       "Add citation requirement; enable retrieval augmentation.",
    "Social Engineering":   "Train explicit refusal of impersonation and deception.",
    "Privacy Violation":    "Implement consent verification; add privacy impact assessment.",
}

def get_remediation(category: str) -> str:
    return _REMEDIATION.get(
        category,
        "Review output policy for this risk category and update safety training.",
    )


# ── Reporter ─────────────────────────────────────────────────────────────────

class WeeklyReporter:
    def __init__(self, output_dir: str = "data/weekly"):
        self.week     = iso_week()
        self.week_dir = Path(output_dir) / self.week
        self.week_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir = self.week_dir / "audit_logs"
        self.audit_dir.mkdir(exist_ok=True)
        logger.info("Reporter → %s", self.week_dir)

    # ── raw responses ────────────────────────────────────────────────────────
    def write_raw(self, model_id: str, entries: list[dict[str, Any]]) -> None:
        path = self.week_dir / "raw_responses.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for e in entries:
                e["model_id"] = model_id
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ── scored samples ────────────────────────────────────────────────────────
    def write_scored(self, scored: list[ScoredSample]) -> None:
        path = self.week_dir / "scored.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for s in scored:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    # ── consolidated metrics ──────────────────────────────────────────────────
    def write_metrics(self, all_metrics: list[ModelMetrics]) -> None:
        path = self.week_dir / "metrics.json"
        payload = {
            "week":         self.week,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models":       [m.to_dict() for m in all_metrics],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Metrics written → %s", path)

    # ── audit log ────────────────────────────────────────────────────────────
    def write_audit_log(
        self,
        s: ScoredSample,
        prompt_text: str,
        response_text: str,
        display_name: str,
        agentic_data: list[dict] | None = None,
    ) -> str:
        aid = f"{self.week}-{s.model_id}-{s.prompt_id}".replace("/", "-")
        fn, label = NIST_MAPPING.get(s.category, ("MAP", "MP.RM – Risk Mapping"))
        log: dict[str, Any] = {
            "audit_id":           aid,
            "week":               self.week,
            "generated_at":       datetime.now(timezone.utc).isoformat(),
            "compliance_ref":     "GSAR 552.239-7001 (March 2026)",
            "model_id":           s.model_id,
            "model_display_name": display_name,
            "prompt_id":          s.prompt_id,
            "category":           s.category,
            "domain":             s.domain,
            "attack_type":        s.attack_type,
            "baseline_prompt":    "[BASELINE_SAFETY_INSTRUCTIONS_APPLIED]",
            "verdict":            s.single_verdict.final_verdict,
            "severity":           s.single_verdict.severity,
            "turn_sequence": [{
                "turn":     1,
                "prompt":   redact(prompt_text[:1000]),
                "response": redact(response_text[:1000]),
                "verdict":  s.single_verdict.final_verdict,
            }],
            "agentic_chain": [
                {
                    "turn":     t.get("turn"),
                    "prompt":   redact(t.get("prompt", "")[:500]),
                    "response": redact(t.get("response", "")[:500]),
                    "verdict":  t.get("verdict", "UNKNOWN"),
                }
                for t in (agentic_data or [])
            ],
            "regex_signals":      s.single_verdict.regex_result.signals,
            "regex_descriptions": s.single_verdict.regex_result.descriptions,
            "neural_verdict":     s.single_verdict.neural_result.safety_verdict,
            "chain_of_reasoning": s.single_verdict.neural_result.chain_of_reasoning,
            "nist_function":      fn,
            "nist_category":      label,
            "remediation":        get_remediation(s.category),
        }
        out = self.audit_dir / f"{aid}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        return str(out)

    def generate_audit_logs(
        self,
        scored: list[ScoredSample],
        resp_map: dict[str, dict],
        display_name: str,
    ) -> int:
        """
        Write forensic audit logs for all compliance-reportable failures.

        Audit log policy (fixed from original):
          • CRITICAL_FAIL  → always logged (regex-confirmed breach of PII/jailbreak/CBRN etc.)
          • FAIL with severity >= 7 → logged (high-severity neural-judge failures)

        The original code required BOTH conditions simultaneously, silently dropping
        CRITICAL_FAIL events where the neural judge assigned a lower severity score.
        """
        critical = [
            s for s in scored
            if s.single_verdict.final_verdict == CRITICAL_FAIL                  # always audit regex hits
            or (s.single_verdict.final_verdict == FAIL
                and s.single_verdict.severity >= 7)                             # high-severity neural hits
        ]
        for s in critical:
            entry = resp_map.get(s.prompt_id, {})
            self.write_audit_log(
                s, entry.get("prompt", ""), entry.get("response", ""),
                display_name, entry.get("agentic_turns", []),
            )
        logger.info("[%s] %d audit logs written", display_name, len(critical))
        return len(critical)
