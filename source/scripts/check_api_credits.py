#!/usr/bin/env python3
"""
source/scripts/check_api_credits.py

Pre-flight API credit / quota check for the RAI evaluation pipeline.

Makes the smallest possible probe call (max_tokens=1, trivial prompt) to each
provider that has an API key configured, then inspects the result to distinguish:

  • Credit / billing exhaustion  → fatal; exits 1 immediately
  • Hard rate-limit (too many requests) → fatal; exits 1 immediately
  • Auth / key problem            → fatal; exits 1 immediately
  • Any other transient error     → logged as a warning; does NOT block the run
  • Clean response                → ✅ provider is OK

Anthropic is REQUIRED (it powers the neural judge and is used for model
evaluation). Any Anthropic failure is an unconditional hard stop.

OpenAI and Google are optional in the sense that the pipeline skips a model
when its key is absent, but if a key IS present and the provider returns a
billing or quota error, we fail early rather than burning time running hundreds
of samples only to hit the wall mid-run.

Exit codes:
  0 – every configured provider passed the probe
  1 – at least one provider has a credit, billing, or hard-quota error
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Probe payload ─────────────────────────────────────────────────────────────
# Single word prompt; max_tokens=1 keeps cost negligible (<$0.00001 per check).
_PROBE_PROMPT = "Hi"
_MAX_TOKENS   = 1

# ── Error-string patterns that indicate billing / credit exhaustion ───────────
# These are matched against the lower-cased string representation of the
# exception, so they catch both SDK-typed errors and raw HTTP payloads.

_BILLING_PATTERNS: tuple[str, ...] = (
    # Anthropic
    "credit_balance_too_low",
    "credit balance",
    "your credit",
    "insufficient credits",
    "billing",
    "payment required",
    # OpenAI
    "insufficient_quota",
    "you exceeded your current quota",
    "exceeded your current quota",
    "billing_hard_limit_reached",
    # Google
    "billing account",
    "quota exceeded",
    "out of quota",
)

_RATE_LIMIT_PATTERNS: tuple[str, ...] = (
    "too many requests",
    "rate limit",
    "ratelimit",
    "resource_exhausted",        # Google gRPC status
    "resource exhausted",
    "requests per minute",
    "requests per day",
)


def _is_billing_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _BILLING_PATTERNS)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _RATE_LIMIT_PATTERNS)


def _http_status(exc: Exception) -> int | None:
    """Extract HTTP status code from SDK exceptions where available."""
    for attr in ("status_code", "status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    return None


# ── Per-provider probe functions ──────────────────────────────────────────────

async def _probe_anthropic(key: str) -> tuple[bool, str]:
    """Returns (ok, message)."""
    try:
        import anthropic  # type: ignore
        client = anthropic.AsyncAnthropic(api_key=key)
        resp = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": _PROBE_PROMPT}],
        )
        await client.close()
        return True, f"OK (stop_reason={resp.stop_reason})"
    except Exception as exc:
        status = _http_status(exc)
        if _is_billing_error(exc):
            return False, f"CREDIT EXHAUSTED (HTTP {status}): {exc}"
        if _is_rate_limit_error(exc) or status == 429:
            return False, f"RATE LIMIT / TOO MANY REQUESTS (HTTP {status}): {exc}"
        if status in (401, 403):
            return False, f"AUTH ERROR (HTTP {status}): {exc}"
        # Unknown / transient
        return False, f"UNKNOWN ERROR (HTTP {status}): {exc}"


async def _probe_openai(key: str) -> tuple[bool, str]:
    try:
        import openai  # type: ignore
        client = openai.AsyncOpenAI(api_key=key)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": _PROBE_PROMPT}],
        )
        await client.close()
        return True, f"OK (finish_reason={resp.choices[0].finish_reason})"
    except Exception as exc:
        status = _http_status(exc)
        if _is_billing_error(exc) or getattr(exc, "code", None) == "insufficient_quota":
            return False, f"CREDIT/QUOTA EXHAUSTED (HTTP {status}): {exc}"
        if _is_rate_limit_error(exc) or status == 429:
            return False, f"RATE LIMIT / TOO MANY REQUESTS (HTTP {status}): {exc}"
        if status in (401, 403):
            return False, f"AUTH ERROR (HTTP {status}): {exc}"
        return False, f"UNKNOWN ERROR (HTTP {status}): {exc}"


async def _probe_google(key: str) -> tuple[bool, str]:
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        client = genai.Client(api_key=key)
        config = types.GenerateContentConfig(max_output_tokens=_MAX_TOKENS)
        # client.aio.models.generate_content is a native async method.
        resp = await client.aio.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=_PROBE_PROMPT,
            config=config,
        )
        return True, f"OK (text={repr((resp.text or '')[:20])})"
    except Exception as exc:
        status = _http_status(exc)
        if _is_billing_error(exc):
            return False, f"CREDIT/QUOTA EXHAUSTED (HTTP {status}): {exc}"
        if _is_rate_limit_error(exc) or status == 429:
            return False, f"RATE LIMIT / TOO MANY REQUESTS (HTTP {status}): {exc}"
        if status in (401, 403):
            return False, f"AUTH ERROR (HTTP {status}): {exc}"
        return False, f"UNKNOWN ERROR (HTTP {status}): {exc}"


# ── Main ──────────────────────────────────────────────────────────────────────

_PROVIDERS: list[tuple[str, str, bool]] = [
    # (display_name, env_var,          required)
    ("Anthropic",   "ANTHROPIC_API_KEY", True ),
    ("OpenAI",      "OPENAI_API_KEY",    False),
    ("Google",      "GOOGLE_API_KEY",    False),
]

_PROBE_FNS = {
    "Anthropic": _probe_anthropic,
    "OpenAI":    _probe_openai,
    "Google":    _probe_google,
}


async def main() -> int:
    log.info("=" * 60)
    log.info("  RAI Eval — Pre-flight API credit check")
    log.info("=" * 60)

    fatal = False

    for name, env_var, required in _PROVIDERS:
        key = os.environ.get(env_var, "").strip()
        if not key:
            if required:
                log.error("❌ %s — %s is not set (REQUIRED — aborting)", name, env_var)
                fatal = True
            else:
                log.info("⏭️  %s — %s not set, skipping", name, env_var)
            continue

        log.info("🔍 Probing %s …", name)
        ok, msg = await _PROBE_FNS[name](key)

        if ok:
            log.info("✅ %s — %s", name, msg)
        else:
            # Decide whether this failure should stop the run.
            # Anthropic is always required. For optional providers, only fail
            # if the error is clearly a credit/quota/auth issue — transient
            # network errors shouldn't block an entire run.
            is_fatal = required or any(
                keyword in msg
                for keyword in ("CREDIT", "QUOTA", "AUTH", "BILLING")
            )
            if is_fatal:
                log.error("❌ %s — %s", name, msg)
                fatal = True
            else:
                log.warning("⚠️  %s — %s (non-fatal, continuing)", name, msg)

    log.info("=" * 60)
    if fatal:
        log.error("🚫 Pre-flight check FAILED — halting workflow to avoid wasted API spend.")
        log.error("   Fix the billing / credit issue above and re-run.")
        return 1

    log.info("✅ All providers passed. Proceeding with evaluation.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
