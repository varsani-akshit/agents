"""Multi-provider router for structured-JSON tasks.

Not every call needs Claude. Classification and edge extraction are high-volume,
narrow, schema-constrained tasks — a fast cheap model does them well. Digests and
`ask` need deep reasoning, tool use, and prompt caching, so they stay on Claude.
Routing by task requirement rather than by preference is what keeps the running
cost proportional to the value of each call.

Model specs are "provider:model", e.g.
    anthropic:claude-haiku-4-5
    gemini:gemini-flash-latest
    groq:openai/gpt-oss-20b
    openai:gpt-4o-mini

Every provider is normalised to the same contract: given a system prompt, a user
payload, and a JSON schema, return a parsed dict. Any provider failure falls back
to the Anthropic model so a routing choice can never take the pipeline down.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

import config
import db

log = logging.getLogger("mia.llm")

# USD per million tokens (input, output). Used for the ledger; non-Anthropic
# providers are cheap enough that these are guardrails, not accounting.
PRICING: dict[str, tuple[float, float]] = {
    "anthropic:claude-opus-5": (5.0, 25.0),
    "anthropic:claude-sonnet-5": (3.0, 15.0),
    "anthropic:claude-haiku-4-5": (1.0, 5.0),
    "gemini:gemini-flash-latest": (0.30, 2.50),
    "gemini:gemini-flash-lite-latest": (0.10, 0.40),
    "groq:openai/gpt-oss-20b": (0.10, 0.50),
    "openai:gpt-4o-mini": (0.15, 0.60),
}


class ProviderError(RuntimeError):
    pass


def parse_spec(spec: str) -> tuple[str, str]:
    """'gemini:gemini-flash-latest' -> ('gemini', 'gemini-flash-latest').

    A bare model name is assumed to be Anthropic, so existing config keeps working.
    """
    if ":" not in spec:
        return "anthropic", spec
    provider, _, model = spec.partition(":")
    return provider.strip().lower(), model.strip()


def price(spec: str, in_tok: int, out_tok: int) -> float:
    inp, out = PRICING.get(spec, (1.0, 5.0))
    return in_tok * inp / 1e6 + out_tok * out / 1e6


def available(spec: str) -> bool:
    provider, _ = parse_spec(spec)
    return bool(
        {
            "anthropic": config.ANTHROPIC_API_KEY,
            "gemini": os.getenv("GEMINI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
        }.get(provider)
    )


# ───────────────────────────── schema translation ───────────────────────────
def _to_gemini_schema(schema: dict) -> dict:
    """JSON Schema -> Gemini's OpenAPI-flavoured dialect (uppercase type names)."""
    types = {
        "object": "OBJECT", "array": "ARRAY", "string": "STRING",
        "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN",
    }
    out: dict[str, Any] = {}
    if "type" in schema:
        out["type"] = types.get(schema["type"], "STRING")
    if "enum" in schema:
        out["enum"] = schema["enum"]
    if "properties" in schema:
        out["properties"] = {k: _to_gemini_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = _to_gemini_schema(schema["items"])
    if "required" in schema:
        out["required"] = schema["required"]
    return out


def _coerce(payload: Any, schema: dict) -> dict:
    """Normalise provider quirks into the shape the caller declared.

    Two are common: a bare array returned where an object envelope was requested
    (Groq does this), and enum values in the wrong case (`high` for `High`).
    Neither is worth failing a batch over.
    """
    props = schema.get("properties", {})
    if isinstance(payload, list):
        array_keys = [k for k, v in props.items() if v.get("type") == "array"]
        payload = {array_keys[0]: payload} if array_keys else {"items": payload}
    if not isinstance(payload, dict):
        raise ProviderError(f"expected object, got {type(payload).__name__}")

    for key, spec in props.items():
        if spec.get("type") != "array" or key not in payload:
            continue
        item_props = spec.get("items", {}).get("properties", {})
        for row in payload[key]:
            if not isinstance(row, dict):
                continue
            for field, fspec in item_props.items():
                allowed = fspec.get("enum")
                val = row.get(field)
                if allowed and isinstance(val, str) and val not in allowed:
                    match = next((a for a in allowed if a.lower() == val.lower()), None)
                    if match:
                        row[field] = match
    return payload


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    return json.loads(text)


# ─────────────────────────────── provider calls ─────────────────────────────
def _call_anthropic(model, system, user, schema, max_tokens) -> tuple[Any, int, int]:
    from brain import client as ac

    resp = ac.complete(
        model=model,
        purpose="_routed",
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
        estimated_usd=0.01,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        _skip_ledger=True,
    )
    return (
        _extract_json(ac.text_of(resp)),
        resp.usage.input_tokens or 0,
        resp.usage.output_tokens or 0,
    )


def _call_gemini(model, system, user, schema, max_tokens) -> tuple[Any, int, int]:
    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": os.environ["GEMINI_API_KEY"]},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _to_gemini_schema(schema),
                "maxOutputTokens": max_tokens,
                "temperature": 0,
            },
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    usage = data.get("usageMetadata", {})
    return (
        _extract_json(text),
        usage.get("promptTokenCount", 0),
        usage.get("candidatesTokenCount", 0),
    )


def _call_openai_compatible(base: str, key: str, model, system, user, schema, max_tokens):
    r = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    return (
        _extract_json(data["choices"][0]["message"]["content"]),
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )


# ───────────────────────────────── public API ───────────────────────────────
def complete_json(
    spec: str,
    *,
    system: str,
    user: str,
    schema: dict,
    purpose: str,
    max_tokens: int = 3000,
    fallback: str | None = None,
) -> dict:
    """Run a schema-constrained completion on the routed provider.

    Falls back to `fallback` (default: the configured Anthropic classify model)
    on any provider error, so a routing decision cannot break the pipeline.
    """
    fallback = fallback or f"anthropic:{config.CLASSIFY_MODEL}"
    attempts = [spec] + ([fallback] if fallback != spec else [])

    last: Exception | None = None
    for attempt in attempts:
        if not available(attempt):
            log.debug("provider for %s unavailable, skipping", attempt)
            continue
        provider, model = parse_spec(attempt)
        try:
            if provider == "anthropic":
                payload, itok, otok = _call_anthropic(model, system, user, schema, max_tokens)
            elif provider == "gemini":
                payload, itok, otok = _call_gemini(model, system, user, schema, max_tokens)
            elif provider == "groq":
                payload, itok, otok = _call_openai_compatible(
                    "https://api.groq.com/openai/v1", os.environ["GROQ_API_KEY"],
                    model, system, user, schema, max_tokens,
                )
            elif provider == "openai":
                payload, itok, otok = _call_openai_compatible(
                    "https://api.openai.com/v1", os.environ["OPENAI_API_KEY"],
                    model, system, user, schema, max_tokens,
                )
            else:
                raise ProviderError(f"unknown provider '{provider}'")

            usd = price(attempt, itok, otok)
            db.execute(
                """INSERT INTO api_calls
                     (provider,model,purpose,input_tokens,output_tokens,usd)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (provider, model, purpose, itok, otok, usd),
            )
            if attempt != spec:
                log.warning("%s fell back to %s", spec, attempt)
            log.info("%s via %s: $%.5f (in=%s out=%s)", purpose, attempt, usd, itok, otok)
            return _coerce(payload, schema)

        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("%s failed on %s: %s", purpose, attempt, str(exc)[:200])

    raise ProviderError(f"all providers failed for {purpose}: {last}")


def _qualify(model: str) -> str:
    """Attach the provider a bare model id actually belongs to.

    Assuming Anthropic here mislabelled `gpt-5.1` as `anthropic:gpt-5.1` in the
    status table and attributed its spend to the wrong provider.
    """
    if ":" in model:
        return model
    if model.startswith(("gpt-", "o3", "o4")):
        return f"openai:{model}"
    if model.startswith("gemini"):
        return f"gemini:{model}"
    return f"anthropic:{model}"


def routing_table() -> list[dict]:
    """What each task is currently routed to — surfaced by `mia status`."""
    return [
        {"task": "classify", "spec": config.CLASSIFY_SPEC, "why": "high volume, narrow schema"},
        {"task": "extract_edges", "spec": config.EXTRACT_SPEC, "why": "high volume, narrow schema"},
        {"task": "alert", "spec": _qualify(config.ALERT_SPEC), "why": "user-facing prose"},
        {"task": "digest", "spec": _qualify(config.DIGEST_MODEL),
         "why": "scheduled; cheapest adequate reasoning"},
        {"task": "ask", "spec": _qualify(config.ASK_MODEL),
         "why": "deep reasoning + web search"},
    ]
