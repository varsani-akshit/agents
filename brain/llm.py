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
    "gemini:gemini-pro-latest": (1.25, 10.0),
    "groq:openai/gpt-oss-20b": (0.10, 0.50),
    "openai:gpt-4o-mini": (0.15, 0.60),
    # Azure Foundry deployments. Estimates on the conservative side — the
    # ledger treats these as guardrails; Azure's invoice is the accounting.
    "azure:gpt-5.4": (2.50, 20.0),
    "azure:gpt-5.4-mini": (0.45, 3.60),
    "azure:gpt-4.1-nano": (0.10, 0.40),
    "azure:Llama-4-Maverick-17B-128E-Instruct-FP8": (0.35, 1.40),
    "azure:gpt-oss-120b": (0.15, 0.60),
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
            "azure": os.getenv("AZURE_FOUNDRY_ENDPOINT") and os.getenv("AZURE_FOUNDRY_API_KEY"),
        }.get(provider)
    )


def _azure_base() -> tuple[str, dict]:
    """Azure Foundry's unified v1 route: one endpoint, model = deployment name.

    The `api-key` header is the auth that works across every deployment on the
    resource; the older per-deployment `?api-version=` route rejects the newer
    model families ("API version not supported"), so it is not used.
    """
    ep = os.environ["AZURE_FOUNDRY_ENDPOINT"].rstrip("/")
    return f"{ep}/openai/v1", {"api-key": os.environ["AZURE_FOUNDRY_API_KEY"]}


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
    generation: dict[str, Any] = {
        "responseMimeType": "application/json",
        "responseSchema": _to_gemini_schema(schema),
        "maxOutputTokens": max_tokens,
        "temperature": 0,
    }
    # Thinking tokens are billed against maxOutputTokens, so a model that
    # reasons for 2,000 tokens leaves only 1,000 for the JSON and the object
    # arrives truncated mid-string. Flash models take thinkingBudget: 0 for
    # mechanical schema work; the Pro family rejects a zero budget outright
    # (400), so Pro requests leave the default and get a larger output budget
    # from their callers instead.
    if "pro" not in model:
        generation["thinkingConfig"] = {"thinkingBudget": 0}
    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": os.environ["GEMINI_API_KEY"]},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": generation,
        },
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    cand = data["candidates"][0]
    if cand.get("finishReason") == "MAX_TOKENS":
        raise ProviderError(
            f"response hit maxOutputTokens ({max_tokens}); JSON would be truncated")
    parts = cand.get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
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


def _call_azure_json(model, system, user, schema, max_tokens):
    """Azure Foundry chat completion returning schema-shaped JSON.

    `response_format: json_object` is honoured by the GPT deployments but not
    reliably by the open-weight ones (Maverick, gpt-oss), which instead get the
    schema spelled out in the system prompt. Either way the reply is parsed and
    coerced, and a malformed reply raises — which is what lets the role router
    escalate a tier instead of accepting a silently broken answer.
    The GPT-5 family rejects both `max_tokens` and non-default temperature, so
    the request uses `max_completion_tokens` and sets no temperature at all.
    """
    base, headers = _azure_base()
    body: dict[str, Any] = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "system",
             "content": system + "\n\nReturn ONLY valid JSON matching this schema, "
                        "no prose, no code fences:\n" + json.dumps(schema)},
            {"role": "user", "content": user},
        ],
    }
    if model.startswith("gpt-"):
        body["response_format"] = {"type": "json_object"}
    r = httpx.post(f"{base}/chat/completions", headers=headers, json=body, timeout=180)
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    text = (d["choices"][0]["message"].get("content") or "").strip()
    if not text:
        raise ProviderError("empty completion (reasoning may have consumed the budget)")
    return _extract_json(text), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


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
    fallback = fallback or config.FALLBACK_SPEC
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
            elif provider == "azure":
                payload, itok, otok = _call_azure_json(model, system, user, schema, max_tokens)
            else:
                raise ProviderError(f"unknown provider '{provider}'")

            usd = price(attempt, itok, otok)
            db.execute(
                """INSERT INTO api_calls
                     (provider,model,purpose,input_tokens,output_tokens,usd)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (provider, model, purpose, itok, otok, usd),
            )
            from brain import observe

            observe.record_llm(spec=attempt, purpose=purpose, input_tokens=itok,
                               output_tokens=otok, usd=usd,
                               prompt={"system_instruction": system, "query": user},
                               completion=payload)
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
    from brain import router

    rows = [
        {"task": "classify", "spec": config.CLASSIFY_SPEC, "why": "high volume, narrow schema"},
        {"task": "extract_edges", "spec": config.EXTRACT_SPEC, "why": "high volume, narrow schema"},
        {"task": "alert", "spec": _qualify(config.ALERT_SPEC), "why": "user-facing prose"},
        {"task": "ask", "spec": _qualify(config.ASK_MODEL), "why": "interactive tool loop"},
    ]
    try:
        for role in ("bulk", "workhorse", "reason", "search", "deep"):
            rows.append({"task": f"role:{role}", "spec": " → ".join(router.chain_for(role)),
                         "why": "escalation chain"})
        rows.append({"task": "role:premium", "spec": " → ".join(
            router.chain_for("premium", premium_site="editor")),
            "why": "editor + research synthesis only"})
    except Exception:  # noqa: BLE001 — status must render even if a key is missing
        pass
    return rows


# ───────────────────────────── plain-text completion ────────────────────────
def _text_anthropic(model, system, user, max_tokens):
    from brain import client

    resp = client.complete(model=model, purpose="_routed", system=system,
                           messages=[{"role": "user", "content": user}],
                           max_tokens=max_tokens, estimated_usd=0.005,
                           _skip_ledger=True)
    return (client.text_of(resp), resp.usage.input_tokens, resp.usage.output_tokens)


def _text_gemini(model, system, user, max_tokens):
    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": os.environ["GEMINI_API_KEY"]},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        },
        timeout=180,
    )
    r.raise_for_status()
    d = r.json()
    parts = (d.get("candidates") or [{}])[0].get("content", {}).get("parts", []) or []
    text = "".join(p.get("text", "") for p in parts).strip()
    u = d.get("usageMetadata", {})
    return (text, u.get("promptTokenCount", 0),
            u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0))


def _text_openai_compatible(base, key, model, system, user, max_tokens):
    r = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "max_completion_tokens": max_tokens,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=180,
    )
    r.raise_for_status()
    d = r.json()
    text = (d["choices"][0]["message"].get("content") or "").strip()
    u = d.get("usage", {})
    return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def complete_text(
    spec: str,
    *,
    system: str,
    user: str,
    purpose: str,
    max_tokens: int = 800,
    fallback: str | None = None,
    with_cost: bool = False,
) -> str | tuple[str, float]:
    """Free-text completion on the routed provider, with the same fallback rules.

    Exists because callers that wanted plain prose were parsing a spec for its
    model name and then calling the Anthropic client regardless — so pointing a
    task at Gemini sent a Gemini model id to Anthropic and 404'd.
    """
    fallback = fallback or config.FALLBACK_SPEC
    attempts = [spec] + ([fallback] if fallback != spec else [])
    last: Exception | None = None
    for attempt in attempts:
        if not available(attempt):
            continue
        provider, model = parse_spec(attempt)
        try:
            if provider == "anthropic":
                text, itok, otok = _text_anthropic(model, system, user, max_tokens)
            elif provider == "gemini":
                text, itok, otok = _text_gemini(model, system, user, max_tokens)
            elif provider in ("groq", "openai"):
                base, key = (("https://api.groq.com/openai/v1", "GROQ_API_KEY")
                             if provider == "groq"
                             else ("https://api.openai.com/v1", "OPENAI_API_KEY"))
                text, itok, otok = _text_openai_compatible(
                    base, os.environ[key], model, system, user, max_tokens)
            elif provider == "azure":
                base, headers = _azure_base()
                r = httpx.post(
                    f"{base}/chat/completions", headers=headers,
                    json={"model": model, "max_completion_tokens": max_tokens,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}]},
                    timeout=300,
                )
                r.raise_for_status()
                d = r.json()
                text = (d["choices"][0]["message"].get("content") or "").strip()
                u = d.get("usage", {})
                itok, otok = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            else:
                raise ProviderError(f"unknown provider '{provider}'")
            if not text:
                raise ProviderError("empty completion")
            db.execute(
                """INSERT INTO api_calls
                     (provider,model,purpose,input_tokens,output_tokens,usd)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (provider, model, purpose, itok, otok, price(attempt, itok, otok)),
            )
            from brain import observe

            observe.record_llm(spec=attempt, purpose=purpose, input_tokens=itok,
                               output_tokens=otok, usd=price(attempt, itok, otok),
                               prompt={"system_instruction": system, "query": user},
                               completion=text)
            # Callers that report cost had to query api_calls afterwards, and
            # the one that did not simply reported zero.
            return (text, price(attempt, itok, otok)) if with_cost else text
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("%s via %s failed: %s", purpose, attempt, exc)
    raise ProviderError(f"{purpose}: all providers failed ({last})")
