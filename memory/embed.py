"""Embeddings with a provider chain, normalised to 1024 dimensions.

Provider order is OpenAI -> Gemini -> Voyage. Voyage's `voyage-finance-2` is the
best fit on paper (finance-tuned) but its no-payment-method tier is capped at
3 requests/minute and 10K tokens/minute, which cannot embed a news corpus or
serve interactive queries. OpenAI's text-embedding-3-small supports native
dimension-shortening to 1024, so the schema is unchanged either way.

Vectors from different models occupy different spaces and must never be compared.
Every stored embedding therefore records the model that produced it, and search
filters on the currently-active model.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time

import httpx

import config

log = logging.getLogger("mia.embed")
DIM = config.EMBED_DIM  # 1024
_cache: dict[str, list[float]] = {}


class EmbeddingUnavailable(RuntimeError):
    pass


# ───────────────────────────────── providers ────────────────────────────────
class _Provider:
    name: str = ""
    model: str = ""
    max_batch: int = 64

    def available(self) -> bool:
        raise NotImplementedError

    def call(self, texts: list[str], input_type: str) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbeddings(_Provider):
    name = "openai"
    model = "text-embedding-3-small"
    max_batch = 96

    def available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def call(self, texts: list[str], input_type: str) -> list[list[float]]:
        r = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": self.model, "input": texts, "dimensions": DIM},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


class GeminiEmbeddings(_Provider):
    name = "gemini"
    model = "gemini-embedding-001"
    max_batch = 32

    def available(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY"))

    def call(self, texts: list[str], input_type: str) -> list[list[float]]:
        key = os.environ["GEMINI_API_KEY"]
        task = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:batchEmbedContents?key={key}"
        )
        requests = [
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": DIM,
                "taskType": task,
            }
            for t in texts
        ]
        r = httpx.post(url, json={"requests": requests}, timeout=120)
        r.raise_for_status()
        return [e["values"] for e in r.json()["embeddings"]]


class VoyageEmbeddings(_Provider):
    name = "voyage"
    model = "voyage-finance-2"
    max_batch = 96

    def available(self) -> bool:
        return bool(config.VOYAGE_API_KEY)

    def call(self, texts: list[str], input_type: str) -> list[list[float]]:
        r = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {config.VOYAGE_API_KEY}"},
            json={
                "model": self.model,
                "input": texts,
                "input_type": input_type,
                "truncation": True,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


_CHAIN: list[_Provider] = [OpenAIEmbeddings(), GeminiEmbeddings(), VoyageEmbeddings()]
_active: _Provider | None = None


def provider() -> _Provider:
    """First available provider. Sticky for the process lifetime."""
    global _active
    if _active is None:
        forced = os.getenv("MIA_EMBED_PROVIDER", "").lower()
        chain = [p for p in _CHAIN if not forced or p.name == forced]
        for p in chain:
            if p.available():
                _active = p
                log.info("embedding provider: %s (%s)", p.name, p.model)
                break
        if _active is None:
            raise EmbeddingUnavailable("no embedding provider configured")
    return _active


def active_model() -> str:
    return provider().model


def available() -> bool:
    try:
        provider()
        return True
    except EmbeddingUnavailable:
        return False


# ─────────────────────────────────── API ────────────────────────────────────
def _key(text: str) -> str:
    return hashlib.sha256(f"{active_model()}|{text}".encode()).hexdigest()


def _call_with_retry(p: _Provider, texts: list[str], input_type: str) -> list[list[float]]:
    last: Exception | None = None
    for attempt in range(4):
        try:
            return p.call(texts, input_type)
        except httpx.HTTPStatusError as exc:
            last = exc
            if exc.response.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            raise EmbeddingUnavailable(
                f"{p.name} HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise EmbeddingUnavailable(f"{p.name} failed after retries: {last}")


def embed(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed texts in order. Empty strings map to zero vectors."""
    if not texts:
        return []
    p = provider()
    out: list[list[float] | None] = [None] * len(texts)
    todo: list[tuple[int, str]] = []

    for i, t in enumerate(texts):
        clean = (t or "").strip()
        if not clean:
            out[i] = [0.0] * DIM
            continue
        k = _key(clean)
        if k in _cache:
            out[i] = _cache[k]
        else:
            todo.append((i, clean[:8000]))

    for start in range(0, len(todo), p.max_batch):
        chunk = todo[start : start + p.max_batch]
        vecs = _call_with_retry(p, [t for _, t in chunk], input_type)
        for (idx, text), vec in zip(chunk, vecs):
            if len(vec) != DIM:
                raise EmbeddingUnavailable(
                    f"{p.name} returned dim {len(vec)}, expected {DIM}"
                )
            _cache[_key(text)] = vec
            out[idx] = vec

    return [v if v is not None else [0.0] * DIM for v in out]


def embed_one(text: str, input_type: str = "query") -> list[float]:
    return embed([text], input_type=input_type)[0]
