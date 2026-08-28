"""The brief pipeline, end to end: one Trodo run, five staged spans.

Scouts run in parallel (context isolation is the point of having them), then
Analysts in parallel, then exactly one Editor. Every stage's full output is
persisted to brief_runs — that is what makes a sentence in the finished brief
clickable down to the quote it came from.

Storage is byte-compatible with the old single-call digest: the same
analyses row, meta shape, chart pack and world-model update, so every page,
share link and rewrite that exists keeps working unchanged.
"""
from __future__ import annotations

import contextvars
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import db
from brain import digest, observe, tools
from brain.pipeline import beats as beats_mod
from brain.pipeline import stages
from memory import store, world_model
from signals import chartdata, stats

log = logging.getLogger("mia.pipeline")

MAX_USD = 1.50  # hard stop for one brief; the design target is under $1
SCOUT_WORKERS = 4
ANALYST_WORKERS = 3  # Gemini Pro rate limits are tighter than Flash


def _parallel(ex: ThreadPoolExecutor, fn, items):
    """Run fn over items in the pool, propagating the caller's context.

    Trodo's run context lives in contextvars, which ThreadPoolExecutor does not
    carry into worker threads — without the copy, every span created inside a
    worker would be an orphan and silently dropped.
    """
    futures = [ex.submit(contextvars.copy_context().run, fn, item) for item in items]
    return [f.result() for f in futures]


def _persist(analysis_id: int | None, stage: str, beat: str | None,
             payload: dict, usd: float, model: str | None) -> int:
    row = db.one(
        """INSERT INTO brief_runs (analysis_id, stage, beat, payload, usd, model)
           VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
        (analysis_id, stage, beat, json.dumps(payload, default=str), round(usd, 5), model),
    )
    return row["id"]


def run(hours: int = 12) -> dict:
    """Produce one brief through the staged pipeline."""
    started = datetime.now(timezone.utc)
    world_model.ensure_seeded()

    with observe.run("brief", trigger="schedule", meta={"hours": hours}) as rec:
        rec.set_input({"hours": hours})

        # ── Quant: everything measured, no model ─────────────────────────────
        with observe.stage("quant", kind="generic") as sp:
            pack = stats.build(persist=True)
            try:
                chart_manifest = chartdata.build_pack()
            except Exception as exc:  # noqa: BLE001
                log.warning("chart rendering failed: %s", exc)
                chart_manifest = {}
            slim = digest.slim_stats(pack)
            chart_lines = digest.chart_link_lines(chart_manifest)
            docs = store.recent_documents(hours=hours, limit=120)
            dlines = digest.doc_lines(docs)
            triggers = db.query(
                """SELECT rule, severity, symbol, detail FROM trigger_events
                   WHERE created_at > now() - make_interval(hours => %s)
                   ORDER BY created_at DESC LIMIT 25""", (hours,))
            prior_wm = world_model.current_body()
            sp.set_output({"docs": len(docs), "charts": len(chart_manifest),
                           "triggers": len(triggers)})

        last = db.one("SELECT title FROM analyses WHERE kind='digest' ORDER BY created_at DESC LIMIT 1")

        # ── Marshal ──────────────────────────────────────────────────────────
        with observe.stage("marshal", kind="generic") as sp:
            marshal_out, marshal_spec = stages.marshal(
                hours=hours, slim=slim, dlines=dlines, prior_wm=prior_wm,
                triggers=triggers, last_headline=last["title"] if last else None)
            north_star = stages.north_star_text(marshal_out)
            sp.set_output(marshal_out)
            sp.set_attribute("priorities", len(marshal_out.get("priorities", [])))

        # ── Scouts, in parallel ──────────────────────────────────────────────
        def _scout(beat: dict):
            with observe.stage(f"scout:{beat['key']}", kind="generic") as sp:
                payload, result = stages.scout(
                    beat=beat, north_star=north_star, hours=hours, dlines=dlines)
                sp.set_output(payload)
                sp.set_attribute("leads", len(payload.get("leads", [])))
                sp.set_attribute("usd", result.get("usd", 0))
                return beat["key"], payload, result

        scout_out: dict[str, dict] = {}
        scout_usd: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=SCOUT_WORKERS) as ex:
            for key, payload, result in _parallel(ex, _scout, beats_mod.BEATS):
                scout_out[key] = payload
                scout_usd[key] = result.get("usd", 0.0)

        # Budget guard before the expensive half. Scouts and marshal are the
        # cheap stages; if they somehow burned past the cap, publishing a brief
        # is no longer worth what it would cost to finish.
        if _stage_spend(started) > MAX_USD:
            rec.set_output({"ok": False, "error": f"budget exceeded pre-analysis: ${_stage_spend(started):.2f}"})
            return {"ok": False, "error": "budget exceeded before analysis stage"}

        # ── Analysts, in parallel ────────────────────────────────────────────
        chart_keys = sorted(chart_manifest.keys())

        def _analyst(beat: dict):
            leads = scout_out.get(beat["key"], {"leads": []})
            if not leads.get("leads"):
                return beat["key"], None, None  # quiet beat: nothing to analyse
            graph_ctx = {}
            try:
                graph_ctx = tools.HANDLERS["query_relationships"](
                    beat["section"].split(" ")[0], depth=1, limit=10)
            except Exception:  # noqa: BLE001
                pass
            with observe.stage(f"analyst:{beat['key']}", kind="generic") as sp:
                payload, spec = stages.analyst(
                    beat=beat, leads=leads, north_star=north_star, slim=slim,
                    graph_ctx=graph_ctx, prior_wm=prior_wm, chart_keys=chart_keys)
                sp.set_output(payload)
                sp.set_attribute("findings", len(payload.get("findings", [])))
                return beat["key"], payload, spec

        findings: dict[str, dict] = {}
        analyst_spec: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=ANALYST_WORKERS) as ex:
            for key, payload, spec in _parallel(ex, _analyst, beats_mod.BEATS):
                if payload:
                    findings[key] = payload
                    analyst_spec[key] = spec or ""

        if not findings:
            rec.set_output({"ok": False, "error": "no beat produced findings"})
            return {"ok": False, "error": "no beat produced findings"}

        # ── Editor: the single writer ────────────────────────────────────────
        citations = []
        seen = set()
        for payload in scout_out.values():
            for c in payload.get("citations", []):
                if c.get("url") and c["url"] not in seen:
                    seen.add(c["url"])
                    citations.append(c)
            for lead in payload.get("leads", []):
                for s in lead.get("sources", []) or []:
                    u = s.get("url") if isinstance(s, dict) else None
                    if u and u not in seen:
                        seen.add(u)
                        citations.append({"url": u, "title": s.get("title")})

        with observe.stage("editor", kind="generic") as sp:
            text, editor_spec, editor_usd = stages.editor(
                findings_by_beat=findings, north_star=north_star, slim=slim,
                chart_lines=chart_lines, prior_wm=prior_wm, hours=hours,
                citation_urls=citations[:60], role=digest.ROLE, fmt=digest.FORMAT)
            sp.set_output({"chars": len(text), "spec": editor_spec, "usd": editor_usd})
        if not text.strip():
            rec.set_output({"ok": False, "error": "editor returned nothing"})
            return {"ok": False, "error": "editor returned nothing"}

        text = tools.resolve_grounding_links(text, citations)
        body, wm_block, regime = digest._split_world_model(text)
        body, headline, standfirst = digest._split_headline(body)
        title = headline or f"Brief — {started.strftime('%Y-%m-%d %H:%M UTC')}"

        # ── Verifier ─────────────────────────────────────────────────────────
        with observe.stage("verifier", kind="generic") as sp:
            try:
                verified, fixed = stages.verifier(body=body, slim=slim)
                body = verified["body"]
                audit = verified["audit"]
            except Exception as exc:  # noqa: BLE001
                log.warning("verifier failed, publishing unaudited: %s", exc)
                audit, fixed = {"error": str(exc)}, 0
            sp.set_output(audit)
            sp.set_attribute("fixed", fixed)

        # Scout spend is already in api_calls (the agent loop records each
        # turn), so the ledger sum is the whole truth — adding scout_usd on
        # top would double-count the cheapest stage.
        total_usd = round(_stage_spend(started), 5)

        # ── Persist: same shape the whole dashboard already reads ────────────
        analysis_id = store.save_analysis(
            "digest", title, body,
            meta={
                "hours": hours,
                "pipeline": True,
                "usd": total_usd,
                "regime": regime,
                "citations": citations[:80],
                "words": len(body.split()),
                "headline": headline,
                "standfirst": standfirst,
                "charts": sorted(chart_manifest.keys()),
                "model": f"pipeline(editor={editor_spec})",
                "provider": "multi",
                "beats": {k: v.get("beat_summary") for k, v in findings.items()},
                "verifier": {"checked": audit.get("checked_claims"),
                             "issues": len(audit.get("issues", [])), "fixed": fixed},
                "tool_calls": [],
                "turns": 1,
            })

        for key, payload in scout_out.items():
            _persist(analysis_id, "scout", key, payload, scout_usd.get(key, 0),
                     "gemini:gemini-flash-latest")
        for key, payload in findings.items():
            _persist(analysis_id, "analyst", key, payload,
                     _purpose_spend(f"brief.analyst.{key}", started), analyst_spec.get(key))
        _persist(analysis_id, "marshal", None, marshal_out,
                 _purpose_spend("brief.marshal", started), marshal_spec)
        _persist(analysis_id, "editor", None,
                 {"north_star": north_star, "citations": len(citations)},
                 editor_usd, editor_spec)
        _persist(analysis_id, "verifier", None, audit,
                 _purpose_spend("brief.verifier", started), None)

        if chart_manifest:
            try:
                chartdata.save_pack(analysis_id, chart_manifest)
            except Exception as exc:  # noqa: BLE001
                log.warning("chart pack not stored: %s", exc)

        wm_version = None
        if wm_block:
            wm_version = world_model.save(wm_block, regime=regime, analysis_id=analysis_id)

        edges = 0
        try:
            from brain import extract

            edges = extract.run(hours=hours)
        except Exception as exc:  # noqa: BLE001
            log.warning("edge extraction failed: %s", exc)

        out = {
            "ok": True,
            "analysis_id": analysis_id,
            "title": title,
            "body": body,
            "regime": regime,
            "world_model_version": wm_version,
            "edges_extracted": edges,
            "usd": total_usd,
            "beats": sorted(findings.keys()),
            "leads": sum(len(p.get("leads", [])) for p in scout_out.values()),
            "findings": sum(len(f.get("findings", [])) for f in findings.values()),
            "verifier_fixed": fixed,
            "turns": 1,
            "tool_calls": [],
        }
        rec.set_output({k: v for k, v in out.items() if k != "body"})
        rec.set_metadata(usd=total_usd, beats=len(findings))
        return out


def _stage_spend(since: datetime) -> float:
    row = db.one(
        "SELECT coalesce(sum(usd),0) s FROM api_calls WHERE purpose LIKE 'brief.%%' AND created_at >= %s",
        (since,))
    return float(row["s"]) if row else 0.0


def _purpose_spend(purpose: str, since: datetime) -> float:
    row = db.one(
        "SELECT coalesce(sum(usd),0) s FROM api_calls WHERE purpose LIKE %s AND created_at >= %s",
        (purpose + "%", since))
    return float(row["s"]) if row else 0.0
