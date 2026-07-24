"""Server-owned canonical handoff contract (``jdatamunch.handoff/v1``).

Suite parity with jcodemunch-mcp v1.108.162 (issue jcodemunch-mcp#374): a
multi-step data audit ends with one authoritative Markdown result. The
assistant authors the analysis (the server never writes conclusions); this
module owns everything downstream: deterministic assembly, evidence
attestation, session-scoped persistence, identity, hashing, and immutable
serving via the ``munch://handoff/<id>`` resource.

jDataMunch's retrieval-record substrate: column ids
(``<dataset>::<column>#column``) and dataset names served this session by
``search_data`` / ``describe_dataset`` / ``describe_column`` (recorded at
the server response chokepoint). An ``evidence_refs`` entry is attested when
it matches a served column id, a served dataset name, or the dataset
component of a served column id. Unknown refs fail closed.

Contract invariants (shared suite-wide):
- Deterministic: same inputs -> byte-identical body, same id, same sha256.
- Each appendix exactly once; duplicate names rejected. No character limit.
- Session-scoped (process == session), in-memory only — never writes to the
  user's data files or index store.
- ``canonical: true`` in the receipt is advisory metadata only.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Iterable, Optional

HANDOFF_SCHEMA = "jdatamunch.handoff/v1"
HANDOFF_SCHEMA_V2 = "jdatamunch.handoff/v2"
HANDOFF_URI_PREFIX = "munch://handoff/"
HANDOFF_CONTENT_TYPE = "text/markdown"

_lock = threading.Lock()
_handoffs: dict[str, dict] = {}

_SERVED_MAXSIZE = 10000
_served_ids: dict[str, None] = {}
_served_datasets: dict[str, None] = {}


def note_served_rows(rows, dataset: Optional[str] = None) -> None:
    """Record served column rows ({id}) + the queried dataset name."""
    with _lock:
        if isinstance(dataset, str) and dataset.strip():
            _served_datasets[dataset.strip()] = None
        for row in rows or ():
            if not isinstance(row, dict):
                continue
            sid = row.get("id")
            if isinstance(sid, str) and sid.strip():
                _served_ids[sid.strip()] = None
                while len(_served_ids) > _SERVED_MAXSIZE:
                    _served_ids.pop(next(iter(_served_ids)))
        while len(_served_datasets) > _SERVED_MAXSIZE:
            _served_datasets.pop(next(iter(_served_datasets)))


def note_served_column(dataset, column) -> None:
    """Record a directly described column (describe_column)."""
    if isinstance(dataset, str) and dataset.strip() and isinstance(column, str) and column.strip():
        note_served_rows(
            [{"id": f"{dataset.strip()}::{column.strip()}#column"}],
            dataset=dataset,
        )


def served_refs() -> tuple[frozenset, frozenset]:
    """Snapshot of (column ids, dataset names) served this session."""
    with _lock:
        return frozenset(_served_ids), frozenset(_served_datasets)


def clear_session_record() -> None:
    with _lock:
        _served_ids.clear()
        _served_datasets.clear()


# --- absence evidence (jcodemunch-mcp#377 phase 3) ---------------------------
#
# A zero-result search cannot be cited under the v1/v2 rules: nothing was
# served, so there is no id to reference. But "we searched the dataset and no
# such column/value is there" is exactly the claim an audit most needs
# attested, and the one agents most often assert with no proof at all.
#
# jData's ``verdict.build_verdict`` reports a state (ok / absent / degraded),
# per-channel status, index coverage, and a scorer pin. This records those
# verdicts under a deterministic ref so a claim can cite the scan itself.
#
# The refusal rules are the whole point and are deliberately strict:
#   - only ``absent`` can prove absence;
#   - ``degraded`` cannot (a keyword-only fallback is a partial scan);
#   - a truncated row walk cannot (the target may sit in the dropped tail).
#
# HONEST DIVERGENCE from jcm/jdoc, disclosed in every rendered proof: jData's
# index does not model freshness — its verdict has no ``index`` channel — so
# the stale-index refusal its siblings enforce cannot fire here. Rather than
# ship a guarantee that reads enforced and isn't, every jData absence proof
# states "index freshness: not tracked by this product" in the body. Absence
# stays citable; the reader is told exactly what was and was not checked.
ABSENCE_REF_PREFIX = "absent:"
_ABSENCE_MAXSIZE = 500
_absences: dict[str, dict] = {}

# Args that NARROW a search. They belong in the ref identity and in the
# rendered proof: "not found" means nothing without the scope it was not found
# in. ``search_scope`` picks which column facets (name / value / type) were
# searched; "all" is the default and narrows nothing.
_SCOPE_ARGS = ("search_scope",)


def _absence_ref(tool: str, dataset: str, query: str, scope: dict) -> str:
    """Deterministic ref: the same scan in the same scope is the same proof."""
    payload = json.dumps(
        {"tool": tool, "dataset": dataset, "query": query, "scope": scope},
        sort_keys=True,
        separators=(",", ":"),
    )
    return ABSENCE_REF_PREFIX + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def absence_refusal(record: Optional[dict]) -> Optional[str]:
    """Why this recorded scan may NOT prove absence, or None when it may.

    jData models no index freshness, so there is deliberately NO stale-index
    rule here — that limitation is disclosed in the rendered proof instead of
    being silently enforced. The rules that DO apply are the ones jData's
    verdict can actually back.
    """
    if not record:
        return None
    state = record.get("state")
    if state != "absent":
        return (
            f"the scan's verdict was '{state}', and only 'absent' can prove absence "
            "(a keyword-only or partial scan is not evidence of nothing)"
        )
    if (record.get("coverage") or {}).get("walk") == "truncated":
        return (
            "the row walk was truncated at index time, so the target may sit in "
            "the rows the ingest pass dropped"
        )
    return None


def note_absence(tool: str, dataset, query, verdict, arguments=None):
    """Record a search verdict so an absence claim can cite the scan itself.

    Returns ``(ref, refusal)``: a citable ref when this scan can prove absence,
    otherwise the reason it cannot. The record is kept either way, so a caller
    that cites a refused scan gets the reason instead of a bare unknown-ref
    error, and the live response can say why no token was offered.
    """
    if not isinstance(verdict, dict) or not isinstance(query, str) or not query.strip():
        return None, None
    state = verdict.get("state")
    # `ok` scans are not absence evidence and would only bloat the map. jData
    # has no `low_confidence` state (its scores are rank-normalized).
    if state not in ("absent", "degraded"):
        return None, None
    scope = {}
    for key in _SCOPE_ARGS:
        val = (arguments or {}).get(key)
        if val not in (None, "", [], {}, "all"):
            scope[key] = val
    dataset_s = dataset if isinstance(dataset, str) else ""
    ref = _absence_ref(tool, dataset_s, query.strip(), scope)
    record = {
        "ref": ref,
        "tool": tool,
        "dataset": dataset_s,
        "query": query.strip(),
        "scope": scope,
        "state": state,
        "channels": verdict.get("channels") or {},
        "coverage": verdict.get("coverage"),
        "scorer": verdict.get("scorer"),
    }
    with _lock:
        _absences[ref] = record
        while len(_absences) > _ABSENCE_MAXSIZE:
            _absences.pop(next(iter(_absences)))
    refusal = absence_refusal(record)
    return (None, refusal) if refusal else (ref, None)


def absence_record(ref: str) -> Optional[dict]:
    with _lock:
        rec = _absences.get(ref)
        return dict(rec) if rec else None


def clear_absences() -> None:
    """Test hook: drop the session absence record."""
    with _lock:
        _absences.clear()


def _validate_evidence(refs, served_ids: Iterable[str], served_datasets: Iterable[str]):
    served = set(served_ids or ())
    datasets = set(served_datasets or ())
    for sid in served:
        datasets.add(sid.split("::", 1)[0])
    seen: set[str] = set()
    ordered: list[str] = []
    unknown: list[str] = []
    refused: list[dict] = []
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip():
            unknown.append(repr(ref))
            continue
        ref = ref.strip()
        if ref in seen:
            continue
        seen.add(ref)
        ordered.append(ref)
        if ref.startswith(ABSENCE_REF_PREFIX):
            # An absence ref attests against the recorded scan, not the served
            # set — by construction nothing was served (#377 phase 3).
            record = absence_record(ref)
            if record is None:
                unknown.append(ref)
                continue
            reason = absence_refusal(record)
            if reason:
                refused.append({"ref": ref, "reason": reason})
            continue
        if ref not in served and ref not in datasets:
            unknown.append(ref)
    return ordered, unknown, refused


def _validate_claims(raw, si, seen_ids):
    """Validate one section's caller-authored claims (jcodemunch-mcp#377 phase 1).

    Claim ids are unique across the WHOLE handoff, not per section: they are
    the machine-readable anchor a caller cites, and two sections owning the
    same id would make that citation ambiguous. Returns (claims, error).
    """
    if raw is None:
        return [], None
    if not isinstance(raw, list) or not raw:
        return None, f"sections[{si}].claims must be a non-empty list when present"
    out = []
    for j, claim in enumerate(raw):
        where = f"sections[{si}].claims[{j}]"
        if not isinstance(claim, dict):
            return None, f"{where} must be an object with 'id', 'statement' and 'evidence_refs'"
        cid = claim.get("id")
        statement = claim.get("statement")
        if not isinstance(cid, str) or not cid.strip():
            return None, f"{where}.id must be a non-empty string"
        if not isinstance(statement, str) or not statement.strip():
            return None, f"{where}.statement must be a non-empty string"
        cid = cid.strip()
        if cid in seen_ids:
            return None, f"duplicate claim id: {cid!r} (claim ids must be unique across the handoff)"
        seen_ids.add(cid)
        refs = claim.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            return None, (
                f"{where}.evidence_refs must be a non-empty list of session "
                "retrieval references (a claim with no evidence is not attestable)"
            )
        classification = claim.get("classification")
        if classification is not None and (
            not isinstance(classification, str) or not classification.strip()
        ):
            return None, f"{where}.classification must be a non-empty string when present"
        out.append(
            {
                "id": cid,
                # Caller-authored text is preserved verbatim; the server never
                # rewrites a statement or a classification.
                "statement": statement.strip(),
                "classification": classification.strip() if classification else None,
                "raw_refs": refs,
            }
        )
    return out, None


def _validate_sections(sections):
    if not isinstance(sections, list) or not sections:
        return None, "sections must be a non-empty list of {heading, content} objects"
    out = []
    seen_ids: set[str] = set()
    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            return None, f"sections[{i}] must be an object with 'heading' and 'content'"
        heading = sec.get("heading")
        content = sec.get("content")
        if not isinstance(heading, str) or not heading.strip():
            return None, f"sections[{i}].heading must be a non-empty string"
        claims, err = _validate_claims(sec.get("claims"), i, seen_ids)
        if err:
            return None, err
        # Content stays required in the v1 shape; a claims-carrying section may
        # omit it, since the claims themselves are then the section's body.
        if content is None and claims:
            content = ""
        if not isinstance(content, str) or (not content.strip() and not claims):
            return None, f"sections[{i}].content must be a non-empty string"
        out.append((heading.strip(), content.rstrip(), claims))
    return out, None


def _validate_appendices(appendices):
    if appendices is None:
        return [], None
    if not isinstance(appendices, list):
        return None, "appendices must be a list of {name, content} objects"
    out = []
    names: set[str] = set()
    for i, app in enumerate(appendices):
        if not isinstance(app, dict):
            return None, f"appendices[{i}] must be an object with 'name' and 'content'"
        name = app.get("name")
        content = app.get("content")
        if not isinstance(name, str) or not name.strip():
            return None, f"appendices[{i}].name must be a non-empty string"
        if not isinstance(content, str) or not content.strip():
            return None, f"appendices[{i}].content must be a non-empty string"
        name = name.strip()
        if name in names:
            return None, f"duplicate appendix name: {name!r} (each appendix appears exactly once)"
        names.add(name)
        ctype = app.get("content_type") or "text/markdown"
        out.append((name, str(ctype), content.rstrip()))
    return out, None


def _render_absence_detail(ref: str, indent: str) -> list:
    """The scan behind an absence ref, rendered so a reader can audit it.

    A bare token proves nothing to a human. What makes an absence claim
    checkable is the scope it was not found in, how much was scanned, and what
    the index does and does not guarantee — so all of that goes in the body,
    including jData's freshness disclosure.
    """
    rec = absence_record(ref)
    if not rec:
        return []
    lines = [f"{indent}- absence proof: `{rec['tool']}` query {rec['query']!r}"]
    scope = rec.get("scope") or {}
    if scope:
        rendered = ", ".join(f"{k}={scope[k]!r}" for k in sorted(scope))
        lines.append(f"{indent}- scope: {rendered}")
    else:
        lines.append(f"{indent}- scope: whole indexed dataset")
    channels = rec.get("channels") or {}
    if channels:
        rendered = ", ".join(f"{k}={channels[k]}" for k in sorted(channels))
        lines.append(f"{indent}- channels: {rendered}")
    coverage = rec.get("coverage") or {}
    if coverage:
        if coverage.get("rows_indexed") is not None:
            lines.append(f"{indent}- scanned: {coverage['rows_indexed']} rows indexed")
        if coverage.get("walk"):
            lines.append(f"{indent}- walk: {coverage['walk']}")
        excluded = coverage.get("excluded") or {}
        if excluded:
            rendered = ", ".join(f"{k}={excluded[k]}" for k in sorted(excluded))
            lines.append(f"{indent}  - excluded: {rendered}")
        generation = coverage.get("generation") or {}
        if generation.get("indexed_at"):
            lines.append(f"{indent}  - index generation: {generation['indexed_at']}")
    else:
        # Never render unknown coverage as if it were a complete scope.
        lines.append(f"{indent}- coverage: not recorded for this index (scope unknown)")
    # The honest weakening, stated in-band on every jData absence proof: this
    # product does not model index freshness, so unlike its siblings it cannot
    # certify the scan ran against an up-to-date tree.
    lines.append(f"{indent}- index freshness: not tracked by this product")
    if rec.get("scorer") is not None:
        lines.append(f"{indent}- scorer: {rec['scorer']}")
    return lines


def render_handoff(
    dataset: str,
    task: str,
    profile: str,
    sections,
    evidence_refs,
    appendices,
    schema: str = HANDOFF_SCHEMA,
) -> str:
    """Deterministic canonical Markdown. No timestamps, no randomness."""
    lines = [
        f"# Handoff: {task}",
        "",
        f"- Schema: {schema}",
        f"- Dataset: {dataset}",
        f"- Profile: {profile}",
        "",
    ]
    # Absence detail renders once. Under its claim when it has one, else in the
    # global index — repeating the whole scan block twice is noise in the exact
    # artifact meant to be read.
    detailed: set[str] = set()
    for _, _, claims in sections:
        for claim in claims:
            detailed.update(claim["refs"])

    for heading, content, claims in sections:
        lines += [f"## {heading}", ""]
        if content:
            lines += [content, ""]
        for claim in claims:
            # Evidence renders BESIDE the claim it supports - the whole point
            # of v2. The global index below stays for v1 compatibility.
            lines += [f"### {claim['statement']}", ""]
            lines.append(f"- Claim id: `{claim['id']}`")
            if claim["classification"]:
                lines.append(f"- Classification: {claim['classification']}")
            lines.append("- Evidence:")
            for ref in claim["refs"]:
                lines.append(f"  - `{ref}`")
                lines += _render_absence_detail(ref, "    ")
            lines.append("")
    lines += [
        "## Evidence",
        "",
        "Every reference below was validated against this session's retrieval",
        "record at finalization time (server-attested).",
        "",
    ]
    for ref in evidence_refs:
        lines.append(f"- `{ref}`")
        if ref not in detailed:
            lines += _render_absence_detail(ref, "  ")
    lines.append("")
    for name, ctype, content in appendices:
        lines += [f"## Appendix: {name}", "", f"_Content type: {ctype}_", "", content, ""]
    return "\n".join(lines).rstrip() + "\n"


def finalize_handoff(
    *,
    dataset,
    task,
    sections,
    evidence_refs,
    profile: str = "general",
    appendices=None,
    served: Optional[tuple] = None,
) -> dict:
    """Assemble, attest, persist, and return the compact receipt.

    Validation failures return ``{"error": ...}`` in-band (jdata's error
    convention). The server never authors content.
    """
    if not isinstance(dataset, str) or not dataset.strip():
        return {"error": "dataset must be a non-empty string"}
    if not isinstance(task, str) or not task.strip():
        return {"error": "task must be a non-empty string"}
    if not isinstance(profile, str) or not profile.strip():
        return {"error": "profile must be a non-empty string"}
    sec, err = _validate_sections(sections)
    if err:
        return {"error": err}
    apps, err = _validate_appendices(appendices)
    if err:
        return {"error": err}
    if served is None:
        served = served_refs()
    claim_count = sum(len(claims) for _, _, claims in sec)

    # Attest each claim's refs on their own, so an unknown ref names the claim
    # that cited it instead of vanishing into one global failure list (#377).
    invalid_claims = []
    refused_claims = []
    for _, _, claims in sec:
        for claim in claims:
            claim_refs, claim_unknown, claim_refused = _validate_evidence(
                claim["raw_refs"], served[0], served[1]
            )
            claim["refs"] = claim_refs
            if claim_unknown:
                invalid_claims.append({"claim_id": claim["id"], "unknown_refs": claim_unknown})
            if claim_refused:
                refused_claims.append({"claim_id": claim["id"], "refused": claim_refused})
    if refused_claims:
        # Distinct from an unknown ref: the scan is real, it just cannot prove
        # absence. Saying so by name is the point of the contract (#377).
        return {
            "error": (
                "absence evidence refused: the following claims cite a recorded "
                "scan that cannot prove absence"
            ),
            "refused_absence_claims": refused_claims,
            "hint": (
                "Only a verdict of 'absent' over a non-truncated dataset scan "
                "proves absence. Widen the scope or re-index, re-run search_data, "
                "then cite the new scan."
            ),
        }
    if invalid_claims:
        return {
            "error": (
                "claim evidence attestation failed: the following claims cite "
                "refs that do not correspond to anything retrieved in this session"
            ),
            "invalid_claims": invalid_claims,
            "hint": (
                "Every claim-scoped ref must be a column id (or its dataset) "
                "this session actually served. Retrieve the evidence that "
                "supports the claim, then finalize."
            ),
        }

    # `evidence_refs` stays required in the v1 shape, but claims can satisfy it:
    # a caller who scoped everything to claims should not have to restate it.
    claim_refs_flat = [ref for _, _, claims in sec for claim in claims for ref in claim["refs"]]
    if not isinstance(evidence_refs, list) or (not evidence_refs and not claim_refs_flat):
        return {
            "error": (
                "evidence_refs must be a non-empty list of session retrieval "
                "references (column ids like '<dataset>::<column>#column' or "
                "dataset names served this session by search_data / "
                "describe_dataset / describe_column), or every claim must "
                "carry its own evidence_refs"
            )
        }
    # The canonical index is the union, caller order first: every claim ref is
    # discoverable from the global list, which keeps v1 consumers whole.
    refs, unknown, refused = _validate_evidence(
        list(evidence_refs) + claim_refs_flat, served[0], served[1]
    )
    if refused:
        return {
            "error": "absence evidence refused: a cited scan cannot prove absence",
            "refused_absence": refused,
            "hint": (
                "Only a verdict of 'absent' over a non-truncated dataset scan "
                "proves absence. Widen the scope or re-index, re-run search_data, "
                "then cite the new scan."
            ),
        }
    if unknown:
        return {
            "error": (
                "evidence attestation failed: the following refs do not "
                "correspond to anything retrieved in this session"
            ),
            "unknown_refs": unknown,
            "hint": (
                "Evidence refs must be column ids (or their dataset names) "
                "that this session actually served. Retrieve the evidence "
                "first, then finalize."
            ),
        }

    # The input picks the contract: no claims anywhere is still a v1 handoff,
    # and its body is byte-identical to what v1 rendered.
    schema = HANDOFF_SCHEMA_V2 if claim_count else HANDOFF_SCHEMA
    body = render_handoff(
        dataset.strip(), task.strip(), profile.strip(), sec, refs, apps, schema
    )
    raw = body.encode("utf-8")
    sha256 = hashlib.sha256(raw).hexdigest()
    handoff_id = sha256[:16]
    receipt = {
        "schema": schema,
        "handoff_id": handoff_id,
        "dataset": dataset.strip(),
        "profile": profile.strip(),
        "content_type": HANDOFF_CONTENT_TYPE,
        "resource_uri": f"{HANDOFF_URI_PREFIX}{handoff_id}",
        "sha256": sha256,
        "length": len(raw),
        "canonical": True,
        "evidence_attested": True,
        "evidence_count": len(refs),
        "appendices": [name for name, _, _ in apps],
    }
    if claim_count:
        # Omitted entirely on a v1 handoff, so v1 receipts stay unchanged.
        receipt["claims_attested"] = claim_count
    absence_count = sum(1 for r in refs if r.startswith(ABSENCE_REF_PREFIX))
    if absence_count:
        receipt["absence_attested"] = absence_count
    with _lock:
        _handoffs[handoff_id] = {"body": body, "receipt": receipt}
    return dict(receipt)


def get_handoff(handoff_id: str) -> Optional[dict]:
    with _lock:
        rec = _handoffs.get(handoff_id)
        return {"body": rec["body"], "receipt": dict(rec["receipt"])} if rec else None


def handoff_for_uri(uri: str) -> Optional[dict]:
    s = str(uri)
    if not s.startswith(HANDOFF_URI_PREFIX):
        return None
    return get_handoff(s[len(HANDOFF_URI_PREFIX):])


def list_handoff_resources() -> list[dict]:
    with _lock:
        return [
            {
                "uri": rec["receipt"]["resource_uri"],
                "name": f"handoff-{hid}",
                "description": (
                    f"Canonical handoff for {rec['receipt']['dataset']} "
                    f"({rec['receipt']['profile']}); immutable, "
                    f"sha256 {rec['receipt']['sha256'][:12]}…"
                ),
            }
            for hid, rec in _handoffs.items()
        ]


def clear_handoffs() -> None:
    with _lock:
        _handoffs.clear()
