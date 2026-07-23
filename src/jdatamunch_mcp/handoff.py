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
import threading
from typing import Iterable, Optional

HANDOFF_SCHEMA = "jdatamunch.handoff/v1"
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


def _validate_evidence(refs, served_ids: Iterable[str], served_datasets: Iterable[str]):
    served = set(served_ids or ())
    datasets = set(served_datasets or ())
    for sid in served:
        datasets.add(sid.split("::", 1)[0])
    seen: set[str] = set()
    ordered: list[str] = []
    unknown: list[str] = []
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip():
            unknown.append(repr(ref))
            continue
        ref = ref.strip()
        if ref in seen:
            continue
        seen.add(ref)
        ordered.append(ref)
        if ref not in served and ref not in datasets:
            unknown.append(ref)
    return ordered, unknown


def _validate_sections(sections):
    if not isinstance(sections, list) or not sections:
        return None, "sections must be a non-empty list of {heading, content} objects"
    out = []
    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            return None, f"sections[{i}] must be an object with 'heading' and 'content'"
        heading = sec.get("heading")
        content = sec.get("content")
        if not isinstance(heading, str) or not heading.strip():
            return None, f"sections[{i}].heading must be a non-empty string"
        if not isinstance(content, str) or not content.strip():
            return None, f"sections[{i}].content must be a non-empty string"
        out.append((heading.strip(), content.rstrip()))
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


def render_handoff(dataset: str, task: str, profile: str, sections, evidence_refs, appendices) -> str:
    """Deterministic canonical Markdown. No timestamps, no randomness."""
    lines = [
        f"# Handoff: {task}",
        "",
        f"- Schema: {HANDOFF_SCHEMA}",
        f"- Dataset: {dataset}",
        f"- Profile: {profile}",
        "",
    ]
    for heading, content in sections:
        lines += [f"## {heading}", "", content, ""]
    lines += [
        "## Evidence",
        "",
        "Every reference below was validated against this session's retrieval",
        "record at finalization time (server-attested).",
        "",
    ]
    lines += [f"- `{ref}`" for ref in evidence_refs]
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
    if not isinstance(evidence_refs, list) or not evidence_refs:
        return {
            "error": (
                "evidence_refs must be a non-empty list of session retrieval "
                "references (column ids like '<dataset>::<column>#column' or "
                "dataset names served this session by search_data / "
                "describe_dataset / describe_column)"
            )
        }
    if served is None:
        served = served_refs()
    refs, unknown = _validate_evidence(evidence_refs, served[0], served[1])
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

    body = render_handoff(dataset.strip(), task.strip(), profile.strip(), sec, refs, apps)
    raw = body.encode("utf-8")
    sha256 = hashlib.sha256(raw).hexdigest()
    handoff_id = sha256[:16]
    receipt = {
        "schema": HANDOFF_SCHEMA,
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
