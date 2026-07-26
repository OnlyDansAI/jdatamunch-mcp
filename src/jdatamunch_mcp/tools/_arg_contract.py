"""Argument contract: an argument we ignored cannot back an absence claim.

Suite parity with jcodemunch-mcp v1.108.175, where the defect was found live: a
`search_text` call passed `regex=true` when the parameter is `is_regex`. Every
tool in all three servers reads its arguments key-by-key
(``arguments.get("limit", 20)``), so a key the tool does not know is dropped in
silence. That is fine for a typo on an optional flag and NOT fine for one
consequence: the call the agent believed it made is not the call that ran, and
the result can still reach ``state: "absent"`` and mint a citable absence ref.

The rule:

  * DISCLOSE on every state — a caller reading an ``ok`` result still deserves to
    know part of its call was discarded.
  * REFUSE only the absence CLAIM, by downgrading ``absent`` to ``degraded``, so
    the existing "only ``absent`` proves absence" check in
    ``handoff.absence_refusal`` does the refusing. No second rule to keep in sync.

⚠ jData difference from jcm: the disclosure is TOP-LEVEL, not under ``_meta``.
This server strips ``_meta`` entirely by default (``get_meta_fields()`` returns
``[]``), so a notice placed there would be deleted before the agent ever saw it —
the same trap that forced the top-level ``empty``/``hint`` keys in v1.28.0 and
the post-filter re-attach of the absence ref in v1.26.0.

Deliberately never rejects the call: under the 1.x zero-surprise contract an
unknown key has always been accepted, and a hard reject would turn a recoverable
mistake into a dead call.
"""

from __future__ import annotations

from typing import Optional, Sequence

# Keys the MCP layer itself may attach to an arguments object. `_meta` is
# reserved by the protocol, so it is never the caller getting a name wrong.
PROTOCOL_KEYS = frozenset({"_meta"})


def unrecognized_keys(
    arguments: Optional[dict], declared: Optional[Sequence[str]]
) -> list[str]:
    """Argument keys this tool does not declare, sorted.

    Returns empty when the schema is unknown: an absent declaration is not
    evidence that a key is wrong, and guessing would manufacture false warnings
    on every tool whose schema we failed to read.
    """
    if not isinstance(arguments, dict) or not declared:
        return []
    known = set(declared) | PROTOCOL_KEYS
    return sorted(k for k in arguments if k not in known)


def note(ignored: list[str]) -> str:
    names = ", ".join(ignored)
    subject = (
        f"{len(ignored)} arguments this tool does not accept were ignored"
        if len(ignored) > 1
        else "An argument this tool does not accept was ignored"
    )
    return (
        f"{subject}: {names}. The call that ran is not the call that was "
        "requested, so this result cannot be read as evidence the target is "
        "absent. Check the parameter name against the tool schema and search "
        "again."
    )


def degrade_absent_verdict(result, ignored: list[str]) -> None:
    """Strip an absence claim built on a misunderstood call (in place).

    Runs BEFORE meta_fields filtering, while `_meta.verdict` still exists, so
    the absence-evidence step downstream refuses to mint a ref.
    """
    if not ignored or not isinstance(result, dict):
        return
    meta = result.get("_meta")
    if not isinstance(meta, dict):
        return
    verdict = meta.get("verdict")
    if isinstance(verdict, dict) and verdict.get("state") == "absent":
        verdict["state"] = "degraded"
        verdict["note"] = note(ignored)
        verdict["ignored_arguments"] = list(ignored)


def disclose(result, ignored: list[str]) -> None:
    """Attach the top-level disclosure (in place), AFTER meta_fields filtering."""
    if not ignored or not isinstance(result, dict):
        return
    result["ignored_arguments"] = list(ignored)
    result["ignored_arguments_note"] = note(ignored)
