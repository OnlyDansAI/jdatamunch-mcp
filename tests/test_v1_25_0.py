"""Claim-scoped evidence in the handoff contract (handoff/v2 phase 1).

Suite parity with jcodemunch-mcp v1.108.165 (jcodemunch-mcp#377 phase 1).

v1 proved every cited ref was retrieved this session. It could not say WHICH
retrieval backs which sentence: refs landed in one global block at the end.
Phase 1 lets a section carry caller-authored claims, each with its own
`evidence_refs`, attested separately and rendered beside the claim.

The load-bearing guarantee is backward compatibility: an input with no claims
anywhere must produce the byte-identical v1 body, id, and receipt.
"""

from __future__ import annotations

import pytest

from jdatamunch_mcp import handoff


@pytest.fixture(autouse=True)
def _clear():
    handoff.clear_handoffs()
    handoff.clear_session_record()
    handoff.note_served_rows([{"id": sid} for sid in SERVED], dataset="orders")
    yield
    handoff.clear_handoffs()
    handoff.clear_session_record()


SERVED = [
    "orders::customer_id#column",
    "orders::total#column",
    "customers::email#column",
]


def _claim(cid, statement, refs, classification=None):
    claim = {"id": cid, "statement": statement, "evidence_refs": refs}
    if classification:
        claim["classification"] = classification
    return claim


def _finalize(sections, evidence_refs=None, **kw):
    return handoff.finalize_handoff(
        dataset="orders",
        task="Audit the orders dataset",
        sections=sections,
        evidence_refs=[] if evidence_refs is None else evidence_refs,
        **kw,
    )


class TestV1Unchanged:
    """A no-claims input must be indistinguishable from pre-#377 output."""

    def test_v1_receipt_carries_v1_schema_and_no_claim_key(self):
        receipt = _finalize(
            [{"heading": "Findings", "content": "Body text."}],
            evidence_refs=[SERVED[0]],
        )
        assert receipt["schema"] == "jdatamunch.handoff/v1"
        assert "claims_attested" not in receipt

    def test_v1_body_has_no_claim_scaffolding(self):
        receipt = _finalize(
            [{"heading": "Findings", "content": "Body text."}],
            evidence_refs=[SERVED[0]],
        )
        body = handoff.get_handoff(receipt["handoff_id"])["body"]
        assert "- Claim id:" not in body
        assert "handoff/v1" in body
        # The v1 shape: heading, content, then the single global Evidence block.
        assert body.index("## Findings") < body.index("## Evidence")

    def test_v1_content_still_required_without_claims(self):
        result = _finalize([{"heading": "Findings"}], evidence_refs=[SERVED[0]])
        assert "error" in result
        assert "content" in result["error"]

    def test_evidence_refs_still_required_without_claims(self):
        result = _finalize([{"heading": "Findings", "content": "Body."}])
        assert "error" in result
        assert "evidence_refs" in result["error"]


class TestClaimScopedEvidence:
    def test_claims_produce_v2_schema_and_count(self):
        receipt = _finalize(
            [
                {
                    "heading": "Column quality",
                    "content": "Overview.",
                    "claims": [
                        _claim("orders.total.not_null", "orders.total has no nulls.", [SERVED[1]]),
                        _claim("orders.customer_id.key", "orders.customer_id is the join key.", [SERVED[0]]),
                    ],
                }
            ],
            evidence_refs=[SERVED[0]],
        )
        assert receipt["schema"] == "jdatamunch.handoff/v2"
        assert receipt["claims_attested"] == 2

    def test_evidence_renders_beside_the_claim(self):
        receipt = _finalize(
            [
                {
                    "heading": "Column quality",
                    "claims": [
                        _claim(
                            "orders.total.not_null",
                            "orders.total has no nulls.",
                            [SERVED[1]],
                            classification="verified_existing_behavior",
                        )
                    ],
                }
            ]
        )
        body = handoff.get_handoff(receipt["handoff_id"])["body"]
        claim_at = body.index("### orders.total has no nulls.")
        ref_at = body.index(f"  - `{SERVED[1]}`")
        global_at = body.index("## Evidence")
        # The ref appears under its claim, above the global index.
        assert claim_at < ref_at < global_at
        assert "- Claim id: `orders.total.not_null`" in body
        assert "- Classification: verified_existing_behavior" in body

    def test_claim_refs_join_the_canonical_evidence_index(self):
        receipt = _finalize(
            [
                {
                    "heading": "Column quality",
                    "claims": [_claim("c1", "A claim.", [SERVED[1]])],
                }
            ],
            evidence_refs=[SERVED[0]],
        )
        # Caller's global ref plus the claim's ref, deduped, caller order first.
        assert receipt["evidence_count"] == 2
        body = handoff.get_handoff(receipt["handoff_id"])["body"]
        tail = body[body.index("## Evidence"):]
        assert f"- `{SERVED[0]}`" in tail
        assert f"- `{SERVED[1]}`" in tail

    def test_claims_alone_satisfy_the_evidence_requirement(self):
        receipt = _finalize(
            [
                {
                    "heading": "Column quality",
                    "claims": [_claim("c1", "A claim.", [SERVED[1]])],
                }
            ]
        )
        assert "error" not in receipt
        assert receipt["evidence_count"] == 1

    def test_duplicate_ref_across_claims_indexed_once(self):
        receipt = _finalize(
            [
                {
                    "heading": "S",
                    "claims": [
                        _claim("c1", "First.", [SERVED[0]]),
                        _claim("c2", "Second.", [SERVED[0]]),
                    ],
                }
            ]
        )
        assert receipt["evidence_count"] == 1
        assert receipt["claims_attested"] == 2

    def test_statement_preserved_verbatim(self):
        statement = "`orders.total` contains no null values (see #377)."
        receipt = _finalize(
            [{"heading": "S", "claims": [_claim("c1", statement, [SERVED[1]])]}]
        )
        body = handoff.get_handoff(receipt["handoff_id"])["body"]
        assert f"### {statement}" in body


class TestFailsClosed:
    def test_unknown_claim_ref_names_its_claim(self):
        result = _finalize(
            [
                {
                    "heading": "S",
                    "claims": [
                        _claim("c1", "Good.", [SERVED[0]]),
                        _claim("c2", "Bad.", ["ghosts::phantom#column"]),
                    ],
                }
            ]
        )
        assert "error" in result
        assert result["invalid_claims"] == [
            {"claim_id": "c2", "unknown_refs": ["ghosts::phantom#column"]}
        ]

    def test_duplicate_claim_id_across_sections_rejected(self):
        result = _finalize(
            [
                {"heading": "A", "claims": [_claim("dup", "One.", [SERVED[0]])]},
                {"heading": "B", "claims": [_claim("dup", "Two.", [SERVED[1]])]},
            ]
        )
        assert "error" in result
        assert "duplicate claim id" in result["error"]

    def test_claim_without_evidence_rejected(self):
        result = _finalize([{"heading": "S", "claims": [_claim("c1", "Unbacked.", [])]}])
        assert "error" in result
        assert "evidence_refs" in result["error"]

    def test_claim_without_statement_rejected(self):
        result = _finalize(
            [{"heading": "S", "claims": [{"id": "c1", "evidence_refs": [SERVED[0]]}]}]
        )
        assert "error" in result
        assert "statement" in result["error"]

    def test_claim_without_id_rejected(self):
        result = _finalize(
            [{"heading": "S", "claims": [{"statement": "No id.", "evidence_refs": [SERVED[0]]}]}]
        )
        assert "error" in result
        assert "id" in result["error"]


class TestDeterminism:
    def test_same_claims_produce_same_id(self):
        sections = [
            {
                "heading": "S",
                "content": "Body.",
                "claims": [_claim("c1", "A claim.", [SERVED[0]])],
            }
        ]
        first = _finalize(sections)
        handoff.clear_handoffs()
        second = _finalize(sections)
        assert first["handoff_id"] == second["handoff_id"]
        assert first["sha256"] == second["sha256"]

    def test_adding_a_claim_changes_the_id(self):
        base = _finalize([{"heading": "S", "content": "Body."}], evidence_refs=[SERVED[0]])
        handoff.clear_handoffs()
        with_claim = _finalize(
            [
                {
                    "heading": "S",
                    "content": "Body.",
                    "claims": [_claim("c1", "A claim.", [SERVED[0]])],
                }
            ],
            evidence_refs=[SERVED[0]],
        )
        assert base["handoff_id"] != with_claim["handoff_id"]

    def test_resource_read_is_byte_identical_across_reads(self):
        receipt = _finalize(
            [{"heading": "S", "claims": [_claim("c1", "A claim.", [SERVED[0]])]}]
        )
        uri = receipt["resource_uri"]
        assert handoff.handoff_for_uri(uri)["body"] == handoff.handoff_for_uri(uri)["body"]
