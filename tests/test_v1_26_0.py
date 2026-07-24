"""Absence evidence in the handoff contract (handoff/v2 phase 3).

Suite parity with jcodemunch-mcp v1.108.166 / jdocmunch-mcp v1.117.0
(jcodemunch-mcp#377 phase 3, design by @mightydanp).

A zero-result search cannot be cited under v1/v2: nothing was served, so there
is no id. This lets a search_data verdict of ``absent`` be recorded under a
deterministic ref and cited as proof that no such column/value is present.

The refusal rules are the feature and they are strict: only ``absent`` proves
absence; ``degraded`` cannot; a truncated row walk cannot.

jData's HONEST DIVERGENCE (the DISCLOSE product decision): jData models no index
freshness, so the stale-index refusal its siblings enforce cannot fire here.
Rather than ship a guarantee that reads enforced and isn't, every jData absence
proof discloses "index freshness: not tracked by this product" in the body.
Absence stays citable; the reader is told exactly what was and was not checked.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os

import pytest

from jdatamunch_mcp import handoff


@pytest.fixture(autouse=True)
def _clear():
    handoff.clear_handoffs()
    handoff.clear_session_record()
    handoff.clear_absences()
    yield
    handoff.clear_handoffs()
    handoff.clear_session_record()
    handoff.clear_absences()


def _verdict(
    state="absent",
    walk="full",
    rows_indexed=1000,
    excluded=None,
    semantic="off",
    scorer=1,
    coverage=True,
):
    v = {
        "state": state,
        "scorer": scorer,
        "channels": {"lexical": "ok", "semantic": semantic},
    }
    if coverage:
        cov = {
            "walk": walk,
            "rows_indexed": rows_indexed,
            "generation": {"indexed_at": "2026-07-24T00:00:00Z", "index_version": 3},
        }
        if excluded:
            cov["excluded"] = excluded
        v["coverage"] = cov
    return v


def _record(tool="search_data", dataset="orders", query="ghost_col", arguments=None, **vkw):
    """Record a verdict, return (ref, refusal)."""
    return handoff.note_absence(tool, dataset, query, _verdict(**vkw), arguments=arguments)


def _finalize_citing(ref, statement="No ghost_col column exists in orders."):
    return handoff.finalize_handoff(
        dataset="orders",
        task="Confirm a column is absent",
        sections=[
            {"heading": "Absence", "claims": [
                {"id": "c1", "statement": statement, "evidence_refs": [ref]}
            ]}
        ],
        evidence_refs=[],
    )


class TestOnlyAbsentProves:
    def test_absent_verdict_is_citable(self):
        ref, refusal = _record(state="absent")
        assert ref is not None and ref.startswith("absent:")
        assert refusal is None

    def test_degraded_verdict_refused(self):
        ref, refusal = _record(state="degraded", semantic="unavailable")
        assert ref is None
        assert "degraded" in refusal

    def test_ok_verdict_not_recorded_at_all(self):
        # `ok` is not absence evidence; note_absence declines to record it.
        ref, refusal = _record(state="ok")
        assert ref is None and refusal is None
        computed = handoff._absence_ref("search_data", "orders", "ghost_col", {})
        assert handoff.absence_record(computed) is None


class TestTruncatedGates:
    def test_truncated_walk_refused(self):
        ref, refusal = _record(walk="truncated")
        assert ref is None
        assert "truncated" in refusal

    def test_full_walk_citable(self):
        ref, refusal = _record(walk="full")
        assert ref is not None and refusal is None

    def test_refused_scan_is_still_recorded(self):
        # A refused scan is kept, so citing it later returns the reason rather
        # than a bare unknown-ref error.
        _record(walk="truncated")
        computed = handoff._absence_ref("search_data", "orders", "ghost_col", {})
        rec = handoff.absence_record(computed)
        assert rec is not None
        assert handoff.absence_refusal(rec) is not None


class TestFreshnessDisclosed:
    """The DISCLOSE decision: jData tracks no freshness, and says so in band."""

    def test_no_stale_rule_blocks_a_fresh_looking_absent_scan(self):
        # jcm/jdoc would consult an index channel here; jData has none, so a
        # clean absent+full scan is citable, with the limitation disclosed.
        ref, refusal = _record(state="absent", walk="full")
        assert ref is not None and refusal is None

    def test_proof_discloses_freshness_not_tracked(self):
        ref, _ = _record()
        receipt = _finalize_citing(ref)
        body = handoff.get_handoff(receipt["handoff_id"])["body"]
        assert "index freshness: not tracked by this product" in body

    def test_disclosure_present_even_when_coverage_missing(self):
        ref, _ = _record(coverage=False)
        receipt = _finalize_citing(ref)
        body = handoff.get_handoff(receipt["handoff_id"])["body"]
        assert "coverage: not recorded for this index (scope unknown)" in body
        assert "index freshness: not tracked by this product" in body


class TestCitation:
    def test_citable_ref_attests_in_finalize(self):
        ref, _ = _record()
        receipt = _finalize_citing(ref)
        assert "error" not in receipt
        assert receipt["schema"] == "jdatamunch.handoff/v2"
        assert receipt["absence_attested"] == 1

    def test_refused_ref_named_at_finalize(self):
        # Record a refused (degraded) scan, then fetch its ref and try to cite
        # it. The record exists, so finalize returns the REASON, not unknown.
        handoff.note_absence(
            "search_data", "orders", "ghost_col", _verdict(state="degraded", semantic="unavailable")
        )
        ref = handoff._absence_ref("search_data", "orders", "ghost_col", {})
        receipt = _finalize_citing(ref)
        assert "error" in receipt
        assert receipt["refused_absence_claims"][0]["claim_id"] == "c1"

    def test_unknown_absence_ref_fails_closed(self):
        receipt = _finalize_citing("absent:deadbeef0000")
        assert "error" in receipt
        assert "absent:deadbeef0000" in receipt["invalid_claims"][0]["unknown_refs"]

    def test_live_response_flags_absent_but_not_citable(self):
        # The refusal string a live search_data response would surface in band.
        ref, refusal = _record(state="degraded", semantic="unavailable")
        assert ref is None
        assert refusal and "absent" in refusal


class TestRenderedProofIsAuditable:
    def test_proof_carries_query_and_scanned_rows(self):
        ref, _ = _record(rows_indexed=42)
        body = handoff.get_handoff(_finalize_citing(ref)["handoff_id"])["body"]
        assert "absence proof: `search_data` query 'ghost_col'" in body
        assert "42 rows indexed" in body

    def test_proof_carries_scope_narrowing(self):
        ref, _ = _record(arguments={"search_scope": "name"})
        body = handoff.get_handoff(_finalize_citing(ref)["handoff_id"])["body"]
        assert "scope: search_scope='name'" in body

    def test_default_scope_renders_as_whole_dataset(self):
        ref, _ = _record(arguments={"search_scope": "all"})
        body = handoff.get_handoff(_finalize_citing(ref)["handoff_id"])["body"]
        assert "scope: whole indexed dataset" in body

    def test_excluded_counts_rendered(self):
        ref, _ = _record(excluded={"malformed_rows": 7})
        body = handoff.get_handoff(_finalize_citing(ref)["handoff_id"])["body"]
        assert "malformed_rows=7" in body

    def test_detail_renders_once_under_its_claim(self):
        ref, _ = _record()
        body = handoff.get_handoff(_finalize_citing(ref)["handoff_id"])["body"]
        assert body.count("absence proof: `search_data`") == 1

    def test_detail_in_global_index_when_no_claim(self):
        # An absence ref cited only at the top level renders its proof in the
        # global Evidence block.
        ref, _ = _record()
        receipt = handoff.finalize_handoff(
            dataset="orders",
            task="t",
            sections=[{"heading": "S", "content": "Body."}],
            evidence_refs=[ref],
        )
        body = handoff.get_handoff(receipt["handoff_id"])["body"]
        tail = body[body.index("## Evidence"):]
        assert "absence proof: `search_data`" in tail


class TestRefIdentity:
    def test_same_scan_same_ref(self):
        a, _ = _record()
        handoff.clear_absences()
        b, _ = _record()
        assert a == b

    def test_different_query_different_ref(self):
        a, _ = _record(query="ghost_col")
        b, _ = _record(query="other_col")
        assert a != b

    def test_different_scope_different_ref(self):
        a, _ = _record(arguments={"search_scope": "name"})
        b, _ = _record(arguments={"search_scope": "value"})
        assert a != b


class TestV1AndV2Unaffected:
    SERVED = ["orders::total#column"]

    def test_symbol_only_handoff_has_no_absence_attested(self):
        handoff.note_served_rows([{"id": self.SERVED[0]}], dataset="orders")
        receipt = handoff.finalize_handoff(
            dataset="orders",
            task="t",
            sections=[{"heading": "S", "content": "Body."}],
            evidence_refs=[self.SERVED[0]],
        )
        assert "absence_attested" not in receipt

    def test_no_absence_detail_in_symbol_handoff(self):
        handoff.note_served_rows([{"id": self.SERVED[0]}], dataset="orders")
        receipt = handoff.finalize_handoff(
            dataset="orders",
            task="t",
            sections=[{"heading": "S", "content": "Body."}],
            evidence_refs=[self.SERVED[0]],
        )
        body = handoff.get_handoff(receipt["handoff_id"])["body"]
        assert "absence proof" not in body


class TestServerChokepoint:
    """The live wiring: an absent search_data hands back a citable token that
    survives the default _meta strip, and finalize attests it end to end."""

    def test_absent_search_reattaches_citable_ref_and_finalizes(self, tmp_path):
        csv_path = tmp_path / "orders_v1260.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["order_total", "customer_ref"])
            w.writerows([(10, "a"), (20, "b")])
        storage = tmp_path / "data-index"
        storage.mkdir()
        from jdatamunch_mcp.tools.index_local import index_local
        index_local(path=str(csv_path), name="orders_v1260", storage_path=str(storage))

        from jdatamunch_mcp import server
        old = os.environ.get("DATA_INDEX_PATH")
        os.environ["DATA_INDEX_PATH"] = str(storage)
        try:
            out = asyncio.run(server.call_tool("search_data", {
                "dataset": "orders_v1260", "query": "zzqqxnothingmatchesthis",
            }))
            payload = json.loads(out[0].text)
            # Zero matches -> absent -> a citable token survives the _meta strip.
            assert payload.get("result") == [], payload
            ref = payload["_meta"]["absence_evidence"]["ref"]
            assert ref.startswith("absent:")

            receipt = asyncio.run(server.call_tool("finalize_handoff", {
                "dataset": "orders_v1260",
                "task": "Confirm a column is absent",
                "sections": [{"heading": "Absence", "claims": [{
                    "id": "c1",
                    "statement": "No column matches 'zzqqxnothingmatchesthis'.",
                    "evidence_refs": [ref],
                }]}],
                "evidence_refs": [],
            }))
            r = json.loads(receipt[0].text)
            assert "error" not in r, r
            assert r["absence_attested"] == 1
            body = handoff.get_handoff(r["handoff_id"])["body"]
            assert "index freshness: not tracked by this product" in body
        finally:
            if old is None:
                os.environ.pop("DATA_INDEX_PATH", None)
            else:
                os.environ["DATA_INDEX_PATH"] = old
        handoff.clear_absences()
