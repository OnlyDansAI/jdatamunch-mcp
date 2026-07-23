"""v1.24.0 — canonical handoff contract (jdatamunch.handoff/v1).

Suite parity with jcodemunch-mcp v1.108.162 (#374 there) and jdocmunch-mcp
v1.114.2: finalize_handoff assembles a deterministic Markdown handoff from
caller-authored sections, attests evidence_refs against the session retrieval
record (column ids / dataset names served by search_data / describe_dataset /
describe_column), and serves the immutable body via munch://handoff/<id>.
"""

import asyncio
import csv
import hashlib
import json

import pytest

from jdatamunch_mcp import handoff


SERVED = (
    frozenset({"sales.csv::amount#column", "sales.csv::region#column"}),
    frozenset({"sales.csv"}),
)


def _finalize(**over):
    args = dict(
        dataset="sales.csv",
        task="Audit the revenue columns",
        sections=[{"heading": "Findings", "content": "Amounts are clean."}],
        evidence_refs=["sales.csv::amount#column"],
        served=SERVED,
    )
    args.update(over)
    return handoff.finalize_handoff(**args)


@pytest.fixture(autouse=True)
def _clean():
    handoff.clear_handoffs()
    handoff.clear_session_record()
    yield
    handoff.clear_handoffs()
    handoff.clear_session_record()


class TestFinalize:
    def test_receipt_shape(self):
        r = _finalize()
        assert r["schema"] == "jdatamunch.handoff/v1"
        assert r["canonical"] is True
        assert r["content_type"] == "text/markdown"
        assert r["resource_uri"] == f"munch://handoff/{r['handoff_id']}"
        assert len(r["sha256"]) == 64 and r["length"] > 0

    def test_deterministic(self):
        a, b = _finalize(), _finalize()
        assert (a["handoff_id"], a["sha256"]) == (b["handoff_id"], b["sha256"])
        assert _finalize(task="Other")["handoff_id"] != a["handoff_id"]

    def test_sha_matches_body(self):
        r = _finalize()
        body = handoff.get_handoff(r["handoff_id"])["body"]
        assert hashlib.sha256(body.encode("utf-8")).hexdigest() == r["sha256"]

    def test_dataset_ref_attests(self):
        assert _finalize(evidence_refs=["sales.csv"])["evidence_attested"] is True

    def test_unknown_ref_fails_closed(self):
        r = _finalize(evidence_refs=["ghost.csv::x#column"])
        assert "error" in r and r["unknown_refs"] == ["ghost.csv::x#column"]

    def test_empty_evidence_and_sections_rejected(self):
        assert "error" in _finalize(evidence_refs=[])
        assert "error" in _finalize(sections=[])

    def test_duplicate_appendix_rejected_and_exactly_once(self):
        assert "error" in _finalize(appendices=[
            {"name": "R", "content": "a"}, {"name": "R", "content": "b"},
        ])
        r = _finalize(appendices=[{"name": "Profiling report", "content": "raw"}])
        body = handoff.get_handoff(r["handoff_id"])["body"]
        assert body.count("## Appendix: Profiling report") == 1

    def test_no_char_limit(self):
        r = _finalize(sections=[{"heading": "Big", "content": "x" * 300_000}])
        assert "error" not in r


class TestSessionRecord:
    def test_note_served_feeds_attestation(self):
        handoff.note_served_column("hr.csv", "salary")
        ids, datasets = handoff.served_refs()
        assert "hr.csv::salary#column" in ids and "hr.csv" in datasets
        r = _finalize(dataset="hr.csv",
                      evidence_refs=["hr.csv::salary#column"], served=None)
        assert r["evidence_attested"] is True

    def test_search_chokepoint_records(self, tmp_path):
        csv_path = tmp_path / "orders_v1240.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["order_total", "customer_ref"])
            w.writerows([(10, "a"), (20, "b")])
        storage = tmp_path / "data-index"
        storage.mkdir()
        from jdatamunch_mcp.tools.index_local import index_local
        index_local(path=str(csv_path), name="orders_v1240", storage_path=str(storage))
        import os
        from jdatamunch_mcp import server
        old = os.environ.get("DATA_INDEX_PATH")
        os.environ["DATA_INDEX_PATH"] = str(storage)
        try:
            out = asyncio.run(server.call_tool("search_data", {
                "dataset": "orders_v1240", "query": "order total",
            }))
            payload = json.loads(out[0].text)
            assert payload.get("result"), payload
        finally:
            if old is None:
                os.environ.pop("DATA_INDEX_PATH", None)
            else:
                os.environ["DATA_INDEX_PATH"] = old
        ids, datasets = handoff.served_refs()
        assert "orders_v1240" in datasets
        assert any(i.startswith("orders_v1240::") for i in ids)


class TestResource:
    def test_repeated_reads_byte_identical(self):
        r = _finalize()
        from jdatamunch_mcp import server
        a = asyncio.run(server.read_resource(r["resource_uri"]))
        b = asyncio.run(server.read_resource(r["resource_uri"]))
        assert a[0].content == b[0].content
        assert a[0].mime_type == "text/markdown"

    def test_advertised_and_identity_unaffected(self):
        r = _finalize()
        from jdatamunch_mcp import server
        uris = [str(x.uri) for x in asyncio.run(server.list_resources())]
        assert r["resource_uri"] in uris
        assert "munch://runtime/identity" in uris

    def test_unknown_id_raises(self):
        from jdatamunch_mcp import server
        with pytest.raises(ValueError):
            asyncio.run(server.read_resource("munch://handoff/0000000000000000"))


class TestRegistration:
    def test_tool_registered_and_write_annotated(self):
        import jdatamunch_mcp.server as srv
        tools = asyncio.run(srv.list_tools())
        t = next((x for x in tools if x.name == "finalize_handoff"), None)
        assert t is not None
        assert t.annotations is not None and t.annotations.readOnlyHint is False
        assert "finalize_handoff" in srv._TOOL_TIER_STANDARD
        assert "finalize_handoff" in srv._NON_READONLY_TOOLS

    def test_dispatch_error_shape(self):
        from jdatamunch_mcp import server
        res = asyncio.run(server.call_tool("finalize_handoff", {
            "dataset": "d", "task": "t",
            "sections": [{"heading": "H", "content": "C"}],
            "evidence_refs": ["never-served.csv"],
        }))
        body = json.loads(res[0].text)
        assert body["unknown_refs"] == ["never-served.csv"]
