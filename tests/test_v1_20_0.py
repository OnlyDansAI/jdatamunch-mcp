"""v1.20.0 — coverage contract on absence claims.

An `absent` verdict backed only by scan counts lies by omission when data was
excluded at index time. index_local now persists a coverage block in
index.json; search_data attaches a coverage disclosure to non-ok verdicts.
"""

import json
from pathlib import Path

from jdatamunch_mcp.storage.data_store import DataStore
from jdatamunch_mcp.tools.index_local import index_local
from jdatamunch_mcp.tools.search_data import search_data
from jdatamunch_mcp.verdict import (
    SCORER_VERSION,
    build_coverage_disclosure,
    build_verdict,
)


def _write_tripwire_jsonl(tmp_path) -> str:
    """JSONL with one malformed line — the trip-wire skip."""
    path = tmp_path / "tripwire.jsonl"
    lines = [
        json.dumps({"id": 1, "name": "Alice", "city": "Hollywood"}),
        json.dumps({"id": 2, "name": "Bob", "city": "Central"}),
        "this is not json {{{",
        json.dumps({"id": 3, "name": "Charlie", "city": "Pacific"}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class TestScorerPin:
    def test_scorer_version_is_pinned(self):
        assert SCORER_VERSION == 1

    def test_every_state_carries_scorer(self):
        assert build_verdict(result_count=3)["scorer"] == 1
        assert build_verdict(result_count=0)["scorer"] == 1
        assert build_verdict(
            result_count=0, semantic_requested=True, semantic_available=False
        )["scorer"] == 1


class TestCoverageDisclosureUnit:
    def test_none_coverage_yields_no_block(self):
        assert build_coverage_disclosure(None) is None
        assert build_coverage_disclosure({}) is None

    def test_empty_skip_counts_omits_excluded(self):
        block = build_coverage_disclosure(
            {"walk": "full", "rows_indexed": 10, "skip_counts": {}},
            indexed_at="2026-07-19T00:00:00",
            index_version=3,
        )
        assert block["walk"] == "full"
        assert block["rows_indexed"] == 10
        assert block["generation"] == {
            "indexed_at": "2026-07-19T00:00:00",
            "index_version": 3,
        }
        assert "excluded" not in block

    def test_verdict_attaches_coverage_on_absent_only(self):
        cov = {"walk": "full", "rows_indexed": 10}
        v_ok = build_verdict(result_count=3, coverage=cov)
        v_absent = build_verdict(result_count=0, coverage=cov)
        assert "coverage" not in v_ok
        assert v_absent["coverage"] == cov

    def test_verdict_attaches_coverage_on_degraded(self):
        cov = {"walk": "truncated", "rows_indexed": 5}
        v = build_verdict(
            result_count=2,
            semantic_requested=True,
            semantic_available=False,
            coverage=cov,
        )
        assert v["state"] == "degraded"
        assert v["coverage"] == cov


class TestCoveragePersistedAtIngest:
    def test_tripwire_skip_recorded(self, tmp_path, storage_dir):
        jsonl = _write_tripwire_jsonl(tmp_path)
        res = index_local(path=jsonl, name="tripwire", storage_path=storage_dir)
        assert "error" not in res

        idx = DataStore(base_path=storage_dir).load("tripwire")
        cov = idx.coverage
        assert cov is not None
        assert cov["walk"] == "full"
        assert cov["rows_indexed"] == 3
        assert cov["skip_counts"] == {"malformed_rows": 1}
        assert cov["recorded_at"]

    def test_clean_ingest_has_full_walk_no_skips(self, sample_csv, storage_dir):
        res = index_local(path=sample_csv, name="clean", storage_path=storage_dir)
        assert "error" not in res
        cov = DataStore(base_path=storage_dir).load("clean").coverage
        assert cov["walk"] == "full"
        assert cov["rows_indexed"] == 10
        assert "skip_counts" not in cov

    def test_row_cap_truncation_counted(self, sample_csv, storage_dir, monkeypatch):
        monkeypatch.setenv("JDATAMUNCH_MAX_ROWS", "5")
        res = index_local(path=sample_csv, name="capped", storage_path=storage_dir)
        assert "error" not in res
        cov = DataStore(base_path=storage_dir).load("capped").coverage
        assert cov["walk"] == "truncated"
        assert cov["rows_indexed"] == 5
        assert cov["skip_counts"]["rows_over_cap"] == 5

    def test_shallow_truncation_never_fabricates_count(
        self, sample_csv, storage_dir, monkeypatch
    ):
        monkeypatch.setenv("JDATAMUNCH_MAX_ROWS", "5")
        res = index_local(
            path=sample_csv, name="shallow", depth="shallow", storage_path=storage_dir
        )
        assert "error" not in res
        cov = DataStore(base_path=storage_dir).load("shallow").coverage
        assert cov["walk"] == "truncated"
        assert "rows_over_cap" not in cov.get("skip_counts", {})


class TestRepoDiscoverySkipCounts:
    def test_discovery_tallies_exclusions_by_reason(self):
        from jdatamunch_mcp.tools.index_repo import MAX_FILE_SIZE, _discover_data_files

        tree = [
            {"type": "blob", "path": "data/sales.csv", "size": 100},
            {"type": "blob", "path": "src/main.py", "size": 100},
            {"type": "blob", "path": "node_modules/x/d.csv", "size": 100},
            {"type": "blob", "path": "huge.csv", "size": MAX_FILE_SIZE + 1},
        ]
        files, skips = _discover_data_files(tree)
        assert [f["path"] for f in files] == ["data/sales.csv"]
        assert skips == {
            "unsupported_extension": 1,
            "skipped_path": 1,
            "oversize": 1,
        }


class TestVerdictCoverageDisclosure:
    def test_absent_verdict_discloses_coverage(self, tmp_path, storage_dir):
        jsonl = _write_tripwire_jsonl(tmp_path)
        index_local(path=jsonl, name="tripwire", storage_path=storage_dir)

        res = search_data(
            dataset="tripwire", query="zzz_nomatch_qqq", storage_path=storage_dir
        )
        v = res["_meta"]["verdict"]
        assert v["state"] == "absent"
        cov = v["coverage"]
        assert cov["generation"]["indexed_at"]
        assert cov["generation"]["index_version"] >= 3
        assert cov["rows_indexed"] == 3
        assert cov["excluded"] == {"malformed_rows": 1}

    def test_ok_verdict_stays_lean(self, tmp_path, storage_dir):
        jsonl = _write_tripwire_jsonl(tmp_path)
        index_local(path=jsonl, name="tripwire", storage_path=storage_dir)

        res = search_data(dataset="tripwire", query="city", storage_path=storage_dir)
        v = res["_meta"]["verdict"]
        assert v["state"] == "ok"
        assert "coverage" not in v

    def test_legacy_index_without_coverage_yields_no_block(
        self, indexed_sample
    ):
        # Simulate an index written before the coverage contract.
        index_json = Path(indexed_sample) / "sample" / "index.json"
        data = json.loads(index_json.read_text(encoding="utf-8"))
        data.pop("coverage", None)
        index_json.write_text(json.dumps(data), encoding="utf-8")

        res = search_data(
            dataset="sample", query="zzz_nomatch_qqq", storage_path=indexed_sample
        )
        v = res["_meta"]["verdict"]
        assert v["state"] == "absent"
        assert "coverage" not in v
