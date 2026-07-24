"""v1.27.0 — a rewrite underneath a scan cannot prove absence (5th refusal rule).

Suite parity with jcodemunch-mcp v1.108.168 / jdocmunch-mcp v1.119.0.

The interesting part is that this rule is genuinely ENFORCED here, unlike the
stale-index rule. jData models no index freshness, so v1.26.0 disclosed that
limitation in the rendered proof rather than shipping a gate that reads as
enforced and isn't. But "was this rewritten under me" is a filesystem fact, not
a freshness model — so it can be backed for real, the same way the truncation
gate is real here.

The dataset mtime deliberately covers `data.sqlite` (and its WAL), not just
`index.json`: the rows a search scans live in the SQLite store, so a reindex
that rewrites rows must register even when the metadata monolith is untouched.
"""

import pytest

from jdatamunch_mcp import handoff
from jdatamunch_mcp.storage.data_store import index_changed_since_load
from jdatamunch_mcp.verdict import build_verdict


class _FakeIndex:
    pass


@pytest.fixture()
def clean_absences():
    handoff.clear_absences()
    yield
    handoff.clear_absences()


def _stamped(tmp_path, loaded_mtime=None):
    idx_json = tmp_path / "index.json"
    sqlite = tmp_path / "data.sqlite"
    idx_json.write_text("{}")
    sqlite.write_bytes(b"rows")
    idx = _FakeIndex()
    idx._dataset_paths = (str(idx_json), str(sqlite))
    from jdatamunch_mcp.storage.data_store import _dataset_mtime_ns

    idx._loaded_mtime_ns = (
        _dataset_mtime_ns(idx_json, sqlite) if loaded_mtime is None else loaded_mtime
    )
    return idx, idx_json, sqlite


def _verdict(state, index_channel=None):
    channels = {"lexical": "ok", "semantic": "off"}
    if index_channel:
        channels["index"] = index_channel
    return {
        "state": state,
        "scanned": {"columns": 10},
        "channels": channels,
        "scorer": 1,
    }


class TestChangeDetection:
    def test_unstamped_index_is_not_changed(self):
        """Unknown must never mean changed, or every verdict degrades."""
        assert index_changed_since_load(_FakeIndex()) is False

    def test_settled_dataset_reports_false(self, tmp_path):
        idx, _, _ = _stamped(tmp_path)
        assert index_changed_since_load(idx) is False

    def test_rewritten_dataset_reports_true(self, tmp_path):
        idx, _, _ = _stamped(tmp_path, loaded_mtime=1)
        assert index_changed_since_load(idx) is True

    def test_row_store_rewrite_registers_even_if_metadata_is_untouched(self, tmp_path):
        """The rows are in data.sqlite; watching only index.json would miss this."""
        idx, _, sqlite = _stamped(tmp_path)
        assert index_changed_since_load(idx) is False
        import os

        future = idx._loaded_mtime_ns + 10_000_000_000
        os.utime(sqlite, ns=(future, future))
        assert index_changed_since_load(idx) is True

    def test_missing_files_never_raise(self, tmp_path):
        idx = _FakeIndex()
        idx._dataset_paths = (str(tmp_path / "a.json"), str(tmp_path / "b.sqlite"))
        idx._loaded_mtime_ns = 12345
        assert index_changed_since_load(idx) is False


class TestVerdictGate:
    def test_zero_results_mid_rewrite_is_degraded_not_absent(self):
        v = build_verdict(result_count=0, index_changed=True)
        assert v["state"] == "degraded"
        assert v["channels"]["index"] == "rebuilding"

    def test_zero_results_on_a_settled_dataset_still_proves_absence(self):
        v = build_verdict(result_count=0, index_changed=False)
        assert v["state"] == "absent"

    def test_index_channel_absent_when_not_rewriting(self):
        """jData cannot claim `fresh` — it tracks no freshness. Positive only."""
        v = build_verdict(result_count=0)
        assert "index" not in v["channels"]

    def test_results_are_still_returned_mid_rewrite(self):
        v = build_verdict(result_count=5, index_changed=True)
        assert v["state"] == "ok"
        assert v["channels"]["index"] == "rebuilding"

    def test_default_is_byte_identical_to_pre_1_27_0(self):
        assert build_verdict(result_count=0) == build_verdict(
            result_count=0, index_changed=False
        )


class TestAbsenceRefusal:
    def test_rewrite_refusal_names_the_cause(self):
        reason = handoff.absence_refusal(
            {"state": "degraded", "channels": {"index": "rebuilding"}}
        )
        assert reason is not None
        assert "rewritten" in reason

    def test_rewrite_scan_yields_no_citable_ref(self, clean_absences):
        ref, refusal = handoff.note_absence(
            "search_data", "sales", "widget",
            _verdict("degraded", index_channel="rebuilding"),
        )
        assert ref is None
        assert "rewritten" in refusal

    def test_settled_dataset_still_mints_a_ref(self, clean_absences):
        ref, refusal = handoff.note_absence(
            "search_data", "sales", "widget", _verdict("absent")
        )
        assert refusal is None
        assert ref and ref.startswith("absent:")

    def test_prior_rules_unaffected(self):
        assert handoff.absence_refusal({"state": "low_confidence"}) is not None
        assert handoff.absence_refusal(
            {"state": "absent", "coverage": {"walk": "truncated"}}
        ) is not None
        assert handoff.absence_refusal({"state": "absent"}) is None

    def test_freshness_is_still_disclosed_as_untracked(self):
        """The v1.26.0 DISCLOSE decision stands: this rule adds, never replaces."""
        import inspect

        assert "not tracked by this product" in inspect.getsource(handoff)
