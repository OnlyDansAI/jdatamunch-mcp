"""An ignored argument cannot back an absence claim (v1.29.0).

Suite parity with jcodemunch-mcp v1.108.175, where the defect was found live: a
`search_text` call passed `regex=true` when the parameter is `is_regex`. The
flag was dropped in silence, the call that ran was not the call that was asked
for, and the response still reached `state: "absent"` and minted a citable
absence ref.

jData difference: the disclosure is TOP-LEVEL, because this server strips
`_meta` entirely by default -- a notice under `_meta` would be deleted before
the agent saw it (same call as `empty`/`hint` in v1.28.0).
"""

from __future__ import annotations

import json

import pytest

from jdatamunch_mcp.tools._arg_contract import (
    degrade_absent_verdict,
    disclose,
    note,
    unrecognized_keys,
)

DECLARED = ("dataset", "query", "limit")


# --- which keys count as unrecognized -----------------------------------------

def test_a_misspelled_parameter_is_reported():
    assert unrecognized_keys({"dataset": "d", "lmit": 5}, DECLARED) == ["lmit"]


def test_declared_parameters_are_silent():
    assert unrecognized_keys({"dataset": "d", "limit": 5}, DECLARED) == []


def test_protocol_meta_is_never_the_callers_mistake():
    assert unrecognized_keys({"dataset": "d", "_meta": {"progressToken": 1}}, DECLARED) == []


def test_unknown_schema_accuses_nobody():
    """An absent declaration is not evidence that a key is wrong."""
    assert unrecognized_keys({"whatever": 1}, None) == []
    assert unrecognized_keys({"whatever": 1}, ()) == []


def test_multiple_unknown_keys_are_sorted():
    assert unrecognized_keys({"zeta": 1, "alpha": 2}, DECLARED) == ["alpha", "zeta"]


# --- the absence claim is stripped, other states are not ----------------------

def _absent():
    return {"result": [], "_meta": {"verdict": {"state": "absent", "note": "none found"}}}


def test_absence_claim_is_downgraded():
    r = _absent()
    degrade_absent_verdict(r, ["lmit"])
    assert r["_meta"]["verdict"]["state"] == "degraded"
    assert "lmit" in r["_meta"]["verdict"]["note"]


def test_a_confident_result_is_not_downgraded():
    r = {"result": [1], "_meta": {"verdict": {"state": "ok", "note": "Confident."}}}
    degrade_absent_verdict(r, ["lmit"])
    assert r["_meta"]["verdict"]["state"] == "ok"
    assert r["_meta"]["verdict"]["note"] == "Confident."


def test_clean_calls_change_nothing():
    r = _absent()
    degrade_absent_verdict(r, [])
    disclose(r, [])
    assert r["_meta"]["verdict"]["state"] == "absent"
    assert "ignored_arguments" not in r


def test_missing_verdict_is_survivable():
    r = {"result": []}
    degrade_absent_verdict(r, ["lmit"])
    assert r == {"result": []}


def test_non_dict_results_never_raise():
    degrade_absent_verdict(["a"], ["x"])
    disclose(None, ["x"])


def test_note_reads_correctly_for_one_and_for_many():
    assert note(["a"]).startswith("An argument this tool does not accept was ignored")
    assert note(["a", "b"]).startswith("2 arguments this tool does not accept were ignored")


# --- the disclosure must survive the default _meta strip ----------------------

def test_disclosure_is_top_level_not_under_meta():
    """`_meta` is stripped by default here, so a notice there is invisible."""
    r = {"result": []}
    disclose(r, ["lmit"])
    assert r["ignored_arguments"] == ["lmit"]
    assert "lmit" in r["ignored_arguments_note"]
    assert "ignored_arguments" not in r.get("_meta", {})


# --- end to end through the real dispatcher -----------------------------------

@pytest.mark.asyncio
async def test_dispatcher_discloses_and_refuses_the_absence_ref(tmp_path, monkeypatch):
    """A bogus argument survives the default `_meta` strip as a top-level key."""
    import csv

    from jdatamunch_mcp import server
    from jdatamunch_mcp.tools.index_local import index_local

    store = tmp_path / "store"
    store.mkdir()
    csv_path = tmp_path / "rows.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "city"])
        w.writerow(["1", "Vilnius"])
        w.writerow(["2", "Kaunas"])
    index_local(path=str(csv_path), name="rows", storage_path=str(store))
    monkeypatch.setenv("DATA_INDEX_PATH", str(store))

    out = await server.call_tool(
        "search_data",
        {"dataset": "rows", "query": "Vilnius", "lmit": 5},
    )
    body = json.loads(out[0].text)
    assert body.get("ignored_arguments") == ["lmit"]
    # An absence ref must not be minted from a misunderstood call.
    assert "ref" not in (body.get("_meta", {}).get("absence_evidence") or {})


@pytest.mark.asyncio
async def test_a_correct_call_gains_no_key(tmp_path, monkeypatch):
    """Non-vacuity + zero blast radius."""
    import csv

    from jdatamunch_mcp import server
    from jdatamunch_mcp.tools.index_local import index_local

    store = tmp_path / "store"
    store.mkdir()
    csv_path = tmp_path / "rows.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "city"])
        w.writerow(["1", "Vilnius"])
    index_local(path=str(csv_path), name="rows", storage_path=str(store))
    monkeypatch.setenv("DATA_INDEX_PATH", str(store))

    out = await server.call_tool(
        "search_data", {"dataset": "rows", "query": "Vilnius"}
    )
    body = json.loads(out[0].text)
    assert "ignored_arguments" not in body


def test_declared_keys_come_from_the_published_catalog():
    from jdatamunch_mcp import server

    declared = server._declared_arg_keys("search_data")
    assert declared is not None
    assert "dataset" in declared and "lmit" not in declared
    assert server._declared_arg_keys("no_such_tool") is None
