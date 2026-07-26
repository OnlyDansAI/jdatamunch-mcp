"""Empty-store nudge (suite parity; jcodemunch-mcp#375 correspondence, suggestion C).

A user ran this server for months holding ZERO datasets and only found out by
going looking: `list_datasets` returned a bare `[]`, which reads identically to
"installed and broken". Their words: "we would have fed both tools months ago."

Top-level rather than `_meta`, because `get_meta_fields()` returns [] by default
here, so a hint in `_meta` would be stripped before the agent ever saw it. Same
key names as jcodemunch-mcp and jdocmunch-mcp.
"""

from __future__ import annotations

from jdatamunch_mcp.tools.list_datasets import list_datasets


def test_empty_store_says_so(tmp_path):
    result = list_datasets(storage_path=str(tmp_path))
    assert result["result"] == []
    assert result["empty"] is True
    assert "hint" in result


def test_the_hint_names_the_command_that_fixes_it(tmp_path):
    hint = list_datasets(storage_path=str(tmp_path))["hint"]
    assert "index_local" in hint
    assert "empty" in hint.lower()


def test_existing_shape_is_preserved(tmp_path):
    """Additive only: `result` and the `_meta` envelope survive untouched."""
    result = list_datasets(storage_path=str(tmp_path))
    assert "result" in result and "_meta" in result
    assert "total_tokens_saved" in result["_meta"]


def test_populated_store_stays_silent(tmp_path, monkeypatch):
    """No key and no token cost once anything is indexed."""
    import jdatamunch_mcp.tools.list_datasets as mod

    class _FakeStore:
        base_path = str(tmp_path)

        def __init__(self, *a, **k):
            pass

        def list_datasets(self):
            return [{"name": "sales", "rows": 10}]

    monkeypatch.setattr(mod, "DataStore", _FakeStore)
    result = list_datasets(storage_path=str(tmp_path))
    assert result["result"], "fixture should report one dataset"
    assert "empty" not in result
    assert "hint" not in result
