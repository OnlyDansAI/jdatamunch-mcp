"""v1.23.0 — tool-surface schema receipt in get_session_stats (jcm v1.108.153 parity).

`_tool_surface_stats()` reports the schema token weight of the visible tool
surface (after tool_profile + disabled_tools filtering) vs the full catalog,
at the meter's bytes/4 scale. Attached inside get_session_stats' `result` as
an advisory `tool_surface` block; a helper failure omits the block, never
fails the call. jData has no Counter surface, so the block carries `profile`
but no `surface` key.
"""

import json

import pytest

from jdatamunch_mcp import server


class TestToolSurfaceStats:
    def test_shape_and_invariants(self, monkeypatch):
        monkeypatch.delenv("JDATAMUNCH_TOOL_PROFILE", raising=False)
        monkeypatch.delenv("JDATAMUNCH_DISABLED_TOOLS", raising=False)
        stats = server._tool_surface_stats()
        assert stats["visible_tools"] > 0
        assert stats["catalog_tools"] >= stats["visible_tools"]
        assert stats["schema_tokens_visible"] > 0
        assert stats["schema_tokens_catalog"] >= stats["schema_tokens_visible"]
        assert stats["schema_tokens_avoided"] == (
            stats["schema_tokens_catalog"] - stats["schema_tokens_visible"]
        )
        assert stats["estimator"] == "bytes/4"
        assert stats["profile"] == "full"
        assert "surface" not in stats

    def test_heaviest_tools_capped_and_sorted(self):
        stats = server._tool_surface_stats(top_n=5)
        heaviest = stats["heaviest_tools"]
        assert 0 < len(heaviest) <= 5
        weights = list(heaviest.values())
        assert weights == sorted(weights, reverse=True)
        assert all(isinstance(w, int) and w > 0 for w in weights)

    def test_disabled_tools_reduce_visible(self, monkeypatch):
        monkeypatch.delenv("JDATAMUNCH_TOOL_PROFILE", raising=False)
        monkeypatch.delenv("JDATAMUNCH_DISABLED_TOOLS", raising=False)
        baseline = server._tool_surface_stats()
        victim = next(iter(baseline["heaviest_tools"]))
        monkeypatch.setenv("JDATAMUNCH_DISABLED_TOOLS", victim)
        stats = server._tool_surface_stats()
        assert stats["visible_tools"] == baseline["visible_tools"] - 1
        assert victim not in stats["heaviest_tools"]


class TestServerWiring:
    @pytest.mark.asyncio
    async def test_get_session_stats_carries_tool_surface(self):
        res = await server.call_tool("get_session_stats", {})
        body = json.loads(res[0].text)
        ts = body["result"]["tool_surface"]
        assert ts["visible_tools"] > 0
        assert ts["schema_tokens_visible"] > 0
        assert ts["estimator"] == "bytes/4"

    @pytest.mark.asyncio
    async def test_helper_failure_never_breaks_stats(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("surface probe failed")

        monkeypatch.setattr(server, "_tool_surface_stats", _boom)
        res = await server.call_tool("get_session_stats", {})
        body = json.loads(res[0].text)
        assert "tool_surface" not in body["result"]
        assert "total_tokens_saved" in body["result"]
