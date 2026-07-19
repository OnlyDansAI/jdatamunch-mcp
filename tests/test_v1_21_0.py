"""v1.21.0 — advisory session token budget (suite parity with jcm v1.108.146).

JDATAMUNCH_SESSION_TOKEN_BUDGET sets an advisory ceiling over response tokens
served; responses carry _meta.budget at approaching (>=80%) / over (>=100%),
attached AFTER meta_fields filtering so the token-efficient default (_meta
stripped) can't delete the warning. get_session_stats reports the block when
configured. Never blocks or truncates.
"""

import json

import pytest

from jdatamunch_mcp.storage import token_tracker


@pytest.fixture(autouse=True)
def _reset_counter():
    token_tracker.reset_session_response_tokens()
    yield
    token_tracker.reset_session_response_tokens()


class TestBudgetStatus:
    def test_unset_env_disables(self, monkeypatch):
        monkeypatch.delenv("JDATAMUNCH_SESSION_TOKEN_BUDGET", raising=False)
        assert token_tracker.budget_status() is None

    def test_garbage_and_zero_disable(self, monkeypatch):
        monkeypatch.setenv("JDATAMUNCH_SESSION_TOKEN_BUDGET", "banana")
        assert token_tracker.budget_status() is None
        monkeypatch.setenv("JDATAMUNCH_SESSION_TOKEN_BUDGET", "0")
        assert token_tracker.budget_status() is None

    def test_state_edges(self, monkeypatch):
        monkeypatch.setenv("JDATAMUNCH_SESSION_TOKEN_BUDGET", "1000")
        token_tracker.record_response_text("x" * (799 * 4))
        assert token_tracker.budget_status()["state"] == "ok"
        token_tracker.record_response_text("x" * 4)  # 800 = 80%
        assert token_tracker.budget_status()["state"] == "approaching"
        token_tracker.record_response_text("x" * (200 * 4))  # 1000
        assert token_tracker.budget_status() == {
            "limit": 1000, "spent": 1000, "state": "over",
        }

    def test_record_returns_cumulative(self):
        assert token_tracker.record_response_text("x" * 400) == 100
        assert token_tracker.record_response_text("x" * 400) == 200
        assert token_tracker.get_session_response_tokens() == 200


class TestSessionStats:
    def test_budget_absent_when_unconfigured(self, monkeypatch, tmp_path):
        monkeypatch.delenv("JDATAMUNCH_SESSION_TOKEN_BUDGET", raising=False)
        from jdatamunch_mcp.tools.get_session_stats import get_session_stats
        out = get_session_stats(storage_path=str(tmp_path))
        assert "budget" not in out["result"]
        assert out["result"]["session_response_tokens"] == 0

    def test_budget_block_when_configured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("JDATAMUNCH_SESSION_TOKEN_BUDGET", "500")
        from jdatamunch_mcp.tools.get_session_stats import get_session_stats
        out = get_session_stats(storage_path=str(tmp_path))
        assert out["result"]["budget"] == {"limit": 500, "spent": 0, "state": "ok"}


class TestServerWiring:
    @pytest.mark.asyncio
    async def test_meta_budget_survives_default_meta_strip(self, monkeypatch, tmp_path):
        monkeypatch.setenv("JDATAMUNCH_SESSION_TOKEN_BUDGET", "100")
        monkeypatch.delenv("JDATAMUNCH_META_FIELDS", raising=False)  # default: strip _meta
        monkeypatch.setenv("DATA_INDEX_PATH", str(tmp_path))
        token_tracker.record_response_text("x" * 400)  # 100 tokens = over
        from jdatamunch_mcp.server import call_tool
        res = await call_tool("get_session_stats", {})
        body = json.loads(res[0].text)
        assert body["_meta"]["budget"]["state"] == "over"

    @pytest.mark.asyncio
    async def test_no_meta_budget_below_threshold(self, monkeypatch, tmp_path):
        monkeypatch.setenv("JDATAMUNCH_SESSION_TOKEN_BUDGET", "1000000")
        monkeypatch.setenv("DATA_INDEX_PATH", str(tmp_path))
        from jdatamunch_mcp.server import call_tool
        res = await call_tool("get_session_stats", {})
        body = json.loads(res[0].text)
        assert "budget" not in body.get("_meta", {})

    @pytest.mark.asyncio
    async def test_responses_accumulate(self, monkeypatch, tmp_path):
        monkeypatch.delenv("JDATAMUNCH_SESSION_TOKEN_BUDGET", raising=False)
        monkeypatch.setenv("DATA_INDEX_PATH", str(tmp_path))
        from jdatamunch_mcp.server import call_tool
        assert token_tracker.get_session_response_tokens() == 0
        await call_tool("get_session_stats", {})
        assert token_tracker.get_session_response_tokens() > 0
