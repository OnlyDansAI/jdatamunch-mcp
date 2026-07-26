"""list_datasets tool: List all indexed datasets."""

import time
from typing import Optional

from ..config import get_index_path
from ..storage.data_store import DataStore
from ..storage.token_tracker import get_total_saved


def list_datasets(storage_path: Optional[str] = None) -> dict:
    """Return summary info for all indexed datasets."""
    t0 = time.time()
    store = DataStore(base_path=storage_path or str(get_index_path()))
    datasets = store.list_datasets()
    total_saved = get_total_saved(str(store.base_path))

    result = {
        "result": datasets,
        "_meta": {
            "timing_ms": round((time.time() - t0) * 1000, 1),
            "tokens_saved": 0,
            "total_tokens_saved": total_saved,
        },
    }

    # Empty-store nudge (suite parity; jcodemunch-mcp#375 correspondence,
    # suggestion C). A user ran the suite for months with this server holding
    # ZERO datasets and only found out by going looking: `list_datasets`
    # returned a bare `[]`, which reads identically to "installed and broken".
    # Their words: "we would have fed both tools months ago."
    #
    # Top-level rather than under `_meta`, because `get_meta_fields()` returns
    # [] by default here — a nudge placed in `_meta` would be stripped before
    # the agent ever saw it. Same key names as the sibling servers.
    if not datasets:
        result["empty"] = True
        result["hint"] = (
            "No datasets are indexed yet, so every search will come back empty "
            "regardless of the query. Index one with index_local(path=...)."
        )
    return result
