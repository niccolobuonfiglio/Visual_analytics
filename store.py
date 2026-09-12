"""
Run persistence.

Results are written to a small JSON file so an analysis survives a server
restart and the interface can restore the last run on page load. Only the
twenty most recent runs are kept; the store is a convenience, not a database.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).parent
STORE = APP_DIR / "runs_store.json"
UPLOAD_DIR = APP_DIR / "user_csvs"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_RUNS = 20


def load_store() -> dict[str, Any]:
    if STORE.exists():
        try:
            return json.loads(STORE.read_text())
        except json.JSONDecodeError:
            return {"runs": []}
    return {"runs": []}


def save_store(data: dict[str, Any]) -> None:
    STORE.write_text(json.dumps(data))


def remember(result: dict[str, Any]) -> None:
    """Insert a result at the front of the history and trim to MAX_RUNS."""
    store = load_store()
    store["runs"] = ([result] + store.get("runs", []))[:MAX_RUNS]
    store["last"] = result
    save_store(store)


def clear_store() -> None:
    save_store({"runs": []})
