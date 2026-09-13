"""
Flask backend for the projection-fidelity toolkit.

Routing only. The analysis itself lives in ``pipeline.py`` and persistence in
``store.py``, so this file stays short enough to read in one go.

The interface builds every figure in the browser from raw arrays, so the run
endpoint returns coordinates and scores rather than rendered images: brushing,
recolouring and filtering then need no server round trip.
"""

from __future__ import annotations

import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from toolkit.datasets import DATASET_NAMES
from toolkit.pipeline import run_pipeline
from toolkit.store import UPLOAD_DIR, load_store, remember, clear_store

APP_DIR = Path(__file__).parent
app = Flask(__name__, static_folder=str(APP_DIR / "static"), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/datasets")
def datasets():
    return jsonify({"datasets": DATASET_NAMES})


@app.route("/api/run", methods=["POST"])
def run():
    params = request.get_json(force=True)
    try:
        result = run_pipeline(params)
    except Exception as exc:
        # Surface the failure in the interface's console panel rather than as an
        # opaque 500, so a bad run is diagnosable from the browser.
        return jsonify({
            "ok": False,
            "error": str(exc),
            "console": f"[error] {exc}\n\n{traceback.format_exc()}",
        }), 200

    remember(result)
    return jsonify(result)


@app.route("/api/last")
def last():
    return jsonify(load_store().get("last") or {"ok": False, "empty": True})


@app.route("/api/history")
def history():
    runs = load_store().get("runs", [])
    return jsonify({"runs": [{
        "timestamp": r.get("timestamp"),
        "name": r.get("info", {}).get("name"),
        "metric": r.get("metric"),
        "params": r.get("params"),
        "summary": r.get("summary"),
    } for r in runs]})


@app.route("/api/clear", methods=["POST"])
def clear():
    clear_store()
    return jsonify({"ok": True})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "no file"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"ok": False, "error": "must be a .csv"}), 400
    dest = UPLOAD_DIR / f.filename
    f.save(dest)
    import pandas as pd
    return jsonify({"ok": True, "path": str(dest),
                    "columns": list(pd.read_csv(dest, nrows=1).columns)})


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
