#!/usr/bin/env python3
"""
app.py — Local web UI for the checkers AI pipeline.

Replaces the old terminal runner (run_simplified_trace.py) with a small
Flask API + single-page frontend, so the board and the AI's reasoning are
shown in a browser instead of printed to the terminal.

Run via:  python main.py
Then open the URL it prints (default http://127.0.0.1:5050).
"""
from __future__ import annotations

from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

from webapp.game_service import GameSession

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
session = GameSession()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/state")
def api_state():
    return jsonify(session.board_view())


@app.post("/api/new_game")
def api_new_game():
    session.reset()
    return jsonify(session.board_view())


@app.get("/api/legal_moves")
def api_legal_moves():
    return jsonify(session.legal_moves_view())


@app.post("/api/red_move")
def api_red_move():
    trace = session.play_red_ply()
    return jsonify({"trace": trace, "state": session.board_view()})


@app.post("/api/black_move")
def api_black_move():
    data = request.get_json(force=True) or {}
    index = int(data.get("index", -1))
    result = session.play_black_move(index)
    return jsonify({"result": result, "state": session.board_view()})


def run(host: str = "127.0.0.1", port: int = 5050, debug: bool = False) -> None:
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run(debug=True)
