#!/usr/bin/env python3
"""
Entry point — launches a local web UI instead of the terminal game.

Run:
  python main.py

It starts a small Flask server and opens your browser to it. RED (AI) moves
are made with a button click, BLACK (you) moves are picked from a list of
legal moves — the board, piece counts, move history, and the AI's
Scorer/Proposer/Explainer reasoning trace are all shown in the page.

For the old terminal-based trace runners, see:
  python run_simplified_trace.py --help
  python run_simplified_trace_reasoning.py --help
"""
import os
import threading
import webbrowser

from webapp.app import run

# Inside Docker, bind 0.0.0.0 so the port mapping can reach Flask, and skip
# webbrowser.open() since there's no browser in the container.
IN_DOCKER = os.environ.get("DOCKER") == "1"
HOST = "0.0.0.0" if IN_DOCKER else "127.0.0.1"  # nosec
PORT = 5050

if __name__ == "__main__":
    display_host = "127.0.0.1" if IN_DOCKER else HOST
    url = f"http://{display_host}:{PORT}"
    if not IN_DOCKER:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Checkers AI \u2014 web UI running at {url}  (Ctrl+C to stop)")
    run(host=HOST, port=PORT, debug=False)
