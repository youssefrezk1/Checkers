#!/usr/bin/env python3
"""
Entry point — runs an interactive game: AI (RED) vs human (BLACK).

For more options see:
  python run_simplified_trace.py --help
  python run_simplified_trace_reasoning.py --help
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "run_simplified_trace.py"] + sys.argv[1:])
