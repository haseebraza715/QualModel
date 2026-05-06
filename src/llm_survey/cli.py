"""Console-script entrypoint for the `llm-survey` command.

Re-exports the argparse `main()` from the top-level `main.py` so that an
installed package can call it without sys.path tricks. We add the repo root
to sys.path *only* if `main` is not already importable, which keeps editable
installs and source checkouts working uniformly.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _resolve_main():
    try:
        return importlib.import_module("main")
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        return importlib.import_module("main")


def main() -> None:
    mod = _resolve_main()
    mod.main()


if __name__ == "__main__":
    main()
