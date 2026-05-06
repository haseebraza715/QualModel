"""Hugging Face Spaces / local entry: `streamlit run app.py` from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
# No-op once the package is installed via `pip install -e .`; required only for
# bare source checkouts (e.g. HF Spaces uploading raw files).
try:  # pragma: no cover
    import llm_survey  # noqa: F401
except ModuleNotFoundError:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from ui.dashboard import main

if __name__ == "__main__":
    main()
