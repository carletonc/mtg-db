#!/usr/bin/env python3
"""Quick script: run the embedding pipeline (downstream of foundation ETL)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etl.embed import run

if __name__ == "__main__":
    run()
