#!/usr/bin/env python3
"""Quick script: create all tables without loading data."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.schema import create_all_tables

if __name__ == "__main__":
    create_all_tables()
