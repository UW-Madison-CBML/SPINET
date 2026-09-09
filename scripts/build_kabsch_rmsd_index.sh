#!/bin/bash
python -m ruff check . --select F821,E9 || exit 1
python build_kabsch_rmsd_index.py
