"""Shared fixtures: one small synthetic warehouse per test session (built in a temp folder,
so the real data, results and models are never touched)."""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pytest

from couponroi import data, warehouse


@pytest.fixture(scope="session")
def synth_dirs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("synthetic")
    raw = root / "raw"
    data.fetch(raw_dir=raw, synthetic=True, seed=7)
    db = root / "warehouse.duckdb"
    warehouse.build(raw_dir=raw, db_path=db, report_dir=None)
    return {"root": root, "raw": raw, "db": db}


@pytest.fixture()
def con(synth_dirs):
    c = duckdb.connect(str(synth_dirs["db"]), read_only=True)
    yield c
    c.close()


@pytest.fixture()
def writable_copy(synth_dirs, tmp_path):
    """A throwaway copy of the warehouse that a test may modify."""
    dst = tmp_path / "copy.duckdb"
    shutil.copy(synth_dirs["db"], dst)
    c = duckdb.connect(str(dst))
    yield c
    c.close()
