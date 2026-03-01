"""Tests for src.clients.binance_client data utility functions."""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import pytest

from src.clients.binance_client import BinanceClient


# ---------------------------------------------------------------------------
# _datetime_to_ms
# ---------------------------------------------------------------------------

class TestDatetimeToMs:
    def test_epoch_start(self):
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        assert BinanceClient._datetime_to_ms(dt) == 0

    def test_known_timestamp(self):
        # 2024-01-01 00:00:00 UTC => 1704067200000 ms
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert BinanceClient._datetime_to_ms(dt) == 1704067200000

    def test_naive_datetime_treated_as_utc(self):
        dt_naive = datetime(2024, 1, 1)
        dt_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert BinanceClient._datetime_to_ms(dt_naive) == BinanceClient._datetime_to_ms(dt_utc)

    def test_subsecond_precision(self):
        dt = datetime(2024, 6, 15, 12, 30, 45, 500_000, tzinfo=timezone.utc)
        ms = BinanceClient._datetime_to_ms(dt)
        assert ms == int(dt.timestamp() * 1000)


# ---------------------------------------------------------------------------
# _atomic_write_csv / _load_existing_csv
# ---------------------------------------------------------------------------

class TestAtomicWriteCsv:
    def test_write_and_read_back(self, tmp_dir):
        filepath = str(tmp_dir / "test_rates.csv")
        rows = [
            (1000, "2024-01-01T00:00:00Z", 42000.0),
            (2000, "2024-01-01T00:01:00Z", 42100.5),
            (3000, "2024-01-01T00:02:00Z", 42050.25),
        ]
        BinanceClient._atomic_write_csv(filepath, rows)

        assert os.path.exists(filepath)
        # No .tmp file should remain
        assert not os.path.exists(filepath + ".tmp")

        loaded = BinanceClient._load_existing_csv(filepath)
        assert len(loaded) == 3
        assert 1000 in loaded
        assert loaded[1000] == ("2024-01-01T00:00:00Z", 42000.0)
        assert loaded[2000] == ("2024-01-01T00:01:00Z", 42100.5)

    def test_overwrite_existing_file(self, tmp_dir):
        filepath = str(tmp_dir / "overwrite.csv")
        BinanceClient._atomic_write_csv(filepath, [(100, "iso1", 1.0)])
        BinanceClient._atomic_write_csv(filepath, [(200, "iso2", 2.0)])
        loaded = BinanceClient._load_existing_csv(filepath)
        assert len(loaded) == 1
        assert 200 in loaded

    def test_csv_header_correct(self, tmp_dir):
        filepath = str(tmp_dir / "header_check.csv")
        BinanceClient._atomic_write_csv(filepath, [(1, "iso", 99.9)])
        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["ts_ms", "ts_iso", "close"]


# ---------------------------------------------------------------------------
# _load_existing_csv
# ---------------------------------------------------------------------------

class TestLoadExistingCsv:
    def test_nonexistent_file_returns_empty(self, tmp_dir):
        filepath = str(tmp_dir / "nope.csv")
        assert BinanceClient._load_existing_csv(filepath) == {}

    def test_parse_known_format(self, tmp_dir):
        filepath = str(tmp_dir / "known.csv")
        with open(filepath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts_ms", "ts_iso", "close"])
            w.writerow([1704067200000, "2024-01-01T00:00:00Z", 42000.0])
            w.writerow([1704067260000, "2024-01-01T00:01:00Z", 42100.5])

        loaded = BinanceClient._load_existing_csv(filepath)
        assert len(loaded) == 2
        assert 1704067200000 in loaded
        ts_iso, close = loaded[1704067200000]
        assert ts_iso == "2024-01-01T00:00:00Z"
        assert close == 42000.0

    def test_skip_malformed_rows(self, tmp_dir):
        filepath = str(tmp_dir / "bad.csv")
        with open(filepath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts_ms", "ts_iso", "close"])
            w.writerow([1000, "ok", 50.0])
            w.writerow(["bad_ts", "bad", "bad"])
            w.writerow([2000, "ok2", 60.0])

        loaded = BinanceClient._load_existing_csv(filepath)
        assert len(loaded) == 2
        assert 1000 in loaded
        assert 2000 in loaded

    def test_headerless_csv_still_parsed(self, tmp_dir):
        """If file has no matching header, rows are parsed as data."""
        filepath = str(tmp_dir / "noheader.csv")
        with open(filepath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([5000, "2024-06-01T00:00:00Z", 100.0])
            w.writerow([6000, "2024-06-01T00:01:00Z", 101.0])

        loaded = BinanceClient._load_existing_csv(filepath)
        assert len(loaded) == 2

    def test_empty_file(self, tmp_dir):
        filepath = str(tmp_dir / "empty.csv")
        with open(filepath, "w") as f:
            pass
        loaded = BinanceClient._load_existing_csv(filepath)
        assert loaded == {}
