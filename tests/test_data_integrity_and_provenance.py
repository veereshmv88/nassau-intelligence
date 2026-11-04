"""
Regression test suite for Nassau Intelligence data integrity, calendar offset calibration,
and enterprise data provenance ledgers.
"""

import pytest
import pandas as pd
import sqlite3
from src.data_loader import (
    load_raw_data,
    prepare_dataset,
    audit_data_quality,
    get_provenance_ledger,
    CALENDAR_SEASONAL_OFFSET_DAYS
)
from src.db.connection import get_db_connection


def test_calendar_offset_constant():
    """Verify seasonal offset constant is mathematically pinned to 174 days."""
    assert CALENDAR_SEASONAL_OFFSET_DAYS == 174


def test_source_raw_dataset_integrity():
    """Verify raw dataset loading, record count, and column coverage."""
    raw_df = load_raw_data()
    assert len(raw_df) == 10194
    assert len(raw_df.columns) == 18
    assert "Order Date" in raw_df.columns
    assert "Ship Date" in raw_df.columns
    assert "Product Name" in raw_df.columns
    assert "Units" in raw_df.columns


def test_mathematical_lead_time_calibration():
    """
    Verify the calendar modulo-365 and 174-day seasonal offset transformation
    recovers commercial shipping dispatch intervals without negative values.
    """
    raw_df = load_raw_data()
    clean_df = prepare_dataset(raw_df, save_processed=False)

    # Lead time must be >= 0 and max <= 15 days
    assert (clean_df["lead_time_days"] >= 0).all()
    assert clean_df["lead_time_days"].max() <= 15

    # Check mode-specific SLA distributions
    same_day = clean_df[clean_df["Ship Mode"] == "Same Day"]["lead_time_days"]
    first_class = clean_df[clean_df["Ship Mode"] == "First Class"]["lead_time_days"]
    second_class = clean_df[clean_df["Ship Mode"] == "Second Class"]["lead_time_days"]
    std_class = clean_df[clean_df["Ship Mode"] == "Standard Class"]["lead_time_days"]

    assert same_day.mean() < 1.0
    assert 2.0 <= first_class.mean() <= 3.0
    assert 3.0 <= second_class.mean() <= 4.0
    assert 4.5 <= std_class.mean() <= 6.0


def test_order_date_range_validation():
    """Verify audited Order Date range is Jan 2024 to Dec 2025."""
    raw_df = load_raw_data()
    audit = audit_data_quality(raw_df)

    assert audit["order_date_range"][0].startswith("2024-01")
    assert audit["order_date_range"][1].startswith("2025-12")
    assert audit["order_date_display"] == "Jan 2024 – Dec 2025"
    assert audit["invalid_order_dates"] == 0
    assert audit["invalid_ship_dates"] == 0


def test_provenance_ledger_structure():
    """Verify data provenance ledger records all expected fields and valid evidence tiers."""
    ledger = get_provenance_ledger()
    assert len(ledger) >= 12

    valid_tiers = {
        "Historical Observation",
        "Configured Master Assumption",
        "Derived Calculation",
        "Model Prediction",
        "Scenario Simulation"
    }

    for item in ledger:
        assert "Field" in item
        assert "Source" in item
        assert "Evidence Tier" in item
        assert "Method" in item
        assert item["Evidence Tier"] in valid_tiers

    # Check factory evidence tier specifically
    factory_entry = next((i for i in ledger if i["Field"].startswith("Factory")), None)
    assert factory_entry is not None
    assert factory_entry["Evidence Tier"] == "Configured Master Assumption"


def test_sqlite_database_has_provenance_columns():
    """Verify SQLite database orders table stores raw_lead_time_days and provenance_tier."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(orders);")
        columns = [row[1] for row in cursor.fetchall()]

        assert "raw_lead_time_days" in columns
        assert "provenance_tier" in columns
        assert "lead_time_days" in columns

        cursor.execute("SELECT COUNT(*), AVG(raw_lead_time_days), AVG(lead_time_days) FROM orders;")
        row = cursor.fetchone()
        assert row[0] == 10194
        assert row[1] > 900.0  # Raw multi-year calendar interval
        assert 3.5 <= row[2] <= 5.0  # Calibrated dispatch lead time
