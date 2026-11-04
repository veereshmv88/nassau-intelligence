"""
Regression and Integration Test Suite for Nassau Intelligence Recommendation Data Contract.
Covers:
- Recommendation dictionary normalization.
- Database-retrieved recommendations conformity.
- Missing optional and partial fields handling.
- Empty recommendations and edge cases.
- Validated DataFrame construction.
- Executive Overview and Decision Center DataFrame pipelines.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.recommendation_contract import (
    normalize_recommendation,
    normalize_recommendations_list,
    build_recommendations_dataframe,
    DEFAULT_SUMMARY_COLUMNS,
    SUMMARY_COLUMN_RENAME,
    DEFAULT_FULL_COLUMNS
)
from src.optimization_engine import OptimizationEngine
from src.db.repositories import RecommendationRepository


def test_normalize_recommendation_from_engine_dict():
    """Verify that engine-generated recommendations conform to the full data contract."""
    engine = OptimizationEngine()
    recs = engine.generate_all_recommendations()
    assert len(recs) > 0, "Engine should generate recommendations"
    
    sample = recs[0]
    required_keys = [
        "Product Name", "Current Factory", "Recommended Factory",
        "Destination Region", "Recommendation Category", "Expected Lead Time Reduction (%)",
        "Recommendation ID", "Confidence Score", "Optimization Rationale",
        # Lowercase / DB equivalents
        "product_name", "destination_region", "current_factory", "recommended_factory",
        "recommendation_category", "expected_lead_time_reduction_pct"
    ]
    for key in required_keys:
        assert key in sample, f"Missing required key: {key} in recommendation dictionary"


def test_normalize_recommendation_from_database_dict():
    """Verify that database records (with snake_case fields) are correctly normalized with display aliases."""
    db_raw_record = {
        "recommendation_id": "REC-001",
        "created_at": "2026-11-04T12:00:00",
        "product_name": "Wonka Bar - Milk Chocolate",
        "division": "Chocolate",
        "current_factory": "Wicked Choccy's",
        "recommended_factory": "Secret Factory",
        "destination_region": "Atlantic",
        "operational_issue": "Sub-optimal dispatch route",
        "supporting_evidence": "Historical lead time 5.3 days",
        "predicted_lead_time": 4.1,
        "lead_time_delta": 22.6,
        "distance_reduction_km": 420.0,
        "cost_impact_pct": 3.5,
        "evidential_category": "Potential opportunity",
        "confidence_score": 92.0,
        "priority_rank": 1,
        "capacity_feasible": 1,
        "recommended_action": "Reallocate to Secret Factory",
        "business_explanation": "Reduces regional haulage mileage"
    }

    normalized = normalize_recommendation(db_raw_record, index=1)
    
    # Must have both lower snake_case and UI Title Case keys
    assert normalized["Destination Region"] == "Atlantic"
    assert normalized["destination_region"] == "Atlantic"
    assert normalized["Recommendation Category"] == "Potential opportunity"
    assert normalized["evidential_category"] == "Potential opportunity"
    assert normalized["Expected Lead Time Reduction (%)"] == 22.6
    assert normalized["Product Name"] == "Wonka Bar - Milk Chocolate"
    assert normalized["Current Factory"] == "Wicked Choccy's"
    assert normalized["Recommended Factory"] == "Secret Factory"
    assert normalized["Capacity Feasible"] == "Yes"


def test_empty_recommendation_list_dataframe_construction():
    """Verify that building a DataFrame from an empty list never raises KeyError."""
    empty_list = []
    
    df_summary = build_recommendations_dataframe(
        empty_list,
        columns=DEFAULT_SUMMARY_COLUMNS,
        rename_map=SUMMARY_COLUMN_RENAME
    )
    assert isinstance(df_summary, pd.DataFrame)
    assert len(df_summary) == 0
    expected_cols = ["Product SKU", "Current Plant", "Candidate Plant", "Market", "Category", "Est. Delta (%)"]
    assert list(df_summary.columns) == expected_cols

    df_full = build_recommendations_dataframe(empty_list, columns=DEFAULT_FULL_COLUMNS)
    assert isinstance(df_full, pd.DataFrame)
    assert len(df_full) == 0
    assert list(df_full.columns) == DEFAULT_FULL_COLUMNS


def test_missing_optional_fields_dataframe_construction():
    """Verify that recommendations with missing fields populate clean fallbacks without crashing."""
    incomplete_recs = [
        {"Product Name": "Wonka Bar - Nutty Crunch Surprise"},
        {"product_name": "Fun Dip", "current_factory": "Sugar Shack"},
        {"Product SKU": "Nerds", "Category": "Insufficient evidence"}
    ]

    df = build_recommendations_dataframe(
        incomplete_recs,
        columns=DEFAULT_SUMMARY_COLUMNS,
        rename_map=SUMMARY_COLUMN_RENAME
    )
    assert len(df) == 3
    assert list(df["Product SKU"]) == ["Wonka Bar - Nutty Crunch Surprise", "Fun Dip", "Nerds"]
    assert list(df["Category"]) == ["Needs validation", "Needs validation", "Insufficient evidence"]
    # Check that no NaNs or unhandled crashes occurred
    assert not df["Product SKU"].isna().any()


def test_executive_overview_pipeline_line_429():
    """Simulate Executive Overview line 429 DataFrame pipeline."""
    engine = OptimizationEngine()
    all_recs = engine.generate_all_recommendations()

    recs_df = build_recommendations_dataframe(
        all_recs[:6] if all_recs else [],
        columns=DEFAULT_SUMMARY_COLUMNS,
        rename_map=SUMMARY_COLUMN_RENAME
    )

    expected_cols = ["Product SKU", "Current Plant", "Candidate Plant", "Market", "Category", "Est. Delta (%)"]
    assert list(recs_df.columns) == expected_cols
    assert len(recs_df) <= 6
    if len(all_recs) > 0:
        assert len(recs_df) > 0


def test_decision_center_filtering_pipeline():
    """Simulate Decision Center category filtering and DataFrame pipeline."""
    engine = OptimizationEngine()
    all_recs = engine.generate_all_recommendations()

    categories = [
        "All Categories",
        "Potential opportunity",
        "Needs validation",
        "Insufficient evidence",
        "Operationally infeasible"
    ]

    cols_to_show = [
        "Recommendation ID", "Product Name", "Current Factory", "Recommended Factory", 
        "Destination Region", "Recommendation Category", "Confidence Score", "Capacity Feasible", "Optimization Rationale"
    ]

    for cat in categories:
        filtered = all_recs if cat == "All Categories" else [
            r for r in all_recs 
            if r.get("Recommendation Category") == cat or r.get("Category") == cat or r.get("evidential_category") == cat
        ]
        df = build_recommendations_dataframe(filtered, columns=cols_to_show)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == cols_to_show
        # If category is empty, it should still have the expected columns and 0 rows
        if not filtered:
            assert len(df) == 0


def test_recommendation_repository_live_retrieval():
    """Verify live database-retrieved recommendations satisfy the data contract."""
    db_recs = RecommendationRepository.get_all_recommendations()
    assert len(db_recs) > 0, "Database should contain seeded/synced recommendations"
    
    first = db_recs[0]
    assert "Destination Region" in first
    assert "Recommendation Category" in first
    assert "Expected Lead Time Reduction (%)" in first
    assert "Product Name" in first
    assert "Current Factory" in first
    assert "Recommended Factory" in first
    
    # Test building DataFrame directly from DB-retrieved recommendations
    df = build_recommendations_dataframe(
        db_recs,
        columns=DEFAULT_SUMMARY_COLUMNS,
        rename_map=SUMMARY_COLUMN_RENAME
    )
    assert len(df) == len(db_recs)
    assert "Market" in df.columns
    assert "Category" in df.columns
