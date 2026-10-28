"""
Comprehensive Automated Test Suite for Nassau Intelligence.
Validates data ingestion, lead-time normalization, feature engineering,
model inference, optimization ranking, and route clustering.
"""

import pytest
import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import FACTORIES, DEFAULT_PRODUCT_ASSIGNMENTS, MODEL_PIPELINE_PATH
from src.data_loader import load_raw_data, prepare_dataset, audit_data_quality
from src.preprocessing import (
    haversine_distance, get_factory_destination_distance, 
    add_geographic_features, get_feature_columns
)
from src.prediction_service import LeadTimePredictor
from src.optimization_engine import OptimizationEngine
from src.route_analytics import aggregate_route_performance, perform_route_clustering
from src.reporting import generate_executive_summary, generate_research_report


def test_data_ingestion_and_audit():
    """Verify raw dataset loading, row counts, and data quality audit."""
    df = load_raw_data()
    assert len(df) == 10194, f"Expected 10194 rows, got {len(df)}"
    assert "Product Name" in df.columns
    assert "Ship Mode" in df.columns
    assert "Sales" in df.columns
    
    audit = audit_data_quality(df)
    assert audit["row_count"] == 10194
    assert audit["duplicate_rows"] == 0
    assert audit["clean_lead_time_min"] >= 0.0


def test_lead_time_cleaning_and_preparation():
    """Verify operational lead time calculation and factory mapping."""
    df = load_raw_data()
    df_clean = prepare_dataset(df, save_processed=False)
    
    assert "lead_time_days" in df_clean.columns
    assert df_clean["lead_time_days"].min() >= 0
    assert df_clean["lead_time_days"].max() <= 15
    assert df_clean["lead_time_days"].mean() > 3.0
    assert "Factory" in df_clean.columns
    assert df_clean["Factory"].isnull().sum() == 0


def test_geographic_distance_calculation():
    """Verify Haversine formula and factory distance calculations."""
    # Distance from Savannah GA to Atlanta GA is approx 350-400 km
    dist = haversine_distance(32.076, -81.088, 33.749, -84.388)
    assert 300 < dist < 450
    
    # Check factory destination distance
    dist_factory = get_factory_destination_distance("Lot's O' Nuts", "California")
    assert 400 < dist_factory < 1200
    
    # Fallback to region
    dist_reg = get_factory_destination_distance("Sugar Shack", "Unknown State", "Interior")
    assert dist_reg > 0


def test_trained_model_pipeline_prediction():
    """Verify that persisted pipeline loads and predicts valid lead times."""
    assert MODEL_PIPELINE_PATH.exists(), "Model pipeline joblib must exist"
    predictor = LeadTimePredictor.get_instance()
    
    pred = predictor.predict_single(
        product_name="Wonka Bar - Milk Chocolate",
        factory_name="Wicked Choccy's",
        destination_region="Interior",
        ship_mode="Standard Class",
        units=2
    )
    assert isinstance(pred, float)
    assert 0.5 <= pred <= 10.0


def test_optimization_engine_simulation():
    """Verify optimization engine counterfactual evaluation and ranking."""
    engine = OptimizationEngine()
    sim = engine.simulate_optimization(
        product_name="Wonka Bar - Milk Chocolate",
        destination_region="Interior",
        ship_mode="Standard Class",
        priority_slider=70.0
    )
    
    assert "factories" in sim
    assert len(sim["factories"]) == len(FACTORIES)
    
    # Check that exactly one factory is marked recommended
    recommended_count = sum(1 for f in sim["factories"] if f["is_recommended"])
    assert recommended_count == 1
    
    # Check that composite scores are bounded
    for f in sim["factories"]:
        assert 0 <= f["composite_score"] <= 100
        assert f["predicted_lead_time"] > 0


def test_system_recommendations_generation():
    """Verify generation of system-wide ranked reassignment opportunities."""
    engine = OptimizationEngine()
    recs = engine.generate_all_recommendations(priority_slider=70.0)
    
    assert isinstance(recs, list)
    assert len(recs) > 0
    
    first = recs[0]
    assert "Product Name" in first
    assert "Current Factory" in first
    assert "Recommended Factory" in first
    assert "Composite Score" in first
    # Verify sorted descending
    scores = [r["Composite Score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_route_analytics_and_clustering():
    """Verify route aggregation and KMeans clustering."""
    df = load_raw_data()
    df_clean = prepare_dataset(df, save_processed=False)
    
    routes = aggregate_route_performance(df_clean)
    assert len(routes) > 0
    assert "Efficiency_Status" in routes.columns
    
    clustered = perform_route_clustering(routes, n_clusters=3)
    assert "Cluster" in clustered.columns
    assert "Cluster_Archetype" in clustered.columns


def test_reporting_generators():
    """Verify executive briefing and technical research report generation."""
    metrics_dummy = {
        "champion_model": "Linear Regression",
        "models": {
            "Linear Regression": {"mae": 0.93, "rmse": 1.15, "r2": 0.61}
        },
        "audit_summary": {
            "total_sales": 141783.63,
            "total_profit": 93442.80,
            "avg_lead_time": 4.29
        }
    }
    recs_dummy = [{"Product Name": "Test SKU"}]
    
    exec_summary = generate_executive_summary(metrics_dummy, recs_dummy)
    assert "EXECUTIVE BRIEFING" in exec_summary
    assert "Linear Regression" in exec_summary
    
    tech_report = generate_research_report(metrics_dummy, {"Distance_km": 0.25})
    assert "TECHNICAL METHODOLOGY" in tech_report
