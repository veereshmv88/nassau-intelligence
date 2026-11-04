"""
Comprehensive tests for Nassau Intelligence Database Architecture, Repositories,
Plant Capacity Constraints, Tooling Eligibility, and Calibrated Prediction Intervals.
"""

import pytest
import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import BASE_DIR
from src.db.connection import get_db_connection, DB_PATH
from src.db.repositories import (
    OrderRepository, ProductRepository, FactoryRepository,
    RecommendationRepository, ScenarioRepository, ModelEvaluationRepository,
    AuditRepository
)
from src.optimization_engine import OptimizationEngine
from src.prediction_service import LeadTimePredictor


def test_sqlite_database_file_exists_and_wal_mode():
    """Verify database file exists and PRAGMA journal_mode is WAL."""
    assert DB_PATH.exists(), f"Database not found at {DB_PATH}"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.upper() == "WAL", f"Expected WAL mode, got {mode}"


def test_foreign_keys_pragma_enabled():
    """Verify SQLite foreign key enforcement is actively turned ON."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys;")
        fk_status = cursor.fetchone()[0]
        assert fk_status == 1, "Foreign key constraint enforcement is disabled."


def test_schema_table_counts_and_integrity():
    """Verify that all 10 normalized tables exist and have expected data rows."""
    expected_min_counts = {
        "factories": 5,
        "carriers": 4,
        "products": 15,
        "destinations": 40,
        "orders": 10194,
        "factory_eligibility": 70,
        "recommendations": 3,
        "model_evaluations": 4,
        "data_audit_log": 1
    }
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for table, min_count in expected_min_counts.items():
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            assert count >= min_count, f"Table {table} has {count} rows, expected at least {min_count}"


def test_indexes_exist():
    """Verify that database performance indexes are established."""
    expected_indexes = [
        "idx_orders_product",
        "idx_orders_factory",
        "idx_orders_carrier",
        "idx_orders_region",
        "idx_orders_date",
        "idx_recommendations_category"
    ]
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'index';")
        existing_indexes = {row[0] for row in cursor.fetchall()}
        for idx in expected_indexes:
            assert idx in existing_indexes, f"Missing index: {idx}"


def test_order_repository_queries():
    """Verify OrderRepository summary statistics and filtering."""
    stats = OrderRepository.get_summary_stats()
    assert stats["total_orders"] == 10194
    assert stats["total_sales"] > 100_000
    assert 3.5 <= stats["avg_lead_time"] <= 4.5

    # Filtered order query
    filtered = OrderRepository.get_filtered_orders(region="Pacific", limit=10)
    assert len(filtered) == 10
    for o in filtered:
        assert o["region"] == "Pacific"
        assert "lead_time_days" in o


def test_product_repository_queries():
    """Verify ProductRepository returns all 15 SKUs with assigned factory."""
    products = ProductRepository.get_all_products()
    assert len(products) == 15
    sample = products[0]
    assert "product_name" in sample
    assert "division" in sample
    assert "assigned_factory" in sample
    assert "profit_margin_pct" in sample


def test_factory_repository_and_capacity_parameters():
    """Verify FactoryRepository returns 5 plants with rated capacities and workloads."""
    factories = FactoryRepository.get_all_factories()
    assert len(factories) == 5
    for f in factories:
        assert f["rated_daily_capacity"] > 0
        assert 40 <= f["baseline_workload_pct"] <= 95
        assert 75 <= f["max_feasible_workload_pct"] <= 95


def test_factory_eligibility_repository():
    """Verify tooling eligibility checks for SKUs."""
    elig = FactoryRepository.get_factory_eligibility("Wonka Bar - Nutty Crunch Surprise")
    assert len(elig) == 5
    # Nutty crunch should have specific factory eligibility
    eligible_plants = [e["factory_name"] for e in elig if e["is_eligible"]]
    assert len(eligible_plants) >= 1


def test_recommendation_repository_and_scenario_saving():
    """Verify RecommendationRepository and ScenarioRepository persistence."""
    recs = RecommendationRepository.get_all_recommendations()
    assert len(recs) >= 3
    sample_rec = recs[0]
    assert "recommendation_id" in sample_rec
    assert "operational_issue" in sample_rec
    assert "evidential_category" in sample_rec

    # Test saving a scenario
    test_scenario = {
        "scenario_id": "SCEN-TEST-999",
        "created_at": "2026-11-04T12:00:00",
        "product_id": 1,
        "destination_region": "Pacific",
        "ship_mode": "Standard Class",
        "batch_units": 5,
        "priority_slider": 70.0,
        "baseline_factory_id": 1,
        "recommended_factory_id": 2,
        "baseline_lead_time": 4.5,
        "recommended_lead_time": 3.2,
        "lead_time_delta": 1.3,
        "lead_time_reduction_pct": 28.9,
        "distance_delta_km": -450.0,
        "profit_impact_pct": 5.4,
        "recommendation_category": "Potential opportunity",
        "confidence_score": 92.0,
        "user_notes": "Automated unit test scenario"
    }
    ScenarioRepository.save_scenario(test_scenario)
    saved = ScenarioRepository.get_saved_scenarios(limit=10)
    saved_ids = [s["scenario_id"] for s in saved]
    assert "SCEN-TEST-999" in saved_ids


def test_prediction_service_prediction_intervals():
    """Verify calibrated empirical prediction intervals in LeadTimePredictor."""
    predictor = LeadTimePredictor.get_instance()
    res = predictor.predict_with_interval(
        product_name="Wonka Bar - Milk Chocolate",
        factory_name="Wicked Choccy's",
        destination_region="Pacific",
        ship_mode="Standard Class",
        confidence_level=0.90
    )
    assert "point_prediction" in res
    assert "interval_lower" in res
    assert "interval_upper" in res
    assert res["interval_lower"] <= res["point_prediction"] <= res["interval_upper"]
    assert res["confidence_level"] == "90%"


def test_optimization_engine_dynamic_capacity_and_tooling():
    """Verify dynamic plant capacity and tooling constraints in optimization engine."""
    engine = OptimizationEngine()
    sim = engine.simulate_optimization(
        product_name="Wonka Bar - Nutty Crunch Surprise",
        destination_region="Pacific",
        ship_mode="Standard Class",
        priority_slider=70.0,
        units=5
    )
    assert "factories" in sim
    for f in sim["factories"]:
        assert "workload_pct" in f
        assert "capacity_feasible" in f
        assert "capacity_status" in f
        assert "tooling_eligible" in f
        assert "interval_lower" in f
        assert "interval_upper" in f
