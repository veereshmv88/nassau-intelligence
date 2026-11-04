"""
Tests for domain models data contract.
Verifies strongly typed dataclasses and dual-contract key access.
"""

import pytest
from src.domain_models import (
    OrderRecord, ProductRecord, FactoryRecord, PredictionResult,
    FactoryEvaluation, OptimizationScenario, Recommendation, ModelEvaluation
)

def test_order_record_serialization():
    order = OrderRecord(
        order_row_id=1,
        order_id="ORD-101",
        order_date="2024-01-15",
        ship_date="2024-01-20",
        customer_id=12,
        city="Seattle",
        state_province="WA",
        region="Pacific",
        postal_code="98101",
        division="Chocolate",
        product_id=5,
        product_name="Wonka Bar - Nutty Crunch Surprise",
        carrier_id=1,
        ship_mode="Standard Class",
        factory_id=3,
        factory_name="Lot's O' Nuts",
        units=5,
        sales=125.0,
        gross_profit=45.0,
        cost=80.0,
        lead_time_days=5.0
    )
    d = order.to_dict()
    assert d["order_id"] == "ORD-101"
    assert d["lead_time_days"] == 5.0
    assert d["factory_name"] == "Lot's O' Nuts"

def test_factory_evaluation_contract():
    fe = FactoryEvaluation(
        factory="Lot's O' Nuts",
        current_factory="Secret Factory",
        is_current=False,
        is_recommended=True,
        predicted_lead_time=5.2,
        interval_lower=4.5,
        interval_upper=6.0,
        interval_str="5.2d [4.5 – 6.0d 90% PI]",
        lead_time_delta=2.0,
        lead_time_reduction_pct=27.8,
        distance_km=970,
        distance_delta_km=-1221,
        estimated_profit_per_unit=14.5,
        profit_impact_pct=8.5,
        speed_score=38.5,
        profit_score=45.0,
        evidence_factor=0.92,
        composite_score=38.4,
        workload_pct=68,
        capacity_feasible=True,
        capacity_status="✓ Feasible",
        tooling_eligible=True,
        tooling_note="Eligible",
        recommendation_category="High-Impact Shift",
        evidence_quality="High",
        confidence_score=0.92,
        risk_level="Low",
        risk_flag="Operational clearance confirmed",
        rationale="Superior freight distance and verified tooling.",
        selection_justification="Champion selected based on composite score."
    )
    d = fe.to_dict()
    assert d["factory"] == "Lot's O' Nuts"
    assert d["is_recommended"] is True
    assert d["composite_score"] == 38.4
    assert d["capacity_feasible"] is True

def test_recommendation_contract():
    rec = Recommendation(
        recommendation_id="REC-001",
        product_id=5,
        product_name="Wonka Bar - Nutty Crunch Surprise",
        division="Chocolate",
        destination_region="Pacific",
        current_factory="Secret Factory",
        recommended_factory="Lot's O' Nuts",
        recommendation_category="High-Impact Shift",
        predicted_current_lead_time=7.2,
        predicted_alternative_lead_time=5.2,
        expected_lead_time_reduction=2.0,
        expected_lead_time_reduction_pct=27.8,
        distance_km=970,
        distance_delta_km=-1221,
        workload_pct=68,
        capacity_feasible=True,
        evidence_level="High",
        operational_risk="Low",
        priority_score=92.0,
        explanation="Cuts freight distance by 1,221 km with high empirical sample."
    )
    d = rec.to_display_dict()
    assert d["Product Name"] == "Wonka Bar - Nutty Crunch Surprise"
    assert d["Destination Region"] == "Pacific"
    assert d["Recommended Factory"] == "Lot's O' Nuts"
    assert d["Expected Lead Time Reduction (%)"] == 27.8
    assert d["expected_lead_time_reduction_pct"] == 27.8
    assert d["Candidate Plant"] == "Lot's O' Nuts"

def test_model_evaluation_contract():
    me = ModelEvaluation(
        model_name="RandomForestRegressor",
        parameters_count=100,
        holdout_mae=0.82,
        holdout_rmse=1.05,
        holdout_r2=0.89,
        cv_rmse_mean=1.04,
        cv_rmse_std=0.03,
        status_notes="Primary champion model"
    )
    d = me.to_dict()
    assert d["model_name"] == "RandomForestRegressor"
    assert d["holdout_mae"] == 0.82
    assert d["holdout_rmse"] == 1.05
    assert d["holdout_r2"] == 0.89
