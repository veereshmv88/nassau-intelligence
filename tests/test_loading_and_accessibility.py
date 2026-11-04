"""
Tests for form accessibility, ML loading state, and session management.
"""

import pytest
from pathlib import Path
from src.prediction_service import LeadTimePredictor
from src.optimization_engine import OptimizationEngine

BASE_DIR = Path(__file__).resolve().parent.parent


def test_global_search_input_accessibility():
    app_text = (BASE_DIR / "src" / "app.py").read_text(encoding="utf-8")
    assert 'id="global-search-input"' in app_text
    assert 'name="global_search"' in app_text
    assert 'autocomplete="off"' in app_text
    assert '<label for="global-search-input"' in app_text
    assert 'aria-label=' in app_text


def test_sidebar_and_tab_radios_have_keys_and_labels():
    app_text = (BASE_DIR / "src" / "app.py").read_text(encoding="utf-8")
    assert 'key="sidebar_nav_selection"' in app_text
    assert 'key="opt_workspace_view_radio"' in app_text


def test_all_optimization_inputs_have_keys_and_labels():
    app_text = (BASE_DIR / "src" / "app.py").read_text(encoding="utf-8")
    assert 'key="opt_product_sku_select"' in app_text
    assert 'key="opt_destination_market_select"' in app_text
    assert 'key="opt_shipping_mode_select"' in app_text
    assert 'key="opt_batch_units_input"' in app_text
    assert 'key="opt_priority_slider_input"' in app_text
    assert 'key="btn_run_optimization"' in app_text


def test_live_ml_inference_console_present_with_status_stages():
    app_text = (BASE_DIR / "src" / "app.py").read_text(encoding="utf-8")
    assert 'key="ml_inf_product_sku"' in app_text
    assert 'key="ml_inf_factory_origin"' in app_text
    assert 'key="ml_inf_destination_market"' in app_text
    assert 'key="ml_inf_order_units"' in app_text
    assert 'key="btn_execute_ml_inference"' in app_text
    assert 'st.status("Analyzing operational data' in app_text


def test_css_visually_hidden_and_focus_styles():
    css_text = (BASE_DIR / "styles" / "main.css").read_text(encoding="utf-8")
    assert ".visually-hidden" in css_text
    assert "outline: 2px solid" in css_text
    assert 'div[data-testid="stStatusWidget"]' in css_text


def test_prediction_service_inference_success():
    predictor = LeadTimePredictor.get_instance()
    res = predictor.predict_with_interval(
        product_name="Wonka Bar - Nutty Crunch Surprise",
        factory_name="Lot's O' Nuts",
        destination_region="Pacific",
        ship_mode="Standard Class",
        units=2,
        confidence_level=0.90
    )
    assert "point_prediction" in res
    assert "interval_lower" in res
    assert "interval_upper" in res
    assert res["point_prediction"] > 0
    assert res["interval_lower"] <= res["point_prediction"] <= res["interval_upper"]


def test_prediction_service_error_handling():
    predictor = LeadTimePredictor.get_instance()
    # Test handling of unexpected/unknown product
    res = predictor.predict_with_interval(
        product_name="Nonexistent Experimental Candy",
        factory_name="Lot's O' Nuts",
        destination_region="Pacific",
        ship_mode="Standard Class",
        units=2
    )
    # Even on fallback, predictor safely predicts without crashing
    assert "point_prediction" in res
    assert res["point_prediction"] > 0


def test_optimization_engine_execution_and_caching_consistency():
    engine = OptimizationEngine()
    sim1 = engine.simulate_optimization(
        product_name="Wonka Bar - Nutty Crunch Surprise",
        destination_region="Pacific",
        ship_mode="Standard Class",
        priority_slider=70.0,
        units=2
    )
    sim2 = engine.simulate_optimization(
        product_name="Wonka Bar - Nutty Crunch Surprise",
        destination_region="Pacific",
        ship_mode="Standard Class",
        priority_slider=70.0,
        units=2
    )
    assert sim1["factories"][0]["factory"] == sim2["factories"][0]["factory"]
    assert sim1["factories"][0]["predicted_lead_time"] == sim2["factories"][0]["predicted_lead_time"]
    assert sim1["factories"][0]["composite_score"] == sim2["factories"][0]["composite_score"]


def test_optimization_engine_handles_edge_units():
    engine = OptimizationEngine()
    # High batch units
    sim_large = engine.simulate_optimization(
        product_name="Wonka Bar - Nutty Crunch Surprise",
        destination_region="Pacific",
        ship_mode="Standard Class",
        priority_slider=70.0,
        units=50
    )
    assert "factories" in sim_large
    assert len(sim_large["factories"]) == 5

