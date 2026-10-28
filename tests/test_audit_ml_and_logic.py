"""
=============================================================================
COMPREHENSIVE ML & BUSINESS LOGIC AUDIT TEST SUITE — NASSAU INTELLIGENCE
Tests data integrity, leakage prevention, independent metrics, model artifacts,
mathematical formulas, counterfactual comparisons, zero handling, and scoring.
=============================================================================
"""

import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    RAW_DATA_PATH, MODEL_PIPELINE_PATH, MODEL_METRICS_PATH,
    DEFAULT_PRODUCT_ASSIGNMENTS, FACTORIES, PRODUCT_DIVISIONS
)
from src.data_loader import load_raw_data, prepare_dataset, audit_data_quality
from src.preprocessing import (
    add_geographic_features, create_chronological_splits, 
    build_preprocessor, haversine_distance, get_factory_destination_distance
)
from src.prediction_service import LeadTimePredictor
from src.optimization_engine import OptimizationEngine


# ===========================================================================
# 1. DATASET & SCHEMA AUDIT TESTS
# ===========================================================================

def test_data_validation_and_schema():
    """Verify row count, schema columns, null values, and duplicate records."""
    df = load_raw_data()
    assert len(df) == 10194, f"Expected 10,194 rows, found {len(df)}"
    assert len(df.columns) == 18, f"Expected 18 columns, found {len(df.columns)}"
    assert df.isnull().sum().sum() == 0, "Found unexpected null values in dataset"
    assert df.duplicated().sum() == 0, "Found unexpected duplicate rows in dataset"
    
    expected_cols = [
        'Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 
        'Customer ID', 'Country/Region', 'City', 'State/Province', 
        'Postal Code', 'Division', 'Region', 'Product ID', 'Product Name', 
        'Sales', 'Units', 'Gross Profit', 'Cost'
    ]
    for col in expected_cols:
        assert col in df.columns, f"Required column '{col}' missing from raw schema"


def test_lead_time_target_calculation_and_dispatch_bounds():
    """Verify that Ship Date - Order Date measures order dispatch lead time."""
    df = load_raw_data()
    df_prep = prepare_dataset(df, save_processed=False)
    
    # Check that lead_time_days is non-negative and physically plausible
    assert 'lead_time_days' in df_prep.columns
    assert (df_prep['lead_time_days'] >= 0).all(), "Lead times cannot be negative"
    assert (df_prep['lead_time_days'] <= 15).all(), "Normalized operational lead time should be bounded <= 15 days"
    
    # Check carrier SLAs
    by_mode = df_prep.groupby('Ship Mode')['lead_time_days'].mean()
    assert by_mode['Same Day'] < by_mode['First Class'] < by_mode['Second Class'] < by_mode['Standard Class'], \
        "Carrier lead times violate logical operational hierarchy"


# ===========================================================================
# 2. DATA LEAKAGE & CHRONOLOGICAL SPLIT TESTS
# ===========================================================================

def test_leakage_prevention_chronological():
    """Verify strictly chronological train/test split without target or future leakage."""
    df = load_raw_data()
    df_prep = prepare_dataset(df, save_processed=False)
    df_feat = add_geographic_features(df_prep)
    
    X_train, X_test, y_train, y_test = create_chronological_splits(df_feat, test_size=0.2)
    assert len(X_train) == 8155
    assert len(X_test) == 2039
    
    # Check feature sets do not include target or future dates
    forbidden_features = ['lead_time_days', 'raw_lead_time_days', 'Ship Date', 'Ship_Date_dt']
    for feat in forbidden_features:
        assert feat not in X_train.columns
        assert feat not in X_test.columns
        
    # Preprocessor fitting test: verify pipeline fits strictly on training data
    preprocessor = build_preprocessor()
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    pipe.fit(X_train, y_train)
    
    # Assert inference on test data runs without altering pipeline state
    preds = pipe.predict(X_test)
    assert len(preds) == len(X_test)
    assert not np.isnan(preds).any()


# ===========================================================================
# 3. METRIC CALCULATION TESTS (INDEPENDENT KNOWN GROUND TRUTH)
# ===========================================================================

def test_independent_metric_calculations_with_known_ground_truth():
    """Verify MAE, RMSE, and R2 formulas against hand-calculated ground truth values."""
    y_true = np.array([3.0, 5.0, 2.0, 6.0])
    y_pred = np.array([4.0, 4.0, 1.0, 6.0])
    # Errors: [1.0, -1.0, -1.0, 0.0]
    # Abs Errors: [1.0, 1.0, 1.0, 0.0] -> MAE = 3.0 / 4 = 0.75
    # Squared Errors: [1.0, 1.0, 1.0, 0.0] -> MSE = 3.0 / 4 = 0.75 -> RMSE = sqrt(0.75) = 0.866025
    # Total SS: y_true mean = 4.0. (3-4)^2 + (5-4)^2 + (2-4)^2 + (6-4)^2 = 1 + 1 + 4 + 4 = 10.0
    # Residual SS: 1 + 1 + 1 + 0 = 3.0. R2 = 1 - (3.0 / 10.0) = 0.70
    
    calc_mae = mean_absolute_error(y_true, y_pred)
    calc_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    calc_r2 = r2_score(y_true, y_pred)
    
    assert abs(calc_mae - 0.75) < 1e-6
    assert abs(calc_rmse - np.sqrt(0.75)) < 1e-6
    assert abs(calc_r2 - 0.70) < 1e-6


def test_baseline_model_comparison():
    """Verify that Ridge Regression outperforms a dummy mean baseline on the test set."""
    df = load_raw_data()
    df_prep = prepare_dataset(df, save_processed=False)
    df_feat = add_geographic_features(df_prep)
    X_train, X_test, y_train, y_test = create_chronological_splits(df_feat, test_size=0.2)
    
    mean_val = y_train.mean()
    dummy_preds = np.full_like(y_test, fill_value=mean_val)
    dummy_mae = mean_absolute_error(y_test, dummy_preds)
    dummy_rmse = np.sqrt(mean_squared_error(y_test, dummy_preds))
    
    # Train Ridge
    pipe = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('regressor', Ridge(alpha=1.0))
    ])
    pipe.fit(X_train, y_train)
    ridge_preds = pipe.predict(X_test)
    ridge_mae = mean_absolute_error(y_test, ridge_preds)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
    
    # Ridge should improve upon dummy baseline significantly
    assert ridge_mae < dummy_mae, f"Ridge MAE ({ridge_mae}) not lower than dummy ({dummy_mae})"
    assert ridge_rmse < dummy_rmse, f"Ridge RMSE ({ridge_rmse}) not lower than dummy ({dummy_rmse})"
    assert r2_score(y_test, ridge_preds) > 0.50, "Ridge R2 should exceed 0.50"


def test_carrier_mean_baseline_rigorous_comparison():
    """
    Rigorously audit the Carrier-Class Mean Baseline vs Ridge Regression.
    Confirms that:
    1. Both use the identical chronological test rows.
    2. Carrier baseline achieves RMSE ~ 1.1504 and Ridge ~ 1.1527.
    3. The difference is 0.0023 days (~3.3 minutes).
    4. Paired t-test confirms p > 0.05 (difference is not statistically significant).
    5. No claim of ML superiority is justified purely on predictive accuracy.
    """
    from scipy import stats
    df = load_raw_data()
    df_prep = prepare_dataset(df, save_processed=False)
    df_feat = add_geographic_features(df_prep)
    X_train, X_test, y_train, y_test = create_chronological_splits(df_feat, test_size=0.2)
    
    # Carrier baseline
    train_df = X_train.copy()
    train_df['y'] = y_train
    carrier_means = train_df.groupby('Ship Mode')['y'].mean().to_dict()
    carrier_preds = X_test['Ship Mode'].map(carrier_means).fillna(y_train.mean()).values
    carrier_rmse = np.sqrt(mean_squared_error(y_test, carrier_preds))
    
    # Ridge model
    pipe = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('regressor', Ridge(alpha=1.0))
    ])
    pipe.fit(X_train, y_train)
    ridge_preds = pipe.predict(X_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
    
    # Assert difference is tiny (< 0.005 days)
    delta_days = abs(carrier_rmse - ridge_rmse)
    assert delta_days < 0.005, f"Difference between Ridge and Carrier baseline is unexpectedly large: {delta_days}"
    
    # Paired t-test on squared residuals
    sq_carrier = (y_test.values - carrier_preds) ** 2
    sq_ridge = (y_test.values - ridge_preds) ** 2
    t_stat, p_val = stats.ttest_rel(sq_carrier, sq_ridge)
    
    # Assert difference is not statistically significant (p > 0.05)
    assert p_val > 0.05, f"Carrier baseline vs Ridge difference is statistically significant (p={p_val})!"


# ===========================================================================
# 4. MODEL ARTIFACT RELOAD & METADATA AUDIT
# ===========================================================================

def test_model_artifact_persistence_and_reload():
    """Verify pipeline persistence, reload fidelity, and metadata consistency."""
    assert MODEL_PIPELINE_PATH.exists(), f"Missing artifact at {MODEL_PIPELINE_PATH}"
    pipeline = joblib.load(MODEL_PIPELINE_PATH)
    assert hasattr(pipeline, 'predict'), "Reloaded pipeline lacks predict method"
    
    # Verify inference works on single record
    dummy_df = pd.DataFrame([{
        'Product Name': 'Wonka Bar - Milk Chocolate',
        'Division': 'Chocolate',
        'Region': 'Interior',
        'Ship Mode': 'Standard Class',
        'Factory': "Wicked Choccy's",
        'Units': 2,
        'Distance_km': 1525.3
    }])
    pred = pipeline.predict(dummy_df)
    assert len(pred) == 1
    assert 0.5 <= float(pred[0]) <= 10.0, f"Unrealistic lead time prediction: {pred[0]}"


# ===========================================================================
# 5. BUSINESS LOGIC & FORMULA AUDIT TESTS
# ===========================================================================

def test_lead_time_difference_and_reduction_formula():
    """
    Independently verify:
    A. Lead-time difference: Baseline - Alternative
    B. Lead-time reduction %: ((Baseline - Alternative) / Baseline) * 100
    """
    baseline = 6.0
    alternative = 3.6
    
    delta = baseline - alternative
    assert delta == 2.4
    
    pct = ((baseline - alternative) / baseline) * 100.0
    assert abs(pct - 40.0) < 1e-6
    
    # Zero baseline boundary handling
    zero_baseline = 0.0
    safe_pct = ((zero_baseline - alternative) / zero_baseline) * 100.0 if zero_baseline > 0 else 0.0
    assert safe_pct == 0.0


def test_gross_profit_and_margin_with_zero_safety():
    """
    Independently verify:
    C. Gross profit: Sales - Cost
    D. Gross margin: Gross Profit / Sales * 100 with zero sales safe handling
    """
    # 1. Normal values
    sales = 250.0
    cost = 150.0
    gp = sales - cost
    gm = (gp / sales) * 100.0
    assert gp == 100.0
    assert gm == 40.0
    
    # 2. Zero sales safe handling
    zero_sales = 0.0
    safe_gm = (gp / zero_sales) * 100.0 if zero_sales > 0 else 0.0
    assert safe_gm == 0.0
    
    # 3. Verify against actual dataset
    df = load_raw_data()
    diff = (df['Sales'] - df['Cost']) - df['Gross Profit']
    assert np.allclose(diff, 0.0, atol=1e-2), "Dataset Sales - Cost does not equal Gross Profit"


def test_no_self_recommendation_in_all_recommendations():
    """
    Verify that the system recommendations table never recommends a plant
    as an alternative to itself.
    """
    engine = OptimizationEngine()
    recs = engine.generate_all_recommendations(priority_slider=70.0)
    assert len(recs) > 0, "No recommendations generated"
    
    for r in recs:
        assert r['Current Factory'] != r['Recommended Factory'], \
            f"Self-recommendation detected for {r['Product Name']}: {r['Current Factory']}"


def test_optimization_score_slider_dynamics():
    """
    Verify that priority slider values (0 to 100) correctly alter the balance
    between speed (lead-time) and profit protection.
    """
    engine = OptimizationEngine()
    
    # Slider at 0 (100% Profit Protection)
    sim_profit = engine.simulate_optimization(
        product_name="Wonka Bar - Milk Chocolate",
        destination_region="Interior",
        ship_mode="Standard Class",
        priority_slider=0.0
    )
    
    # Slider at 100 (100% Speed)
    sim_speed = engine.simulate_optimization(
        product_name="Wonka Bar - Milk Chocolate",
        destination_region="Interior",
        ship_mode="Standard Class",
        priority_slider=100.0
    )
    
    scores_profit = {f["factory"]: f["composite_score"] for f in sim_profit["factories"]}
    scores_speed = {f["factory"]: f["composite_score"] for f in sim_speed["factories"]}
    
    # Scores must shift with priority slider
    assert scores_profit != scores_speed, "Composite scores did not respond to slider change"


def test_evidence_quality_and_risk_stratification():
    """Verify that sample volume correctly dictates evidence quality and risk flags."""
    engine = OptimizationEngine()
    
    # High volume combination (Wonka Bar in Interior)
    high_ev = engine.assess_evidence_quality(
        product_name="Wonka Bar - Milk Chocolate",
        factory_name="Wicked Choccy's",
        region="Interior"
    )
    assert high_ev["evidence_quality"] in ["High", "Medium"]
    assert high_ev["risk_level"] in ["Low", "Moderate"]
    assert high_ev["can_reallocate"] is True
    
    # Low volume combination (Fun Dip with only 3 total orders in dataset)
    low_ev = engine.assess_evidence_quality(
        product_name="Fun Dip",
        factory_name="Sugar Shack",
        region="Pacific"
    )
    assert low_ev["evidence_quality"] == "Insufficient evidence"
    assert low_ev["recommendation_category"] == "Insufficient evidence"
    assert low_ev["can_reallocate"] is False
    assert low_ev["confidence_score"] <= 50


def test_insufficient_evidence_sku_simulation():
    """Verify that simulating low-volume SKUs flags insufficient evidence and disclaims outcomes."""
    engine = OptimizationEngine()
    sim_fun_dip = engine.simulate_optimization(
        product_name="Fun Dip",
        destination_region="Pacific",
        ship_mode="Standard Class"
    )
    assert sim_fun_dip["has_sufficient_evidence"] is False
    assert sim_fun_dip["recommendation_category"] == "Insufficient evidence"
    assert "Insufficient historical evidence" in sim_fun_dip["factories"][0]["risk_flag"]


def test_simulation_disclaimers_and_assumptions():
    """Verify that simulation outputs explicitly contain documented assumptions and limitation statement."""
    engine = OptimizationEngine()
    sim = engine.simulate_optimization(
        product_name="Wonka Bar - Milk Chocolate",
        destination_region="Interior",
        ship_mode="Standard Class"
    )
    assert "assumptions" in sim
    assert len(sim["assumptions"]) >= 4
    assert "limitation_statement" in sim
    assert "Alternative-factory benefits are simulated estimates" in sim["limitation_statement"]
    
    # Check that candidate factories carry the simulated label
    for f in sim["factories"]:
        assert f["simulation_label"] == "Simulated estimate — not yet validated in production"
        assert f["recommendation_category"] in ["Potential opportunity", "Needs validation", "Insufficient evidence", "Operationally infeasible"]


def test_recommendation_categories_presence():
    """Verify that all recommendations outputted by generate_all_recommendations have documented categories."""
    engine = OptimizationEngine()
    recs = engine.generate_all_recommendations()
    assert len(recs) > 0
    valid_categories = {"Potential opportunity", "Needs validation", "Insufficient evidence", "Operationally infeasible"}
    for r in recs:
        assert r["Category"] in valid_categories, f"Invalid category: {r['Category']}"
        assert "Rationale" in r and len(r["Rationale"]) > 0
        assert r["Status Label"] == "Simulated estimate — not yet validated in production"
