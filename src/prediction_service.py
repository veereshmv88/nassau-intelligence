"""
Prediction Service for Nassau Intelligence.
Loads the trained ML pipeline and evaluates lead-time for candidate factory configurations
with calibrated empirical prediction intervals computed from chronological holdout residuals.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.config import (
    MODEL_PIPELINE_PATH, FACTORIES, DEFAULT_PRODUCT_ASSIGNMENTS, 
    PRODUCT_DIVISIONS
)
from src.preprocessing import get_factory_destination_distance


class LeadTimePredictor:
    """
    Predicts shipping lead time using the persisted model pipeline
    with empirical prediction intervals derived from chronological test residuals.
    """
    
    _instance = None
    _loaded_mtime = None

    # Calibrated empirical residual percentiles from chronological holdout (N = 2,039)
    # Residuals = y_actual - y_predicted
    # 90% Prediction Interval: [y_pred + RESIDUAL_P05, y_pred + RESIDUAL_P95]
    # 80% Prediction Interval: [y_pred + RESIDUAL_P10, y_pred + RESIDUAL_P90]
    RESIDUAL_P05 = -1.31
    RESIDUAL_P10 = -1.23
    RESIDUAL_MEDIAN = 0.49
    RESIDUAL_P90 = 1.77
    RESIDUAL_P95 = 2.54
    RESIDUAL_STD = 1.09

    def __init__(self):
        if not MODEL_PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Model pipeline not found at {MODEL_PIPELINE_PATH}. Run 'python src/train.py' first."
            )
        self.pipeline = joblib.load(MODEL_PIPELINE_PATH)
        self._loaded_mtime = MODEL_PIPELINE_PATH.stat().st_mtime

    @classmethod
    def get_instance(cls, force_reload: bool = False):
        current_mtime = MODEL_PIPELINE_PATH.stat().st_mtime if MODEL_PIPELINE_PATH.exists() else 0
        if cls._instance is None or force_reload or cls._instance._loaded_mtime != current_mtime:
            cls._instance = cls()
        return cls._instance

    def predict_single(
        self,
        product_name: str,
        factory_name: str,
        destination_region: str,
        ship_mode: str,
        state_province: str = "",
        units: int = 2
    ) -> float:
        """Predict expected shipping lead time (days) for a single configuration."""
        division = PRODUCT_DIVISIONS.get(product_name, "Chocolate")
        distance_km = get_factory_destination_distance(factory_name, state_province, destination_region)
        
        input_df = pd.DataFrame([{
            'Product Name': product_name,
            'Division': division,
            'Region': destination_region,
            'Ship Mode': ship_mode,
            'Factory': factory_name,
            'Units': units,
            'Distance_km': distance_km
        }])
        
        pred = self.pipeline.predict(input_df)[0]
        # Ensure lead time is physically bounded within operational limits (0.5 to 10 days)
        return min(10.0, max(0.5, round(float(pred), 2)))

    def predict_with_interval(
        self,
        product_name: str,
        factory_name: str,
        destination_region: str,
        ship_mode: str,
        state_province: str = "",
        units: int = 2,
        confidence_level: float = 0.90
    ) -> Dict[str, Any]:
        """
        Predict shipping lead time with calibrated empirical prediction intervals.
        Avoids false precision by reporting upper and lower error bounds.
        """
        point_pred = self.predict_single(
            product_name=product_name,
            factory_name=factory_name,
            destination_region=destination_region,
            ship_mode=ship_mode,
            state_province=state_province,
            units=units
        )

        if confidence_level >= 0.90:
            lower = max(0.5, round(point_pred + self.RESIDUAL_P05, 1))
            upper = min(10.0, round(point_pred + self.RESIDUAL_P95, 1))
            ci_pct = 90
        else:
            lower = max(0.5, round(point_pred + self.RESIDUAL_P10, 1))
            upper = min(10.0, round(point_pred + self.RESIDUAL_P90, 1))
            ci_pct = 80

        return {
            "point_prediction": point_pred,
            "interval_lower": lower,
            "interval_upper": upper,
            "confidence_level": f"{ci_pct}%",
            "interval_str": f"{point_pred:.1f}d [{lower:.1f} – {upper:.1f}d {ci_pct}% PI]"
        }

    def evaluate_all_factories(
        self,
        product_name: str,
        destination_region: str,
        ship_mode: str,
        state_province: str = "",
        units: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Evaluate and return lead-time predictions across all candidate factories
        including calibrated prediction intervals.
        """
        current_factory = DEFAULT_PRODUCT_ASSIGNMENTS.get(product_name, "Sugar Shack")
        results = []
        
        for f_name in FACTORIES.keys():
            interval_res = self.predict_with_interval(
                product_name=product_name,
                factory_name=f_name,
                destination_region=destination_region,
                ship_mode=ship_mode,
                state_province=state_province,
                units=units
            )
            dist = get_factory_destination_distance(f_name, state_province, destination_region)
            
            results.append({
                "factory": f_name,
                "is_current": (f_name == current_factory),
                "predicted_lead_time": interval_res["point_prediction"],
                "interval_lower": interval_res["interval_lower"],
                "interval_upper": interval_res["interval_upper"],
                "interval_str": interval_res["interval_str"],
                "distance_km": dist,
                "city": FACTORIES[f_name]["city"],
                "state": FACTORIES[f_name]["state"],
                "specialty": FACTORIES[f_name]["specialty"],
                "color": FACTORIES[f_name]["color"]
            })
            
        return results
