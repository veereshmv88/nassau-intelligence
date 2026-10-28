"""
Prediction Service for Nassau Intelligence.
Loads the trained ML pipeline and evaluates lead-time for candidate factory configurations.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.config import (
    MODEL_PIPELINE_PATH, FACTORIES, DEFAULT_PRODUCT_ASSIGNMENTS, 
    PRODUCT_DIVISIONS
)
from src.preprocessing import get_factory_destination_distance


class LeadTimePredictor:
    """Predicts shipping lead time using the persisted model pipeline."""
    
    _instance = None
    _loaded_mtime = None

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

    def evaluate_all_factories(
        self,
        product_name: str,
        destination_region: str,
        ship_mode: str,
        state_province: str = "",
        units: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Evaluate and return lead-time predictions across all 5 candidate factories.
        """
        current_factory = DEFAULT_PRODUCT_ASSIGNMENTS.get(product_name, "Sugar Shack")
        results = []
        
        for f_name in FACTORIES.keys():
            pred_lead = self.predict_single(
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
                "predicted_lead_time": pred_lead,
                "distance_km": dist,
                "city": FACTORIES[f_name]["city"],
                "state": FACTORIES[f_name]["state"],
                "specialty": FACTORIES[f_name]["specialty"],
                "color": FACTORIES[f_name]["color"]
            })
            
        return results
