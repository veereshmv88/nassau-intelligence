"""
Feature Engineering and Preprocessing Pipeline for Nassau Intelligence.
Prevents target leakage and integrates geographic distance.
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.config import FACTORIES, STATE_COORDINATES, REGION_COORDINATES


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians
    r = 6371.0  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def get_factory_destination_distance(factory_name: str, state_name: str, region_name: str = None) -> float:
    """
    Calculate geographic distance (km) between origin factory and destination state.
    Falls back to regional centroid if state is unknown.
    """
    if factory_name not in FACTORIES:
        return 1000.0  # Default nominal distance
    
    f_lat = FACTORIES[factory_name]["lat"]
    f_lon = FACTORIES[factory_name]["lon"]
    
    if state_name in STATE_COORDINATES:
        dest_lat, dest_lon = STATE_COORDINATES[state_name]
    elif region_name in REGION_COORDINATES:
        dest_lat, dest_lon = REGION_COORDINATES[region_name]
    else:
        dest_lat, dest_lon = 39.8283, -98.5795  # Geographic center of contiguous US
        
    return round(haversine_distance(f_lat, f_lon, dest_lat, dest_lon), 1)


def add_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich dataframe with calculated geographic distance."""
    df_feat = df.copy()
    
    distances = []
    for _, row in df_feat.iterrows():
        f_name = row.get('Factory', "Sugar Shack")
        state = row.get('State/Province', "")
        region = row.get('Region', "")
        dist = get_factory_destination_distance(f_name, state, region)
        distances.append(dist)
        
    df_feat['Distance_km'] = distances
    return df_feat


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """Return categorical and numerical feature column names."""
    cat_cols = ['Product Name', 'Division', 'Region', 'Ship Mode', 'Factory']
    num_cols = ['Units', 'Distance_km']
    return cat_cols, num_cols


def build_preprocessor() -> ColumnTransformer:
    """Build unified scikit-learn ColumnTransformer."""
    cat_cols, num_cols = get_feature_columns()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', StandardScaler(), num_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def create_chronological_splits(
    df: pd.DataFrame, 
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sort chronologically by Order Date and perform holdout split without leakage.
    Ensures that shipment dates and targets are strictly separated from feature set.
    """
    if 'Order_Date_dt' not in df.columns:
        df['Order_Date_dt'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
        
    df_sorted = df.sort_values('Order_Date_dt').reset_index(drop=True)
    
    # Feature columns
    cat_cols, num_cols = get_feature_columns()
    all_features = cat_cols + num_cols
    
    target_col = 'lead_time_days'
    
    split_idx = int(len(df_sorted) * (1.0 - test_size))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    X_train = train_df[all_features].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[all_features].copy()
    y_test = test_df[target_col].copy()
    
    return X_train, X_test, y_train, y_test
