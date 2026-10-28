"""
Route Analytics and Clustering Module for Nassau Intelligence.
Analyzes shipping routes, geographic distributions, underperforming lanes,
and route clustering using KMeans.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.config import FACTORIES, REGION_COORDINATES, STATE_COORDINATES
from src.preprocessing import get_factory_destination_distance


def aggregate_route_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate historical shipping performance grouped by (Factory, Region, Ship Mode).
    Computes mean lead time, total orders, total sales, and avg distance.
    """
    df_calc = df.copy()
    if 'Distance_km' not in df_calc.columns:
        df_calc['Distance_km'] = [
            get_factory_destination_distance(r.get('Factory', "Sugar Shack"), r.get('State/Province', ''), r.get('Region', ''))
            for _, r in df_calc.iterrows()
        ]
        
    routes = df_calc.groupby(['Factory', 'Region', 'Ship Mode']).agg(
        Order_Count=('Order ID', 'count'),
        Avg_Lead_Time=('lead_time_days', 'mean'),
        Std_Lead_Time=('lead_time_days', 'std'),
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Gross Profit', 'sum'),
        Avg_Distance_km=('Distance_km', 'mean')
    ).reset_index()
    
    routes['Std_Lead_Time'] = routes['Std_Lead_Time'].fillna(0)
    routes['Avg_Lead_Time'] = routes['Avg_Lead_Time'].round(2)
    routes['Std_Lead_Time'] = routes['Std_Lead_Time'].round(2)
    routes['Total_Sales'] = routes['Total_Sales'].round(2)
    routes['Total_Profit'] = routes['Total_Profit'].round(2)
    routes['Avg_Distance_km'] = routes['Avg_Distance_km'].round(1)
    
    # Identify underperforming lanes (higher lead time compared to ship mode benchmark)
    mode_benchmarks = df_calc.groupby('Ship Mode')['lead_time_days'].mean().to_dict()
    routes['Benchmark_Lead_Time'] = routes['Ship Mode'].map(mode_benchmarks).round(2)
    routes['Lead_Time_Deviation'] = (routes['Avg_Lead_Time'] - routes['Benchmark_Lead_Time']).round(2)
    
    # Tag status
    routes['Efficiency_Status'] = np.where(
        routes['Lead_Time_Deviation'] > 0.8, 'Underperforming',
        np.where(routes['Lead_Time_Deviation'] < -0.4, 'High Efficiency', 'Nominal')
    )
    
    return routes


def perform_route_clustering(routes_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Cluster routes based on operational performance:
    Avg_Lead_Time, Order_Count, Avg_Distance_km, Total_Sales.
    Returns routes dataframe with Cluster label and descriptive cluster archetype.
    """
    if len(routes_df) < n_clusters:
        routes_df['Cluster'] = 0
        routes_df['Cluster_Archetype'] = "Standard Route"
        return routes_df
        
    features = ['Avg_Lead_Time', 'Order_Count', 'Avg_Distance_km', 'Total_Sales']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(routes_df[features].fillna(0))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    routes_df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Name clusters based on centroid characteristics
    centroids = kmeans.cluster_centers_
    # Sort clusters by average lead time and volume
    cluster_profiles = {}
    for c in range(n_clusters):
        c_mean_lt = routes_df[routes_df['Cluster'] == c]['Avg_Lead_Time'].mean()
        c_mean_vol = routes_df[routes_df['Cluster'] == c]['Order_Count'].mean()
        c_mean_dist = routes_df[routes_df['Cluster'] == c]['Avg_Distance_km'].mean()
        
        if c_mean_lt > 4.5 and c_mean_dist > 1800:
            cluster_profiles[c] = "High-Distance / High-Lead Time Lane (Optimization Candidate)"
        elif c_mean_vol > routes_df['Order_Count'].median():
            cluster_profiles[c] = "High-Volume Core Distribution Backbone"
        else:
            cluster_profiles[c] = "Regional Short-Haul / Express Corridor"
            
    routes_df['Cluster_Archetype'] = routes_df['Cluster'].map(cluster_profiles)
    return routes_df


def get_factory_network_map_data() -> Dict[str, Any]:
    """
    Prepare data for factory nodes and active shipping routes for Plotly geo visualizations.
    """
    factory_nodes = []
    for f_name, info in FACTORIES.items():
        factory_nodes.append({
            "name": f_name,
            "lat": info["lat"],
            "lon": info["lon"],
            "city": info["city"],
            "state": info["state"],
            "region": info["region"],
            "specialty": info["specialty"],
            "color": info["color"]
        })
        
    return {
        "factories": factory_nodes,
        "region_centroids": REGION_COORDINATES
    }
