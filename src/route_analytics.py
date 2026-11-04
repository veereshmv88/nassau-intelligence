"""
Route Analytics and Clustering Module for Nassau Intelligence.
Analyzes shipping routes, geographic distributions, underperforming lanes,
and route clustering using KMeans.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.config import FACTORIES, REGION_COORDINATES, STATE_COORDINATES
from src.preprocessing import get_factory_destination_distance

logger = logging.getLogger(__name__)

ROUTE_SCHEMA_COLUMNS = [
    "Factory",
    "Destination_State",
    "Region",
    "Ship Mode",
    "Total_Orders",
    "Total_Volume",
    "Avg_Lead_Time",
    "Std_Lead_Time",
    "Avg_Distance_km",
    "Total_Sales",
    "Total_Profit",
    "Dest_Lat",
    "Dest_Lon",
    "Factory_Lat",
    "Factory_Lon",
    "Benchmark_Lead_Time",
    "Lead_Time_Deviation",
    "Efficiency_Status"
]


def aggregate_route_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate historical shipping performance grouped by (Factory, Destination_State, Region, Ship Mode).
    Computes mean lead time, total orders, total volume, total sales, and avg distance.
    Returns a validated, schema-compliant DataFrame.
    """
    if df is None or df.empty:
        empty_df = pd.DataFrame(columns=ROUTE_SCHEMA_COLUMNS)
        empty_df["Order_Count"] = pd.Series(dtype=int)
        empty_df["Total_Units"] = pd.Series(dtype=int)
        return empty_df

    df_calc = df.copy()

    # Map Destination State from State/Province or fallback to Region
    if "Destination_State" not in df_calc.columns:
        if "State/Province" in df_calc.columns and df_calc["State/Province"].notna().any():
            df_calc["Destination_State"] = df_calc["State/Province"].fillna(df_calc.get("Region", "Unknown"))
        elif "Region" in df_calc.columns:
            df_calc["Destination_State"] = df_calc["Region"]
        else:
            df_calc["Destination_State"] = "National"

    if "Region" not in df_calc.columns:
        df_calc["Region"] = "Interior"

    if "Ship Mode" not in df_calc.columns:
        df_calc["Ship Mode"] = "Standard Class"

    if "Factory" not in df_calc.columns:
        df_calc["Factory"] = "Sugar Shack"

    # Calculate Distance if not precomputed
    if "Distance_km" not in df_calc.columns:
        df_calc["Distance_km"] = [
            get_factory_destination_distance(
                r.get("Factory", "Sugar Shack"),
                r.get("State/Province", r.get("Destination_State", "")),
                r.get("Region", "")
            )
            for _, r in df_calc.iterrows()
        ]

    # Handle Units / Volume safely
    if "Units" not in df_calc.columns:
        # If Units missing, count rows as 1 unit per order
        df_calc["Units"] = 1

    if "Sales" not in df_calc.columns:
        df_calc["Sales"] = 0.0

    if "Gross Profit" not in df_calc.columns:
        df_calc["Gross Profit"] = 0.0

    if "lead_time_days" not in df_calc.columns:
        df_calc["lead_time_days"] = 5.0

    routes = df_calc.groupby(["Factory", "Destination_State", "Region", "Ship Mode"]).agg(
        Order_Count=("Order ID" if "Order ID" in df_calc.columns else "Factory", "count"),
        Total_Volume=("Units", "sum"),
        Avg_Lead_Time=("lead_time_days", "mean"),
        Std_Lead_Time=("lead_time_days", "std"),
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Gross Profit", "sum"),
        Avg_Distance_km=("Distance_km", "mean")
    ).reset_index()

    # Aliases and formatting
    routes["Total_Orders"] = routes["Order_Count"].astype(int)
    routes["Total_Volume"] = routes["Total_Volume"].fillna(0).astype(int)
    routes["Total_Units"] = routes["Total_Volume"]  # backwards compatibility alias
    routes["Std_Lead_Time"] = routes["Std_Lead_Time"].fillna(0).round(2)
    routes["Avg_Lead_Time"] = routes["Avg_Lead_Time"].round(2)
    routes["Total_Sales"] = routes["Total_Sales"].round(2)
    routes["Total_Profit"] = routes["Total_Profit"].round(2)
    routes["Avg_Distance_km"] = routes["Avg_Distance_km"].round(1)

    # Attach Geographic Coordinates
    routes["Dest_Lat"] = routes["Destination_State"].map(
        lambda s: STATE_COORDINATES.get(s, REGION_COORDINATES.get(s, (39.8, -98.5)))[0]
    )
    routes["Dest_Lon"] = routes["Destination_State"].map(
        lambda s: STATE_COORDINATES.get(s, REGION_COORDINATES.get(s, (39.8, -98.5)))[1]
    )
    routes["Factory_Lat"] = routes["Factory"].map(lambda f: FACTORIES.get(f, {}).get("lat", 39.8))
    routes["Factory_Lon"] = routes["Factory"].map(lambda f: FACTORIES.get(f, {}).get("lon", -98.5))

    # Identify underperforming lanes (compared to carrier ship mode benchmark)
    mode_benchmarks = df_calc.groupby("Ship Mode")["lead_time_days"].mean().to_dict()
    routes["Benchmark_Lead_Time"] = routes["Ship Mode"].map(mode_benchmarks).round(2)
    routes["Lead_Time_Deviation"] = (routes["Avg_Lead_Time"] - routes["Benchmark_Lead_Time"]).round(2)

    routes["Efficiency_Status"] = np.where(
        routes["Lead_Time_Deviation"] > 0.8, "Underperforming",
        np.where(routes["Lead_Time_Deviation"] < -0.4, "High Efficiency", "Nominal")
    )

    return routes


def perform_route_clustering(routes_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Cluster routes based on operational performance:
    Avg_Lead_Time, Total_Orders, Avg_Distance_km, Total_Volume.
    Returns routes dataframe with Cluster label and descriptive cluster archetype.
    """
    if routes_df is None or routes_df.empty:
        res = pd.DataFrame(columns=ROUTE_SCHEMA_COLUMNS + ["Cluster", "Cluster_Archetype"])
        return res

    result_df = routes_df.copy()

    # If insufficient samples for clustering, return gracefully
    if len(result_df) < n_clusters:
        result_df["Cluster"] = 0
        result_df["Cluster_Archetype"] = "Standard Regional Route"
        return result_df

    features = ["Avg_Lead_Time", "Total_Orders", "Avg_Distance_km", "Total_Volume"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(result_df[features].fillna(0))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    result_df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Name clusters dynamically based on centroid characteristics
    cluster_profiles = {}
    median_orders = result_df["Total_Orders"].median()
    median_dist = result_df["Avg_Distance_km"].median()

    for c in range(n_clusters):
        c_subset = result_df[result_df["Cluster"] == c]
        if c_subset.empty:
            cluster_profiles[c] = f"Cluster {c+1} (Standard Corridor)"
            continue

        c_mean_lt = c_subset["Avg_Lead_Time"].mean()
        c_mean_vol = c_subset["Total_Orders"].mean()
        c_mean_dist = c_subset["Avg_Distance_km"].mean()

        if c_mean_dist > median_dist and c_mean_lt > 4.8:
            cluster_profiles[c] = "Long-Haul Inter-Regional Corridor (Optimization Candidate)"
        elif c_mean_vol >= median_orders and c_mean_dist <= median_dist:
            cluster_profiles[c] = "High-Volume Core Distribution Trunk"
        elif c_mean_vol >= median_orders and c_mean_dist > median_dist:
            cluster_profiles[c] = "High-Volume Transcontinental Freight Lane"
        elif c_mean_dist < 800:
            cluster_profiles[c] = "Regional Short-Haul / Express Corridor"
        else:
            cluster_profiles[c] = "Balanced Secondary Carrier Route"

    result_df["Cluster_Archetype"] = result_df["Cluster"].map(cluster_profiles)
    return result_df


def get_route_cluster_summaries(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate aggregate summary statistics for each route cluster.
    """
    if clustered_df is None or clustered_df.empty or "Cluster" not in clustered_df.columns:
        return pd.DataFrame()

    summary = clustered_df.groupby(["Cluster", "Cluster_Archetype"]).agg(
        Route_Count=("Factory", "count"),
        Total_Orders=("Total_Orders", "sum"),
        Total_Volume=("Total_Volume", "sum"),
        Avg_Lead_Time=("Avg_Lead_Time", "mean"),
        Avg_Distance_km=("Avg_Distance_km", "mean"),
        Underperforming_Count=("Efficiency_Status", lambda s: (s == "Underperforming").sum())
    ).reset_index()

    summary["Avg_Lead_Time"] = summary["Avg_Lead_Time"].round(2)
    summary["Avg_Distance_km"] = summary["Avg_Distance_km"].round(1)
    summary["Underperforming_Rate"] = (
        (summary["Underperforming_Count"] / summary["Route_Count"]) * 100
    ).round(1).astype(str) + "%"

    return summary


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
