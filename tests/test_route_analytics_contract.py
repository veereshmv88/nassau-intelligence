"""
Tests for Route Analytics Data Contract and K-Means Clustering Workflow.
Validates schema compliance, missing field handling, coordinate resilience,
and frontend presentation integration.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_raw_data, prepare_dataset
from src.route_analytics import (
    aggregate_route_performance,
    perform_route_clustering,
    get_route_cluster_summaries,
    ROUTE_SCHEMA_COLUMNS
)


@pytest.fixture
def clean_dataset():
    raw_df = load_raw_data()
    return prepare_dataset(raw_df, save_processed=False)


def test_route_aggregation_schema_compliance(clean_dataset):
    """Verify that aggregate_route_performance returns all required schema fields."""
    routes = aggregate_route_performance(clean_dataset)
    assert not routes.empty
    
    # Assert critical columns required by frontend and contracts
    required_cols = [
        "Factory", "Destination_State", "Region", "Ship Mode",
        "Total_Orders", "Total_Volume", "Avg_Lead_Time", "Avg_Distance_km",
        "Efficiency_Status", "Dest_Lat", "Dest_Lon"
    ]
    for col in required_cols:
        assert col in routes.columns, f"Missing expected column: {col}"

    assert routes["Total_Orders"].dtype in [np.int64, int, np.int32]
    assert routes["Total_Volume"].dtype in [np.int64, int, np.int32]
    assert (routes["Total_Orders"] > 0).all()


def test_kmeans_clustering_output_and_alignment(clean_dataset):
    """Verify KMeans produces valid cluster assignments and archetypes properly aligned."""
    routes = aggregate_route_performance(clean_dataset)
    clustered = perform_route_clustering(routes, n_clusters=4)
    
    assert len(clustered) == len(routes)
    assert "Cluster" in clustered.columns
    assert "Cluster_Archetype" in clustered.columns
    
    # Clusters must be valid integers between 0 and 3
    assert set(clustered["Cluster"].unique()).issubset({0, 1, 2, 3})
    assert clustered["Cluster_Archetype"].notna().all()


def test_route_cluster_summaries_generation(clean_dataset):
    """Verify get_route_cluster_summaries returns aggregated statistics per cluster."""
    routes = aggregate_route_performance(clean_dataset)
    clustered = perform_route_clustering(routes, n_clusters=4)
    summaries = get_route_cluster_summaries(clustered)
    
    assert not summaries.empty
    assert "Cluster" in summaries.columns
    assert "Cluster_Archetype" in summaries.columns
    assert "Total_Orders" in summaries.columns
    assert "Total_Volume" in summaries.columns
    assert "Underperforming_Rate" in summaries.columns


def test_frontend_dataframe_column_selection(clean_dataset):
    """Verify frontend DataFrame column selection from line 1576 does not raise KeyError."""
    routes = aggregate_route_performance(clean_dataset)
    clustered = perform_route_clustering(routes, n_clusters=4)
    
    # This was the exact failing line in the bug report:
    cols_needed = [
        "Factory", "Destination_State", "Total_Orders", "Total_Volume",
        "Cluster_Archetype", "Efficiency_Status"
    ]
    frontend_slice = clustered[cols_needed].head(12)
    assert len(frontend_slice) == 12
    assert list(frontend_slice.columns) == cols_needed


def test_empty_route_dataset_handling():
    """Verify empty DataFrames are handled gracefully without exceptions."""
    empty_df = pd.DataFrame()
    routes = aggregate_route_performance(empty_df)
    assert routes.empty
    assert "Total_Orders" in routes.columns
    assert "Total_Volume" in routes.columns
    
    clustered = perform_route_clustering(routes, n_clusters=4)
    assert clustered.empty
    
    summaries = get_route_cluster_summaries(clustered)
    assert summaries.empty


def test_missing_optional_and_required_fields_handling():
    """Verify resilience when fields like State/Province, Units, or Distance_km are absent."""
    minimal_df = pd.DataFrame({
        "Factory": ["Sugar Shack", "Wicked Choccy's"],
        "Region": ["Pacific", "Interior"],
        "Order ID": ["ORD-1", "ORD-2"],
        "lead_time_days": [4, 6]
    })
    
    routes = aggregate_route_performance(minimal_df)
    assert not routes.empty
    assert "Destination_State" in routes.columns
    assert routes["Destination_State"].iloc[0] == "Pacific"  # fell back to Region
    assert "Total_Volume" in routes.columns
    assert routes["Total_Volume"].iloc[0] == 1  # defaulted to 1 unit
    
    clustered = perform_route_clustering(routes, n_clusters=2)
    assert len(clustered) == 2
    assert "Cluster" in clustered.columns


def test_unknown_or_invalid_coordinates_resilience():
    """Verify coordinates fallback to regional centroids when destination states are unknown."""
    df_unknown = pd.DataFrame({
        "Factory": ["Sugar Shack"],
        "State/Province": ["Unknown Land"],
        "Region": ["Atlantic"],
        "Order ID": ["ORD-99"],
        "Units": [5],
        "lead_time_days": [3]
    })
    
    routes = aggregate_route_performance(df_unknown)
    assert routes["Dest_Lat"].notna().all()
    assert routes["Dest_Lon"].notna().all()
    assert routes["Factory_Lat"].notna().all()
    assert routes["Factory_Lon"].notna().all()

