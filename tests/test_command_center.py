"""
Tests for Command Center dashboard components and visualizations.
Verifies alignment with the primary enterprise reference design.
"""

import pytest
from pathlib import Path
import plotly.graph_objects as go
from src.chart_builder import create_command_center_network_map, create_operational_pulse_chart
from src.data_loader import load_raw_data, prepare_dataset

BASE_DIR = Path(__file__).resolve().parent.parent


def test_command_center_network_map_traces():
    fig = create_command_center_network_map()
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data if t.name]
    assert "Historical route (observed)" in trace_names
    assert "Simulated alternative (evaluated)" in trace_names


def test_operational_pulse_chart_execution():
    df = prepare_dataset(load_raw_data())
    fig = create_operational_pulse_chart(df, time_range="Last 12 Months")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    # Check that annotations contain callout badge
    assert len(fig.layout.annotations) >= 1
    assert "days" in fig.layout.annotations[0].text


def test_command_center_page_content_in_app():
    app_text = (BASE_DIR / "src" / "app.py").read_text(encoding="utf-8")
    assert "Supply Chain Command Center" in app_text
    assert "Orders Analyzed" in app_text
    assert "Avg. Dispatch Lead Time" in app_text
    assert "Active Factories" in app_text
    assert "Capacity Feasible" in app_text
    assert "Factory Performance" in app_text
    assert "Decision Brief" in app_text
    assert "No verified improvement over current assignment." in app_text
    assert "Model Health" in app_text
    assert "Data Quality" in app_text

