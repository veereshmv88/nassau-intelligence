"""
Regression tests: import integrity and application startup.

Guards against the ImportError that caused Render deployment failure:
  "ImportError: cannot import name 'create_factory_network_map' from src.chart_builder"

These tests must continue to pass from both:
  - The project root (where app.py lives)
  - The src/ directory (where src/app.py is the Render entry-point)
"""

import sys
import importlib
from pathlib import Path

import pytest

# Ensure the project root is in sys.path before any imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ---------------------------------------------------------------------------
# 1. All src sub-modules import without error
# ---------------------------------------------------------------------------
SRC_MODULES = [
    "src.config",
    "src.preprocessing",
    "src.data_loader",
    "src.prediction_service",
    "src.optimization_engine",
    "src.chart_builder",
    "src.route_analytics",
    "src.reporting",
]


@pytest.mark.parametrize("module_name", SRC_MODULES)
def test_src_module_importable(module_name):
    """Every module in the src package must be importable without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None, f"{module_name} imported as None"


# ---------------------------------------------------------------------------
# 2. The exact symbol that caused the Render ImportError is importable
# ---------------------------------------------------------------------------
def test_create_factory_network_map_importable():
    """
    Regression: Render deployment failed because create_factory_network_map
    could not be imported from src.chart_builder.
    This must never regress.
    """
    from src.chart_builder import create_factory_network_map
    assert callable(create_factory_network_map), (
        "create_factory_network_map must be a callable function"
    )


def test_all_chart_builder_functions_importable():
    """All six chart-builder functions consumed by app.py must be importable."""
    from src.chart_builder import (
        create_factory_network_map,
        create_lead_time_regional_bar,
        create_profit_vs_lead_time_scatter,
        create_feature_importance_chart,
        create_actual_vs_predicted_chart,
        create_lead_time_distribution_chart,
    )
    for fn in [
        create_factory_network_map,
        create_lead_time_regional_bar,
        create_profit_vs_lead_time_scatter,
        create_feature_importance_chart,
        create_actual_vs_predicted_chart,
        create_lead_time_distribution_chart,
    ]:
        assert callable(fn), f"{fn.__name__} must be callable"


# ---------------------------------------------------------------------------
# 3. chart_builder functions execute without exception (smoke-test)
# ---------------------------------------------------------------------------
def test_create_factory_network_map_executes():
    """create_factory_network_map must render a figure without exception."""
    import pandas as pd
    from src.chart_builder import create_factory_network_map
    import plotly.graph_objects as go

    fig = create_factory_network_map(df=pd.DataFrame())
    assert isinstance(fig, go.Figure), "create_factory_network_map must return a go.Figure"
    assert len(fig.data) >= 1, "Factory map must contain at least one trace"


def test_create_lead_time_regional_bar_executes():
    """create_lead_time_regional_bar must produce a figure without exception."""
    import pandas as pd
    from src.chart_builder import create_lead_time_regional_bar
    import plotly.graph_objects as go

    fig = create_lead_time_regional_bar(df=pd.DataFrame())
    assert isinstance(fig, go.Figure)


def test_feature_importance_chart_executes():
    """create_feature_importance_chart must accept a dict and return a figure."""
    from src.chart_builder import create_feature_importance_chart
    import plotly.graph_objects as go

    feat_imp = {"Distance_km": 0.45, "Units": 0.22, "Ship_Mode_Standard Class": 0.18}
    fig = create_feature_importance_chart(feat_imp)
    assert isinstance(fig, go.Figure)


def test_actual_vs_predicted_chart_executes():
    """create_actual_vs_predicted_chart must accept lists and return a figure."""
    from src.chart_builder import create_actual_vs_predicted_chart
    import plotly.graph_objects as go

    fig = create_actual_vs_predicted_chart([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# 4. src/app.py and root app.py both exist and are syntactically valid
# ---------------------------------------------------------------------------
def test_app_py_exists_and_compiles():
    """The root app.py must exist and compile without syntax errors."""
    import ast
    app_path = BASE_DIR / "app.py"
    assert app_path.exists(), "app.py must exist in the project root"
    source = app_path.read_text(encoding="utf-8")
    # Will raise SyntaxError on bad code
    tree = ast.parse(source)
    assert tree is not None


def test_src_app_py_exists_and_compiles():
    """
    src/app.py must exist (the Render entry-point) and compile without syntax errors.
    This is the file referenced in the Render start command:
      streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0
    """
    import ast
    src_app_path = BASE_DIR / "src" / "app.py"
    assert src_app_path.exists(), (
        "src/app.py must exist. Render uses 'streamlit run src/app.py' as the start command. "
        "Without this file the deployment fails before any imports are attempted."
    )
    source = src_app_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert tree is not None


# ---------------------------------------------------------------------------
# 5. Import chain: app.py imports chart_builder, config, etc. (no circular deps)
# ---------------------------------------------------------------------------
def test_app_imports_resolve():
    """
    Verify the complete import chain used in app.py resolves without errors.
    This mirrors exactly what Streamlit does when it starts src/app.py.
    """
    from src.config import (
        FACTORIES, DEFAULT_PRODUCT_ASSIGNMENTS, PRODUCT_DIVISIONS,
        REGION_ALIASES, MODEL_PIPELINE_PATH, MODEL_METRICS_PATH,
        FEATURE_IMPORTANCE_PATH, THEME,
    )
    from src.data_loader import load_raw_data, prepare_dataset, audit_data_quality
    from src.optimization_engine import OptimizationEngine
    from src.prediction_service import LeadTimePredictor
    from src.route_analytics import aggregate_route_performance, perform_route_clustering
    from src.reporting import generate_executive_summary, generate_research_report
    from src.chart_builder import (
        create_factory_network_map,
        create_lead_time_regional_bar,
        create_profit_vs_lead_time_scatter,
        create_feature_importance_chart,
        create_actual_vs_predicted_chart,
        create_lead_time_distribution_chart,
    )
    # If we reach here, no ImportError was raised
    assert True
