"""
Database DDL Schema Definition for Nassau Intelligence.
Normalized 3NF relational schema with strict foreign keys, checks, and performance indexes.
"""

import sqlite3
import logging
from src.db.connection import get_db_connection

logger = logging.getLogger("NassauSchema")

DDL_STATEMENTS = [
    # 1. Factories Master
    """
    CREATE TABLE IF NOT EXISTS factories (
        factory_id INTEGER PRIMARY KEY AUTOINCREMENT,
        factory_name TEXT NOT NULL UNIQUE,
        city TEXT NOT NULL,
        state TEXT NOT NULL,
        region TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        specialty TEXT NOT NULL,
        rated_daily_capacity INTEGER NOT NULL DEFAULT 30,
        baseline_workload_pct REAL NOT NULL DEFAULT 70.0,
        max_feasible_workload_pct REAL NOT NULL DEFAULT 85.0,
        color_hex TEXT NOT NULL DEFAULT '#087F5B'
    );
    """,

    # 2. Destinations Master
    """
    CREATE TABLE IF NOT EXISTS destinations (
        destination_id INTEGER PRIMARY KEY AUTOINCREMENT,
        state_province TEXT NOT NULL UNIQUE,
        region TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL
    );
    """,

    # 3. Carrier Service Master
    """
    CREATE TABLE IF NOT EXISTS carriers (
        carrier_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ship_mode TEXT NOT NULL UNIQUE,
        mean_lead_time_days REAL NOT NULL,
        sla_target_days REAL NOT NULL,
        cost_multiplier REAL NOT NULL DEFAULT 1.0
    );
    """,

    # 4. Products Master
    """
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT NOT NULL UNIQUE,
        division TEXT NOT NULL,
        default_factory_id INTEGER NOT NULL,
        allergen_profile TEXT NOT NULL DEFAULT 'Standard',
        avg_sales_per_unit REAL NOT NULL DEFAULT 0.0,
        avg_cost_per_unit REAL NOT NULL DEFAULT 0.0,
        avg_profit_per_unit REAL NOT NULL DEFAULT 0.0,
        profit_margin_pct REAL NOT NULL DEFAULT 0.0,
        FOREIGN KEY (default_factory_id) REFERENCES factories(factory_id) ON DELETE RESTRICT
    );
    """,

    # 5. Orders Historical Manifest (10,194 records)
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_row_id INTEGER PRIMARY KEY,
        order_id TEXT NOT NULL,
        order_date TEXT NOT NULL,
        ship_date TEXT NOT NULL,
        customer_id INTEGER NOT NULL,
        city TEXT NOT NULL,
        state_province TEXT NOT NULL,
        region TEXT NOT NULL,
        postal_code TEXT,
        division TEXT NOT NULL,
        product_id INTEGER NOT NULL,
        carrier_id INTEGER NOT NULL,
        factory_id INTEGER NOT NULL,
        units INTEGER NOT NULL CHECK (units > 0),
        sales REAL NOT NULL CHECK (sales >= 0),
        gross_profit REAL NOT NULL,
        cost REAL NOT NULL,
        lead_time_days REAL NOT NULL CHECK (lead_time_days >= 0),
        FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE RESTRICT,
        FOREIGN KEY (carrier_id) REFERENCES carriers(carrier_id) ON DELETE RESTRICT,
        FOREIGN KEY (factory_id) REFERENCES factories(factory_id) ON DELETE RESTRICT
    );
    """,

    # 6. Factory-Product Tooling Compatibility
    """
    CREATE TABLE IF NOT EXISTS factory_eligibility (
        eligibility_id INTEGER PRIMARY KEY AUTOINCREMENT,
        factory_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        is_eligible INTEGER NOT NULL DEFAULT 1 CHECK (is_eligible IN (0, 1)),
        tooling_notes TEXT,
        UNIQUE (factory_id, product_id),
        FOREIGN KEY (factory_id) REFERENCES factories(factory_id) ON DELETE CASCADE,
        FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
    );
    """,

    # 7. Saved Optimization Scenarios
    """
    CREATE TABLE IF NOT EXISTS optimization_scenarios (
        scenario_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        product_id INTEGER NOT NULL,
        destination_region TEXT NOT NULL,
        ship_mode TEXT NOT NULL,
        batch_units INTEGER NOT NULL,
        priority_slider REAL NOT NULL,
        baseline_factory_id INTEGER NOT NULL,
        recommended_factory_id INTEGER NOT NULL,
        baseline_lead_time REAL NOT NULL,
        recommended_lead_time REAL NOT NULL,
        lead_time_delta REAL NOT NULL,
        lead_time_reduction_pct REAL NOT NULL,
        distance_delta_km REAL NOT NULL,
        profit_impact_pct REAL NOT NULL,
        recommendation_category TEXT NOT NULL,
        confidence_score INTEGER NOT NULL,
        user_notes TEXT,
        FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
        FOREIGN KEY (baseline_factory_id) REFERENCES factories(factory_id) ON DELETE RESTRICT,
        FOREIGN KEY (recommended_factory_id) REFERENCES factories(factory_id) ON DELETE RESTRICT
    );
    """,

    # 8. Intelligent Recommendations Board
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        recommendation_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        product_id INTEGER NOT NULL,
        current_factory_id INTEGER NOT NULL,
        candidate_factory_id INTEGER NOT NULL,
        destination_region TEXT NOT NULL,
        operational_issue TEXT NOT NULL,
        supporting_evidence TEXT NOT NULL,
        predicted_lead_time REAL NOT NULL,
        lead_time_delta REAL NOT NULL,
        distance_reduction_km REAL NOT NULL,
        cost_impact_pct REAL NOT NULL,
        evidential_category TEXT NOT NULL,
        confidence_score INTEGER NOT NULL,
        priority_rank INTEGER NOT NULL,
        capacity_feasible INTEGER NOT NULL CHECK (capacity_feasible IN (0, 1)),
        recommended_action TEXT NOT NULL,
        business_explanation TEXT NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
        FOREIGN KEY (current_factory_id) REFERENCES factories(factory_id) ON DELETE RESTRICT,
        FOREIGN KEY (candidate_factory_id) REFERENCES factories(factory_id) ON DELETE RESTRICT
    );
    """,

    # 9. Model Evaluation Benchmark Registry
    """
    CREATE TABLE IF NOT EXISTS model_evaluations (
        evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL UNIQUE,
        parameters_count INTEGER NOT NULL,
        holdout_mae REAL NOT NULL,
        holdout_rmse REAL NOT NULL,
        holdout_r2 REAL NOT NULL,
        cv_rmse_mean REAL NOT NULL,
        cv_rmse_std REAL NOT NULL,
        status_notes TEXT NOT NULL,
        evaluated_at TEXT NOT NULL
    );
    """,

    # 10. Data Governance & Audit Log
    """
    CREATE TABLE IF NOT EXISTS data_audit_log (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        source_file TEXT NOT NULL,
        total_rows INTEGER NOT NULL,
        valid_rows INTEGER NOT NULL,
        duplicate_count INTEGER NOT NULL,
        mean_lead_time REAL NOT NULL,
        anomaly_notes TEXT NOT NULL
    );
    """
]

INDEX_STATEMENTS = [
    # Performance indexes for dashboard query acceleration
    "CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(product_id);",
    "CREATE INDEX IF NOT EXISTS idx_orders_factory ON orders(factory_id);",
    "CREATE INDEX IF NOT EXISTS idx_orders_carrier ON orders(carrier_id);",
    "CREATE INDEX IF NOT EXISTS idx_orders_region ON orders(region);",
    "CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_category ON recommendations(evidential_category);",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_rank ON recommendations(priority_rank);"
]


def init_database_schema(conn: sqlite3.Connection = None):
    """Execute all DDL statements and create indexes within a transactional block."""
    if conn is None:
        with get_db_connection() as managed_conn:
            _execute_schema(managed_conn)
    else:
        _execute_schema(conn)


def _execute_schema(conn: sqlite3.Connection):
    cursor = conn.cursor()
    for ddl in DDL_STATEMENTS:
        cursor.execute(ddl)
    for idx in INDEX_STATEMENTS:
        cursor.execute(idx)
    cursor.close()
    logger.info("Nassau relational schema and indexes verified successfully.")
