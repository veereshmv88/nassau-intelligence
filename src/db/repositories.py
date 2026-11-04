"""
Repository Layer for Nassau Intelligence Database.
Provides clean, optimized, typed data access for Orders, Products, Factories,
Recommendations, Scenarios, and Model Evaluations.
"""

import sqlite3
from typing import List, Dict, Any, Optional
from src.db.connection import get_db_connection
from src.recommendation_contract import normalize_recommendations_list, normalize_recommendation


class OrderRepository:
    """Data access repository for historical order records and aggregations."""

    @staticmethod
    def get_total_count() -> int:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM orders;")
            return cursor.fetchone()[0]

    @staticmethod
    def get_summary_stats() -> Dict[str, Any]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) AS total_orders,
                    ROUND(SUM(sales), 2) AS total_sales,
                    ROUND(SUM(gross_profit), 2) AS total_profit,
                    ROUND(AVG(lead_time_days), 2) AS avg_lead_time,
                    ROUND(AVG(units), 1) AS avg_units
                FROM orders;
            """)
            row = cursor.fetchone()
            return dict(row) if row else {}

    @staticmethod
    def get_orders(
        limit: int = 100, 
        offset: int = 0,
        product_name: Optional[str] = None,
        factory_name: Optional[str] = None,
        region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT 
                    o.order_row_id, o.order_id, o.order_date, o.ship_date,
                    p.product_name, p.division,
                    f.factory_name, o.city, o.state_province, o.region,
                    c.ship_mode, o.units, o.sales, o.gross_profit, o.cost,
                    o.raw_lead_time_days, o.lead_time_days, o.provenance_tier
                FROM orders o
                JOIN products p ON o.product_id = p.product_id
                JOIN factories f ON o.factory_id = f.factory_id
                JOIN carriers c ON o.carrier_id = c.carrier_id
                WHERE 1=1
            """
            params = []
            if product_name:
                query += " AND p.product_name = ?"
                params.append(product_name)
            if factory_name:
                query += " AND f.factory_name = ?"
                params.append(factory_name)
            if region:
                query += " AND o.region = ?"
                params.append(region)

            query += " ORDER BY o.order_row_id ASC LIMIT ? OFFSET ?;"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def get_filtered_orders(
        cls,
        limit: int = 100,
        offset: int = 0,
        product_name: Optional[str] = None,
        factory_name: Optional[str] = None,
        region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return cls.get_orders(limit=limit, offset=offset, product_name=product_name, factory_name=factory_name, region=region)

    @staticmethod
    def get_factory_volume_distribution() -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    f.factory_name,
                    f.city,
                    f.state,
                    f.region,
                    f.rated_daily_capacity,
                    f.baseline_workload_pct,
                    COUNT(o.order_row_id) AS total_orders,
                    ROUND(SUM(o.sales), 2) AS total_sales,
                    ROUND(AVG(o.lead_time_days), 2) AS mean_lead_time
                FROM factories f
                LEFT JOIN orders o ON f.factory_id = o.factory_id
                GROUP BY f.factory_id
                ORDER BY total_orders DESC;
            """)
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_carrier_lead_time_summary() -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    c.ship_mode,
                    c.sla_target_days,
                    COUNT(o.order_row_id) AS order_count,
                    ROUND(AVG(o.lead_time_days), 2) AS empirical_mean_lead_time,
                    ROUND(MIN(o.lead_time_days), 2) AS min_lead_time,
                    ROUND(MAX(o.lead_time_days), 2) AS max_lead_time
                FROM carriers c
                LEFT JOIN orders o ON c.carrier_id = o.carrier_id
                GROUP BY c.carrier_id
                ORDER BY empirical_mean_lead_time ASC;
            """)
            return [dict(row) for row in cursor.fetchall()]


class ProductRepository:
    """Repository for products master and SKU profile queries."""

    @staticmethod
    def get_all_products() -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    p.product_id, p.product_name, p.division, p.allergen_profile,
                    f.factory_name AS assigned_factory,
                    p.avg_sales_per_unit, p.avg_cost_per_unit, p.avg_profit_per_unit, p.profit_margin_pct
                FROM products p
                JOIN factories f ON p.default_factory_id = f.factory_id
                ORDER BY p.division, p.product_name;
            """)
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_product_by_name(product_name: str) -> Optional[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    p.product_id, p.product_name, p.division, p.allergen_profile,
                    f.factory_name AS assigned_factory,
                    p.avg_sales_per_unit, p.avg_cost_per_unit, p.avg_profit_per_unit, p.profit_margin_pct
                FROM products p
                JOIN factories f ON p.default_factory_id = f.factory_id
                WHERE p.product_name = ?;
            """, (product_name,))
            row = cursor.fetchone()
            return dict(row) if row else None


class FactoryRepository:
    """Repository for factory metadata, capacity, and eligibility queries."""

    @staticmethod
    def get_all_factories() -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    factory_id, factory_name, city, state, region,
                    latitude, longitude, specialty, rated_daily_capacity,
                    baseline_workload_pct, max_feasible_workload_pct, color_hex
                FROM factories
                ORDER BY factory_id ASC;
            """)
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_factory_by_name(factory_name: str) -> Optional[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM factories WHERE factory_name = ?;", (factory_name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_factory_eligibility(product_name: str) -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    f.factory_name, fe.is_eligible, fe.tooling_notes
                FROM factory_eligibility fe
                JOIN factories f ON fe.factory_id = f.factory_id
                JOIN products p ON fe.product_id = p.product_id
                WHERE p.product_name = ?;
            """, (product_name,))
            return [dict(row) for row in cursor.fetchall()]


class RecommendationRepository:
    """Repository for managing ranked reallocation recommendations."""

    @staticmethod
    def get_all_recommendations(category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT 
                    r.recommendation_id, r.created_at,
                    p.product_name, p.division,
                    f_cur.factory_name AS current_factory,
                    f_cand.factory_name AS recommended_factory,
                    r.destination_region, r.operational_issue, r.supporting_evidence,
                    r.predicted_lead_time, r.lead_time_delta, r.distance_reduction_km,
                    r.cost_impact_pct, r.evidential_category, r.confidence_score,
                    r.priority_rank, r.capacity_feasible, r.recommended_action, r.business_explanation
                FROM recommendations r
                JOIN products p ON r.product_id = p.product_id
                JOIN factories f_cur ON r.current_factory_id = f_cur.factory_id
                JOIN factories f_cand ON r.candidate_factory_id = f_cand.factory_id
                WHERE 1=1
            """
            params = []
            if category_filter and category_filter != "All Categories":
                query += " AND r.evidential_category = ?"
                params.append(category_filter)

            query += " ORDER BY r.priority_rank ASC;"
            cursor.execute(query, params)
            raw_rows = [dict(row) for row in cursor.fetchall()]
            return normalize_recommendations_list(raw_rows)

    @staticmethod
    def replace_all_recommendations(recs: List[Dict[str, Any]]):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM recommendations;")
            for r in recs:
                cursor.execute("""
                    INSERT INTO recommendations (
                        recommendation_id, created_at, product_id, current_factory_id, candidate_factory_id,
                        destination_region, operational_issue, supporting_evidence, predicted_lead_time,
                        lead_time_delta, distance_reduction_km, cost_impact_pct, evidential_category,
                        confidence_score, priority_rank, capacity_feasible, recommended_action, business_explanation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    r["recommendation_id"], r["created_at"], r["product_id"], r["current_factory_id"],
                    r["candidate_factory_id"], r["destination_region"], r["operational_issue"],
                    r["supporting_evidence"], r["predicted_lead_time"], r["lead_time_delta"],
                    r["distance_reduction_km"], r["cost_impact_pct"], r["evidential_category"],
                    r["confidence_score"], r["priority_rank"], r["capacity_feasible"],
                    r["recommended_action"], r["business_explanation"]
                ))
            cursor.close()


class ScenarioRepository:
    """Repository for user-saved optimization scenarios."""

    @staticmethod
    def save_scenario(scenario: Dict[str, Any]):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO optimization_scenarios (
                    scenario_id, created_at, product_id, destination_region, ship_mode,
                    batch_units, priority_slider, baseline_factory_id, recommended_factory_id,
                    baseline_lead_time, recommended_lead_time, lead_time_delta, lead_time_reduction_pct,
                    distance_delta_km, profit_impact_pct, recommendation_category, confidence_score, user_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                scenario["scenario_id"], scenario["created_at"], scenario["product_id"],
                scenario["destination_region"], scenario["ship_mode"], scenario["batch_units"],
                scenario["priority_slider"], scenario["baseline_factory_id"], scenario["recommended_factory_id"],
                scenario["baseline_lead_time"], scenario["recommended_lead_time"], scenario["lead_time_delta"],
                scenario["lead_time_reduction_pct"], scenario["distance_delta_km"], scenario["profit_impact_pct"],
                scenario["recommendation_category"], scenario["confidence_score"], scenario.get("user_notes", "")
            ))
            cursor.close()

    @staticmethod
    def get_saved_scenarios(limit: int = 50) -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    s.scenario_id, s.created_at, p.product_name, s.destination_region, s.ship_mode,
                    s.batch_units, s.priority_slider,
                    f_base.factory_name AS current_factory,
                    f_rec.factory_name AS recommended_factory,
                    s.baseline_lead_time, s.recommended_lead_time, s.lead_time_delta,
                    s.lead_time_reduction_pct, s.distance_delta_km, s.profit_impact_pct,
                    s.recommendation_category, s.confidence_score, s.user_notes
                FROM optimization_scenarios s
                JOIN products p ON s.product_id = p.product_id
                JOIN factories f_base ON s.baseline_factory_id = f_base.factory_id
                JOIN factories f_rec ON s.recommended_factory_id = f_rec.factory_id
                ORDER BY s.created_at DESC
                LIMIT ?;
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]


class ModelEvaluationRepository:
    """Repository for querying stored ML evaluation benchmarks."""

    @staticmethod
    def get_all_benchmarks() -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    model_name, parameters_count, holdout_mae, holdout_rmse,
                    holdout_r2, cv_rmse_mean, cv_rmse_std, status_notes, evaluated_at
                FROM model_evaluations
                ORDER BY holdout_rmse ASC;
            """)
            return [dict(row) for row in cursor.fetchall()]


class AuditRepository:
    """Repository for data governance and audit log inspections."""

    @staticmethod
    def get_latest_audit() -> Optional[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM data_audit_log ORDER BY audit_id DESC LIMIT 1;")
            row = cursor.fetchone()
            return dict(row) if row else None
