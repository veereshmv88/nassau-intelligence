"""
Factory Reallocation and Optimization Engine for Nassau Intelligence.
Evaluates counterfactual factory assignments, applies multi-objective scoring,
estimates profit sensitivity, and calculates risk and evidence quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.config import (
    FACTORIES, DEFAULT_PRODUCT_ASSIGNMENTS, PRODUCT_DIVISIONS
)
from src.prediction_service import LeadTimePredictor
from src.data_loader import load_raw_data, prepare_dataset


class OptimizationEngine:
    """
    Supply Chain Decision-Intelligence Engine for Factory Reallocation.
    """
    
    def __init__(self, data_df: Optional[pd.DataFrame] = None):
        self.predictor = LeadTimePredictor.get_instance()
        if data_df is None:
            raw = load_raw_data()
            self.df = prepare_dataset(raw, save_processed=False)
        else:
            self.df = data_df
            
        # Precompute empirical benchmarks for profit sensitivity
        self._compute_product_financial_benchmarks()

    def _compute_product_financial_benchmarks(self):
        """Compute empirical sales, cost, and gross profit per unit for each product."""
        grouped = self.df.groupby('Product Name').agg({
            'Sales': 'sum',
            'Gross Profit': 'sum',
            'Cost': 'sum',
            'Units': 'sum'
        }).reset_index()
        
        self.product_benchmarks = {}
        for _, row in grouped.iterrows():
            units = max(1, row['Units'])
            self.product_benchmarks[row['Product Name']] = {
                "avg_sales_per_unit": round(row['Sales'] / units, 2),
                "avg_profit_per_unit": round(row['Gross Profit'] / units, 2),
                "avg_cost_per_unit": round(row['Cost'] / units, 2),
                "profit_margin_pct": round((row['Gross Profit'] / row['Sales']) * 100, 1)
            }

    def assess_evidence_quality(self, product_name: str, factory_name: str, region: str) -> Dict[str, Any]:
        """
        Assess empirical historical support for the proposed configuration.
        Assigns transparent recommendation categories:
        - Potential opportunity: Favorable under high empirical volume and consistent geographic alignment.
        - Needs validation: Benefit depends on unvalidated transport & capacity assumptions.
        - Insufficient evidence: Low historical order count (< 20 total or < 5 in region).
        - Operationally infeasible: Severe operational penalty (e.g., cross-country distance > 3,800 km).
        """
        # Total product orders in dataset
        prod_total_count = len(self.df[self.df['Product Name'] == product_name])
        
        # Exact product-region count in data
        prod_region_count = len(self.df[
            (self.df['Product Name'] == product_name) & 
            (self.df['Region'] == region)
        ])
        
        # Factory-region count (assigned or observed)
        factory_region_count = len(self.df[
            (self.df['Factory'] == factory_name) & 
            (self.df['Region'] == region)
        ])
        
        # Insufficient evidence condition
        if prod_total_count < 20 or prod_region_count < 5:
            category = "Insufficient evidence"
            quality = "Insufficient evidence"
            confidence_score = 45
            risk_level = "Elevated"
            risk_flag = f"Insufficient historical evidence: only {prod_total_count} total orders ({prod_region_count} in {region}). Cannot support reliable reallocation."
            can_reallocate = False
        elif prod_region_count >= 50 and factory_region_count >= 100:
            category = "Potential opportunity"
            quality = "High"
            confidence_score = 92
            risk_level = "Low"
            risk_flag = "Well-supported by historical regional shipping volume; favorable under stated assumptions."
            can_reallocate = True
        else:
            category = "Needs validation"
            quality = "Medium"
            confidence_score = 78
            risk_level = "Moderate"
            risk_flag = "Moderate volume; predicted benefit depends on unvalidated transport & capacity assumptions."
            can_reallocate = True
            
        return {
            "evidence_quality": quality,
            "recommendation_category": category,
            "confidence_score": confidence_score,
            "risk_level": risk_level,
            "risk_flag": risk_flag,
            "can_reallocate": can_reallocate,
            "prod_total_count": prod_total_count,
            "prod_region_count": prod_region_count,
            "factory_region_count": factory_region_count
        }

    def simulate_optimization(
        self,
        product_name: str,
        destination_region: str,
        ship_mode: str,
        priority_slider: float = 70.0,  # 0: Profit protection, 50: Balanced, 100: Speed
        state_province: str = "",
        units: int = 2
    ) -> Dict[str, Any]:
        """
        Run multi-criteria optimization simulation across candidate factories.
        Returns ranked alternatives, metrics deltas, transparent assumptions,
        and explicit disclaimers: 'Simulated estimate — not yet validated in production.'
        """
        current_factory = DEFAULT_PRODUCT_ASSIGNMENTS.get(product_name, "Sugar Shack")
        
        # Predict baseline
        baseline_lead_time = self.predictor.predict_single(
            product_name=product_name,
            factory_name=current_factory,
            destination_region=destination_region,
            ship_mode=ship_mode,
            state_province=state_province,
            units=units
        )
        
        # Financial benchmarks
        bench = self.product_benchmarks.get(product_name, {
            "avg_sales_per_unit": 10.0,
            "avg_profit_per_unit": 5.0,
            "avg_cost_per_unit": 5.0,
            "profit_margin_pct": 50.0
        })
        
        baseline_profit_per_unit = bench["avg_profit_per_unit"]
        
        # Weights from slider (0 - 100)
        weight_speed = priority_slider / 100.0
        weight_profit = 1.0 - weight_speed
        
        # Evidence check for current product
        current_evidence = self.assess_evidence_quality(product_name, current_factory, destination_region)
        has_sufficient_evidence = current_evidence["can_reallocate"]
        
        factory_evaluations = []
        all_options = self.predictor.evaluate_all_factories(
            product_name=product_name,
            destination_region=destination_region,
            ship_mode=ship_mode,
            state_province=state_province,
            units=units
        )
        
        baseline_distance = [o["distance_km"] for o in all_options if o["factory"] == current_factory][0]
        
        # Standardized documented simulation assumptions
        documented_assumptions = [
            "Carrier Mode SLA is the primary determinant of dispatch lead times (~98% of explainable variation).",
            "Marginal logistics transit sensitivity is assumed at $0.00020 per km per unit (~$0.20 per 1,000 km).",
            "Faster dispatch saves an assumed $0.08 per day per unit in operational inventory holding and warehouse efficiency.",
            "Candidate plant is assumed to possess available production capacity, tooling, and labor for the SKU.",
            "Dispatch lead time represents plant-to-carrier handoff, not door-to-door customer delivery time."
        ]
        
        for opt in all_options:
            f_name = opt["factory"]
            cand_lead_time = opt["predicted_lead_time"]
            distance_km = opt["distance_km"]
            is_cur = (f_name == current_factory)
            
            # Improvement calculation (A. Delta = Baseline - Alternative, B. % = (Delta / Baseline)*100)
            lead_time_delta = round(baseline_lead_time - cand_lead_time, 2)
            if baseline_lead_time > 0:
                lead_time_reduction_pct = round((lead_time_delta / baseline_lead_time) * 100, 1)
            else:
                lead_time_reduction_pct = 0.0
                
            # Profit Sensitivity model under documented assumptions
            dist_savings = (baseline_distance - distance_km) * 0.00020
            speed_savings = max(0, lead_time_delta) * 0.08
            
            est_profit_per_unit = max(0.5, round(baseline_profit_per_unit + dist_savings + speed_savings, 2))
            profit_impact_pct = round(((est_profit_per_unit - baseline_profit_per_unit) / max(0.1, baseline_profit_per_unit)) * 100, 1)
            
            # Evidence quality and recommendation category
            ev = self.assess_evidence_quality(product_name, f_name, destination_region)
            
            # Infeasibility check: extreme cross-country haul > 3,800 km creating massive penalty
            if distance_km > 3800 and lead_time_reduction_pct < -25:
                cat = "Operationally infeasible"
                rationale = "Excessive distance penalty (>3,800 km) impairs dispatch SLAs."
            elif not has_sufficient_evidence:
                cat = "Insufficient evidence"
                rationale = "Insufficient historical orders for this SKU to validate alternative sourcing."
            elif lead_time_delta > 0 and profit_impact_pct > 0 and ev["confidence_score"] >= 85:
                cat = "Potential opportunity"
                rationale = f"Shorter transit route ({abs(distance_km - baseline_distance):.0f} km closer) and faster predicted dispatch under {ship_mode}."
            elif lead_time_delta > 0 or profit_impact_pct > 0:
                cat = "Needs validation"
                rationale = f"Favorable simulated balance; requires plant floor capacity and transport cost validation."
            else:
                cat = "Needs validation"
                rationale = "No operational improvement over current plant assignment."
                
            # Composite optimization score
            speed_score = max(0, min(100, 100 - (cand_lead_time * 12)))
            profit_score = max(0, min(100, 50 + (profit_impact_pct * 1.5)))
            evidence_factor = ev["confidence_score"] / 100.0
            
            # Penalty for insufficient evidence or infeasibility
            if cat == "Insufficient evidence":
                evidence_factor *= 0.5
            elif cat == "Operationally infeasible":
                evidence_factor *= 0.2
                
            composite_score = round(
                (weight_speed * speed_score + weight_profit * profit_score) * evidence_factor, 1
            )
            
            factory_evaluations.append({
                "factory": f_name,
                "current_factory": current_factory,
                "is_current": is_cur,
                "is_recommended": False,
                "predicted_lead_time": cand_lead_time,
                "lead_time_delta": lead_time_delta,
                "lead_time_reduction_pct": lead_time_reduction_pct,
                "distance_km": distance_km,
                "estimated_profit_per_unit": est_profit_per_unit,
                "profit_impact_pct": profit_impact_pct,
                "composite_score": composite_score,
                "recommendation_category": cat,
                "evidence_quality": ev["evidence_quality"],
                "confidence_score": ev["confidence_score"],
                "risk_level": ev["risk_level"],
                "risk_flag": ev["risk_flag"],
                "rationale": rationale,
                "simulation_label": "Simulated estimate — not yet validated in production",
                "city": opt["city"],
                "state": opt["state"],
                "specialty": opt["specialty"],
                "color": opt["color"]
            })
            
        # Sort by composite score descending
        factory_evaluations.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # Mark champion
        best_option = factory_evaluations[0]
        best_option["is_recommended"] = True
        
        return {
            "product_name": product_name,
            "destination_region": destination_region,
            "ship_mode": ship_mode,
            "priority_slider": priority_slider,
            "baseline_factory": current_factory,
            "baseline_lead_time": baseline_lead_time,
            "baseline_profit_per_unit": baseline_profit_per_unit,
            "recommended_factory": best_option["factory"],
            "recommendation_category": best_option["recommendation_category"],
            "has_sufficient_evidence": has_sufficient_evidence,
            "expected_lead_time_reduction_pct": best_option["lead_time_reduction_pct"],
            "expected_profit_increase_pct": best_option["profit_impact_pct"],
            "confidence_score": best_option["confidence_score"],
            "factories": factory_evaluations,
            "assumptions": documented_assumptions,
            "limitation_statement": (
                "Alternative-factory benefits are simulated estimates, not verified operational outcomes. "
                "Historical product assignments do not establish what would happen if a product were manufactured at a different factory. "
                "Validate promising scenarios using factory capacity, production compatibility, actual transportation costs, and a controlled operational pilot."
            )
        }

    def generate_all_recommendations(
        self,
        priority_slider: float = 70.0
    ) -> List[Dict[str, Any]]:
        """
        Generate ranked system-wide factory reassignment recommendations
        across all products and destination regions with transparent categorization.
        """
        regions = self.df['Region'].unique().tolist()
        products = list(DEFAULT_PRODUCT_ASSIGNMENTS.keys())
        default_ship_mode = "Standard Class"
        
        recs = []
        for prod in products:
            for reg in regions:
                sim = self.simulate_optimization(
                    product_name=prod,
                    destination_region=reg,
                    ship_mode=default_ship_mode,
                    priority_slider=priority_slider
                )
                
                # Check top alternative
                best = sim["factories"][0]
                
                # Only include when not current factory
                if not best["is_current"]:
                    lt_pct = best["lead_time_reduction_pct"]
                    lt_delta_str = f"-{lt_pct:.1f}%" if lt_pct >= 0 else f"+{abs(lt_pct):.1f}%"
                    prof_pct = best["profit_impact_pct"]
                    prof_impact_str = f"+{prof_pct:.1f}%" if prof_pct >= 0 else f"{prof_pct:.1f}%"
                    
                    recs.append({
                        "Product Name": prod,
                        "Division": PRODUCT_DIVISIONS.get(prod, "Other"),
                        "Region": reg,
                        "Current Factory": sim["baseline_factory"],
                        "Recommended Factory": sim["recommended_factory"],
                        "Baseline Lead Time": f"{sim['baseline_lead_time']} days",
                        "Alternative Lead Time": f"{best['predicted_lead_time']} days",
                        "Lead Time Delta": lt_delta_str,
                        "Profit Impact": prof_impact_str,
                        "Confidence": f"{best['confidence_score']}%",
                        "Category": best["recommendation_category"],
                        "Evidence Status": best["evidence_quality"],
                        "Risk": best["risk_level"],
                        "Rationale": best["rationale"],
                        "Composite Score": best["composite_score"],
                        "Status Label": "Simulated estimate — not yet validated in production",
                        "raw_lead_time_reduction": best["lead_time_reduction_pct"],
                        "raw_profit_increase": best["profit_impact_pct"]
                    })
                    
        # Sort recommendations: Potential opportunity first, then Needs validation, then Insufficient evidence
        category_order = {
            "Potential opportunity": 3,
            "Needs validation": 2,
            "Insufficient evidence": 1,
            "Operationally infeasible": 0
        }
        recs.sort(key=lambda x: (category_order.get(x["Category"], 0), x["Composite Score"]), reverse=True)
        return recs
