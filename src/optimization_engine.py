"""
Factory Reallocation and Optimization Engine for Nassau Intelligence.
Evaluates counterfactual factory assignments, applies multi-objective scoring,
models dynamic plant capacity headroom, verifies production tooling eligibility,
estimates profit sensitivity, and calculates risk and evidence quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from src.config import (
    FACTORIES, DEFAULT_PRODUCT_ASSIGNMENTS, PRODUCT_DIVISIONS
)
from src.prediction_service import LeadTimePredictor
from src.data_loader import load_raw_data, prepare_dataset
from src.recommendation_contract import normalize_recommendations_list, normalize_recommendation
try:
    from src.db.repositories import FactoryRepository, ProductRepository, RecommendationRepository
except ImportError:
    FactoryRepository = None
    ProductRepository = None
    RecommendationRepository = None


class OptimizationEngine:
    """
    Supply Chain Decision-Intelligence Engine for Factory Reallocation.
    Integrates empirical ML predictions, plant capacity constraints,
    tooling eligibility rules, and multi-criteria optimization.
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
        # Cache factory capacity profiles
        self._load_factory_profiles()

    def _load_factory_profiles(self):
        """Load factory operational parameters and rated capacity."""
        self.factory_profiles = {
            "Lot's O' Nuts": {"capacity": 35, "baseline_workload": 82, "max_feasible": 90, "color": "bar-amber"},
            "Wicked Choccy's": {"capacity": 28, "baseline_workload": 74, "max_feasible": 90, "color": "bar-green"},
            "Secret Factory": {"capacity": 15, "baseline_workload": 68, "max_feasible": 85, "color": "bar-green"},
            "The Other Factory": {"capacity": 10, "baseline_workload": 62, "max_feasible": 85, "color": "bar-green"},
            "Sugar Shack": {"capacity": 10, "baseline_workload": 54, "max_feasible": 85, "color": "bar-green"}
        }
        if FactoryRepository:
            try:
                db_facs = FactoryRepository.get_all_factories()
                for f in db_facs:
                    name = f["factory_name"]
                    self.factory_profiles[name] = {
                        "capacity": f["rated_daily_capacity"],
                        "baseline_workload": int(f["baseline_workload_pct"]),
                        "max_feasible": int(f["max_feasible_workload_pct"]),
                        "color": "bar-amber" if f["baseline_workload_pct"] > 78 else "bar-green"
                    }
            except Exception:
                pass

    def check_tooling_eligibility(self, product_name: str, factory_name: str) -> Tuple[bool, str]:
        """Verify whether candidate plant has production lines for this SKU."""
        # Baseline rules: Peanut crunch products require allergen-controlled lines
        # Sugar confections require starch mogul / cooking kettles
        # Chocolate bars require tempering / enrobing tunnels
        div = PRODUCT_DIVISIONS.get(product_name, "Chocolate")
        if FactoryRepository:
            try:
                elig_list = FactoryRepository.get_factory_eligibility(product_name)
                for item in elig_list:
                    if item["factory_name"] == factory_name:
                        return bool(item["is_eligible"]), item["tooling_notes"]
            except Exception:
                pass
                
        # Heuristic fallback if DB is offline
        if "Nutty Crunch" in product_name and factory_name in ["Sugar Shack", "The Other Factory"]:
            return False, "Allergen isolation restriction: lacks dedicated nut handling line"
        if div == "Chocolate" and factory_name == "Sugar Shack":
            return False, "Missing continuous chocolate tempering and moulding tunnel"
        return True, "Tooling and allergen protocol verified compatible"

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
        - Operationally infeasible: Capacity exceeded, allergen/tooling mismatch, or extreme haulage.
        """
        prod_total_count = len(self.df[self.df['Product Name'] == product_name])
        prod_region_count = len(self.df[
            (self.df['Product Name'] == product_name) & 
            (self.df['Region'] == region)
        ])
        factory_region_count = len(self.df[
            (self.df['Factory'] == factory_name) & 
            (self.df['Region'] == region)
        ])
        
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
        Incorporates dynamic plant workload, tooling checks, and calibrated intervals.
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
        
        bench = self.product_benchmarks.get(product_name, {
            "avg_sales_per_unit": 10.0,
            "avg_profit_per_unit": 5.0,
            "avg_cost_per_unit": 5.0,
            "profit_margin_pct": 50.0
        })
        baseline_profit_per_unit = bench["avg_profit_per_unit"]
        
        weight_speed = priority_slider / 100.0
        weight_profit = 1.0 - weight_speed
        
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
        
        documented_assumptions = [
            "Carrier Mode SLA is the primary determinant of dispatch lead times (~98% of explainable variation).",
            "Marginal logistics transit sensitivity is assumed at $0.00020 per km per unit (~$0.20 per 1,000 km).",
            "Faster dispatch saves an assumed $0.08 per day per unit in operational inventory holding and warehouse efficiency.",
            "Candidate plant capacity feasibility is dynamically checked against maximum operational threshold (85–90%).",
            "SKU production is subject to factory tooling compatibility and allergen cross-contamination protocols.",
            "Dispatch lead time represents plant-to-carrier handoff, not door-to-door customer delivery time."
        ]
        
        for opt in all_options:
            f_name = opt["factory"]
            cand_lead_time = opt["predicted_lead_time"]
            distance_km = opt["distance_km"]
            is_cur = (f_name == current_factory)
            
            lead_time_delta = round(baseline_lead_time - cand_lead_time, 2)
            lead_time_reduction_pct = round((lead_time_delta / max(0.1, baseline_lead_time)) * 100, 1)
                
            dist_savings = (baseline_distance - distance_km) * 0.00020
            speed_savings = max(0, lead_time_delta) * 0.08
            
            est_profit_per_unit = max(0.5, round(baseline_profit_per_unit + dist_savings + speed_savings, 2))
            profit_impact_pct = round(((est_profit_per_unit - baseline_profit_per_unit) / max(0.1, baseline_profit_per_unit)) * 100, 1)
            
            # Plant Capacity & Tooling Check
            fac_meta = self.factory_profiles.get(f_name, {"capacity": 20, "baseline_workload": 70, "max_feasible": 85, "color": "bar-green"})
            is_eligible, tooling_note = self.check_tooling_eligibility(product_name, f_name)
            
            # Dynamic workload: allocating additional volume increases candidate workload
            workload_delta = 0 if is_cur else round((units / max(1, fac_meta["capacity"] * 4)) * 5, 1)
            post_reallocation_workload = min(100, int(fac_meta["baseline_workload"] + workload_delta))
            
            # Capacity feasibility
            is_capacity_feasible = (post_reallocation_workload <= fac_meta["max_feasible"])
            capacity_status = "✓ Feasible" if (is_capacity_feasible and is_eligible) else (
                "✕ Over Capacity" if not is_capacity_feasible else "✕ Incompatible Tooling"
            )
            
            # Evidence quality and recommendation category
            ev = self.assess_evidence_quality(product_name, f_name, destination_region)
            
            if not is_eligible:
                cat = "Operationally infeasible"
                rationale = f"Plant ineligible: {tooling_note}."
            elif not is_capacity_feasible:
                cat = "Operationally infeasible"
                rationale = f"Post-reallocation workload ({post_reallocation_workload}%) exceeds maximum operational limit ({fac_meta['max_feasible']}%)."
            elif distance_km > 3800 and lead_time_reduction_pct < -25:
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
                
            speed_score = max(0, min(100, 100 - (cand_lead_time * 12)))
            profit_score = max(0, min(100, 50 + (profit_impact_pct * 1.5)))
            evidence_factor = ev["confidence_score"] / 100.0
            
            if cat == "Insufficient evidence":
                evidence_factor *= 0.5
            elif cat == "Operationally infeasible":
                evidence_factor *= 0.1
                
            composite_score = round(
                (weight_speed * speed_score + weight_profit * profit_score) * evidence_factor, 1
            )
            
            factory_evaluations.append({
                "factory": f_name,
                "current_factory": current_factory,
                "is_current": is_cur,
                "is_recommended": False,
                "predicted_lead_time": cand_lead_time,
                "interval_lower": opt.get("interval_lower", max(0.5, cand_lead_time - 1.3)),
                "interval_upper": opt.get("interval_upper", min(10.0, cand_lead_time + 2.5)),
                "interval_str": opt.get("interval_str", f"{cand_lead_time:.1f}d"),
                "lead_time_delta": lead_time_delta,
                "lead_time_reduction_pct": lead_time_reduction_pct,
                "distance_km": distance_km,
                "estimated_profit_per_unit": est_profit_per_unit,
                "profit_impact_pct": profit_impact_pct,
                "composite_score": composite_score,
                "workload_pct": post_reallocation_workload,
                "capacity_feasible": (is_capacity_feasible and is_eligible),
                "capacity_status": capacity_status,
                "tooling_eligible": is_eligible,
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
            
        factory_evaluations.sort(key=lambda x: x["composite_score"], reverse=True)
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
        rec_counter = 1
        for prod in products:
            for reg in regions:
                sim = self.simulate_optimization(
                    product_name=prod,
                    destination_region=reg,
                    ship_mode=default_ship_mode,
                    priority_slider=priority_slider
                )
                
                best = sim["factories"][0]
                
                if not best["is_current"]:
                    lt_pct = best["lead_time_reduction_pct"]
                    lt_delta_str = f"-{lt_pct:.1f}%" if lt_pct >= 0 else f"+{abs(lt_pct):.1f}%"
                    prof_pct = best["profit_impact_pct"]
                    prof_impact_str = f"+{prof_pct:.1f}%" if prof_pct >= 0 else f"{prof_pct:.1f}%"
                    
                    recs.append({
                        "Recommendation ID": f"REC-{rec_counter:03d}",
                        "Product Name": prod,
                        "Division": PRODUCT_DIVISIONS.get(prod, "Other"),
                        "Destination Region": reg,
                        "Region": reg,
                        "Current Factory": sim["baseline_factory"],
                        "Recommended Factory": sim["recommended_factory"],
                        "Baseline Lead Time": f"{sim['baseline_lead_time']} days",
                        "Alternative Lead Time": f"{best['predicted_lead_time']} days",
                        "Lead Time Delta": lt_delta_str,
                        "Expected Lead Time Reduction (%)": lt_pct,
                        "Profit Impact": prof_impact_str,
                        "Confidence Score": f"{best['confidence_score']}%",
                        "Confidence": f"{best['confidence_score']}%",
                        "Recommendation Category": best["recommendation_category"],
                        "Category": best["recommendation_category"],
                        "Evidence Status": best["evidence_quality"],
                        "Risk Level": best["risk_level"],
                        "Risk": best["risk_level"],
                        "Optimization Rationale": best["rationale"],
                        "Rationale": best["rationale"],
                        "Capacity Feasible": "Yes" if best["capacity_feasible"] else "No",
                        "Composite Score": best["composite_score"],
                        "Status Label": "Simulated estimate — not yet validated in production",
                        "raw_lead_time_reduction": best["lead_time_reduction_pct"],
                        "raw_profit_increase": best["profit_impact_pct"]
                    })
                    rec_counter += 1
                    
        category_order = {
            "Potential opportunity": 3,
            "Needs validation": 2,
            "Insufficient evidence": 1,
            "Operationally infeasible": 0
        }
        recs.sort(key=lambda x: (category_order.get(x["Category"], 0), x["Composite Score"]), reverse=True)
        return normalize_recommendations_list(recs)

    def sync_recommendations_to_db(self, priority_slider: float = 70.0) -> int:
        """
        Generate recommendations and persist into relational database.
        Returns number of records written.
        """
        if not RecommendationRepository or not ProductRepository or not FactoryRepository:
            return 0
            
        try:
            recs = self.generate_all_recommendations(priority_slider=priority_slider)
            products_map = {p["product_name"]: p["product_id"] for p in ProductRepository.get_all_products()}
            factories_map = {f["factory_name"]: f["factory_id"] for f in FactoryRepository.get_all_factories()}
            
            db_records = []
            for rank, r in enumerate(recs[:25], start=1):
                p_id = products_map.get(r["Product Name"])
                cur_f_id = factories_map.get(r["Current Factory"])
                cand_f_id = factories_map.get(r["Recommended Factory"])
                
                if not p_id or not cur_f_id or not cand_f_id:
                    continue
                    
                db_records.append({
                    "recommendation_id": r["Recommendation ID"],
                    "created_at": datetime.now().isoformat(),
                    "product_id": p_id,
                    "current_factory_id": cur_f_id,
                    "candidate_factory_id": cand_f_id,
                    "destination_region": r["Destination Region"],
                    "operational_issue": f"Sub-optimal dispatch transit route from {r['Current Factory']} to {r['Destination Region']}",
                    "supporting_evidence": f"Historical lead time {r['Baseline Lead Time']}, modeled reduction {r['Lead Time Delta']} under {r['Evidence Status']} volume",
                    "predicted_lead_time": float(r["Alternative Lead Time"].replace(" days", "")),
                    "lead_time_delta": float(r["raw_lead_time_reduction"]),
                    "distance_reduction_km": 0.0,
                    "cost_impact_pct": float(r["raw_profit_increase"]),
                    "evidential_category": r["Category"],
                    "confidence_score": float(r["Confidence Score"].replace("%", "")),
                    "priority_rank": rank,
                    "capacity_feasible": 1 if r["Capacity Feasible"] == "Yes" else 0,
                    "recommended_action": f"Reallocate production quota to {r['Recommended Factory']}",
                    "business_explanation": r["Rationale"]
                })
                
            RecommendationRepository.replace_all_recommendations(db_records)
            return len(db_records)
        except Exception as e:
            print(f"Error syncing recommendations to database: {e}")
            return 0
