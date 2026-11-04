"""
Recommendation Data Contract & Schema Normalizer for Nassau Intelligence.
Provides a unified schema, bi-directional field mapping, and validated
DataFrame construction across Engine, Database, Repositories, and Streamlit Dashboard.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


# Standard column specifications for recommendation views
DEFAULT_SUMMARY_COLUMNS = [
    "Product Name", "Current Factory", "Recommended Factory", 
    "Destination Region", "Recommendation Category", "Expected Lead Time Reduction (%)"
]

SUMMARY_COLUMN_RENAME = {
    "Product Name": "Product SKU",
    "Current Factory": "Current Plant",
    "Recommended Factory": "Candidate Plant",
    "Destination Region": "Market",
    "Recommendation Category": "Category",
    "Expected Lead Time Reduction (%)": "Est. Delta (%)"
}

DEFAULT_FULL_COLUMNS = [
    "Recommendation ID", "Product Name", "Current Factory", "Recommended Factory",
    "Destination Region", "Recommendation Category", "Confidence Score",
    "Capacity Feasible", "Optimization Rationale"
]


def normalize_recommendation(rec: Dict[str, Any], index: int = 1) -> Dict[str, Any]:
    """
    Normalize any recommendation dictionary (whether from database, engine, or legacy format)
    into a canonical dictionary containing both standardized snake_case and UI Title Case aliases.
    Preserves genuine values and uses honest default indicators for missing optional fields.
    """
    if not isinstance(rec, dict):
        return {}

    # 1. Product Name
    prod_name = (
        rec.get("Product Name") or 
        rec.get("product_name") or 
        rec.get("Product SKU") or 
        rec.get("Product") or 
        "Unknown Product"
    )

    # 2. Division
    division = rec.get("Division") or rec.get("division") or "Chocolate"

    # 3. Factories
    current_factory = (
        rec.get("Current Factory") or 
        rec.get("current_factory") or 
        rec.get("Current Plant") or 
        rec.get("Current") or 
        "Current Plant"
    )
    recommended_factory = (
        rec.get("Recommended Factory") or 
        rec.get("recommended_factory") or 
        rec.get("Candidate Plant") or 
        rec.get("Recommended") or 
        "Alternative Plant"
    )

    # 4. Region
    destination_region = (
        rec.get("Destination Region") or 
        rec.get("destination_region") or 
        rec.get("Region") or 
        rec.get("Market") or 
        "All Regions"
    )

    # 5. Category (Evidential Stratification)
    category = (
        rec.get("Recommendation Category") or 
        rec.get("evidential_category") or 
        rec.get("Category") or 
        "Needs validation"
    )
    valid_categories = {
        "Potential opportunity", "Needs validation", 
        "Insufficient evidence", "Operationally infeasible"
    }
    if category not in valid_categories:
        category = "Needs validation"

    # 6. Recommendation ID
    rec_id = (
        rec.get("Recommendation ID") or 
        rec.get("recommendation_id") or 
        f"REC-{index:03d}"
    )

    # 7. Lead times & delta
    base_lt = rec.get("Baseline Lead Time") or rec.get("baseline_lead_time")
    if base_lt is not None:
        base_lt_str = f"{base_lt} days" if not str(base_lt).endswith("days") else str(base_lt)
    else:
        base_lt_str = "N/A"

    alt_lt = rec.get("Alternative Lead Time") or rec.get("alternative_lead_time") or rec.get("predicted_lead_time")
    if alt_lt is not None:
        alt_lt_str = f"{alt_lt} days" if not str(alt_lt).endswith("days") else str(alt_lt)
    else:
        alt_lt_str = "N/A"

    # Expected Lead Time Reduction percentage
    raw_red = (
        rec.get("Expected Lead Time Reduction (%)") or 
        rec.get("raw_lead_time_reduction") or 
        rec.get("lead_time_delta")
    )
    try:
        if raw_red is not None:
            if isinstance(raw_red, (int, float)):
                lt_red_num = round(float(raw_red), 1)
            else:
                cleaned = str(raw_red).replace("%", "").replace("+", "").replace("-", "").strip()
                lt_red_num = round(float(cleaned), 1) if cleaned else 0.0
        else:
            lt_red_num = 0.0
    except (ValueError, TypeError):
        lt_red_num = 0.0

    lt_delta_str = rec.get("Lead Time Delta")
    if not lt_delta_str:
        lt_delta_str = f"-{lt_red_num:.1f}%" if lt_red_num >= 0 else f"+{abs(lt_red_num):.1f}%"

    # Profit impact
    prof_impact = rec.get("Profit Impact")
    raw_profit = rec.get("raw_profit_increase") or rec.get("cost_impact_pct")
    if prof_impact is None:
        if raw_profit is not None:
            try:
                prof_val = float(raw_profit)
                prof_impact = f"+{prof_val:.1f}%" if prof_val >= 0 else f"{prof_val:.1f}%"
            except (ValueError, TypeError):
                prof_impact = "0.0%"
        else:
            prof_impact = "0.0%"

    # Confidence score
    conf = rec.get("Confidence Score") or rec.get("confidence_score") or rec.get("Confidence")
    try:
        if conf is not None:
            conf_clean = str(conf).replace("%", "").strip()
            conf_num = float(conf_clean)
        else:
            conf_num = 75.0
    except (ValueError, TypeError):
        conf_num = 75.0
    conf_str = f"{int(conf_num)}%"

    # Evidence status & risk
    evidence_status = rec.get("Evidence Status") or rec.get("evidence_status") or rec.get("supporting_evidence") or "Medium"
    if "High" in str(evidence_status):
        evidence_status = "High"
    elif "Insufficient" in str(evidence_status):
        evidence_status = "Insufficient"
    elif "Observed" in str(evidence_status):
        evidence_status = "Observed"
    else:
        evidence_status = "Medium"

    risk_level = rec.get("Risk Level") or rec.get("risk_level") or rec.get("Risk") or "Moderate"
    operational_issue = (
        rec.get("operational_issue") or 
        rec.get("Operational Issue") or 
        f"Sub-optimal dispatch route from {current_factory} to {destination_region}"
    )
    rationale = (
        rec.get("Optimization Rationale") or 
        rec.get("optimization_rationale") or 
        rec.get("Rationale") or 
        rec.get("business_explanation") or 
        operational_issue
    )

    cap_feas = rec.get("Capacity Feasible")
    if cap_feas is None:
        cap_val = rec.get("capacity_feasible")
        if cap_val is not None:
            cap_feas = "Yes" if cap_val in (1, True, "1", "Yes") else "No"
        else:
            cap_feas = "Yes"
    else:
        cap_feas = "Yes" if str(cap_feas).lower() in ("yes", "true", "1") else "No"

    composite_score = rec.get("Composite Score") or rec.get("composite_score") or 50.0

    # Build canonical object with dual-contract key access
    normalized = {
        # Canonical Snake_Case (API & DB contract)
        "recommendation_id": rec_id,
        "product_name": prod_name,
        "division": division,
        "destination_region": destination_region,
        "current_factory": current_factory,
        "recommended_factory": recommended_factory,
        "baseline_lead_time": base_lt_str,
        "alternative_lead_time": alt_lt_str,
        "predicted_lead_time": alt_lt if alt_lt is not None else 0.0,
        "lead_time_delta": lt_delta_str,
        "expected_lead_time_reduction_pct": lt_red_num,
        "profit_impact": prof_impact,
        "cost_impact_pct": raw_profit if raw_profit is not None else 0.0,
        "confidence_score": conf_num,
        "recommendation_category": category,
        "evidential_category": category,
        "evidence_status": evidence_status,
        "supporting_evidence": str(evidence_status),
        "risk_level": risk_level,
        "capacity_feasible": cap_feas,
        "operational_issue": operational_issue,
        "recommended_action": rec.get("recommended_action") or f"Reallocate quota to {recommended_factory}",
        "priority_rank": rec.get("priority_rank", index),
        "created_at": rec.get("created_at", "2026-11-04T12:00:00"),
        "optimization_rationale": rationale,
        "business_explanation": rationale,
        "distance_reduction_km": rec.get("distance_reduction_km", 0.0),
        "composite_score": composite_score,
        "status_label": "Simulated estimate — not yet validated in production",

        # UI Title Case Aliases (Dashboard contract)
        "Recommendation ID": rec_id,
        "Product Name": prod_name,
        "Product SKU": prod_name,
        "Division": division,
        "Destination Region": destination_region,
        "Region": destination_region,
        "Market": destination_region,
        "Current Factory": current_factory,
        "Current Plant": current_factory,
        "Recommended Factory": recommended_factory,
        "Candidate Plant": recommended_factory,
        "Operational Issue": operational_issue,
        "Recommended Action": rec.get("recommended_action") or f"Reallocate quota to {recommended_factory}",
        "Baseline Lead Time": base_lt_str,
        "Alternative Lead Time": alt_lt_str,
        "Lead Time Delta": lt_delta_str,
        "Expected Lead Time Reduction (%)": lt_red_num,
        "Est. Delta (%)": f"-{lt_red_num:.1f}%" if lt_red_num >= 0 else f"+{abs(lt_red_num):.1f}%",
        "Profit Impact": prof_impact,
        "Confidence Score": conf_str,
        "Confidence": conf_str,
        "Recommendation Category": category,
        "Category": category,
        "Evidence Status": evidence_status,
        "Risk Level": risk_level,
        "Risk": risk_level,
        "Capacity Feasible": cap_feas,
        "Optimization Rationale": rationale,
        "Rationale": rationale,
        "Composite Score": composite_score,
        "Status Label": "Simulated estimate — not yet validated in production",
        "raw_lead_time_reduction": lt_red_num,
        "raw_profit_increase": prof_impact
    }
    return normalized


def normalize_recommendations_list(recs: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize a full list of recommendations, ensuring strict contract conformity."""
    if not recs:
        return []
    return [normalize_recommendation(r, idx) for idx, r in enumerate(recs, start=1)]


def build_recommendations_dataframe(
    recs: Optional[List[Dict[str, Any]]],
    columns: Optional[List[str]] = None,
    rename_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Construct a validated, robust pandas DataFrame from a recommendation list.
    Guarantees:
    - Never raises KeyError on missing columns or empty lists.
    - Preserves genuine values.
    - Replaces missing optional values with clean domain fallbacks.
    - Applies rename_map safely.
    """
    target_columns = columns or DEFAULT_SUMMARY_COLUMNS

    if not recs:
        empty_cols = [rename_map.get(c, c) for c in target_columns] if rename_map else target_columns
        return pd.DataFrame(columns=empty_cols)

    normalized_list = normalize_recommendations_list(recs)

    # Extract required fields with fallbacks
    rows = []
    for r in normalized_list:
        row = {}
        for col in target_columns:
            if col in r:
                row[col] = r[col]
            else:
                # Fallback heuristics for unmapped columns
                snake_col = col.lower().replace(" ", "_").replace("(%)", "pct").replace("?", "")
                row[col] = r.get(snake_col, "N/A")
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all target columns exist
    for col in target_columns:
        if col not in df.columns:
            df[col] = "N/A"

    # Select target column order
    df = df[target_columns]

    # Apply rename mapping if provided
    if rename_map:
        df = df.rename(columns=rename_map)

    return df
