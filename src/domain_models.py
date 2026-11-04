"""
Typed Domain Models for Nassau Intelligence.
Provides validated, strongly-typed data structures for:
1. OrderRecord
2. ProductRecord
3. FactoryRecord
4. PredictionResult
5. FactoryEvaluation
6. OptimizationScenario
7. Recommendation
8. ModelEvaluation
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class OrderRecord:
    """Historical shipping order transaction."""
    order_row_id: int
    order_id: str
    order_date: str
    ship_date: str
    customer_id: int
    city: str
    state_province: str
    region: str
    postal_code: str
    division: str
    product_id: int
    product_name: str
    carrier_id: int
    ship_mode: str
    factory_id: int
    factory_name: str
    units: int
    sales: float
    gross_profit: float
    cost: float
    lead_time_days: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProductRecord:
    """Product master record with unit financial economics."""
    product_id: int
    product_name: str
    division: str
    allergen_profile: str
    assigned_factory: str
    avg_sales_per_unit: float = 0.0
    avg_cost_per_unit: float = 0.0
    avg_profit_per_unit: float = 0.0
    profit_margin_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactoryRecord:
    """Manufacturing facility profile, capacity, and spatial coordinates."""
    factory_id: int
    factory_name: str
    city: str
    state: str
    region: str
    latitude: float
    longitude: float
    specialty: str
    rated_daily_capacity: int = 30
    baseline_workload_pct: float = 70.0
    max_feasible_workload_pct: float = 85.0
    color_hex: str = "#087F5B"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionResult:
    """Single ML prediction with empirical prediction intervals."""
    product_name: str
    factory_name: str
    destination_region: str
    ship_mode: str
    point_prediction: float
    interval_lower: float
    interval_upper: float
    confidence_level: str = "90%"
    interval_str: str = ""
    distance_km: float = 0.0

    def __post_init__(self):
        if not self.interval_str:
            self.interval_str = f"{self.point_prediction:.1f}d [{self.interval_lower:.1f} – {self.interval_upper:.1f}d {self.confidence_level} PI]"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactoryEvaluation:
    """Multi-criteria evaluation for a candidate factory configuration."""
    factory: str
    current_factory: str
    is_current: bool
    is_recommended: bool
    predicted_lead_time: float
    interval_lower: float
    interval_upper: float
    interval_str: str
    lead_time_delta: float
    lead_time_reduction_pct: float
    distance_km: float
    distance_delta_km: float
    estimated_profit_per_unit: float
    profit_impact_pct: float
    speed_score: float
    profit_score: float
    evidence_factor: float
    composite_score: float
    workload_pct: int
    capacity_feasible: bool
    capacity_status: str
    tooling_eligible: bool
    tooling_note: str
    recommendation_category: str
    evidence_quality: str
    confidence_score: float
    risk_level: str
    risk_flag: str
    rationale: str
    selection_justification: str
    city: str = ""
    state: str = ""
    specialty: str = ""
    color: str = "#087F5B"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationScenario:
    """Persisted user what-if simulation scenario."""
    scenario_id: str
    created_at: str
    product_id: int
    product_name: str
    destination_region: str
    ship_mode: str
    batch_units: int
    priority_slider: float
    baseline_factory: str
    recommended_factory: str
    baseline_lead_time: float
    recommended_lead_time: float
    lead_time_delta: float
    lead_time_reduction_pct: float
    distance_delta_km: float
    profit_impact_pct: float
    recommendation_category: str
    confidence_score: float
    user_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Recommendation:
    """Intelligent operational factory reallocation proposal."""
    recommendation_id: str
    product_id: int
    product_name: str
    division: str
    destination_region: str
    current_factory: str
    recommended_factory: str
    recommendation_category: str
    predicted_current_lead_time: float
    predicted_alternative_lead_time: float
    expected_lead_time_reduction: float
    expected_lead_time_reduction_pct: float
    distance_km: float
    distance_delta_km: float
    workload_pct: int
    capacity_feasible: bool
    evidence_level: str
    operational_risk: str
    priority_score: float
    explanation: str
    assumptions: List[str] = field(default_factory=list)
    status_label: str = "Simulated estimate — not yet validated in production"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to frontend dual-contract dictionary with formatted aliases."""
        lt_delta_str = f"-{self.expected_lead_time_reduction_pct:.1f}%" if self.expected_lead_time_reduction_pct >= 0 else f"+{abs(self.expected_lead_time_reduction_pct):.1f}%"
        return {
            "Recommendation ID": self.recommendation_id,
            "Product Name": self.product_name,
            "Product SKU": self.product_name,
            "Division": self.division,
            "Destination Region": self.destination_region,
            "Region": self.destination_region,
            "Market": self.destination_region,
            "Current Factory": self.current_factory,
            "Current Plant": self.current_factory,
            "Recommended Factory": self.recommended_factory,
            "Candidate Plant": self.recommended_factory,
            "Baseline Lead Time": f"{self.predicted_current_lead_time:.2f} days",
            "Alternative Lead Time": f"{self.predicted_alternative_lead_time:.2f} days",
            "Lead Time Delta": lt_delta_str,
            "Expected Lead Time Reduction (%)": self.expected_lead_time_reduction_pct,
            "Est. Delta (%)": lt_delta_str,
            "Confidence Score": f"{int(self.priority_score)}%",
            "Confidence": f"{int(self.priority_score)}%",
            "Recommendation Category": self.recommendation_category,
            "Category": self.recommendation_category,
            "Evidence Status": self.evidence_level,
            "Risk Level": self.operational_risk,
            "Risk": self.operational_risk,
            "Capacity Feasible": "Yes" if self.capacity_feasible else "No",
            "Optimization Rationale": self.explanation,
            "Rationale": self.explanation,
            "Composite Score": self.priority_score,
            "Status Label": self.status_label,
            # Snake_case keys for API compatibility
            "recommendation_id": self.recommendation_id,
            "product_name": self.product_name,
            "destination_region": self.destination_region,
            "current_factory": self.current_factory,
            "recommended_factory": self.recommended_factory,
            "recommendation_category": self.recommendation_category,
            "evidential_category": self.recommendation_category,
            "expected_lead_time_reduction_pct": self.expected_lead_time_reduction_pct,
            "confidence_score": self.priority_score,
            "capacity_feasible": self.capacity_feasible,
            "optimization_rationale": self.explanation,
            "business_explanation": self.explanation,
            "distance_km": self.distance_km,
            "distance_delta_km": self.distance_delta_km,
            "workload_pct": self.workload_pct,
            "operational_risk": self.operational_risk
        }


@dataclass
class ModelEvaluation:
    """Persisted machine learning evaluation diagnostics."""
    model_name: str
    parameters_count: int
    holdout_mae: float
    holdout_rmse: float
    holdout_r2: float
    cv_rmse_mean: float
    cv_rmse_std: float
    status_notes: str
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
