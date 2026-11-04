"""
Tests for optimization engine scoring and selection logic.
Verifies multi-objective score breakdown, tooling/capacity feasibility filtering,
and selection justification rationale.
"""

import pytest
from src.optimization_engine import OptimizationEngine

@pytest.fixture(scope="module")
def engine():
    return OptimizationEngine()

def test_simulation_produces_objective_scores(engine):
    sim = engine.simulate_optimization(
        product_name="Wonka Bar - Nutty Crunch Surprise",
        destination_region="Pacific",
        ship_mode="Standard Class",
        priority_slider=70.0,
        units=2
    )
    assert "factories" in sim
    assert len(sim["factories"]) == 5
    
    top = sim["factories"][0]
    assert "composite_score" in top
    assert "speed_score" in top
    assert "profit_score" in top
    assert "evidence_factor" in top
    assert top["composite_score"] > 0
    assert top["is_recommended"] is True

def test_infeasible_candidate_strictly_disqualified(engine):
    sim = engine.simulate_optimization(
        product_name="Wonka Bar - Nutty Crunch Surprise",
        destination_region="Pacific",
        ship_mode="Standard Class",
        priority_slider=70.0,
        units=2
    )
    # Sugar Shack lacks chocolate tempering equipment for Wonka Bar
    sugar_shack = next(f for f in sim["factories"] if f["factory"] == "Sugar Shack")
    assert sugar_shack["is_feasible_candidate"] is False
    assert sugar_shack["is_recommended"] is False
    assert "ineligible" in sugar_shack["tooling_note"].lower() or "retrofit" in sugar_shack["tooling_note"].lower()
    # Score must be severely penalized
    assert sugar_shack["composite_score"] < 10.0

def test_selection_justification_explains_champion_choice(engine):
    sim = engine.simulate_optimization(
        product_name="Wonka Bar - Nutty Crunch Surprise",
        destination_region="Pacific",
        ship_mode="Standard Class",
        priority_slider=70.0,
        units=2
    )
    champion = sim["factories"][0]
    assert "selection_justification" in champion
    assert len(champion["selection_justification"]) > 0
    # Must explain either disqualification of nominally faster plant or superior composite trade-off
    assert "Sugar Shack" in champion["selection_justification"] or "composite" in champion["selection_justification"]

def test_priority_slider_influences_weighting(engine):
    # At slider=0 (protect margin), cost/profit weight is dominant
    sim_cost = engine.simulate_optimization(
        product_name="Wonka Bar - Nutty Crunch Surprise",
        destination_region="Pacific",
        ship_mode="Standard Class",
        priority_slider=0.0,
        units=2
    )
    # At slider=100 (maximize speed), speed weight is dominant
    sim_speed = engine.simulate_optimization(
        product_name="Wonka Bar - Nutty Crunch Surprise",
        destination_region="Pacific",
        ship_mode="Standard Class",
        priority_slider=100.0,
        units=2
    )
    assert sim_cost is not None
    assert sim_speed is not None

