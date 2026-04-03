"""
reward/core.py  –  Dev B reward module (self-contained, no extra deps).
"""
from __future__ import annotations
from typing import Any, Dict, Tuple

DEFAULT_WEIGHTS: Dict[str, float] = {
    "conversion_bonus":   2.0,
    "annoyance_penalty":  1.0,
    "satisfaction_gain":  1.0,
    "obligation_penalty": 0.2,
    "cost_penalty":       0.01,
    "delay_penalty":      0.3,
    "abandoned_penalty":  1.5,
    "no_sale_penalty":    0.5,
}

STAGE_ORDER = [
    "GREETING", "DISCOVERY", "QUALIFICATION",
    "OBJECTION_HANDLING", "NEGOTIATION", "CLOSING", "POST_SALE",
]


def _stage_progress(before: str, after: str) -> float:
    """Positive if stage moved forward, negative if regressed."""
    bi = STAGE_ORDER.index(before) if before in STAGE_ORDER else 0
    ai = STAGE_ORDER.index(after)  if after  in STAGE_ORDER else 0
    return float(ai - bi) * 0.05


def compute_step_reward(
    state_before: Dict[str, Any],
    state_after:  Dict[str, Any],
    action:       Any,
    user_event:   str,
    done:         bool,
    reward_weights: Dict[str, float] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute per-step reward.

    Parameters
    ----------
    state_before / state_after : dicts with keys:
        stage, annoyance, satisfaction, cost_to_business,
        outcome, violation_count
    action       : Action object (uses .action_type)
    user_event   : str  e.g. "positive", "frustrated"
    done         : bool
    reward_weights : optional override dict

    Returns
    -------
    (total_reward: float, components: dict)
    """
    w = {**DEFAULT_WEIGHTS, **(reward_weights or {})}

    # ── continuous deltas ──────────────────────────────────────────────────
    sat_gain   = state_after["satisfaction"]    - state_before["satisfaction"]
    ann_gain   = state_after["annoyance"]       - state_before["annoyance"]
    cost_delta = state_after["cost_to_business"] - state_before["cost_to_business"]
    new_viols  = max(0, state_after["violation_count"] - state_before["violation_count"])

    stage_prog = _stage_progress(state_before["stage"], state_after["stage"])

    # ── per-step components ────────────────────────────────────────────────
    c_satisfaction  =  w["satisfaction_gain"]  * sat_gain
    c_annoyance     = -w["annoyance_penalty"]  * ann_gain
    c_obligation    = -w["obligation_penalty"] * new_viols
    c_cost          = -w["cost_penalty"]       * cost_delta
    c_stage         =  stage_prog

    c_delay = (
        -w["delay_penalty"]
        if getattr(action, "action_type", "") == "DELAY_RESPONSE"
        else 0.0
    )

    # ── terminal bonus / penalty ───────────────────────────────────────────
    c_terminal = 0.0
    if done:
        outcome = state_after["outcome"]
        if outcome == "SALE":
            c_terminal = w["conversion_bonus"] - w["cost_penalty"] * state_after["cost_to_business"]
        elif outcome == "ABANDONED":
            c_terminal = -w["abandoned_penalty"]
        elif outcome == "NO_SALE":
            c_terminal = -w["no_sale_penalty"]
        # ESCALATED → 0 (neutral; human takes over)

    total = (
        c_satisfaction
        + c_annoyance
        + c_obligation
        + c_cost
        + c_stage
        + c_delay
        + c_terminal
    )

    components = {
        "satisfaction_gain": round(c_satisfaction, 4),
        "annoyance_penalty": round(c_annoyance,    4),
        "obligation_penalty": round(c_obligation,  4),
        "cost_penalty":      round(c_cost,         4),
        "stage_progress":    round(c_stage,        4),
        "delay_penalty":     round(c_delay,        4),
        "terminal":          round(c_terminal,     4),
    }
    return round(total, 4), components