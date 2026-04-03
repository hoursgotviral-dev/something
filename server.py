"""
server.py  –  FastAPI OpenEnv server for WhatsApp sales-agent RL.

Endpoints
---------
GET  /health                     → {"status": "ok", ...}
POST /v1/reset?task_id=task1     → Observation JSON
POST /v1/step                    → StepResponse JSON
GET  /v1/state                   → State JSON

Run locally:
    uvicorn server:app --host 0.0.0.0 --port 7860 --reload

Docker:
    docker build -t whatsapp-rl .
    docker run -p 7860:7860 whatsapp-rl
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

# ── make project root importable inside Docker ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env import make_env
from models import Action, ActionType, Observation, State


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ActionRequest(BaseModel):
    """Body accepted by POST /v1/step"""
    action_type: ActionType
    message: str = ""
    discount_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class RewardComponents(BaseModel):
    satisfaction_gain:  float = 0.0
    annoyance_penalty:  float = 0.0
    obligation_penalty: float = 0.0
    cost_penalty:       float = 0.0
    stage_progress:     float = 0.0
    delay_penalty:      float = 0.0
    terminal:           float = 0.0


class StepResponse(BaseModel):
    """Full response returned by POST /v1/step"""
    observation:       Dict[str, Any]
    reward:            float
    done:              bool
    outcome:           str
    time_step:         int
    conversion_prob:   float
    violation_count:   int
    state_snapshot:    Dict[str, Any]
    obligation_events: List[Dict[str, Any]]
    reward_components: Dict[str, float]


class HealthResponse(BaseModel):
    status:       str
    version:      str
    current_task: Optional[str]
    episode_done: Optional[bool]
    uptime_s:     float


# ══════════════════════════════════════════════════════════════════════════════
# APP + GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="WhatsApp Sales RL – OpenEnv",
    description="RL environment server for WhatsApp sales-agent training.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # open for all origins (tighten in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment (one episode at a time)
_current_env = None
_current_task_id: Optional[str] = None
_server_start = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _require_env():
    """Raise 400 if no episode has been started yet."""
    if _current_env is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /v1/reset?task_id=<task> first.",
        )
    return _current_env


def _obs_to_dict(obs: Observation) -> Dict[str, Any]:
    """Serialise Observation to a plain dict (JSON-safe)."""
    return {
        "chat_history":  obs.chat_history,
        "stage":         obs.stage,
        "intent":        obs.intent,
        "sentiment":     round(obs.sentiment, 4),
        "uncertainties": obs.uncertainties,
        "obligations": {
            "pending_count":   len(obs.obligations.pending),
            "fulfilled_count": len(obs.obligations.fulfilled),
            "violation_count": obs.obligations.violation_count,
            "obligations": [
                {
                    "id":          o.obligation_id,
                    "type":        o.type,
                    "description": o.description,
                    "status":      o.status,
                    "importance":  o.importance,
                    "due_at":      o.due_at,
                }
                for o in obs.obligations.obligations
            ],
        },
        "step_count": obs.step_count,
    }


def _state_to_dict(state: State) -> Dict[str, Any]:
    """Serialise full ground-truth State."""
    return {
        "user_type":        state.user_type,
        "true_intent":      state.true_intent,
        "trust":            round(state.trust, 4),
        "patience":         round(state.patience, 4),
        "annoyance":        round(state.annoyance, 4),
        "satisfaction":     round(state.satisfaction, 4),
        "conversion_prob":  round(state.conversion_prob, 4),
        "cost_to_business": round(state.cost_to_business, 4),
        "stage":            state.stage,
        "time_step":        state.time_step,
        "outcome":          state.outcome,
        "episode_done":     state.episode_done,
        "violation_count":  state.obligations.violation_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Server liveness probe. Used by Docker HEALTHCHECK."""
    env = _current_env
    return HealthResponse(
        status="ok",
        version="1.0.0",
        current_task=_current_task_id,
        episode_done=env.state().episode_done if env else None,
        uptime_s=round(time.time() - _server_start, 1),
    )


@app.post("/v1/reset", tags=["Environment"])
def reset(task_id: str = Query(default="task1",
                               description="One of: task1, task2, task3")):
    """
    Start a new episode.

    - **task_id**: task1 (easy) | task2 (medium) | task3 (hard)

    Returns the initial Observation.
    """
    global _current_env, _current_task_id

    valid = {"task1", "task2", "task3", "easy", "medium", "hard"}
    if task_id not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Must be one of: {sorted(valid)}",
        )

    try:
        _current_env    = make_env(task_id=task_id)
        _current_task_id = task_id
        obs = _current_env.reset()
        return _obs_to_dict(obs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/v1/step", response_model=StepResponse, tags=["Environment"])
def step(body: ActionRequest):
    """
    Advance the environment by one step.

    Send an action and receive the next observation, reward, and info.

    Example body:
    ```json
    {"action_type": "ASK_QUESTION", "message": "What is your budget?"}
    ```
    ```json
    {"action_type": "OFFER_DISCOUNT", "discount_pct": 10.0}
    ```
    """
    env = _require_env()

    if env.state().episode_done:
        raise HTTPException(
            status_code=400,
            detail="Episode is already done. Call POST /v1/reset to start a new one.",
        )

    try:
        action = Action(
            action_type=body.action_type,
            message=body.message,
            discount_pct=body.discount_pct,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")

    try:
        obs, reward, done, info = env.step(action)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")

    return StepResponse(
        observation       = _obs_to_dict(obs),
        reward            = round(reward, 4),
        done              = done,
        outcome           = info["outcome"],
        time_step         = info["time_step"],
        conversion_prob   = round(info["conversion_prob"], 4),
        violation_count   = info["violation_count"],
        state_snapshot    = info["state_snapshot"],
        obligation_events = info["obligation_events"],
        reward_components = info["reward_components"],
    )


@app.get("/v1/state", tags=["Environment"])
def get_state():
    """
    Return the full ground-truth State (hidden from agents in real training).
    Useful for debugging and monitoring.
    """
    env = _require_env()
    try:
        return _state_to_dict(env.state())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State fetch failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# DEV / STANDALONE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True)