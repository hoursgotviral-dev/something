from __future__ import annotations

import uuid
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────────────────────────────────────────

ActionType = Literal[
    "ASK_QUESTION",
    "GIVE_PRICE",
    "OFFER_DISCOUNT",
    "PROVIDE_INFO",
    "ESCALATE",
    "DELAY_RESPONSE",
    "END_CONVERSATION",
]

ACTIONS: List[ActionType] = list(ActionType.__args__)  # type: ignore[attr-defined]


class Action(BaseModel):
    action_type: ActionType
    message: str = ""
    discount_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_discount(self) -> "Action":
        if self.action_type == "OFFER_DISCOUNT" and self.discount_pct is None:
            raise ValueError("discount_pct required for OFFER_DISCOUNT")
        if self.discount_pct is not None and self.action_type != "OFFER_DISCOUNT":
            raise ValueError("discount_pct only allowed for OFFER_DISCOUNT")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# OBLIGATIONS
# ─────────────────────────────────────────────────────────────────────────────

ObligationStatus = Literal["PENDING", "FULFILLED", "VIOLATED", "WAIVED", "EXPIRED"]

ObligationType = Literal[
    "follow_up",        # user said "remind me", "I'll tell you later"
    "agent_commitment", # agent said "I'll send X", "I'll check on Y"
    "system",           # internally generated
]


class InternalObligation(BaseModel):
    """
    Represents a trackable commitment made by either the agent or the user.

    Fields
    ------
    obligation_id   : unique identifier (auto-generated UUID if not supplied)
    type            : one of follow_up | agent_commitment | system
    description     : human-readable description
    status          : PENDING | FULFILLED | VIOLATED | WAIVED | EXPIRED
    importance      : 0-1 weight used by the reward module
    related_stage   : the conversation stage at which this obligation is relevant
    created_at_step : time_step when the obligation was created
    due_at          : time_step by which it must be fulfilled (None = no deadline)
    fulfilled_at_step : time_step when fulfilled (None until then)
    """

    obligation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ObligationType = "system"
    description: str
    status: ObligationStatus = "PENDING"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    related_stage: str = ""
    created_at_step: int = Field(ge=0)
    due_at: Optional[int] = Field(default=None, ge=0)
    fulfilled_at_step: Optional[int] = Field(default=None, ge=0)

    # keep backward-compat alias
    @property
    def due_by_step(self) -> Optional[int]:
        return self.due_at

    def is_overdue(self, current_step: int) -> bool:
        return (
            self.status == "PENDING"
            and self.due_at is not None
            and current_step > self.due_at
        )


class ObligationSummary(BaseModel):
    obligations: List[InternalObligation] = Field(default_factory=list)

    # ── read-only views ───────────────────────────────────────────────────────

    @property
    def pending(self) -> List[InternalObligation]:
        return [o for o in self.obligations if o.status == "PENDING"]

    @property
    def fulfilled(self) -> List[InternalObligation]:
        return [o for o in self.obligations if o.status == "FULFILLED"]

    @property
    def violated(self) -> List[InternalObligation]:
        return [o for o in self.obligations if o.status in ("VIOLATED", "EXPIRED")]

    @property
    def violation_count(self) -> int:
        return len(self.violated)

    @property
    def has_pending(self) -> bool:
        return bool(self.pending)

    # ── mutation helpers (return new ObligationSummary) ──────────────────────

    def add(self, obligation: InternalObligation) -> "ObligationSummary":
        return ObligationSummary(obligations=self.obligations + [obligation])

    def update_status(
        self,
        obligation_id: str,
        new_status: ObligationStatus,
        fulfilled_at: Optional[int] = None,
    ) -> "ObligationSummary":
        updated = []
        for o in self.obligations:
            if o.obligation_id == obligation_id:
                data = o.model_dump()
                data["status"] = new_status
                if fulfilled_at is not None:
                    data["fulfilled_at_step"] = fulfilled_at
                updated.append(InternalObligation(**data))
            else:
                updated.append(o)
        return ObligationSummary(obligations=updated)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATED TYPES
# ─────────────────────────────────────────────────────────────────────────────

IntentType = Literal[
    "PURCHASE", "INQUIRY", "COMPLAINT",
    "COMPARISON", "NEGOTIATION", "SUPPORT", "UNKNOWN",
]

StageType = Literal[
    "GREETING",
    "DISCOVERY",
    "QUALIFICATION",
    "OBJECTION_HANDLING",
    "NEGOTIATION",
    "CLOSING",
    "POST_SALE",
    "ESCALATED",
    "ENDED",
]

UserType = Literal[
    "IMPULSIVE", "ANALYTICAL", "SKEPTICAL",
    "LOYAL", "PRICE_SENSITIVE", "UNKNOWN",
]

OutcomeType = Literal[
    "SALE", "NO_SALE", "ESCALATED",
    "ABANDONED", "IN_PROGRESS",
]


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION  (agent-visible)
# ─────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    chat_history: List[str] = Field(default_factory=list)
    stage: StageType = "GREETING"
    intent: IntentType = "UNKNOWN"
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    uncertainties: List[str] = Field(default_factory=list)
    obligations: ObligationSummary = Field(default_factory=ObligationSummary)
    step_count: int = Field(default=0, ge=0)


# ─────────────────────────────────────────────────────────────────────────────
# STATE  (ground truth, hidden from agent)
# ─────────────────────────────────────────────────────────────────────────────

def _unit(v: float) -> float:
    """Clamp to [0, 1]."""
    return max(0.0, min(1.0, v))


class State(BaseModel):
    # ── user profile ──────────────────────────────────────────────────────────
    user_type: UserType = "UNKNOWN"
    true_intent: IntentType = "UNKNOWN"

    # ── continuous signals ────────────────────────────────────────────────────
    trust: float = Field(default=0.5, ge=0.0, le=1.0)
    patience: float = Field(default=0.7, ge=0.0, le=1.0)
    annoyance: float = Field(default=0.0, ge=0.0, le=1.0)
    satisfaction: float = Field(default=0.5, ge=0.0, le=1.0)

    conversion_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    cost_to_business: float = Field(default=0.0, ge=0.0)

    # ── episode meta ──────────────────────────────────────────────────────────
    stage: StageType = "GREETING"
    obligations: ObligationSummary = Field(default_factory=ObligationSummary)
    time_step: int = Field(default=0, ge=0)
    outcome: OutcomeType = "IN_PROGRESS"
    episode_done: bool = False

    # ── immutable helper ──────────────────────────────────────────────────────
    def with_updates(self, **kwargs) -> "State":
        """
        Return a new State with the given fields updated.
        Bounded float fields are automatically clamped to [0, 1].
        Once episode_done is True the outcome is frozen.
        """
        unit_fields = {
            "trust", "patience", "annoyance",
            "satisfaction", "conversion_prob",
        }
        safe: dict = {}
        for k, v in kwargs.items():
            if k == "outcome" and self.episode_done:
                # outcome is frozen once episode ends
                continue
            safe[k] = _unit(v) if k in unit_fields else v
        return self.model_copy(update=safe)