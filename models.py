from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ---------------- ACTION ----------------

ActionType = Literal[
    "ASK_QUESTION",
    "GIVE_PRICE",
    "OFFER_DISCOUNT",
    "PROVIDE_INFO",
    "ESCALATE",
    "DELAY_RESPONSE",
    "END_CONVERSATION",
]

ACTIONS: List[ActionType] = list(ActionType.__args__)


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


# ---------------- OBLIGATIONS ----------------

ObligationStatus = Literal["PENDING", "FULFILLED", "VIOLATED", "WAIVED"]


class InternalObligation(BaseModel):
    obligation_id: str
    description: str
    status: ObligationStatus = "PENDING"
    created_at_step: int = Field(ge=0)
    due_by_step: Optional[int] = Field(default=None, ge=0)
    fulfilled_at_step: Optional[int] = Field(default=None, ge=0)

    def is_overdue(self, current_step: int) -> bool:
        return (
            self.status == "PENDING"
            and self.due_by_step is not None
            and current_step > self.due_by_step
        )


class ObligationSummary(BaseModel):
    obligations: List[InternalObligation] = Field(default_factory=list)

    @property
    def pending(self):
        return [o for o in self.obligations if o.status == "PENDING"]

    @property
    def violated(self):
        return [o for o in self.obligations if o.status == "VIOLATED"]

    @property
    def violation_count(self):
        return len(self.violated)

    @property
    def has_pending(self):
        return bool(self.pending)


# ---------------- TYPES ----------------

IntentType = Literal[
    "PURCHASE", "INQUIRY", "COMPLAINT",
    "COMPARISON", "NEGOTIATION", "SUPPORT", "UNKNOWN"
]

StageType = Literal[
    "GREETING", "DISCOVERY", "OBJECTION_HANDLING",
    "NEGOTIATION", "CLOSING", "POST_SALE",
    "ESCALATED", "ENDED"
]

UserType = Literal[
    "IMPULSIVE", "ANALYTICAL", "SKEPTICAL",
    "LOYAL", "PRICE_SENSITIVE", "UNKNOWN"
]

OutcomeType = Literal[
    "SALE", "NO_SALE", "ESCALATED",
    "ABANDONED", "IN_PROGRESS"
]


# ---------------- OBSERVATION ----------------

class Observation(BaseModel):
    chat_history: List[str] = Field(default_factory=list)
    stage: StageType = "GREETING"
    intent: IntentType = "UNKNOWN"
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    uncertainties: List[str] = Field(default_factory=list)
    obligations: ObligationSummary = Field(default_factory=ObligationSummary)
    step_count: int = Field(default=0, ge=0)


# ---------------- STATE ----------------

def _unit(v: float) -> float:
    return max(0.0, min(1.0, v))


class State(BaseModel):
    user_type: UserType = "UNKNOWN"
    true_intent: IntentType = "UNKNOWN"

    trust: float = Field(default=0.5, ge=0.0, le=1.0)
    patience: float = Field(default=0.7, ge=0.0, le=1.0)
    annoyance: float = Field(default=0.0, ge=0.0, le=1.0)
    satisfaction: float = Field(default=0.5, ge=0.0, le=1.0)

    conversion_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    cost_to_business: float = Field(default=0.0, ge=0.0)

    stage: StageType = "GREETING"
    obligations: ObligationSummary = Field(default_factory=ObligationSummary)

    time_step: int = Field(default=0, ge=0)
    outcome: OutcomeType = "IN_PROGRESS"
    episode_done: bool = False

    def with_updates(self, **kwargs) -> "State":
        unit_fields = {"trust", "patience", "annoyance", "satisfaction", "conversion_prob"}
        safe = {
            k: (_unit(v) if k in unit_fields else v)
            for k, v in kwargs.items()
        }
        return self.model_copy(update=safe)