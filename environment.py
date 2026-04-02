from __future__ import annotations

import random
from typing import Tuple

from models import (
    Action, Observation, State,
    InternalObligation, ObligationSummary
)

# ---- TASK CONFIG ----
TASKS = {
    "easy": 15,
    "medium": 20,
    "hard": 25,
}

CONVERSION_THRESHOLD = 0.85
PATIENCE_THRESHOLD = 0.15


class WhatsAppEnv:

    def __init__(self, task_id: str = "medium"):
        self.task_id = task_id
        self.max_steps = TASKS.get(task_id, 20)
        self._state = None
        self._chat_history = []
        self._rng = random.Random()

    def seed(self, value: int):
        self._rng.seed(value)

    def reset(self) -> Observation:
        self._chat_history = []

        self._state = State(
            user_type=self._rng.choice(["IMPULSIVE", "ANALYTICAL", "PRICE_SENSITIVE"]),
            true_intent="PURCHASE",
            trust=0.5,
            patience=0.8,
            satisfaction=0.5,
            conversion_prob=0.4,
            stage="GREETING"
        )

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:

        if self._state is None:
            raise RuntimeError("Call reset() first")

        if self._state.episode_done:
            raise RuntimeError("Episode done")

        # ---- snapshot ----
        state_before = self._state.model_copy(deep=True)

        # ---- pipeline ----
        self._apply_agent_action_to_state(action)
        self._advance_time()

        user_msg, user_event = self._simulate_user(action)
        self._update_state_from_user(user_msg, user_event)

        obligation_events = self._update_obligations(user_event)

        done = self._check_done()

        reward, components = self._compute_reward(
            state_before, action, user_event, done
        )

        obs = self._build_observation()
        s = self._state

        # ---- INFO (STRICT FORMAT) ----
        info = {
            "outcome": s.outcome,
            "time_step": s.time_step,
            "conversion_prob": s.conversion_prob,
            "violation_count": s.obligations.violation_count,

            "state_snapshot": {
                "outcome": s.outcome,
                "satisfaction": s.satisfaction,
                "annoyance": s.annoyance,
                "cost_to_business": s.cost_to_business,
                "trust": s.trust,
                "patience": s.patience,
                "conversion_prob": s.conversion_prob,
                "stage": s.stage,
                "time_step": s.time_step,
                "violation_count": s.obligations.violation_count,
            },

            "obligation_events": obligation_events,
            "reward_components": components,
        }

        return obs, reward, done, info

    def state(self):
        return self._state

    # ---------- INTERNAL ----------

    def _apply_agent_action_to_state(self, action: Action):
        s = self._state
        self._chat_history.append(f"AGENT: {action.message or action.action_type}")

        conv_delta = 0

        if action.action_type == "OFFER_DISCOUNT":
            conv_delta += 0.1
            self._state = s.with_updates(
                conversion_prob=s.conversion_prob + conv_delta,
                cost_to_business=s.cost_to_business + (action.discount_pct or 0)
            )

        elif action.action_type == "DELAY_RESPONSE":
            self._state = s.with_updates(
                trust=s.trust - 0.1,
                annoyance=s.annoyance + 0.2,
                conversion_prob=s.conversion_prob - 0.1
            )

    def _advance_time(self):
        s = self._state
        self._state = s.with_updates(
            time_step=s.time_step + 1,
            patience=s.patience - 0.05
        )

    def _simulate_user(self, action: Action):
        if action.action_type == "OFFER_DISCOUNT":
            return "Sounds good", "positive"
        if action.action_type == "DELAY_RESPONSE":
            return "You're late!", "frustrated"
        return "Tell me more", "neutral"

    def _update_state_from_user(self, msg, event):
        self._chat_history.append(f"USER: {msg}")
        s = self._state

        if event == "positive":
            self._state = s.with_updates(
                satisfaction=s.satisfaction + 0.1,
                conversion_prob=s.conversion_prob + 0.1
            )
        elif event == "frustrated":
            self._state = s.with_updates(
                annoyance=s.annoyance + 0.2,
                patience=s.patience - 0.1
            )

    def _update_obligations(self, user_event):
        s = self._state
        events = []
        obs = list(s.obligations.obligations)

        for o in obs:
            if o.is_overdue(s.time_step):
                o.status = "VIOLATED"
                events.append({"type": "expired", "id": o.obligation_id, "importance": 1.0})

        return events

    def _compute_reward(self, before, action, user_event, done):
        try:
            from reward.core import compute_step_reward
            reward, components = compute_step_reward(
                state_before={
                    "stage": before.stage,
                    "annoyance": before.annoyance,
                    "satisfaction": before.satisfaction,
                    "cost_to_business": before.cost_to_business,
                    "outcome": before.outcome,
                },
                state_after={
                    "stage": self._state.stage,
                    "annoyance": self._state.annoyance,
                    "satisfaction": self._state.satisfaction,
                    "cost_to_business": self._state.cost_to_business,
                    "outcome": self._state.outcome,
                    "violation_count": self._state.obligations.violation_count,
                },
                action=action,
                user_event=user_event,
                done=done,
            )
        except ImportError:
            reward = self._state.satisfaction - self._state.annoyance
            components = {"fallback": reward}

        return reward, components

    def _check_done(self):
        s = self._state

        if s.conversion_prob >= CONVERSION_THRESHOLD:
            self._state = s.with_updates(outcome="SALE", episode_done=True)
            return True

        if s.patience <= PATIENCE_THRESHOLD:
            self._state = s.with_updates(outcome="ABANDONED", episode_done=True)
            return True

        if s.time_step >= self.max_steps:
            self._state = s.with_updates(outcome="NO_SALE", episode_done=True)
            return True

        return False

    def _build_observation(self):
        s = self._state

        uncertainties = []
        if s.trust < 0.3:
            uncertainties.append("low trust")
        if s.patience < 0.3:
            uncertainties.append("low patience")

        return Observation(
            chat_history=self._chat_history,
            stage=s.stage,
            intent="INQUIRY",
            sentiment=(s.satisfaction - 0.5) * 2,
            uncertainties=uncertainties,
            obligations=s.obligations,
            step_count=s.time_step
        )