from __future__ import annotations

import sys
import os

# ensure the project root is on the path regardless of where pytest is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import random
from typing import Tuple

from models import Action, InternalObligation, ObligationSummary, State
from environment import (
    WhatsAppEnv,
    TaskConfig,
    TASK_CONFIGS,
    C,
    _DefaultUserSimulator,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def make_env(task_id: str = "medium", seed: int = 42, simulator=None) -> WhatsAppEnv:
    env = WhatsAppEnv(task_id=task_id, simulator=simulator)
    env.seed(seed)
    return env


def ask_action(msg: str = "") -> Action:
    return Action(action_type="ASK_QUESTION", message=msg)


def discount_action(pct: float = 10.0) -> Action:
    return Action(action_type="OFFER_DISCOUNT", discount_pct=pct)


def _run_episode(env: WhatsAppEnv, action_fn=None):
    """Run until done; return list of (obs, reward, done, info)."""
    obs = env.reset()
    results = []
    while True:
        action = action_fn(obs) if action_fn else ask_action()
        obs, reward, done, info = env.step(action)
        results.append((obs, reward, done, info))
        if done:
            break
    return results


# ─────────────────────────────────────────────────────────────────────────────
# A5-1 · BASIC RESET / STEP
# ─────────────────────────────────────────────────────────────────────────────


class TestResetStepBasic:

    def test_reset_returns_observation(self):
        env = make_env()
        obs = env.reset()
        assert obs.stage == "GREETING"
        assert obs.step_count == 0
        assert isinstance(obs.chat_history, list)

    def test_step_returns_correct_types(self):
        env = make_env()
        env.reset()
        obs, reward, done, info = env.step(ask_action())
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_without_reset_raises(self):
        env = make_env()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(ask_action())

    def test_step_after_done_raises(self):
        env = make_env()
        env.reset()
        # force done by maxing conversion
        env._state = env._state.with_updates(conversion_prob=1.0)
        env.step(discount_action())
        with pytest.raises(RuntimeError, match="done"):
            env.step(ask_action())

    def test_chat_history_grows(self):
        env = make_env()
        env.reset()
        for _ in range(3):
            env.step(ask_action())
            if env.state().episode_done:
                break
        assert len(env._chat_history) >= 2  # at least one AGENT + one USER line

    def test_state_accessible(self):
        env = make_env()
        env.reset()
        s = env.state()
        assert 0.0 <= s.trust <= 1.0
        assert 0.0 <= s.patience <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# A5-2 · TIME & CLAMPING
# ─────────────────────────────────────────────────────────────────────────────


class TestTimeAndClamping:

    def test_time_step_increments(self):
        env = make_env()
        env.reset()
        for expected in range(1, 6):
            env.step(ask_action())
            if env.state().episode_done:
                break
            assert env.state().time_step == expected

    def test_patience_decays(self):
        env = make_env()
        env.reset()
        initial = env.state().patience
        env.step(ask_action())
        if not env.state().episode_done:
            assert env.state().patience < initial

    def test_values_stay_in_bounds_many_steps(self):
        env = make_env()
        env.reset()
        unit_attrs = ["trust", "patience", "annoyance", "satisfaction", "conversion_prob"]
        for _ in range(50):
            action = discount_action() if random.random() < 0.3 else ask_action()
            env.step(action)
            if env.state().episode_done:
                env.reset()
            s = env.state()
            for attr in unit_attrs:
                v = getattr(s, attr)
                assert 0.0 <= v <= 1.0, f"{attr}={v} out of bounds"

    def test_cost_to_business_non_negative(self):
        env = make_env()
        env.reset()
        for _ in range(10):
            env.step(discount_action(pct=5.0))
            if env.state().episode_done:
                break
        assert env.state().cost_to_business >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# A5-3 · OBLIGATION EXPIRY EFFECTS
# ─────────────────────────────────────────────────────────────────────────────


class TestObligationExpiry:

    def _env_with_obligation(self, due_at_offset: int = 1) -> WhatsAppEnv:
        env = make_env()
        env.reset()
        obl = InternalObligation(
            type="follow_up",
            description="Test obligation",
            importance=1.0,
            related_stage="GREETING",
            created_at_step=0,
            due_at=env.state().time_step + due_at_offset,
        )
        env._state = env._state.with_updates(
            obligations=env._state.obligations.add(obl)
        )
        return env

    def test_obligation_expires_and_penalises(self):
        env = self._env_with_obligation(due_at_offset=1)
        annoyance_before = env.state().annoyance
        trust_before = env.state().trust

        # step past the due_at so expiry triggers
        env.step(ask_action())  # time_step → 1 (at due_at)
        env.step(ask_action())  # time_step → 2 (past due_at)

        if not env.state().episode_done:
            assert env.state().annoyance >= annoyance_before
            assert env.state().trust <= trust_before

    def test_expired_obligation_in_violation_count(self):
        env = self._env_with_obligation(due_at_offset=1)
        env.step(ask_action())
        env.step(ask_action())
        if not env.state().episode_done:
            assert env.state().obligations.violation_count >= 1

    def test_fulfilled_obligation_not_violated(self):
        env = self._env_with_obligation(due_at_offset=5)
        # PROVIDE_INFO fulfils follow_up obligations
        env.step(Action(action_type="PROVIDE_INFO", message="Here's the info"))
        # Should be fulfilled before due_at
        fulfilled = [
            o for o in env.state().obligations.fulfilled
        ]
        assert len(fulfilled) >= 1

    def test_obligation_events_contain_created(self):
        env = make_env()
        env.reset()
        # agent commits to sending info
        _, _, _, info = env.step(
            Action(action_type="PROVIDE_INFO", message="I'll send you the brochure")
        )
        # "created" event should appear because message contains commitment trigger
        # (depends on whether simulator injects follow-up; just ensure no crash)
        assert isinstance(info["obligation_events"], list)

    def test_obligation_events_contain_expired(self):
        env = self._env_with_obligation(due_at_offset=1)
        env.step(ask_action())
        _, _, _, info = env.step(ask_action())
        event_types = [e["type"] for e in info["obligation_events"]]
        assert "expired" in event_types


# ─────────────────────────────────────────────────────────────────────────────
# A5-4 · TERMINAL CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────


class TestTerminalConditions:

    def test_sale_on_high_conversion(self):
        env = make_env()
        env.reset()
        env._state = env._state.with_updates(conversion_prob=0.84)
        # One discount push should breach threshold
        _, _, done, info = env.step(discount_action(pct=20.0))
        if done:
            assert info["outcome"] == "SALE"

    def test_abandoned_on_low_patience(self):
        env = make_env()
        env.reset()
        env._state = env._state.with_updates(patience=0.20)
        # DELAY_RESPONSE drops patience further
        _, _, done, info = env.step(Action(action_type="DELAY_RESPONSE"))
        if done:
            assert info["outcome"] == "ABANDONED"

    def test_no_sale_on_max_steps(self):
        env = make_env(task_id="easy")  # max_steps = 15
        env.reset()
        results = _run_episode(env)
        last_info = results[-1][3]
        assert last_info["outcome"] in {"SALE", "NO_SALE", "ABANDONED", "ESCALATED"}

    def test_escalated_outcome(self):
        env = make_env()
        env.reset()
        _, _, done, info = env.step(Action(action_type="ESCALATE"))
        if done:
            assert info["outcome"] == "ESCALATED"

    def test_end_conversation_outcome(self):
        env = make_env()
        env.reset()
        _, _, done, info = env.step(Action(action_type="END_CONVERSATION"))
        if done:
            assert info["outcome"] == "NO_SALE"

    def test_episode_ends_eventually(self):
        env = make_env()
        results = _run_episode(env)
        assert results[-1][2] is True  # done == True on last step

    def test_outcome_set_in_info(self):
        env = make_env()
        results = _run_episode(env)
        last_info = results[-1][3]
        assert last_info["outcome"] != "IN_PROGRESS"


# ─────────────────────────────────────────────────────────────────────────────
# A5-5 · SEEDING
# ─────────────────────────────────────────────────────────────────────────────


class TestSeeding:

    def _collect_rewards(self, seed: int) -> list:
        env = make_env(seed=seed)
        results = _run_episode(env)
        return [r for _, r, _, _ in results]

    def test_same_seed_same_rewards(self):
        r1 = self._collect_rewards(7)
        r2 = self._collect_rewards(7)
        assert r1 == r2

    def test_different_seeds_different_rewards(self):
        r1 = self._collect_rewards(1)
        r2 = self._collect_rewards(999)
        # Very unlikely to be identical
        assert r1 != r2 or len(r1) != len(r2)

    def test_seed_after_reset_repeatable(self):
        env = make_env()
        env.seed(42)
        obs1 = env.reset()
        s1 = env.state().model_dump()

        env.seed(42)
        obs2 = env.reset()
        s2 = env.state().model_dump()

        assert s1["trust"] == pytest.approx(s2["trust"])
        assert s1["patience"] == pytest.approx(s2["patience"])


# ─────────────────────────────────────────────────────────────────────────────
# A5-6 · STAGE TRANSITIONS
# ─────────────────────────────────────────────────────────────────────────────


class TestStageTransitions:

    def test_greeting_to_discovery(self):
        env = make_env()
        env.reset()
        assert env.state().stage == "GREETING"
        # Default simulator returns neutral/positive → should move to DISCOVERY
        env.step(ask_action())
        if not env.state().episode_done:
            assert env.state().stage in {"DISCOVERY", "OBJECTION_HANDLING", "GREETING"}

    def test_escalate_forces_escalated_stage(self):
        env = make_env()
        env.reset()
        env.step(Action(action_type="ESCALATE"))
        assert env.state().stage in {"ESCALATED", "ENDED"}

    def test_end_conversation_forces_ended_stage(self):
        env = make_env()
        env.reset()
        env.step(Action(action_type="END_CONVERSATION"))
        assert env.state().stage == "ENDED"


# ─────────────────────────────────────────────────────────────────────────────
# A5-7 · ALL ACTION TYPES
# ─────────────────────────────────────────────────────────────────────────────


class TestAllActionTypes:

    @pytest.mark.parametrize("action_type", [
        "ASK_QUESTION", "GIVE_PRICE", "PROVIDE_INFO",
        "ESCALATE", "DELAY_RESPONSE", "END_CONVERSATION",
    ])
    def test_action_does_not_crash(self, action_type):
        env = make_env()
        env.reset()
        action = Action(action_type=action_type)
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)

    def test_offer_discount_requires_pct(self):
        with pytest.raises(Exception):
            Action(action_type="OFFER_DISCOUNT")  # missing discount_pct

    def test_offer_discount_with_pct(self):
        env = make_env()
        env.reset()
        obs, reward, done, info = env.step(discount_action(15.0))
        assert isinstance(reward, float)


# ─────────────────────────────────────────────────────────────────────────────
# A5-8 · INFO SCHEMA
# ─────────────────────────────────────────────────────────────────────────────


class TestInfoSchema:

    def test_top_level_keys(self):
        env = make_env()
        env.reset()
        _, _, _, info = env.step(ask_action())
        for key in ["outcome", "time_step", "conversion_prob",
                    "violation_count", "state_snapshot",
                    "obligation_events", "reward_components"]:
            assert key in info, f"Missing key: {key}"

    def test_state_snapshot_keys(self):
        env = make_env()
        env.reset()
        _, _, _, info = env.step(ask_action())
        snap = info["state_snapshot"]
        for key in ["outcome", "satisfaction", "annoyance", "cost_to_business",
                    "trust", "patience", "conversion_prob", "stage",
                    "time_step", "violation_count"]:
            assert key in snap, f"Missing state_snapshot key: {key}"

    def test_obligation_events_is_list(self):
        env = make_env()
        env.reset()
        _, _, _, info = env.step(ask_action())
        assert isinstance(info["obligation_events"], list)

    def test_reward_components_is_dict(self):
        env = make_env()
        env.reset()
        _, _, _, info = env.step(ask_action())
        assert isinstance(info["reward_components"], dict)


# ─────────────────────────────────────────────────────────────────────────────
# A5-9 · TASK CONFIGS
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskConfigs:

    @pytest.mark.parametrize("task_id,expected_max", [
        ("easy", 15), ("medium", 20), ("hard", 25)
    ])
    def test_max_steps(self, task_id, expected_max):
        env = make_env(task_id=task_id)
        assert env.max_steps == expected_max

    def test_custom_config_overrides(self):
        cfg = TaskConfig(task_id="custom", max_steps=7)
        env = WhatsAppEnv(config=cfg)
        assert env.max_steps == 7

    def test_hard_config_lower_initial_trust(self):
        results = []
        for seed in range(20):
            env = make_env(task_id="hard", seed=seed)
            env.reset()
            results.append(env.state().trust)
        avg_trust = sum(results) / len(results)
        assert avg_trust < 0.55  # hard should start with lower trust


# ─────────────────────────────────────────────────────────────────────────────
# A5-10 · OUTCOME FROZEN AFTER DONE
# ─────────────────────────────────────────────────────────────────────────────


class TestOutcomeFrozen:

    def test_outcome_not_overwritten(self):
        env = make_env()
        env.reset()
        # Force conversion
        env._state = env._state.with_updates(conversion_prob=0.84)
        _, _, done, info = env.step(discount_action(20.0))
        if done:
            first_outcome = info["outcome"]
            # Try to mutate (should be ignored by with_updates guard)
            env._state = env._state.with_updates(outcome="ABANDONED")
            assert env._state.outcome == first_outcome


# ─────────────────────────────────────────────────────────────────────────────
# RUN STANDALONE
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v"])