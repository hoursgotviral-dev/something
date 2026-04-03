"""
simulator/user_simulator.py  –  Dev C user simulator (self-contained).

Satisfies UserSimulatorProtocol defined in environment.py.
Plugged in automatically by server.py / make_env().
"""
from __future__ import annotations

import random
from typing import Tuple

# ── response tables ────────────────────────────────────────────────────────

_BASE_RESPONSES = {
    "ASK_QUESTION":     [
        ("That's a good question, let me think…",   "neutral"),
        ("Interesting – tell me more.",              "positive"),
        ("I'm not sure that's relevant.",            "skeptical"),
    ],
    "GIVE_PRICE":       [
        ("That seems expensive.",                    "skeptical"),
        ("Hmm, let me compare with others.",         "neutral"),
        ("Can you do better on the price?",          "skeptical"),
    ],
    "OFFER_DISCOUNT":   [
        ("That sounds reasonable!",                  "positive"),
        ("Okay, that's more like it.",               "positive"),
        ("Deal! Let's go ahead.",                    "very_positive"),
    ],
    "PROVIDE_INFO":     [
        ("Thanks, that's helpful.",                  "neutral"),
        ("Good to know!",                            "positive"),
        ("I appreciate the details.",                "positive"),
    ],
    "ESCALATE":         [
        ("Okay, I'll wait for your manager.",        "neutral"),
        ("Fine, but please be quick.",               "neutral"),
    ],
    "DELAY_RESPONSE":   [
        ("Why is this taking so long?",              "frustrated"),
        ("I don't have all day.",                    "frustrated"),
        ("Still waiting…",                           "frustrated"),
    ],
    "END_CONVERSATION": [
        ("Goodbye.",                                 "neutral"),
        ("Okay, thanks anyway.",                     "neutral"),
    ],
}

# User-type modifiers: override base responses for specific combos
_USER_TYPE_OVERRIDES = {
    ("PRICE_SENSITIVE", "GIVE_PRICE"):    ("That's way too expensive for me.", "frustrated"),
    ("PRICE_SENSITIVE", "OFFER_DISCOUNT"):("Now we're talking!",               "very_positive"),
    ("IMPULSIVE",       "OFFER_DISCOUNT"):("Deal! Let's do it right now.",     "very_positive"),
    ("IMPULSIVE",       "ASK_QUESTION"):  ("Just tell me the price.",          "neutral"),
    ("SKEPTICAL",       "PROVIDE_INFO"):  ("I'll need to verify that myself.", "skeptical"),
    ("SKEPTICAL",       "GIVE_PRICE"):    ("I've seen cheaper elsewhere.",      "skeptical"),
    ("ANALYTICAL",      "PROVIDE_INFO"):  ("Interesting. Got any data on that?","neutral"),
    ("ANALYTICAL",      "ASK_QUESTION"):  ("Good question, here's my situation:","positive"),
    ("LOYAL",           "GIVE_PRICE"):    ("Okay, I trust your pricing.",       "positive"),
    ("LOYAL",           "OFFER_DISCOUNT"):("You didn't have to, but thanks!",   "very_positive"),
}

# Phrases injected occasionally to create follow-up obligations
_FOLLOW_UP_PHRASES = [
    " Remind me tomorrow.",
    " I'll think about it and get back to you.",
    " Let me check with my partner first.",
]


class UserSimulator:
    """
    Production user simulator.

    Satisfies UserSimulatorProtocol:
        __call__(action, state, rng) -> (message: str, event: str)
    """

    def __call__(
        self,
        action,          # models.Action
        state,           # models.State
        rng: random.Random,
    ) -> Tuple[str, str]:

        action_type = action.action_type
        user_type   = state.user_type

        # 1. Check for user-type override
        override_key = (user_type, action_type)
        if override_key in _USER_TYPE_OVERRIDES:
            msg, event = _USER_TYPE_OVERRIDES[override_key]
        else:
            # 2. Pick from base responses
            options = _BASE_RESPONSES.get(action_type, [("Hmm.", "neutral")])
            msg, event = rng.choice(options)

        # 3. Adjust based on current emotional state
        if state.annoyance > 0.7 and event not in ("frustrated",):
            msg += " I'm getting frustrated with this."
            event = "frustrated"
        elif state.trust > 0.75 and event == "skeptical":
            # High trust overrides skepticism
            event = "neutral"

        # 4. Occasionally inject a follow-up promise (creates obligation in env)
        if rng.random() < 0.12:
            msg += rng.choice(_FOLLOW_UP_PHRASES)

        return msg, event


# Singleton used by make_env / server
default_simulator = UserSimulator()