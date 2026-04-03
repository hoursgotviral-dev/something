"""
env/__init__.py
Exports make_env so server.py can do:  from env import make_env
"""
from __future__ import annotations
import sys, os
# ensure project root is importable when called from server.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import WhatsAppEnv, TaskConfig, TASK_CONFIGS
from simulator.user_simulator import default_simulator


def make_env(task_id: str = "medium", config: TaskConfig | None = None) -> WhatsAppEnv:
    """
    Factory used by server.py and tests.

    Maps  task1 → easy,  task2 → medium,  task3 → hard
    so OpenEnv task IDs line up with difficulty presets.
    """
    _TASK_MAP = {
        "task1": "easy",
        "task2": "medium",
        "task3": "hard",
    }
    resolved_id = _TASK_MAP.get(task_id, task_id)

    env = WhatsAppEnv(
        task_id=resolved_id,
        config=config,
        simulator=default_simulator,
    )
    return env


__all__ = ["make_env", "WhatsAppEnv", "TaskConfig", "TASK_CONFIGS"]