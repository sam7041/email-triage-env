"""
Email Triage OpenEnv — HTTP Client
Connects to a running environment server (HF Space or local Docker).
"""
from __future__ import annotations

import requests
from dataclasses import asdict
from typing import Any, Dict, Optional

try:
    from .models import EmailObservation, TriageAction, TriageState
except ImportError:
    from models import EmailObservation, TriageAction, TriageState


class EmailTriageEnvClient:
    """Simple synchronous HTTP client for the Email Triage environment."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._task = "easy"

    # ── OpenEnv-style API ───────────────────────────────────────────────────

    def reset(self, task: str = "easy") -> Dict[str, Any]:
        self._task = task
        r = requests.post(
            f"{self.base_url}/reset", json={"task": task}, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: TriageAction) -> Dict[str, Any]:
        payload = asdict(action)
        payload["task"] = self._task
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        r = requests.get(
            f"{self.base_url}/state", params={"task": self._task}, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def list_tasks(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Quick smoke-test (run: python client.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = EmailTriageEnvClient()
    print("Health:", env.health())

    for task in ("easy", "medium", "hard"):
        print(f"\n── Task: {task} ──")
        obs = env.reset(task=task)
        print(f"  Email: {obs['subject'][:60]}…")
        print(f"  Task : {obs['task_name']}")

        # Dummy action — for real use, replace with LLM output
        action = TriageAction(
            priority="high", category="technical", sentiment="frustrated",
            key_entities=["ORD-001"], requires_human=True,
        )
        while not obs.get("done"):
            obs = env.step(action)
            print(f"  Reward: {obs['reward']}  done={obs['done']}")
