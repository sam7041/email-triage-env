"""
Email Triage Environment — Typed Models
Action / Observation / State for the OpenEnv spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Base stubs (mirrors openenv-core so the file works even without the package)
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    @dataclass
    class Action:          # type: ignore[no-redef]
        pass

    @dataclass(kw_only=True)
    class Observation:     # type: ignore[no-redef]
        done: bool = False
        reward: Union[bool, int, float, None] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class State:           # type: ignore[no-redef]
        episode_id: Optional[str] = None
        step_count: int = 0


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

VALID_PRIORITIES = ("low", "medium", "high", "critical")
VALID_CATEGORIES = ("billing", "technical", "sales", "general", "security")
VALID_SENTIMENTS = ("positive", "neutral", "negative", "frustrated")


@dataclass
class TriageAction(Action):
    """What the agent sends for each email."""
    priority: str = "medium"            # low | medium | high | critical
    category: str = "general"           # billing | technical | sales | general | security
    sentiment: str = "neutral"          # positive | neutral | negative | frustrated
    key_entities: List[str] = field(default_factory=list)   # extracted nouns/IDs
    requires_human: bool = False        # needs human escalation?
    response_draft: Optional[str] = None  # only scored in hard task
    reasoning: Optional[str] = None     # chain-of-thought (not scored)


@dataclass(kw_only=True)
class EmailObservation(Observation):
    """What the agent sees each step."""
    email_id: str = ""
    subject: str = ""
    body: str = ""
    sender: str = ""
    sender_tier: str = "free"           # free | pro | enterprise
    emails_in_thread: int = 1
    task_name: str = ""
    task_description: str = ""
    step_reward_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class TriageState(State):
    """Episode-level bookkeeping."""
    episode_id: Optional[str] = None
    step_count: int = 0
    current_task: str = ""
    emails_processed: int = 0
    cumulative_reward: float = 0.0
