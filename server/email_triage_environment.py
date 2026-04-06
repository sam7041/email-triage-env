"""
Email Triage Environment — Core Logic
Implements reset() / step() / state for the OpenEnv spec.
Three tasks with deterministic graders returning rewards in [0.0, 1.0].
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

try:
    from ..models import EmailObservation, TriageAction, TriageState
except ImportError:
    from models import EmailObservation, TriageAction, TriageState


# ---------------------------------------------------------------------------
# Email dataset — ground-truth labels embedded alongside each email
# ---------------------------------------------------------------------------
EMAILS: List[Dict[str, Any]] = [
    # ── EASY TASK emails (priority classification only) ────────────────────
    {
        "task": "easy",
        "email_id": "E001",
        "subject": "Payment failed – please help urgently",
        "body": (
            "Hi Support,\n\n"
            "My payment of ₹4,999 failed three times today and my account is "
            "now locked. I have a client presentation in 2 hours and need access "
            "immediately. Order ID: ORD-2024-88231. Please escalate!\n\nRahul"
        ),
        "sender": "rahul.mehta@acmecorp.in",
        "sender_tier": "enterprise",
        "emails_in_thread": 1,
        "gt_priority": "critical",
        "gt_category": "billing",
        "gt_sentiment": "frustrated",
        "gt_requires_human": True,
        "gt_entities": ["ORD-2024-88231", "₹4,999", "account locked"],
        "gt_response_keywords": ["escalate", "priority", "account", "payment"],
    },
    {
        "task": "easy",
        "email_id": "E002",
        "subject": "Question about invoice formatting",
        "body": (
            "Hello,\n\n"
            "Could you clarify whether GST is included separately in the invoices "
            "or if it's part of the total? Just want to confirm for our records.\n\nThanks,\nPreeti"
        ),
        "sender": "preeti.sharma@startup.io",
        "sender_tier": "pro",
        "emails_in_thread": 1,
        "gt_priority": "low",
        "gt_category": "billing",
        "gt_sentiment": "neutral",
        "gt_requires_human": False,
        "gt_entities": ["GST", "invoice"],
        "gt_response_keywords": ["GST", "invoice", "total"],
    },
    # ── MEDIUM TASK emails (priority + category + entities) ────────────────
    {
        "task": "medium",
        "email_id": "M001",
        "subject": "API returning 500 errors since last deployment",
        "body": (
            "Team,\n\n"
            "After yesterday's v2.3.1 deploy, our /api/v2/orders endpoint has been "
            "throwing 500 Internal Server Error for ~30% of requests. Error trace:\n"
            "  AttributeError: 'NoneType' object has no attribute 'id'\n"
            "  at order_service.py:142\n\n"
            "We're losing roughly ₹50K/hour in transactions. Need hotfix ASAP.\n\n"
            "Account: ENT-4412\nAryan Kapoor, CTO"
        ),
        "sender": "aryan.kapoor@techfirm.com",
        "sender_tier": "enterprise",
        "emails_in_thread": 3,
        "gt_priority": "critical",
        "gt_category": "technical",
        "gt_sentiment": "frustrated",
        "gt_requires_human": True,
        "gt_entities": ["v2.3.1", "500 Internal Server Error", "ENT-4412", "/api/v2/orders"],
        "gt_response_keywords": ["hotfix", "escalate", "500", "deploy"],
    },
    {
        "task": "medium",
        "email_id": "M002",
        "subject": "Interested in upgrading our team plan",
        "body": (
            "Hi,\n\n"
            "We're currently on the Starter plan (5 seats) and looking to upgrade "
            "to Professional (20 seats). Could you send over a quote and let me know "
            "if there are annual discounts available?\n\nCompany: GreenLeaf Analytics\n"
            "Current plan: PLAN-STARTER-2024\n\nRegards,\nNisha Verma"
        ),
        "sender": "nisha.verma@greenleaf.io",
        "sender_tier": "pro",
        "emails_in_thread": 1,
        "gt_priority": "high",
        "gt_category": "sales",
        "gt_sentiment": "positive",
        "gt_requires_human": False,
        "gt_entities": ["GreenLeaf Analytics", "PLAN-STARTER-2024", "Professional", "20 seats"],
        "gt_response_keywords": ["upgrade", "quote", "annual", "discount"],
    },
    # ── HARD TASK emails (full triage + response draft) ───────────────────
    {
        "task": "hard",
        "email_id": "H001",
        "subject": "Data breach concern — unauthorized login from unknown IP",
        "body": (
            "URGENT — Security Alert\n\n"
            "I received a notification that my account (user: priya.nair@corp.in) "
            "was accessed from IP 185.220.101.45 (Russia) at 03:14 IST. I did NOT "
            "authorize this. Please immediately:\n"
            "1. Lock my account\n"
            "2. Revoke all active sessions\n"
            "3. Provide access logs for last 30 days\n\n"
            "Ticket ref if any prior: TKT-2024-55310\nPriya Nair"
        ),
        "sender": "priya.nair@corp.in",
        "sender_tier": "enterprise",
        "emails_in_thread": 1,
        "gt_priority": "critical",
        "gt_category": "security",
        "gt_sentiment": "frustrated",
        "gt_requires_human": True,
        "gt_entities": ["185.220.101.45", "TKT-2024-55310", "priya.nair@corp.in", "03:14 IST"],
        "gt_response_keywords": ["lock", "session", "security", "logs", "investigate"],
    },
    {
        "task": "hard",
        "email_id": "H002",
        "subject": "Monthly report download button not working on Safari",
        "body": (
            "Hi Support Team,\n\n"
            "The 'Download Monthly Report' button on the dashboard doesn't work on "
            "Safari 17.3 (macOS Sonoma 14.4). It works fine on Chrome. I need the "
            "March 2024 report for our board meeting tomorrow.\n\n"
            "User: vikram.joshi@example.org | Plan: PRO-2024\n\nThanks"
        ),
        "sender": "vikram.joshi@example.org",
        "sender_tier": "pro",
        "emails_in_thread": 1,
        "gt_priority": "high",
        "gt_category": "technical",
        "gt_sentiment": "neutral",
        "gt_requires_human": False,
        "gt_entities": ["Safari 17.3", "macOS Sonoma 14.4", "PRO-2024", "March 2024 report"],
        "gt_response_keywords": ["Safari", "Chrome", "workaround", "report", "download"],
    },
]

TASK_META = {
    "easy": {
        "name": "urgency_classification",
        "description": (
            "Classify the email's urgency level: low | medium | high | critical. "
            "Focus solely on how time-sensitive and business-impacting the issue is."
        ),
    },
    "medium": {
        "name": "department_routing",
        "description": (
            "Classify priority AND category (billing/technical/sales/general/security) "
            "AND extract key entities (order IDs, plan names, error codes, etc.). "
            "Partial credit is awarded for each correct component."
        ),
    },
    "hard": {
        "name": "full_triage",
        "description": (
            "Perform a complete triage: priority, category, sentiment, key entities, "
            "requires_human flag, AND write a concise professional response_draft "
            "that addresses the customer's specific concerns."
        ),
    },
}

PRIORITY_ORDER = ["low", "medium", "high", "critical"]


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _priority_score(pred: str, gt: str) -> float:
    """Partial credit based on proximity in urgency ladder."""
    try:
        pi, gi = PRIORITY_ORDER.index(pred.lower()), PRIORITY_ORDER.index(gt.lower())
    except ValueError:
        return 0.0
    diff = abs(pi - gi)
    return max(0.0, 1.0 - diff * 0.5)


def _category_score(pred: str, gt: str) -> float:
    return 1.0 if pred.lower() == gt.lower() else 0.0


def _entity_score(pred_list: List[str], gt_list: List[str]) -> float:
    """Soft match: reward fraction of GT entities found (case-insensitive substring)."""
    if not gt_list:
        return 1.0
    pred_lower = " ".join(pred_list).lower()
    hits = sum(1 for e in gt_list if e.lower() in pred_lower)
    return round(hits / len(gt_list), 4)


def _sentiment_score(pred: str, gt: str) -> float:
    return 1.0 if pred.lower() == gt.lower() else 0.0


def _requires_human_score(pred: bool, gt: bool) -> float:
    return 1.0 if pred == gt else 0.0


def _response_score(draft: Optional[str], keywords: List[str]) -> float:  # type: ignore[name-defined]
    """Keyword coverage as proxy for response quality."""
    if not draft:
        return 0.0
    draft_lower = draft.lower()
    hits = sum(1 for kw in keywords if kw.lower() in draft_lower)
    return round(hits / len(keywords), 4) if keywords else 1.0


def grade_easy(action: TriageAction, email: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    p = _priority_score(action.priority, email["gt_priority"])
    return round(p, 4), {"priority": p}


def grade_medium(action: TriageAction, email: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    p = _priority_score(action.priority, email["gt_priority"])
    c = _category_score(action.category, email["gt_category"])
    e = _entity_score(action.key_entities, email["gt_entities"])
    breakdown = {"priority": p, "category": c, "entity_extraction": e}
    total = round(0.4 * p + 0.35 * c + 0.25 * e, 4)
    return total, breakdown


def grade_hard(action: TriageAction, email: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    p  = _priority_score(action.priority, email["gt_priority"])
    c  = _category_score(action.category, email["gt_category"])
    e  = _entity_score(action.key_entities, email["gt_entities"])
    s  = _sentiment_score(action.sentiment, email["gt_sentiment"])
    rh = _requires_human_score(action.requires_human, email["gt_requires_human"])
    rd = _response_score(action.response_draft, email["gt_response_keywords"])
    breakdown = {
        "priority": p, "category": c, "entity_extraction": e,
        "sentiment": s, "requires_human": rh, "response_draft": rd,
    }
    total = round(0.25*p + 0.20*c + 0.15*e + 0.15*s + 0.10*rh + 0.15*rd, 4)
    return total, breakdown


GRADERS = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class EmailTriageEnvironment:
    """
    OpenEnv-compatible Email Triage Environment.

    Each episode runs one task (easy / medium / hard).
    step() processes one email; episode ends after all emails for that task.
    """

    def __init__(self, task: str = "easy"):
        assert task in ("easy", "medium", "hard"), f"Unknown task: {task}"
        self._task = task
        self._state = TriageState()
        self._queue: List[Dict[str, Any]] = []
        self._current_email: Dict[str, Any] = {}

    # ── OpenEnv API ─────────────────────────────────────────────────────────

    def reset(self) -> EmailObservation:
        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            current_task=TASK_META[self._task]["name"],
        )
        self._queue = [e for e in EMAILS if e["task"] == self._task]
        return self._next_observation(reward=None, done=False)

    def step(self, action: TriageAction) -> EmailObservation:
        if not self._current_email:
            # reset was not called or queue exhausted
            obs = self._next_observation(reward=0.0, done=True)
            obs.metadata["error"] = "No active email. Call reset() first."
            return obs

        grader = GRADERS[self._task]
        reward, breakdown = grader(action, self._current_email)
        graded_email = self._current_email  # save before queue pop

        self._state.step_count += 1
        self._state.emails_processed += 1
        self._state.cumulative_reward += reward

        # Advance queue
        self._queue.pop(0)
        done = len(self._queue) == 0

        obs = self._next_observation(reward=reward, done=done)
        obs.step_reward_breakdown = breakdown
        obs.metadata["graded_email_id"] = graded_email["email_id"]
        obs.metadata["cumulative_reward"] = round(self._state.cumulative_reward, 4)
        return obs

    @property
    def state(self) -> TriageState:
        return self._state

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _next_observation(self, reward, done: bool) -> EmailObservation:
        if self._queue:
            self._current_email = self._queue[0]
            email = self._current_email
        else:
            self._current_email = {}
            email = {
                "email_id": "", "subject": "", "body": "",
                "sender": "", "sender_tier": "free", "emails_in_thread": 0,
            }

        meta = TASK_META[self._task]
        return EmailObservation(
            email_id=email.get("email_id", ""),
            subject=email.get("subject", ""),
            body=email.get("body", ""),
            sender=email.get("sender", ""),
            sender_tier=email.get("sender_tier", "free"),
            emails_in_thread=email.get("emails_in_thread", 1),
            task_name=meta["name"],
            task_description=meta["description"],
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "emails_remaining": len(self._queue),
            },
        )

    def list_tasks(self) -> List[Dict[str, Any]]:
        return [
            {
                "task_id": task,
                "name": TASK_META[task]["name"],
                "description": TASK_META[task]["description"],
                "difficulty": {"easy": "easy", "medium": "medium", "hard": "hard"}[task],
                "num_emails": len([e for e in EMAILS if e["task"] == task]),
            }
            for task in ("easy", "medium", "hard")
        ]

    def run_grader(self, task_id: str, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Enumerate task emails, run grader, return scores."""
        emails = [e for e in EMAILS if e["task"] == task_id]
        grader = GRADERS[task_id]
        results = []
        for email in emails:
            action = TriageAction(**{k: v for k, v in action_dict.items()
                                     if k in TriageAction.__dataclass_fields__})
            reward, breakdown = grader(action, email)
            results.append({
                "email_id": email["email_id"],
                "reward": reward,
                "breakdown": breakdown,
            })
        avg = sum(r["reward"] for r in results) / len(results) if results else 0.0
        return {"task": task_id, "average_reward": round(avg, 4), "per_email": results}


# Allow optional type hint without runtime circular import
from typing import Optional  # noqa: E402
