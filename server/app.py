"""
Email Triage Environment — FastAPI Server
Exposes /reset, /step, /state, /tasks, /grader, /health endpoints.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Dual-import pattern: relative in-repo, bare in Docker (PYTHONPATH=/app/env)
try:
    from ..models import EmailObservation, TriageAction, TriageState
    from .email_triage_environment import EmailTriageEnvironment
except ImportError:
    from models import EmailObservation, TriageAction, TriageState
    from server.email_triage_environment import EmailTriageEnvironment

# ---------------------------------------------------------------------------
# App factory (one instance per session — adequate for hackathon evaluation)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "A real-world OpenEnv environment where AI agents learn to triage "
        "customer-support emails: prioritise, categorise, extract entities, "
        "and draft responses."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instances keyed by task
_envs: Dict[str, EmailTriageEnvironment] = {
    "easy": EmailTriageEnvironment("easy"),
    "medium": EmailTriageEnvironment("medium"),
    "hard": EmailTriageEnvironment("hard"),
}
_active_task: str = "easy"


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic v2 compatible)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "easy"   # easy | medium | hard


class StepRequest(BaseModel):
    task: str = "easy"
    priority: str = "medium"
    category: str = "general"
    sentiment: str = "neutral"
    key_entities: list = []
    requires_human: bool = False
    response_draft: str | None = None
    reasoning: str | None = None


class GraderRequest(BaseModel):
    task_id: str
    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "email-triage-env", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    global _active_task
    if req.task not in _envs:
        raise HTTPException(400, f"Unknown task '{req.task}'. Choose: easy, medium, hard")
    _active_task = req.task
    obs = _envs[req.task].reset()
    return asdict(obs)


@app.post("/step")
def step(req: StepRequest):
    task = req.task if req.task in _envs else _active_task
    action = TriageAction(
        priority=req.priority,
        category=req.category,
        sentiment=req.sentiment,
        key_entities=req.key_entities,
        requires_human=req.requires_human,
        response_draft=req.response_draft,
        reasoning=req.reasoning,
    )
    obs = _envs[task].step(action)
    return asdict(obs)


@app.get("/state")
def state(task: str = "easy"):
    if task not in _envs:
        raise HTTPException(400, f"Unknown task '{task}'")
    return asdict(_envs[task].state)


@app.get("/tasks")
def list_tasks():
    """Return all available tasks with metadata — required by OpenEnv spec."""
    return _envs["easy"].list_tasks()


@app.post("/grader")
def run_grader(req: GraderRequest):
    """
    Enumerate a task's emails, run the grader on the supplied action,
    and return per-email rewards (all in [0.0, 1.0]).
    """
    if req.task_id not in _envs:
        raise HTTPException(400, f"Unknown task '{req.task_id}'")
    return _envs[req.task_id].run_grader(req.task_id, req.action)


@app.get("/")
def root():
    return {
        "name": "email-triage-env",
        "description": "OpenEnv Email Triage Environment",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks", "/grader"],
        "tasks": ["easy", "medium", "hard"],
    }
