"""
inference.py  —  Baseline Inference Script for Email Triage OpenEnv
===================================================================
- Uses OpenAI-compatible client (API_BASE_URL + MODEL_NAME + HF_TOKEN)
- Runs the LLM agent against all 3 tasks (easy, medium, hard)
- Emits structured stdout logs in [START] / [STEP] / [END] format
- Runtime: < 20 min on vcpu=2, 8 GB RAM
- Reproducible: fixed random seed, deterministic graders

Usage:
    export API_BASE_URL=https://api-inference.huggingface.co/v1
    export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
    export HF_TOKEN=hf_...
    python inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

TASKS = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# Logging helpers — must match [START] / [STEP] / [END] exactly
# ---------------------------------------------------------------------------

def log(tag: str, payload: Dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def ts() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------------------------------------------------------------------------
# HTTP helpers for environment
# ---------------------------------------------------------------------------

def env_reset(task: str) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# System prompt for the triage agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert customer-support triage agent.
Given an email, respond ONLY with a JSON object (no markdown fences) with these fields:
{
  "priority": "<low|medium|high|critical>",
  "category": "<billing|technical|sales|general|security>",
  "sentiment": "<positive|neutral|negative|frustrated>",
  "key_entities": ["<entity1>", "<entity2>"],
  "requires_human": <true|false>,
  "response_draft": "<concise professional reply to the customer>",
  "reasoning": "<one-sentence justification>"
}

Rules:
- priority=critical → payment failure, data breach, complete service outage, < 2 hr deadline
- priority=high → significant impact, same-day resolution needed
- priority=medium → affects work but has workaround
- priority=low → informational, non-urgent
- key_entities: extract order IDs, plan names, error codes, IPs, version numbers
- response_draft: address the customer's actual concern, professional tone, 2-4 sentences
"""

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(observation: Dict[str, Any]) -> Dict[str, Any]:
    user_msg = (
        f"Task: {observation.get('task_name')} — {observation.get('task_description')}\n\n"
        f"From: {observation.get('sender')} [{observation.get('sender_tier')} plan]\n"
        f"Subject: {observation.get('subject')}\n\n"
        f"{observation.get('body')}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as exc:
        # Fallback: heuristic action so the episode can still complete
        print(f"[WARN] LLM parse error ({exc}); using heuristic fallback.", file=sys.stderr)
        body = observation.get("body", "").lower()
        priority = "critical" if any(w in body for w in ["urgent", "immediately", "asap", "breach"]) else \
                   "high"     if any(w in body for w in ["error", "fail", "not working", "broken"]) else \
                   "medium"   if any(w in body for w in ["issue", "problem", "question"]) else "low"
        category = "security"  if "breach" in body or "unauthorized" in body else \
                   "technical" if any(w in body for w in ["error", "500", "api", "deploy"]) else \
                   "billing"   if any(w in body for w in ["payment", "invoice", "gst"]) else \
                   "sales"     if any(w in body for w in ["upgrade", "plan", "quote"]) else "general"
        return {
            "priority": priority, "category": category, "sentiment": "neutral",
            "key_entities": [], "requires_human": priority == "critical",
            "response_draft": "Thank you for contacting support. We will look into this.",
            "reasoning": "heuristic fallback",
        }

# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(task: str) -> Dict[str, Any]:
    log("START", {
        "task": task,
        "model": MODEL_NAME,
        "env_url": ENV_BASE_URL,
        "timestamp": ts(),
    })

    obs = env_reset(task)
    step_num = 0
    total_reward = 0.0

    while not obs.get("done", False):
        # Let LLM decide action
        action_dict = call_llm(obs)
        action_dict["task"] = task  # tell server which env

        # Step environment
        next_obs = env_step(action_dict)
        reward    = next_obs.get("reward") or 0.0
        done      = next_obs.get("done", False)
        step_num += 1
        total_reward += reward

        log("STEP", {
            "step": step_num,
            "task": task,
            "email_id": obs.get("email_id"),
            "action": {k: v for k, v in action_dict.items() if k != "task"},
            "reward": reward,
            "reward_breakdown": next_obs.get("step_reward_breakdown", {}),
            "done": done,
            "cumulative_reward": round(total_reward, 4),
            "timestamp": ts(),
        })

        obs = next_obs
        if done:
            break

    # Normalize reward to number of emails processed
    avg_reward = round(total_reward / max(step_num, 1), 4)

    log("END", {
        "task": task,
        "model": MODEL_NAME,
        "total_steps": step_num,
        "total_reward": round(total_reward, 4),
        "average_reward": avg_reward,
        "success": avg_reward >= 0.5,
        "timestamp": ts(),
    })

    return {"task": task, "average_reward": avg_reward, "steps": step_num}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("Email Triage OpenEnv — Baseline Inference Script", flush=True)
    print(f"Model : {MODEL_NAME}", flush=True)
    print(f"Env   : {ENV_BASE_URL}", flush=True)
    print("=" * 60, flush=True)

    results: List[Dict[str, Any]] = []
    for task in TASKS:
        result = run_task(task)
        results.append(result)
        time.sleep(1)  # brief pause between tasks

    # Summary
    overall = round(sum(r["average_reward"] for r in results) / len(results), 4)
    print("\n" + "=" * 60, flush=True)
    print("FINAL SCORES", flush=True)
    for r in results:
        print(f"  {r['task']:8s}  avg_reward={r['average_reward']:.4f}  steps={r['steps']}", flush=True)
    print(f"\n  OVERALL   avg_reward={overall:.4f}", flush=True)
    print("=" * 60, flush=True)

    # Machine-readable final line for automated evaluation
    log("SUMMARY", {
        "results": results,
        "overall_average_reward": overall,
        "model": MODEL_NAME,
        "timestamp": ts(),
    })


if __name__ == "__main__":
    main()
