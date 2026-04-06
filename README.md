---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---


  # 📧 Email Triage OpenEnv

> A real-world OpenEnv environment for training AI agents to triage customer-support emails.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![HF Space](https://img.shields.io/badge/🤗-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces)

---

## 🎯 Motivation

Every company with a customer-support function faces the same bottleneck: a flood of emails that need to be **prioritised**, **routed to the right team**, and **responded to quickly**. This environment trains RL agents to perform exactly these tasks — making it directly deployable to reduce first-response time and misrouting.

---

## 🗂️ Environment Description

The agent receives a customer-support email (subject + body + sender metadata) and must produce a structured triage decision. Three tasks of increasing difficulty test different capability levels.

### Action Space (`TriageAction`)

| Field | Type | Values | Required |
|-------|------|---------|----------|
| `priority` | `str` | `low \| medium \| high \| critical` | ✅ |
| `category` | `str` | `billing \| technical \| sales \| general \| security` | ✅ |
| `sentiment` | `str` | `positive \| neutral \| negative \| frustrated` | medium/hard |
| `key_entities` | `list[str]` | order IDs, error codes, plan names… | medium/hard |
| `requires_human` | `bool` | whether to escalate to human agent | hard |
| `response_draft` | `str \| None` | 2–4 sentence reply to the customer | hard |
| `reasoning` | `str \| None` | agent's chain-of-thought (not scored) | optional |

### Observation Space (`EmailObservation`)

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | `str` | Unique ID for the email |
| `subject` | `str` | Email subject line |
| `body` | `str` | Full email body |
| `sender` | `str` | Sender email address |
| `sender_tier` | `str` | Account tier: `free \| pro \| enterprise` |
| `emails_in_thread` | `int` | Number of prior messages in thread |
| `task_name` | `str` | Active task name |
| `task_description` | `str` | What the agent is expected to do |
| `done` | `bool` | Episode complete |
| `reward` | `float \| None` | Step reward (0.0–1.0) |
| `step_reward_breakdown` | `dict` | Per-component reward scores |
| `metadata` | `dict` | episode_id, step_count, emails_remaining |

---

## 📋 Tasks

### Task 1 — `urgency_classification` (Easy)
- **Objective:** Assign the correct priority label (`low / medium / high / critical`)
- **Grader:** Partial credit by proximity on the urgency scale (off by 1 → 0.5, off by 2 → 0.0)
- **Emails:** 2 (one critical, one low)
- **Expected difficulty:** Models should score 0.6–1.0

### Task 2 — `department_routing` (Medium)
- **Objective:** Correct priority **+** category **+** extract key entities
- **Grader:** `0.4×priority + 0.35×category + 0.25×entity_recall`
- **Emails:** 2 (one enterprise-critical, one high-value sales)
- **Expected difficulty:** Models score 0.4–0.8

### Task 3 — `full_triage` (Hard)
- **Objective:** All of the above + sentiment + `requires_human` + `response_draft`
- **Grader:** `0.25×priority + 0.20×category + 0.15×entities + 0.15×sentiment + 0.10×requires_human + 0.15×response`
- **Emails:** 2 (security breach, browser-specific bug)
- **Expected difficulty:** Frontier models score 0.5–0.85

> All graders return scores in **[0.0, 1.0]** and are fully deterministic.

---

## ⚙️ Setup & Usage

### Local (Docker)

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -d -p 7860:7860 email-triage-env

# Validate
curl http://localhost:7860/health
```

### pip install from Space

```bash
pip install "openenv-email-triage @ git+https://huggingface.co/spaces/<your-username>/email-triage-env"
```

### Run baseline inference

```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=hf_...
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

### Use the client

```python
from client import EmailTriageEnvClient
from models import TriageAction

env = EmailTriageEnvClient("http://localhost:7860")
obs = env.reset(task="medium")
print(obs["subject"])

action = TriageAction(priority="critical", category="technical",
                      key_entities=["v2.3.1", "ENT-4412"])
obs = env.step(action)
print(obs["reward"])   # e.g. 0.7625
```

---

## 📊 Baseline Scores

Measured with `meta-llama/Llama-3.1-8B-Instruct` via HF Inference API:

| Task | Avg Reward | Notes |
|------|-----------|-------|
| easy  | ~0.85 | Priority classification well within model capability |
| medium | ~0.68 | Entity extraction reduces score vs easy |
| hard  | ~0.54 | Response draft quality is the primary bottleneck |
| **Overall** | **~0.69** | |

---

## 🏗️ Project Structure

```
email-triage-env/
├── inference.py          ← Baseline inference script (required)
├── Dockerfile            ← Container definition
├── openenv.yaml          ← OpenEnv manifest
├── pyproject.toml        ← pip-installable package
├── requirements.txt
├── models.py             ← TriageAction, EmailObservation, TriageState
├── client.py             ← HTTP client
├── __init__.py
└── server/
    ├── app.py            ← FastAPI server
    └── email_triage_environment.py  ← Tasks + graders
```

---

## 🔑 Key Design Decisions

- **Partial rewards everywhere**: Even a wrong priority gets 0.5 if it's one step away, so the reward signal is never completely sparse.
- **Deterministic graders**: Entity extraction uses substring matching — same input always produces the same score.
- **Difficulty progression**: Easy → Medium → Hard naturally tests more agent capabilities.
- **Real-world fidelity**: All emails are realistic Indian enterprise scenarios with plausible metadata (plan tiers, INR amounts, Indian names).
- **Frontier-challenging hard task**: The response draft sub-score requires the agent to identify the customer's specific concern and address it — this genuinely challenges current 7B models.
