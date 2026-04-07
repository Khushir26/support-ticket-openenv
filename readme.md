# Support Ticket Agent — OpenEnv Environment

**Real-world customer support ticket triage environment for RL agent evaluation.**

An AI agent reads incoming support tickets and must classify the correct department, assign priority, and draft a professional first reply. Powered by the [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) dataset on HuggingFace.

---

## Environment Description

Customer support triage is a task every company with a support inbox does daily. An agent must:
1. Read a ticket (subject + body)
2. Route it to the correct department (7 options)
3. Assign urgency priority (1/2/3)
4. Draft a professional first reply

This is a genuine, high-value real-world task — getting routing wrong costs companies hours of delay; a good first reply reduces back-and-forth by 40%.

---

## Tasks

| Task | Name | Difficulty | Reward Signal |
|------|------|-----------|---------------|
| `task1` | Department Classification | Easy | Binary: 1.0 correct, 0.0 wrong |
| `task2` | Classification + Priority | Medium | Dept (60%) + Priority (40%) |
| `task3` | Triage + Draft Reply | Hard | Dept (40%) + Priority (30%) + Reply quality (30%) |

### Task 1 — Department Classification (Easy)
Classify the ticket into exactly one of 7 departments. Binary reward: correct = 1.0, wrong = 0.0.

### Task 2 — Classification + Priority (Medium)
Classify department AND assign priority (1=Low, 2=Medium, 3=High). Partial credit: correct department only → 0.60; correct priority only → 0.40; both correct → 1.00.

### Task 3 — Triage + Draft Reply (Hard)
Three-component reward:
- **Department** (40%): correct routing
- **Priority** (30%): correct urgency
- **Reply quality** (30%): keyword overlap with gold reply + length appropriateness + professionalism signals

---

## Action Space

```json
{
  "department": "Technical",
  "priority": 2,
  "reply": "Dear Customer..."
}
```

**Valid departments:** `Technical`, `Billing`, `Product`, `IT`, `Returns`, `Sales`, `HR`

**Priority:** `1` = Low, `2` = Medium, `3` = High

---

## Observation Space

```json
{
  "ticket_id": "HF-00042",
  "subject": "Login error 403 Forbidden",
  "body": "I cannot log in to my account...",
  "customer_name": "Customer",
  "task_id": "task1",
  "step": 1,
  "max_steps": 20,
  "valid_departments": ["Technical", "Billing", "Product", "IT", "Returns", "Sales", "HR"],
  "instructions": "Classify this ticket..."
}
```

---

## Reward Function

### Task 1
`score = 1.0 if department == gold_department else 0.0`

### Task 2
`score = dept_correct * 0.6 + priority_correct * 0.4`

### Task 3
`score = dept_correct * 0.4 + priority_correct * 0.3 + reply_quality * 0.3`

All scores guaranteed in [0.0, 1.0]. Graders are fully deterministic.

---

## Dataset

**Source:** [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)

Loaded via the `datasets` library. English tickets are filtered and department labels normalised to 7 canonical categories. A curated fallback dataset guarantees all 7 departments are represented even if HF is unreachable.

---

## Setup & Usage

### Prerequisites
```bash
python --version  # 3.10, 3.11, or 3.12
```

### Install
```bash
pip install -r requirements.txt
```

### Local demo (no API key needed)
```bash
python demo.py
```

### Run baseline inference (with LLM)
```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Start API server
```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t support-ticket-agent .
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx support-ticket-agent
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit action, get reward |
| `/state` | GET | Current episode state |
| `/tasks` | GET | List all tasks |
| `/grader` | POST | Score a single action |

---

## Baseline Scores

| Task | Rule-based | LLM (Qwen2.5-72B) |
|------|-----------|-------------------|
| task1 (Easy) | ~0.75 | ~0.88 |
| task2 (Medium) | ~0.55 | ~0.70 |
| task3 (Hard) | ~0.40 | ~0.55 |

---

## Project Structure

```
support-ticket-agent/
├── main.py           # FastAPI server
├── environment.py    # Core environment + dataset loading
├── models.py         # Pydantic models
├── graders.py        # Deterministic graders
├── inference.py      # Baseline inference script
├── demo.py           # Local demo
├── openenv.yaml      # OpenEnv metadata
├── requirements.txt  # Dependencies
├── Dockerfile        # Container definition
└── README.md         # This file
```

---

## Team

**The Avengers** — OpenEnv Hackathon 2026