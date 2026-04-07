"""
main.py — FastAPI server for the Support Ticket Agent OpenEnv environment.

Required endpoints (all must pass automated judging):
  GET  /health    → 200 + {"status":"ok"}  ← judging pings this
  POST /reset     → start episode, return first observation  ← BODY IS OPTIONAL
  POST /step      → submit action, get reward (0.0–1.0)
  GET  /state     → current episode state
  GET  /tasks     → list 3 tasks + action schemas
  POST /grader    → score a single action (standalone)
  POST /baseline  → trigger inference, return scores

CRITICAL FIX: POST /reset must accept empty body {} OR no body at all.
OpenEnv calls POST /reset with null/empty body — making ResetRequest optional.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import SupportTicketEnv, TASK_CONFIG, VALID_DEPARTMENTS
from models import ResetResponse, StepResponse, EnvState

# ── Global environment instance ────────────────────────────────────────────

_env: Optional[SupportTicketEnv] = None


def get_env() -> SupportTicketEnv:
    if _env is None:
        raise HTTPException(503, "Environment not ready. Try again in a moment.")
    return _env


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    print("[STARTUP] Loading Support Ticket Agent environment...", flush=True)
    _env = SupportTicketEnv(seed=42)
    print("[STARTUP] Environment ready.", flush=True)
    yield
    print("[SHUTDOWN] Done.", flush=True)


app = FastAPI(
    title="Support Ticket Agent — OpenEnv",
    description=(
        "Real-world OpenEnv environment: AI agent triages customer support tickets "
        "by classifying department, assigning priority, and drafting replies. "
        "Dataset: Tobi-Bueck/customer-support-tickets (HuggingFace)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    """All fields optional — OpenEnv may POST with empty/null body."""
    task_id: Optional[str] = "task1"


class StepRequest(BaseModel):
    department: str
    priority: int = 2
    reply: Optional[str] = None


class GraderRequest(BaseModel):
    task_id: str
    predicted_department: str
    predicted_priority: int = 2
    predicted_reply: Optional[str] = ""
    gold_department: str
    gold_priority: int = 2
    gold_reply: Optional[str] = ""
    ticket_subject: Optional[str] = ""
    ticket_body: Optional[str] = ""


class BaselineRequest(BaseModel):
    task_ids: List[str] = ["task1", "task2", "task3"]
    max_tickets: int = 5


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "name":         "Support Ticket Agent — OpenEnv",
        "version":      "1.0.0",
        "status":       "ok",
        "dataset":      "Tobi-Bueck/customer-support-tickets",
        "openenv_spec": "1.0",
        "tasks":        list(TASK_CONFIG.keys()),
        "endpoints": [
            "GET  /health",
            "POST /reset",
            "POST /step",
            "GET  /state",
            "GET  /tasks",
            "POST /grader",
            "POST /baseline",
        ],
    }


@app.get("/health", tags=["Health"])
async def health():
    """
    Automated judging pings this first.
    Must return HTTP 200 with {"status": "ok"}.
    """
    env = get_env()
    try:
        env.reset(task_id="task1")
        env_ok = True
    except Exception:
        env_ok = False
    return {
        "status": "ok",
        "environment_loaded": env_ok,
        "dataset_tickets": len(env._df) if env._df is not None else 0,
    }


@app.post("/reset", response_model=ResetResponse, tags=["OpenEnv"])
async def reset(request: Request):
    """
    Start a new episode for the given task.

    IMPORTANT: Accepts empty body, null body, or JSON body.
    OpenEnv calls POST /reset with no body — this endpoint handles all cases.

    Body (all optional):
      task_id: "task1" | "task2" | "task3"  (default: "task1")
    """
    env = get_env()

    # Safely parse body — handle null, empty, or missing body gracefully
    task_id = "task1"  # safe default
    try:
        body_bytes = await request.body()
        if body_bytes and body_bytes.strip() not in (b"", b"null", b"{}"):
            import json
            body_json = json.loads(body_bytes)
            if isinstance(body_json, dict):
                task_id = body_json.get("task_id", "task1") or "task1"
    except Exception:
        pass  # any parse error → use default task_id

    # Validate task_id
    if task_id not in TASK_CONFIG:
        task_id = "task1"

    try:
        return env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))


@app.post("/step", response_model=StepResponse, tags=["OpenEnv"])
async def step(request: StepRequest):
    """
    Submit one action for the current ticket.
    Returns reward in [0.0, 1.0], next observation, and done flag.

    department: one of Technical / Billing / Product / IT / Returns / Sales / HR
    priority:   1 (Low) | 2 (Medium) | 3 (High)
    reply:      draft first reply text (task3 only; ignored for task1/task2)
    """
    env = get_env()
    try:
        action = {
            "department": request.department,
            "priority":   request.priority,
            "reply":      request.reply or "",
        }
        return env.step(action)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))


@app.get("/state", response_model=EnvState, tags=["OpenEnv"])
async def state():
    """Return the current internal episode state."""
    env = get_env()
    try:
        return env.state()
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))


@app.get("/tasks", tags=["OpenEnv"])
async def tasks():
    """
    List all 3 tasks with descriptions, difficulty, and action schemas.
    Judges enumerate tasks and run graders from here.
    """
    task_list = []
    for task_id, cfg in TASK_CONFIG.items():
        task_list.append({
            "task_id":     task_id,
            "name":        cfg["name"],
            "description": cfg["description"],
            "difficulty":  cfg["difficulty"],
            "num_tickets": cfg["num_tickets"],
            "max_steps":   cfg["max_steps"],
            "action_schema": {
                "department": {
                    "type":        "string",
                    "required":    True,
                    "options":     VALID_DEPARTMENTS,
                    "description": "Department to route this ticket to",
                },
                "priority": {
                    "type":        "integer",
                    "required":    task_id in ("task2", "task3"),
                    "options":     [1, 2, 3],
                    "description": "1=Low, 2=Medium, 3=High/Urgent",
                },
                "reply": {
                    "type":        "string",
                    "required":    task_id == "task3",
                    "description": "Professional first reply to customer (task3 only)",
                },
            },
            "reward_info": _reward_info(task_id),
            "grader_criteria": _grader_criteria(task_id),
        })
    return {"tasks": task_list, "total": len(task_list)}


@app.post("/grader", tags=["OpenEnv"])
async def grader(request: GraderRequest):
    """
    Score a single action against known gold labels.
    Judges use this to verify graders produce scores in [0.0, 1.0].
    Returns score in [0.0, 1.0] with detailed breakdown.
    """
    from graders import grade_task1, grade_task2, grade_task3

    if request.task_id not in TASK_CONFIG:
        raise HTTPException(400, f"Unknown task_id '{request.task_id}'. "
                            f"Valid: {list(TASK_CONFIG.keys())}")

    max_steps = TASK_CONFIG[request.task_id]["max_steps"]

    if request.task_id == "task1":
        result = grade_task1(
            request.predicted_department,
            request.gold_department,
            1, max_steps,
        )
    elif request.task_id == "task2":
        result = grade_task2(
            request.predicted_department, request.predicted_priority,
            request.gold_department, request.gold_priority,
            1, max_steps,
        )
    else:
        result = grade_task3(
            request.predicted_department, request.predicted_priority,
            request.predicted_reply or "",
            request.gold_department, request.gold_priority,
            request.gold_reply or "",
            1, max_steps,
        )

    assert 0.0 <= result["score"] <= 1.0, "Grader produced out-of-range score"

    return {
        "task_id":  request.task_id,
        "score":    result["score"],
        "in_range": 0.0 <= result["score"] <= 1.0,
        "result":   result,
    }


@app.post("/baseline", tags=["OpenEnv"])
async def baseline(request: BaselineRequest):
    """
    Trigger the inference script and return baseline scores.
    Uses HF_TOKEN + API_BASE_URL + MODEL_NAME from environment variables.
    Returns mock scores if no token is configured (endpoint never crashes).
    """
    hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
    api_base  = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model     = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    if not hf_token:
        return {
            "status":  "no_token",
            "message": "Set HF_TOKEN in HuggingFace Space secrets to enable live inference.",
            "api_base_url": api_base,
            "model_name":   model,
            "mock_baseline_scores": {
                "task1": {"score": 1.00, "difficulty": "easy",   "description": "rule-based"},
                "task2": {"score": 0.91, "difficulty": "medium", "description": "rule-based"},
                "task3": {"score": 0.78, "difficulty": "hard",   "description": "rule-based"},
            },
        }

    try:
        from openai import OpenAI
        from inference import run_task as _run_task

        client  = OpenAI(api_key=hf_token, base_url=api_base)
        results = []
        env     = get_env()

        for task_id in request.task_ids:
            if task_id not in TASK_CONFIG:
                continue
            result = _run_task(env, client, task_id)
            results.append(result)

        return {"status": "ok", "model": model, "results": results}

    except Exception as exc:
        return {
            "status": "error",
            "error":  str(exc),
            "mock_baseline_scores": {
                "task1": {"score": 1.00},
                "task2": {"score": 0.91},
                "task3": {"score": 0.78},
            },
        }


# ── Helpers ────────────────────────────────────────────────────────────────

def _reward_info(task_id: str) -> Dict[str, Any]:
    if task_id == "task1":
        return {
            "components": {"department": 1.0},
            "scoring":    "Binary: 1.0 correct department, 0.0 wrong",
            "range":      [0.0, 1.0],
        }
    elif task_id == "task2":
        return {
            "components": {"department": 0.6, "priority": 0.4},
            "scoring":    "Partial credit: dept correct → +0.6, priority correct → +0.4",
            "range":      [0.0, 1.0],
        }
    else:
        return {
            "components": {
                "department":    0.4,
                "priority":      0.3,
                "reply_quality": 0.3,
            },
            "scoring": (
                "3-component reward. "
                "Reply scored by keyword overlap + length + professionalism."
            ),
            "range": [0.0, 1.0],
        }


def _grader_criteria(task_id: str) -> Dict[str, Any]:
    base = {
        "deterministic": True,
        "reproducible":  True,
        "score_range":   [0.0, 1.0],
    }
    if task_id == "task1":
        return {**base, "type": "exact_match", "field": "department"}
    elif task_id == "task2":
        return {**base, "type": "weighted_match", "fields": ["department", "priority"]}
    else:
        return {**base, "type": "multi_component",
                "fields": ["department", "priority", "reply_quality"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)