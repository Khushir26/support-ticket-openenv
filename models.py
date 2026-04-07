"""
models.py — Typed Pydantic models for the Support Ticket Agent OpenEnv environment.
Satisfies OpenEnv spec: typed Observation, Action, Reward models.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

VALID_DEPARTMENTS: List[str] = [
    "Technical", "Billing", "Product", "IT", "Returns", "Sales", "HR"
]


# ── Observation: what the agent SEES each step ─────────────────────────────
class TicketObservation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_name: str
    task_id: str
    step: int
    max_steps: int
    valid_departments: List[str] = Field(default_factory=lambda: list(VALID_DEPARTMENTS))
    instructions: str


# ── Action: what the agent SUBMITS ────────────────────────────────────────
class TicketAction(BaseModel):
    department: str = Field(..., description="One of the 7 valid departments")
    priority: int   = Field(2, ge=1, le=3, description="1=Low 2=Medium 3=High")
    reply: Optional[str] = Field("", description="Draft first reply (Task 3 only)")


# ── Reward: what the environment RETURNS after each step ──────────────────
class TicketReward(BaseModel):
    score: float            = Field(..., ge=0.0, le=1.0)
    department_score: float = Field(..., ge=0.0, le=1.0)
    priority_score: float   = Field(..., ge=0.0, le=1.0)
    reply_score: float      = Field(..., ge=0.0, le=1.0)
    feedback: str
    done: bool
    correct_department: Optional[str] = None
    correct_priority: Optional[int]   = None


# ── EnvState: internal episode tracking ───────────────────────────────────
class EnvState(BaseModel):
    task_id: str
    current_ticket_index: int
    step: int
    done: bool
    cumulative_score: float
    total_tickets: int
    scores_history: List[float] = Field(default_factory=list)


# ── API response wrappers ──────────────────────────────────────────────────
class ResetResponse(BaseModel):
    observation: TicketObservation
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    observation: TicketObservation
    reward: TicketReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)