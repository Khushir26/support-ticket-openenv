"""
models.py — Typed Pydantic models for the Support Ticket Agent OpenEnv environment.
Satisfies OpenEnv spec: typed Observation, Action, Reward models.
PHASE 2 FIX: All score fields use gt=0.0, lt=1.0 (strictly between 0 and 1).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

VALID_DEPARTMENTS: List[str] = [
    "Technical", "Billing", "Product", "IT", "Returns", "Sales", "HR"
]


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


class TicketAction(BaseModel):
    department: str = Field(..., description="One of the 7 valid departments")
    priority: int   = Field(2, ge=1, le=3, description="1=Low 2=Medium 3=High")
    reply: Optional[str] = Field("", description="Draft first reply (Task 3 only)")


class TicketReward(BaseModel):
    # CRITICAL: gt/lt (not ge/le) — strictly between 0 and 1
    score:            float = Field(..., gt=0.0, lt=1.0)
    department_score: float = Field(..., gt=0.0, lt=1.0)
    priority_score:   float = Field(..., gt=0.0, lt=1.0)
    reply_score:      float = Field(..., gt=0.0, lt=1.0)
    feedback: str
    done: bool
    correct_department: Optional[str] = None
    correct_priority:   Optional[int] = None


class EnvState(BaseModel):
    task_id: str
    current_ticket_index: int
    step: int
    done: bool
    cumulative_score: float
    total_tickets: int
    scores_history: List[float] = Field(default_factory=list)


class ResetResponse(BaseModel):
    observation: TicketObservation
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    observation: TicketObservation
    reward: TicketReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
