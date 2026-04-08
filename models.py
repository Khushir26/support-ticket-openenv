"""
models.py — Typed Pydantic models for the Support Ticket Agent OpenEnv environment.
Satisfies OpenEnv spec: typed Observation, Action, Reward models.
PHASE 2 FIX: Using ge=0.001, le=0.999 to be more permissive while still ensuring (0,1) range.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

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
    priority: int = Field(2, ge=1, le=3, description="1=Low 2=Medium 3=High")
    reply: Optional[str] = Field("", description="Draft first reply (Task 3 only)")


class TicketReward(BaseModel):
    """
    Reward model with scores strictly between 0 and 1.
    Using validators to ensure compliance.
    """
    score: float = Field(..., description="Overall score strictly between 0 and 1")
    department_score: float = Field(..., description="Department classification score")
    priority_score: float = Field(..., description="Priority classification score")
    reply_score: float = Field(..., description="Reply quality score")
    feedback: str
    done: bool
    correct_department: Optional[str] = None
    correct_priority: Optional[int] = None

    @field_validator('score', 'department_score', 'priority_score', 'reply_score', mode='before')
    @classmethod
    def clamp_score(cls, v):
        """Ensure all scores are strictly between 0 and 1."""
        if v is None:
            return 0.5  # Default neutral score
        v = float(v)
        # Clamp to strictly within (0, 1)
        if v <= 0.0:
            return 0.01
        if v >= 1.0:
            return 0.99
        return round(v, 4)


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
