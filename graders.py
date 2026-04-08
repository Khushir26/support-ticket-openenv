"""
graders.py — Fixed graders for Support Ticket Agent OpenEnv environment.
PHASE 2 FIX: Using 0.01-0.99 bounds to avoid any floating-point edge cases.
All scores are STRICTLY between 0 and 1 (never 0.0 or 1.0).
"""
from __future__ import annotations
import re
from typing import Optional, Set

VALID_DEPARTMENTS = ["Technical", "Billing", "Product", "IT", "Returns", "Sales", "HR"]

# Use wider margins to be absolutely safe
_MIN_SCORE = 0.01
_MAX_SCORE = 0.99


def _clamp(score: float) -> float:
    """Clamp score strictly between 0 and 1 with safe margins."""
    clamped = max(_MIN_SCORE, min(_MAX_SCORE, float(score)))
    return round(clamped, 4)


def _norm_dept(dept: str) -> str:
    return dept.strip().lower()


def _dept_ok(predicted: str, gold: str) -> bool:
    return _norm_dept(predicted) == _norm_dept(gold)


def _prio_ok(predicted, gold) -> bool:
    try:
        return int(predicted) == int(gold)
    except:
        return False


def _keywords(text: str) -> Set[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return set(words)


def _reply_quality(reply: str, gold_reply: str) -> float:
    """Score reply quality - always returns value in (0, 1) range."""
    if not reply or not reply.strip():
        return _MIN_SCORE

    wc = len(reply.split())

    if wc < 5:
        length_score = 0.1
    elif wc < 15:
        length_score = 0.35
    elif wc <= 120:
        length_score = 0.85
    else:
        length_score = 0.65

    gold_kws = _keywords(gold_reply)
    pred_kws = _keywords(reply)

    if not gold_kws:
        overlap = 0.5
    else:
        # Cap at 0.95 to stay safely below 1.0
        overlap = min(len(gold_kws & pred_kws) / len(gold_kws), 0.95)

    final = overlap * 0.7 + length_score * 0.3
    return _clamp(final)


# ── TASK 1 ─────────────────────────────

def grade_task1(pred_dept, gold_dept, step, max_steps):
    """Grade task 1: Department classification only."""
    d_ok = _dept_ok(pred_dept, gold_dept)

    # Use 0.95 for correct, 0.05 for wrong (safe margins)
    dept_score = _clamp(0.95 if d_ok else 0.05)

    score = dept_score  # Task 1 is department only

    return {
        "score": _clamp(score),
        "department_score": dept_score,
        "priority_score": _clamp(0.5),  # Neutral score for unused field
        "reply_score": _clamp(0.5),      # Neutral score for unused field
        "correct_department": gold_dept,
        "correct_priority": None,
        "feedback": f"Dept {'OK' if d_ok else 'WRONG'}"
    }


# ── TASK 2 ─────────────────────────────

def grade_task2(pred_dept, pred_prio, gold_dept, gold_prio, step, max_steps):
    """Grade task 2: Department + Priority classification."""
    d_ok = _dept_ok(pred_dept, gold_dept)
    p_ok = _prio_ok(pred_prio, gold_prio)

    # Use 0.95 for correct, 0.05 for wrong
    dept_score = _clamp(0.95 if d_ok else 0.05)
    prio_score = _clamp(0.95 if p_ok else 0.05)

    raw_score = dept_score * 0.6 + prio_score * 0.4
    score = _clamp(raw_score)

    return {
        "score": score,
        "department_score": dept_score,
        "priority_score": prio_score,
        "reply_score": _clamp(0.5),  # Neutral score for unused field
        "correct_department": gold_dept,
        "correct_priority": int(gold_prio),
        "feedback": f"Dept {'OK' if d_ok else 'WRONG'}, Prio {'OK' if p_ok else 'WRONG'}"
    }


# ── TASK 3 ─────────────────────────────

def grade_task3(pred_dept, pred_prio, pred_reply,
                gold_dept, gold_prio, gold_reply,
                step, max_steps):
    """Grade task 3: Department + Priority + Reply quality."""
    d_ok = _dept_ok(pred_dept, gold_dept)
    p_ok = _prio_ok(pred_prio, gold_prio)

    # Use 0.95 for correct, 0.05 for wrong
    dept_score = _clamp(0.95 if d_ok else 0.05)
    prio_score = _clamp(0.95 if p_ok else 0.05)
    reply_score = _reply_quality(pred_reply or "", gold_reply)

    raw_score = dept_score * 0.4 + prio_score * 0.3 + reply_score * 0.3
    score = _clamp(raw_score)

    return {
        "score": score,
        "department_score": dept_score,
        "priority_score": prio_score,
        "reply_score": reply_score,
        "correct_department": gold_dept,
        "correct_priority": int(gold_prio),
        "feedback": f"Dept {'OK' if d_ok else 'WRONG'}, Prio {'OK' if p_ok else 'WRONG'}, Reply {reply_score:.3f}"
    }
