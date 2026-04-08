from __future__ import annotations
import re
from typing import Optional, Set

VALID_DEPARTMENTS = ["Technical", "Billing", "Product", "IT", "Returns", "Sales", "HR"]

_MIN_SCORE = 0.001
_MAX_SCORE = 0.999

def _clamp(score: float) -> float:
    return round(max(_MIN_SCORE, min(_MAX_SCORE, score)), 4)


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
    if not reply or not reply.strip():
        return _clamp(_MIN_SCORE)

    wc = len(reply.split())

    if wc < 5:
        length_score = 0.05
    elif wc < 15:
        length_score = 0.3
    elif wc <= 120:
        length_score = 0.9
    else:
        length_score = 0.7

    gold_kws = _keywords(gold_reply)
    pred_kws = _keywords(reply)

    if not gold_kws:
        overlap = 0.5
    else:
        overlap = min(len(gold_kws & pred_kws) / len(gold_kws), 0.99)

    final = overlap * 0.7 + length_score * 0.3
    return _clamp(final)


# ── TASK 1 ─────────────────────────────

def grade_task1(pred_dept, gold_dept, step, max_steps):
    d_ok = _dept_ok(pred_dept, gold_dept)

    dept_score = _clamp(0.999 if d_ok else 0.001)

    score = _clamp(dept_score)

    return {
        "score": score,
        "department_score": dept_score,
        "priority_score": _clamp(0.001),
        "reply_score": _clamp(0.001),
        "correct_department": gold_dept,
        "correct_priority": None,
        "feedback": f"Dept {'OK' if d_ok else 'WRONG'}"
    }


# ── TASK 2 ─────────────────────────────

def grade_task2(pred_dept, pred_prio, gold_dept, gold_prio, step, max_steps):
    d_ok = _dept_ok(pred_dept, gold_dept)
    p_ok = _prio_ok(pred_prio, gold_prio)

    dept_score = _clamp(0.999 if d_ok else 0.001)
    prio_score = _clamp(0.999 if p_ok else 0.001)

    raw_score = dept_score * 0.6 + prio_score * 0.4
    score = _clamp(raw_score)

    return {
        "score": score,
        "department_score": dept_score,
        "priority_score": prio_score,
        "reply_score": _clamp(0.001),
        "correct_department": gold_dept,
        "correct_priority": int(gold_prio),
        "feedback": f"Dept {d_ok}, Prio {p_ok}"
    }


# ── TASK 3 ─────────────────────────────

def grade_task3(pred_dept, pred_prio, pred_reply,
                gold_dept, gold_prio, gold_reply,
                step, max_steps):

    d_ok = _dept_ok(pred_dept, gold_dept)
    p_ok = _prio_ok(pred_prio, gold_prio)

    dept_score = _clamp(0.999 if d_ok else 0.001)
    prio_score = _clamp(0.999 if p_ok else 0.001)
    reply_score = _clamp(_reply_quality(pred_reply or "", gold_reply))

    raw_score = dept_score * 0.4 + prio_score * 0.3 + reply_score * 0.3
    score = _clamp(raw_score)

    return {
        "score": score,
        "department_score": dept_score,
        "priority_score": prio_score,
        "reply_score": reply_score,
        "correct_department": gold_dept,
        "correct_priority": int(gold_prio),
        "feedback": f"Dept {d_ok}, Prio {p_ok}, Reply {reply_score:.3f}"
    }
