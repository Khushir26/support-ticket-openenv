"""
graders.py — Deterministic graders for all 3 tasks.

Scoring design (NO step penalty — max_steps=1 per ticket):
  Task 1 — Department only            binary  0.0 or 1.0
  Task 2 — Dept 60% + Priority 40%    partial credit
  Task 3 — Dept 40% + Prio 30%        partial credit
           + Reply quality 30%

Reply quality:
  keyword overlap with gold  55%
  length appropriateness     25%
  professionalism signals    20%

100% deterministic. Scores always in [0.0, 1.0].
"""
from __future__ import annotations

import re
from typing import Optional, Set

VALID_DEPARTMENTS = ["Technical", "Billing", "Product", "IT", "Returns", "Sales", "HR"]

_SYNONYM_GROUPS = [
    {"issue", "problem", "error", "trouble", "fault", "bug", "concern"},
    {"resolve", "fix", "solve", "address", "handle", "investigate", "look into"},
    {"refund", "reimbursement", "credit", "reimburse", "return payment"},
    {"request", "query", "inquiry", "question", "ticket"},
    {"update", "inform", "notify", "follow up", "respond", "get back"},
    {"apologize", "sorry", "regret", "apologies", "apologise"},
    {"replace", "replacement", "exchange", "substitute", "send another"},
    {"urgently", "immediately", "asap", "priority", "promptly"},
    {"dispatch", "ship", "send", "deliver", "forward"},
    {"label", "return label", "prepaid", "shipping label"},
    {"business day", "working day", "calendar day"},
    {"within", "inside", "under", "less than"},
]

_SYNONYM_MAP: dict[str, str] = {}
for _grp in _SYNONYM_GROUPS:
    _canon = sorted(_grp)[0]
    for _w in _grp:
        _SYNONYM_MAP[_w] = _canon

_STOPWORDS: Set[str] = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "its",
    "new", "now", "see", "two", "who", "any", "did", "had", "let", "say",
    "she", "too", "use", "way", "with", "this", "that", "have", "from",
    "they", "been", "were", "there", "their", "what", "which", "when",
    "would", "could", "should", "about", "into", "more", "also", "dear",
    "your", "thank", "please", "customer", "hello", "regards", "sincerely",
    "best", "hope", "trust", "just", "very", "some", "such", "contact",
    "reach", "shortly", "soon", "here", "team", "support", "name",
}


def _norm_dept(dept: str) -> str:
    return dept.strip().lower()


def _dept_ok(predicted: str, gold: str) -> bool:
    return _norm_dept(predicted) == _norm_dept(gold)


def _prio_ok(predicted, gold) -> bool:
    try:
        return int(predicted) == int(gold)
    except (ValueError, TypeError):
        return False


def _keywords(text: str) -> Set[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    result: Set[str] = set()
    for w in words:
        if w not in _STOPWORDS:
            result.add(_SYNONYM_MAP.get(w, w))
    return result


def _reply_quality(reply: str, gold_reply: str) -> float:
    """Score reply quality 0.0-1.0 via keyword overlap + length + professionalism."""
    if not reply or not reply.strip():
        return 0.0

    words   = reply.split()
    wc      = len(words)
    r_lower = reply.lower()

    # Length: optimal 30-120 words
    if   wc < 5:      length_score = 0.05
    elif wc < 15:     length_score = 0.35
    elif wc <= 120:   length_score = 1.00
    elif wc <= 200:   length_score = 0.85
    else:             length_score = 0.65

    # Professionalism signals
    prof = 0.0
    if any(g in r_lower for g in ["dear", "hello", "thank you", "greetings"]):
        prof += 0.35
    if any(a in r_lower for a in ["will", "resolve", "investigate", "assist",
                                   "help", "look into", "process", "address",
                                   "dispatch", "ship", "refund", "credit",
                                   "review", "handle", "escalate"]):
        prof += 0.40
    if any(c in r_lower for c in ["regards", "sincerely", "shortly",
                                   "business day", "hours", "apologize",
                                   "apologise", "sorry", "within"]):
        prof += 0.25
    prof = min(prof, 1.0)

    if not gold_reply or not gold_reply.strip():
        return round(length_score * 0.55 + prof * 0.45, 4)

    gold_kws = _keywords(gold_reply)
    pred_kws = _keywords(reply)

    if not gold_kws:
        overlap = 0.55
    else:
        matched = len(gold_kws & pred_kws)
        overlap = min(matched / len(gold_kws), 1.0)

    final = overlap * 0.55 + length_score * 0.25 + prof * 0.20
    return round(min(final, 1.0), 4)


# ── Task 1 ────────────────────────────────────────────────────────────────

def grade_task1(pred_dept: str, gold_dept: str, step: int, max_steps: int) -> dict:
    """Binary: 1.0 correct department, 0.0 wrong. No step penalty."""
    d_ok  = _dept_ok(pred_dept, gold_dept)
    score = 1.0 if d_ok else 0.0
    return {
        "score":              round(score, 4),
        "department_score":   float(d_ok),
        "priority_score":     0.0,
        "reply_score":        0.0,
        "correct_department": gold_dept,
        "correct_priority":   None,
        "feedback": (
            f"Dept: {'CORRECT' if d_ok else 'WRONG'} "
            f"('{pred_dept}' vs '{gold_dept}'). Score={score:.2f}"
        ),
    }


# ── Task 2 ────────────────────────────────────────────────────────────────

def grade_task2(pred_dept: str, pred_prio, gold_dept: str, gold_prio,
                step: int, max_steps: int) -> dict:
    """Dept (60%) + Priority (40%). No step penalty."""
    d_ok       = _dept_ok(pred_dept, gold_dept)
    p_ok       = _prio_ok(pred_prio, gold_prio)
    dept_score = 1.0 if d_ok else 0.0
    prio_score = 1.0 if p_ok else 0.0
    score      = round(dept_score * 0.6 + prio_score * 0.4, 4)
    return {
        "score":              score,
        "department_score":   dept_score,
        "priority_score":     prio_score,
        "reply_score":        0.0,
        "correct_department": gold_dept,
        "correct_priority":   int(gold_prio),
        "feedback": (
            f"Dept: {'OK' if d_ok else 'WRONG'} ('{pred_dept}' vs '{gold_dept}') "
            f"×0.6={dept_score*0.6:.2f}, "
            f"Prio: {'OK' if p_ok else 'WRONG'} ({pred_prio} vs {gold_prio}) "
            f"×0.4={prio_score*0.4:.2f}. Score={score:.2f}"
        ),
    }


# ── Task 3 ────────────────────────────────────────────────────────────────

def grade_task3(pred_dept: str, pred_prio, pred_reply: Optional[str],
                gold_dept: str, gold_prio, gold_reply: str,
                step: int, max_steps: int) -> dict:
    """Dept (40%) + Priority (30%) + Reply quality (30%). No step penalty."""
    d_ok       = _dept_ok(pred_dept, gold_dept)
    p_ok       = _prio_ok(pred_prio, gold_prio)
    r_score    = _reply_quality(pred_reply or "", gold_reply)
    dept_score = 1.0 if d_ok else 0.0
    prio_score = 1.0 if p_ok else 0.0
    score      = round(dept_score * 0.4 + prio_score * 0.3 + r_score * 0.3, 4)
    return {
        "score":              score,
        "department_score":   dept_score,
        "priority_score":     prio_score,
        "reply_score":        round(r_score, 4),
        "correct_department": gold_dept,
        "correct_priority":   int(gold_prio),
        "feedback": (
            f"Dept={'CORRECT' if d_ok else 'WRONG'} ×0.40={dept_score*0.40:.2f}, "
            f"Prio={'OK' if p_ok else 'WRONG'} ×0.30={prio_score*0.30:.2f}, "
            f"Reply={r_score:.3f} ×0.30={r_score*0.30:.2f}. Score={score:.2f}"
        ),
    }