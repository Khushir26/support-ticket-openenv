"""
tests/test_graders.py — Full grader test suite.
All assertions aligned with the 5%-per-step penalty in graders.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graders import (
    dept_matches, priority_matches, reply_keyword_score,
    grade_task1, grade_task2, grade_task3, _step_penalty
)

PASS = 0
FAIL = 0

def check(label, condition, got=None):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {label}")
        PASS += 1
    else:
        msg = f"  [FAIL] {label}"
        if got is not None:
            msg += f" (got {got})"
        print(msg)
        FAIL += 1

print("=" * 50)
print("SUPPORT TICKET AGENT — GRADER TEST SUITE")
print("=" * 50)

# ── _step_penalty ─────────────────────────────────────────────────────────────
print("\n── _step_penalty ──")
check("step=1 no penalty",     _step_penalty(1.0, 1) == 1.0,   _step_penalty(1.0, 1))
check("step=2 → 0.95",         _step_penalty(1.0, 2) == 0.95,  _step_penalty(1.0, 2))
check("step=3 → 0.90",         _step_penalty(1.0, 3) == 0.90,  _step_penalty(1.0, 3))
check("step=4 capped at 0.90", _step_penalty(1.0, 4) == 0.90,  _step_penalty(1.0, 4))
check("step=10 capped at 0.90",_step_penalty(1.0, 10) == 0.90, _step_penalty(1.0, 10))
check("score=0 stays 0",       _step_penalty(0.0, 2) == 0.0,   _step_penalty(0.0, 2))

# ── dept_matches ──────────────────────────────────────────────────────────────
print("\n── dept_matches ──")
check("exact match",          dept_matches("Technical", "Technical"))
check("case insensitive",     dept_matches("technical", "Technical"))
check("trailing space",       dept_matches("Billing ", "Billing"))
check("wrong dept → False",   not dept_matches("Billing", "Technical"))
check("both lowercase",       dept_matches("billing", "billing"))

# ── priority_matches ──────────────────────────────────────────────────────────
print("\n── priority_matches ──")
check("1==1",   priority_matches(1, 1))
check("2==2",   priority_matches(2, 2))
check("3==3",   priority_matches(3, 3))
check("1!=2",   not priority_matches(1, 2))
check("3!=1",   not priority_matches(3, 1))
check("str '2'==2", priority_matches("2", 2))

# ── reply_keyword_score ───────────────────────────────────────────────────────
print("\n── reply_keyword_score ──")
s1 = reply_keyword_score("We will investigate your billing issue promptly.", "We are looking into your billing problem.")
check("overlapping replies > 0",  s1 > 0, got=round(s1,4))
check("overlapping replies ≤ 1",  s1 <= 1, got=round(s1,4))
check("empty reply → 0.0",        reply_keyword_score("", "some gold") == 0.0)
s2 = reply_keyword_score("Thank you for reaching out. We will resolve your issue.", "")
check("quality score with no gold > 0", s2 > 0, got=round(s2,4))

# ── Task 1 grader ─────────────────────────────────────────────────────────────
print("\n── Task 1 grader ──")
r1 = grade_task1("Technical", "Technical", 1, 1)
check("correct dept step=1 → 1.0",  r1["score"] == 1.0,        r1["score"])
check("dept_score = 1.0",           r1["department_score"] == 1.0)
check("priority_score = 0.0",       r1["priority_score"] == 0.0)
check("reply_score = 0.0",          r1["reply_score"] == 0.0)

r1w = grade_task1("Billing", "Technical", 1, 1)
check("wrong dept → 0.0",           r1w["score"] == 0.0,        r1w["score"])

r1p = grade_task1("Technical", "Technical", 2, 1)
check("step=2 penalty → 0.95",      r1p["score"] == 0.95,       r1p["score"])

r1p3 = grade_task1("Technical", "Technical", 3, 1)
check("step=3 penalty → 0.90",      r1p3["score"] == 0.90,      r1p3["score"])

# ── Task 2 grader ─────────────────────────────────────────────────────────────
print("\n── Task 2 grader ──")
r2 = grade_task2("Technical", 2, "Technical", 2, 1, 2)
check("both correct → 1.0",         r2["score"] == 1.0,         r2["score"])

r2d = grade_task2("Technical", 1, "Technical", 2, 1, 2)
check("dept only → 0.6",            r2d["score"] == 0.6,        r2d["score"])
check("dept_score = 1.0",           r2d["department_score"] == 1.0)
check("priority_score = 0.0",       r2d["priority_score"] == 0.0)

r2p = grade_task2("Billing", 2, "Technical", 2, 1, 2)
check("prio only → 0.4",            r2p["score"] == 0.4,        r2p["score"])
check("dept_score = 0.0",           r2p["department_score"] == 0.0)
check("priority_score = 1.0",       r2p["priority_score"] == 1.0)

r2n = grade_task2("Billing", 1, "Technical", 2, 1, 2)
check("neither → 0.0",              r2n["score"] == 0.0,        r2n["score"])

r2pen = grade_task2("Technical", 2, "Technical", 2, 2, 2)
check("step 2 penalty → 0.95",      r2pen["score"] == 0.95,     r2pen["score"])

for dept, prio in [("Technical",1),("Billing",2),("IT",3),("HR",1)]:
    rr = grade_task2(dept, prio, dept, prio, 1, 2)
    check(f"range check {dept}/{prio}", 0.0 <= rr["score"] <= 1.0, rr["score"])

# ── Task 3 grader ─────────────────────────────────────────────────────────────
print("\n── Task 3 grader ──")
r3 = grade_task3("Technical", 2,
                 "We will investigate and resolve your technical error promptly.",
                 "Technical", 2,
                 "We are looking into your technical issue and will fix it soon.", 1, 3)
check("all correct → score > 0.6",  r3["score"] > 0.6,          r3["score"])
check("dept_score = 1.0",           r3["department_score"] == 1.0)
check("priority_score = 1.0",       r3["priority_score"] == 1.0)
check("reply_score > 0",            r3["reply_score"] > 0,       r3["reply_score"])

r3w = grade_task3("Billing", 2, "some reply about billing",
                  "Technical", 2, "We will fix your technical issue.", 1, 3)
check("wrong dept → dept_score=0",  r3w["department_score"] == 0.0)
check("score ≤ 0.6",                r3w["score"] <= 0.6,         r3w["score"])

r3e = grade_task3("Technical", 2, "",
                  "Technical", 2, "We will fix your technical issue.", 1, 3)
check("empty reply → reply_score=0", r3e["reply_score"] == 0.0,  r3e["reply_score"])

r3pp = grade_task3("Technical", 2, "",
                   "Technical", 2, "", 1, 3)
check("partial score dept+prio",    r3pp["score"] == 0.7,        r3pp["score"])

r3all_wrong = grade_task3("HR", 1, "",
                           "Technical", 3, "We will fix the error urgently.", 1, 3)
check("all wrong → low score",      r3all_wrong["score"] < 0.3,  r3all_wrong["score"])

r3s1 = grade_task3("Technical", 2, "resolving your issue",
                   "Technical", 2, "resolving your issue", 1, 3)
r3s3 = grade_task3("Technical", 2, "resolving your issue",
                   "Technical", 2, "resolving your issue", 3, 3)
check("step 3 < step 1",            r3s3["score"] < r3s1["score"],
      f"{r3s3['score']} vs {r3s1['score']}")

for dept, prio in [("Technical",1),("Billing",3),("HR",2),("Returns",1)]:
    rr = grade_task3(dept, prio, "Thank you for contacting us.",
                     "Technical", 2, "We will fix this technical issue.", 1, 3)
    check(f"range {dept}/{prio}",   0.0 <= rr["score"] <= 1.0,  rr["score"])

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 50)
total = PASS + FAIL
print(f"RESULTS: {PASS} passed, {FAIL} failed" + (" ✓ PERFECT" if FAIL == 0 else ""))
print("=" * 50)

if FAIL > 0:
    sys.exit(1)