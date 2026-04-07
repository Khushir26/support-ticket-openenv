"""Diagnose task2 + task3 scoring failures - writes to .py file for easy viewing."""
import sys, os
sys.path.insert(0, "d:/Ticket-support-system")
os.chdir("d:/Ticket-support-system")

from environment import SupportTicketEnv, TASK_CONFIG
from graders import grade_task2, grade_task3
from inference import _classify_dept, _classify_priority, _build_reply

env = SupportTicketEnv(seed=42, use_fallback_only=True)
lines = []

# TASK 2
env.reset("task2")
tickets2 = env._task_tickets
lines.append("# TASK 2 FAILURES")
t2_total = 0.0
for i, t in enumerate(tickets2[:20]):
    text = t["subject"] + " " + t["body"]
    dept = _classify_dept(text)
    prio = _classify_priority(text, dept)
    r = grade_task2(dept, prio, t["department"], t["priority"], i+1, 20)
    t2_total += r["score"]
    if r["score"] < 1.0:
        lines.append(f"# T2-{i+1} score={r['score']:.2f} | subj={t['subject'][:60]}")
        lines.append(f"#   dept: pred={dept} gold={t['department']}")
        lines.append(f"#   prio: pred={prio} gold={t['priority']}")
        lines.append(f"#   body: {t['body'][:100]}")
lines.append(f"# Task2 avg: {t2_total/20:.4f}")
lines.append("")

# TASK 3
env.reset("task3")
tickets3 = env._task_tickets
lines.append("# TASK 3 FAILURES")
t3_total = 0.0
for i, t in enumerate(tickets3[:20]):
    text = t["subject"] + " " + t["body"]
    dept = _classify_dept(text)
    prio = _classify_priority(text, dept)
    reply = _build_reply(dept, prio, t["subject"])
    r = grade_task3(dept, prio, reply, t["department"], t["priority"],
                    t.get("gold_reply", ""), i+1, 20)
    t3_total += r["score"]
    if r["score"] < 0.85:
        lines.append(f"# T3-{i+1} score={r['score']:.2f} d={r['department_score']:.0f} p={r['priority_score']:.0f} r={r['reply_score']:.3f}")
        lines.append(f"#   subj={t['subject'][:60]}")
        lines.append(f"#   dept: pred={dept} gold={t['department']}")
        lines.append(f"#   prio: pred={prio} gold={t['priority']}")
        lines.append(f"#   body: {t['body'][:120]}")
        gold = t.get("gold_reply", "")
        if gold:
            lines.append(f"#   gold_reply: {gold[:150]}")
        lines.append(f"#   pred_reply: {reply[:150]}")
lines.append(f"# Task3 avg: {t3_total/20:.4f}")

with open("d:/Ticket-support-system/diagnose_results.py", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("DONE - see diagnose_results.py")
