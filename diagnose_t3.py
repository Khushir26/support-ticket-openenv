"""Diagnose task3 scoring — which tickets lose points and why."""
import sys, os
sys.path.insert(0, "d:/Ticket-support-system")
os.chdir("d:/Ticket-support-system")

from environment import SupportTicketEnv, TASK_CONFIG
from graders import grade_task3

# Import the rule agent from inference
# We need to replicate or import it
env = SupportTicketEnv(seed=42, use_fallback_only=True)
reset = env.reset("task3")

# Get all task3 tickets
tdf = env._task_dfs["task3"]
tickets = tdf.sample(frac=1, random_state=42).reset_index(drop=True).to_dict("records")

# Import inference module
import importlib.util
spec = importlib.util.spec_from_file_location("inference", "d:/Ticket-support-system/inference.py")

# Manual rule-based classification (simpler approach)
from inference import _classify_dept, _classify_priority, _build_reply

total = 0.0
for i, t in enumerate(tickets[:20]):
    text = t["subject"] + " " + t["body"]
    dept = _classify_dept(text)
    prio = _classify_priority(text, dept)
    reply = _build_reply(dept, prio, t["subject"])
    
    result = grade_task3(
        dept, prio, reply,
        t["department"], t["priority"], t.get("gold_reply", ""),
        i+1, 20
    )
    
    total += result["score"]
    
    if result["score"] < 0.90:
        print(f"\n--- Ticket {i+1}: score={result['score']:.2f} ---")
        print(f"  Subject: {t['subject'][:60]}")
        print(f"  Gold dept={t['department']}, pred dept={dept} {'OK' if dept==t['department'] else 'WRONG'}")
        print(f"  Gold prio={t['priority']}, pred prio={prio} {'OK' if prio==t['priority'] else 'WRONG'}")
        print(f"  Reply score: {result['reply_score']:.3f}")
        print(f"  Reply (pred): {reply[:120]}...")
        gold = t.get("gold_reply", "")
        if gold:
            print(f"  Reply (gold): {gold[:120]}...")

avg = total / 20
print(f"\n\nOverall task3 avg: {avg:.4f}")
