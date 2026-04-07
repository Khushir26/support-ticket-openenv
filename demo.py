"""
demo.py — Local demo of the Support Ticket Agent environment.

Runs the rule-based agent through all 3 tasks so you can verify the
environment works end-to-end before deploying.

Usage:
    python demo.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from environment import SupportTicketEnv, TASK_CONFIG


def rule_agent(obs, task_id: str) -> dict:
    """Lightweight rule-based agent for demo purposes."""
    body = (obs.subject + " " + obs.body).lower()

    if any(w in body for w in ["vpn", "printer", "laptop setup", "it support",
                                "software license", "new joiner", "workstation"]):
        dept = "IT"
    elif any(w in body for w in ["leave", "payroll", "salary", "wfh", "hr",
                                  "performance review", "health insurance", "expense"]):
        dept = "HR"
    elif any(w in body for w in ["invoice", "billing", "refund", "charge", "payment",
                                  "gst", "subscription", "pro-rated", "credit card"]):
        dept = "Billing"
    elif any(w in body for w in ["return", "damaged", "wrong item", "exchange",
                                  "defective", "replacement", "not as described"]):
        dept = "Returns"
    elif any(w in body for w in ["pricing", "upgrade", "enterprise", "demo",
                                  "reseller", "volume discount", "bulk purchase"]):
        dept = "Sales"
    elif any(w in body for w in ["feature", "feedback", "dark mode", "suggestion",
                                  "roadmap", "ui", "ux", "navigation", "pdf export"]):
        dept = "Product"
    else:
        dept = "Technical"

    if any(w in body for w in ["urgent", "asap", "critical", "outage", "down",
                                "immediately", "production", "double charged",
                                "payment failed", "security breach"]):
        priority = 3
    elif any(w in body for w in ["feedback", "suggestion", "feature request",
                                  "information", "leave balance", "wfh policy"]):
        priority = 1
    else:
        priority = 2

    reply = ""
    if task_id == "task3":
        reply = (
            f"Dear Customer, thank you for contacting us regarding '{obs.subject[:50]}'. "
            f"Our {dept} team will investigate and resolve this issue within "
            f"{'2 hours' if priority == 3 else '24 hours' if priority == 2 else '2 business days'}. "
            f"We apologize for any inconvenience. Best regards, Support Team"
        )

    return {"department": dept, "priority": priority, "reply": reply}


def run_demo():
    print("=" * 70)
    print("  SUPPORT TICKET AGENT — LOCAL DEMO")
    print("  Rule-based agent — no API key needed")
    print("=" * 70)

    env = SupportTicketEnv(seed=42, use_fallback_only=True)
    summary = {}
    SHOW_TICKETS = 4  # tickets to show per task in demo

    for task_id in ["task1", "task2", "task3"]:
        cfg = TASK_CONFIG[task_id]
        print(f"\n{'─' * 70}")
        print(f"  {task_id.upper()} — {cfg['name']} [{cfg['difficulty'].upper()}]")
        print(f"  {cfg['description']}")
        print(f"{'─' * 70}")

        reset_resp = env.reset(task_id=task_id)
        obs = reset_resp.observation

        scores = []
        count  = 0

        while not env.state().done and count < SHOW_TICKETS:
            count += 1
            print(f"\n  Ticket {count}: [{obs.ticket_id}]")
            print(f"  Subject : {obs.subject[:65]}")
            print(f"  Body    : {obs.body[:90]}...")

            action = rule_agent(obs, task_id)
            print(f"  Agent   → dept={action['department']:<12} priority={action['priority']}", end="")
            if task_id == "task3":
                print(f"  reply={len(action['reply'])} chars", end="")
            print()

            step_resp = env.step(action)
            reward    = step_resp.reward
            scores.append(reward.score)

            bar = "█" * int(reward.score * 25) + "░" * (25 - int(reward.score * 25))
            print(f"  Score   : [{bar}] {reward.score:.4f}")
            print(f"  Detail  : {reward.feedback}")

            obs = step_resp.observation

        avg = sum(scores) / len(scores) if scores else 0.0
        summary[task_id] = {"name": cfg["name"], "difficulty": cfg["difficulty"],
                             "avg_score": avg, "tickets": count}
        print(f"\n  {task_id} average (first {count} tickets): {avg:.4f}")

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    for task_id, r in summary.items():
        bar = "█" * int(r["avg_score"] * 35) + "░" * (35 - int(r["avg_score"] * 35))
        print(f"  {task_id} [{r['difficulty']:6s}]: [{bar}] {r['avg_score']:.4f}")
    print(f"{'=' * 70}")
    print("\n  Environment is working correctly!")
    print("  To run baseline with LLM: HF_TOKEN=hf_xxx python inference.py")
    print("  To start the API server:  uvicorn main:app --port 7860")


if __name__ == "__main__":
    run_demo()