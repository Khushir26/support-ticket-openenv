"""
inference.py — Support Ticket Agent Baseline Inference Script

MANDATORY requirements (hackathon spec):
  ✓ Named inference.py in project root
  ✓ OpenAI client for ALL LLM calls
  ✓ API_BASE_URL with default value
  ✓ MODEL_NAME with default value
  ✓ HF_TOKEN read from env (no hardcoded value)
  ✓ Exact [START]/[STEP]/[END] stdout format
  ✓ score strictly in (0.001, 0.999) — never 0.0 or 1.0
  ✓ Runs < 20 min on 2 vCPU / 8 GB RAM
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import List, Optional, Tuple

from openai import OpenAI
# FIXED: removed TICKETS_PER_TASK — it doesn't exist in environment.py
from environment import SupportTicketEnv, TASK_CONFIG, VALID_DEPARTMENTS

# ── Required env vars ─────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str     = os.getenv("HF_TOKEN", "")   # NO hardcoded value
API_KEY: str      = HF_TOKEN or "dummy-key"

TASKS             = ["task1", "task2", "task3"]
BENCHMARK         = "support_ticket_agent"
MAX_TICKETS       = 20
SUCCESS_THRESHOLD = 0.5
_LLM_DISABLED     = False


# ── Log functions ─────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} "
          f"reward={reward:.2f} done={str(done).lower()} "
          f"error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
          flush=True)


# ── Rule-based agent ──────────────────────────────────────────────────────
_DEPT_KW = {
    "Technical": [
        ("not working", 4.0), ("cannot login", 5.0), ("login error", 5.0),
        ("login fail", 5.0), ("403", 5.0), ("app crash", 5.0),
        ("crashes", 4.0), ("server error", 5.0), ("500 error", 5.0),
        ("api", 3.5), ("webhook", 4.0), ("ssl", 4.5), ("certificate", 3.5),
        ("timeout", 4.0), ("not loading", 4.5), ("outage", 5.0),
        ("bug", 4.0), ("broken", 3.5), ("password", 3.0), ("2fa", 4.5),
        ("authentication", 3.5), ("sync", 3.0), ("data loss", 5.0),
        ("error", 3.0), ("fail", 3.0), ("server", 3.0),
        ("security breach", 5.0), ("access denied", 4.5),
        ("session expired", 4.0), ("slow", 2.5), ("performance", 3.0),
    ],
    "Billing": [
        ("invoice", 5.0), ("billing", 5.0), ("billed", 5.0), ("refund", 5.0),
        ("payment", 4.0), ("charge", 4.0), ("charged", 4.5),
        ("overcharged", 5.5), ("double charged", 5.5),
        ("subscription", 3.5), ("cancel subscription", 5.0),
        ("credit card", 4.0), ("receipt", 4.0),
        ("gst", 4.5), ("tax invoice", 5.0), ("pro-rated", 4.5),
        ("money back", 5.0), ("deducted", 4.5), ("payment failed", 5.0),
        ("billing cycle", 4.5), ("cancel", 2.5),
    ],
    "Returns": [
        ("damaged", 3.5), ("wrong item", 4.5), ("wrong product", 4.5),
        ("incorrect item", 4.0), ("not as described", 4.0),
        ("defective", 4.0), ("faulty", 3.5), ("broken product", 3.5),
        ("will not turn on", 3.5), ("does not turn on", 3.5),
        ("cracked screen", 4.0), ("cracked", 3.0),
        ("return request", 4.5), ("initiate an exchange", 4.0),
        ("exchange for", 3.5), ("different size", 3.5),
        ("replacement", 3.0), ("return label", 3.5), ("prepaid label", 3.5),
        ("return", 2.5), ("exchange", 2.0), ("missing item", 4.5),
    ],
    "Product": [
        ("feature request", 4.5), ("please add", 3.5), ("dark mode", 4.0),
        ("pdf export", 4.0), ("bulk export", 3.5), ("push notification", 4.0),
        ("slack integration", 4.0), ("workflow automation", 3.5),
        ("navigation", 3.0), ("confusing", 2.5), ("ux", 2.5),
        ("roadmap", 4.0), ("enhancement", 3.0), ("missing feature", 4.0),
        ("feedback", 2.5), ("suggestion", 3.0),
        ("api rate limit", 4.0), ("rate limit", 3.0),
        ("feature", 2.5), ("improve", 2.0),
    ],
    "IT": [
        ("vpn", 5.0), ("vpn not connecting", 5.5), ("vpn authentication", 5.0),
        ("work email", 5.0), ("email on new computer", 5.0),
        ("new computer", 3.5), ("new laptop", 4.5), ("laptop setup", 5.0),
        ("printer", 5.0), ("printer offline", 5.0), ("office printer", 5.0),
        ("adobe", 4.0), ("software license", 5.0), ("creative suite", 4.5),
        ("microsoft office", 4.5), ("wifi", 4.5), ("wi-fi", 4.5),
        ("new employee", 4.0), ("new joiner", 4.5), ("employee setup", 5.0),
        ("workstation", 4.0), ("it support", 5.0), ("network", 2.5),
    ],
    "Sales": [
        ("enterprise pricing", 5.0), ("enterprise plan", 4.5),
        ("volume discount", 5.0), ("bulk purchase", 4.5),
        ("bulk license", 4.5), ("50 user", 4.0),
        ("upgrade from", 4.5), ("upgrade to pro", 4.5),
        ("from basic to pro", 5.0), ("pro plan", 3.5),
        ("reseller", 5.0), ("partnership", 3.5),
        ("product demo", 4.5), ("demo before", 4.5),
        ("pricing plan", 4.0), ("quote", 3.5),
    ],
    "HR": [
        ("leave balance", 5.0), ("remaining leave", 4.5),
        ("leave policy", 5.0), ("annual leave", 4.0),
        ("sick leave", 4.0), ("carry forward", 4.5),
        ("wfh", 5.0), ("work from home", 5.0), ("wfh policy", 5.0),
        ("payroll", 5.0), ("salary slip", 5.0), ("salary", 3.5),
        ("expense reimbursement", 5.0), ("expense claim", 4.5),
        ("performance review", 5.0), ("appraisal", 5.0),
        ("health insurance", 5.0), ("enrollment", 4.0),
        ("hr policy", 5.0), ("hr portal", 4.5),
    ],
}

_EXPLICIT_LOW = [
    "cancel subscription", "switch to annual", "gst invoice", "tax invoice",
    "feature request", "please add", "dark mode", "pdf export",
    "push notification", "slack integration", "workflow automation",
    "dashboard navigation", "navigation is confusing", "rate limit",
    "software license", "license needed", "adobe creative suite",
    "upgrade from basic to pro", "from basic to pro", "upgrade to pro",
    "upgrade my plan", "enterprise pricing", "volume discount",
    "reseller partner", "product demo", "demo before",
    "performance review timeline", "annual performance review",
    "carry forward", "leave balance", "remaining leave",
    "wfh policy", "work from home policy",
    "how to initiate", "initiate an exchange", "different size",
]
_EXPLICIT_MEDIUM = [
    "wrong item", "wrong product", "received wrong", "not as described",
    "salary slip not received", "salary slip", "refund not received",
    "invoice amount", "amount incorrect", "app crashes", "app crash",
    "slow", "performance issue", "missing item",
]
_EXPLICIT_HIGH = [
    "completely down", "server down", "production down", "outage",
    "security breach", "data breach", "double charged", "charged twice",
    "payment failed but amount deducted", "cannot access email",
    "locked out", "403 forbidden", "403", "2fa codes rejected",
    "ssl certificate", "certificate expired", "ssl error",
    "vpn not connecting", "vpn authentication failure",
    "defective", "will not turn on", "cracked screen",
    "laptop arrived damaged", "completely broken",
]
_CONTEXT_HIGH = ["asap", "urgent", "immediately", "critical", "emergency"]

_REPLY_TPL = {
    "Technical": {
        3: "Dear Customer, we understand the urgency of {issue}. Our engineering team is actively investigating and will restore service within 2 hours. We sincerely apologize for the disruption. Best regards, Technical Team",
        2: "Dear Customer, thank you for reporting {issue}. Our technical team is investigating and will resolve this within 24 hours. We apologize for the inconvenience. Best regards, Technical Team",
        1: "Dear Customer, thank you for reaching out about {issue}. Our team will review and respond within 2 business days. Best regards, Technical Team",
    },
    "Billing": {
        3: "Dear Customer, we have identified the billing issue regarding {issue} and initiated an immediate correction. A refund will be processed within 2-3 business days. We sincerely apologize. Best regards, Billing Team",
        2: "Dear Customer, thank you for contacting us about {issue}. Our billing team will process the adjustment within 3-5 business days and email a confirmation. Best regards, Billing Team",
        1: "Dear Customer, thank you for your inquiry about {issue}. Our billing team will respond within 2 business days. Best regards, Billing Team",
    },
    "Returns": {
        3: "Dear Customer, we sincerely apologize for {issue}. A prepaid return label has been emailed and a replacement will be dispatched within 24 hours of receiving the return. Best regards, Returns Team",
        2: "Dear Customer, we have processed your return request regarding {issue}. A prepaid return label will be emailed shortly and your refund or replacement will be processed within 5 business days. Best regards, Returns Team",
        1: "Dear Customer, you can initiate a return for {issue} from your order history page. We cover return shipping. Best regards, Returns Team",
    },
    "Product": {
        3: "Dear Customer, thank you for the feedback about {issue}. We have escalated this to our product team for immediate review. Best regards, Product Team",
        2: "Dear Customer, thank you for the valuable feedback about {issue}. We have added this to our product backlog for the upcoming cycle. Best regards, Product Team",
        1: "Dear Customer, thank you for the suggestion about {issue}. Our product team reviews all feedback to shape our roadmap. Best regards, Product Team",
    },
    "IT": {
        3: "Dear Customer, we understand the urgency of {issue}. Our IT team is working on this immediately and will resolve it within 4 hours. We apologize for the disruption. Best regards, IT Support",
        2: "Dear Customer, our IT team has received your request about {issue}. A technician will assist you within 1 business day. Best regards, IT Support",
        1: "Dear Customer, thank you for your request about {issue}. Our IT team will process this within 2-3 business days. Best regards, IT Support",
    },
    "Sales": {
        3: "Dear Customer, thank you for your interest in {issue}. Our sales team will contact you within 4 hours with a customized proposal. Best regards, Sales Team",
        2: "Dear Customer, thank you for reaching out about {issue}. Our sales team will contact you within 24 hours. Best regards, Sales Team",
        1: "Dear Customer, thank you for inquiring about {issue}. Our sales team will contact you within 2 business days. Best regards, Sales Team",
    },
    "HR": {
        3: "Dear Customer, we have received your urgent request about {issue} and will address it within 24 hours. Best regards, HR Team",
        2: "Dear Customer, thank you for reaching out about {issue}. Our HR team will process your request within 2 business days. Best regards, HR Team",
        1: "Dear Customer, thank you for your inquiry about {issue}. You can find information on the HR portal. Our team will follow up within 3 business days. Best regards, HR Team",
    },
}


def _classify_dept(subject: str, body: str) -> str:
    t = (subject + " " + body).lower()
    subj = subject.lower()
    scores = {d: 0.0 for d in VALID_DEPARTMENTS}
    for dept, kws in _DEPT_KW.items():
        for phrase, weight in kws:
            if phrase in t:
                scores[dept] += weight * 1.5 if phrase in subj else weight
    # disambiguate
    if scores["Returns"] > 0 and scores["Billing"] > 0:
        physical = any(w in t for w in ["damaged","wrong item","defective","cracked","arrived","faulty"])
        scores["Returns"] += 5.0 if physical else 0
        scores["Billing"]  += 0   if physical else 3.0
    if scores["Technical"] > 0 and scores["Product"] > 0:
        is_req = any(w in t for w in ["feature","suggestion","please add","feedback","roadmap","enhancement"])
        is_bug = any(w in t for w in ["error","crash","not working","bug","fail","cannot","timeout","broken"])
        if is_req and not is_bug: scores["Product"] += 5.0
        elif is_bug:              scores["Technical"] += 5.0
    if scores["IT"] > 0 and scores["Technical"] > 0:
        it_sig = any(w in t for w in ["vpn","printer","laptop","software license","new employee","new joiner","helpdesk"])
        if it_sig: scores["IT"] += 5.0
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "Technical"


def _classify_prio(subject: str, body: str, dept: str) -> int:
    t = (subject + " " + body).lower()
    for p in _EXPLICIT_LOW:
        if p in t: return 1
    for p in _EXPLICIT_MEDIUM:
        if p in t: return 2
    for p in _EXPLICIT_HIGH:
        if p in t: return 3
    for p in _CONTEXT_HIGH:
        if p in t: return 3
    dept_default = {"Technical":2,"Billing":2,"Product":1,"IT":2,"Returns":2,"Sales":1,"HR":1}
    if dept == "HR": return 1
    return dept_default.get(dept, 2)


def _make_reply(dept: str, prio: int, subject: str) -> str:
    issue = (subject or "your request").strip().rstrip(".")[:60]
    tmpls = _REPLY_TPL.get(dept, _REPLY_TPL["Technical"])
    return tmpls.get(prio, tmpls[2]).format(issue=issue)


def _rule_agent(obs, task: str) -> dict:
    dept  = _classify_dept(obs.subject, obs.body)
    prio  = _classify_prio(obs.subject, obs.body, dept)
    reply = _make_reply(dept, prio, obs.subject) if task == "task3" else ""
    return {"department": dept, "priority": prio, "reply": reply}


# ── LLM agent ─────────────────────────────────────────────────────────────
_SYS_T2 = f"""You are a customer support ticket classifier.
Output ONLY valid JSON. No markdown. No explanation.
Fields: "department" (one of {VALID_DEPARTMENTS}), "priority" (1/2/3), "reply" ("")
Departments: Technical=bugs/crashes/API errors/outages | Billing=invoices/refunds/charges/subscriptions | Product=feature requests/feedback/rate limits/navigation | IT=VPN/work email/laptops/printers/licenses/new employee | Returns=damaged/wrong/defective items | Sales=pricing/upgrades/enterprise/demos/volume | HR=leave/WFH/payroll/salary/insurance/reviews
Priority: 1=Low(requests/inquiries/admin/HR) 2=Med(standard bugs/billing errors/wrong item) 3=High(outages/403/2FA/SSL/VPN down/defective/double charged)"""

_SYS_T3 = f"""You are an expert customer support triage agent.
Output ONLY valid JSON. No markdown. No explanation.
Fields: "department", "priority" (1/2/3), "reply" (40-75 word professional reply)
Departments: Technical=bugs/crashes/outages/API/2FA/SSL | Billing=invoices/refunds/charges/subscriptions | Product=feature requests/feedback/rate limits/navigation | IT=VPN/work email/laptops/printers/licenses | Returns=damaged/wrong/defective | Sales=pricing/upgrades/enterprise/demos | HR=leave/WFH/payroll/salary/insurance/reviews
Priority: 1=Low 2=Medium 3=High(outages/403/2FA/SSL/VPN/defective)
Reply MUST: start "Dear Customer," + include will+action verb + timeframe + end "Best regards, [Dept] Team" + apologize if problem + 40-75 words"""


def _llm_agent(client: OpenAI, obs, task: str) -> Tuple[dict, Optional[str]]:
    global _LLM_DISABLED
    system = _SYS_T3 if task == "task3" else _SYS_T2
    temp   = 0.1 if task == "task3" else 0.0
    prompt = f"Subject: {obs.subject}\nBody: {obs.body[:300]}\nJSON only."
    raw = ""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            temperature=temp, max_tokens=400)
        raw = (resp.choices[0].message.content or "").strip()
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"): part = part[4:].strip()
                if part.startswith("{"): raw = part; break
        s = raw.find("{"); e = raw.rfind("}")+1
        if s >= 0 and e > s: raw = raw[s:e]
        parsed = json.loads(raw)
        dept = str(parsed.get("department","")).strip()
        if dept not in VALID_DEPARTMENTS:
            dept = _classify_dept(obs.subject, obs.body)
        try: prio = max(1, min(3, int(parsed.get("priority",2))))
        except: prio = 2
        reply = str(parsed.get("reply","") or "")
        if task != "task3": reply = ""
        elif not reply.strip(): reply = _make_reply(dept, prio, obs.subject)
        return {"department":dept,"priority":prio,"reply":reply}, None
    except json.JSONDecodeError as exc:
        return _rule_agent(obs, task), f"JSON:{exc}"
    except Exception as exc:
        err = str(exc)
        if any(c in err for c in ["402","403","quota","credit"]):
            _LLM_DISABLED = True
            print(f"[WARN] LLM disabled: {err[:80]}", file=sys.stderr, flush=True)
        return _rule_agent(obs, task), err[:120]


def _get_action(client, obs, task):
    if task == "task1" or not HF_TOKEN or _LLM_DISABLED:
        return _rule_agent(obs, task), None
    return _llm_agent(client, obs, task)


# ── Task runner ───────────────────────────────────────────────────────────
def run_task(env: SupportTicketEnv, client, task_id: str) -> dict:
    cfg = TASK_CONFIG[task_id]
    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        reset_resp = env.reset(task_id=task_id)
        obs        = reset_resp.observation
        total      = reset_resp.info.get("total_tickets", MAX_TICKETS)

        for step in range(1, min(total, MAX_TICKETS) + 1):
            if env.state().done: break
            action, error = _get_action(client, obs, task_id)
            step_resp     = env.step(action)
            reward        = step_resp.reward.score
            done          = step_resp.done
            rewards.append(reward)
            steps_taken = step
            action_str = json.dumps(
                {"department":action["department"],"priority":action["priority"],
                 "reply":action.get("reply","")}, separators=(",",":"))
            log_step(step, action_str, reward, done, error)
            if done: break
            obs = step_resp.observation
            if task_id != "task1" and HF_TOKEN and not _LLM_DISABLED:
                time.sleep(0.1)

    except Exception as exc:
        print(f"[ERROR] {task_id}: {exc}", file=sys.stderr, flush=True)
        if not rewards: rewards = [0.001]

    # CRITICAL: clamp episode score strictly between 0 and 1
    raw_score = sum(rewards) / max(len(rewards), 1)
    score     = round(min(max(raw_score, 0.001), 0.999), 4)
    success   = score >= SUCCESS_THRESHOLD

    log_end(success, steps_taken, score, rewards)
    return {"task_id":task_id,"name":cfg["name"],"difficulty":cfg["difficulty"],
            "score":score,"num_tickets":steps_taken}


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    global _LLM_DISABLED
    print(f"[INFO] API_BASE_URL = {API_BASE_URL}", flush=True)
    print(f"[INFO] MODEL_NAME   = {MODEL_NAME}", flush=True)
    print(f"[INFO] HF_TOKEN     = {'SET' if HF_TOKEN else 'NOT SET'}", flush=True)
    if not HF_TOKEN:
        _LLM_DISABLED = True
        print("[INFO] No HF_TOKEN — rule-based mode.", flush=True)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    print("[INFO] Loading environment...", flush=True)
    env = SupportTicketEnv(seed=42, use_fallback_only=True)

    results = {}
    for task_id in TASKS:
        mode = "RULE" if (task_id == "task1" or not HF_TOKEN or _LLM_DISABLED) else "LLM"
        print(f"\n{'='*60}\n[INFO] {task_id} — {TASK_CONFIG[task_id]['name']} [{mode}]\n{'='*60}", flush=True)
        results[task_id] = run_task(env, client, task_id)

    print(f"\n{'='*60}\nFINAL BASELINE RESULTS\n{'='*60}", flush=True)
    for tid, r in results.items():
        bar = "█"*int(r["score"]*30) + "░"*(30-int(r["score"]*30))
        print(f"  {tid} [{r['difficulty']:6s}]: {r['score']:.4f}  [{bar}]", flush=True)
    overall = sum(r["score"] for r in results.values()) / len(results)
    print(f"{'─'*60}\n  Overall: {overall:.4f}\n{'='*60}", flush=True)

    with open("baseline_scores.json","w") as f:
        json.dump(results, f, indent=2)
    print("[INFO] Saved → baseline_scores.json", flush=True)


if __name__ == "__main__":
    main()
