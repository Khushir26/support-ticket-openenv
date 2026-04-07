# Deployment Guide — HuggingFace Spaces

Complete step-by-step instructions to get your environment live and
passing all automated judging checks.

---

## Step 1 — Create a HuggingFace Account + Space

1. Go to https://huggingface.co and sign up (free)
2. Click your profile → **New Space**
3. Fill in:
   - **Space name**: `support-ticket-agent`
   - **License**: MIT
   - **SDK**: Docker  ← IMPORTANT, must be Docker
   - **Visibility**: Public ← judges need to access it
4. Click **Create Space**

---

## Step 2 — Upload Your Files

You need to upload exactly these 9 files to your Space:

```
main.py
environment.py
models.py
graders.py
baseline.py
openenv.yaml
requirements.txt
Dockerfile
README.md
```

**Option A — via the HuggingFace web UI:**
1. In your Space, click **Files** tab
2. Click **Add file → Upload files**
3. Upload all 9 files at once
4. Click **Commit changes**

**Option B — via Git (faster):**
```bash
# Install git-lfs first
git lfs install

# Clone your empty space
git clone https://huggingface.co/spaces/YOUR_USERNAME/support-ticket-agent
cd support-ticket-agent

# Copy all your files here
cp /path/to/your/files/* .

# Push
git add .
git commit -m "Initial deployment"
git push
```

---

## Step 3 — Set Your OpenAI API Key as a Secret

1. In your Space, go to **Settings** tab
2. Scroll to **Repository secrets**
3. Click **New secret**
4. Name: `OPENAI_API_KEY`
5. Value: your `sk-...` key
6. Click **Save**

> The key is injected as an environment variable at runtime.
> It's never visible in your code or logs.

---

## Step 4 — Watch It Build

1. Go to the **App** tab of your Space
2. You'll see "Building..." with Docker logs
3. First build takes ~3-5 minutes (downloads dataset from HuggingFace)
4. Once you see `Application startup complete`, it's live

**If the build fails**, click **Logs** and look for:
- Missing file → check Step 2
- Port error → Dockerfile already uses 7860, should be fine
- Dataset error → HuggingFace dataset download issues (retry)

---

## Step 5 — Verify It's Working

Once live, your Space URL will be:
`https://YOUR_USERNAME-support-ticket-agent.hf.space`

Test each endpoint in your browser or with curl:

```bash
BASE="https://YOUR_USERNAME-support-ticket-agent.hf.space"

# 1. Health check — must return 200
curl $BASE/health

# 2. Tasks list — must return all 3 tasks
curl $BASE/tasks

# 3. Reset — start episode
curl -X POST $BASE/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}'

# 4. Step — submit action
curl -X POST $BASE/step \
  -H "Content-Type: application/json" \
  -d '{"department": "Technical", "priority": 2}'

# 5. State
curl $BASE/state

# 6. Grader — test directly
curl -X POST $BASE/grader \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task2",
    "ticket_body": "My invoice is wrong",
    "ticket_subject": "Billing issue",
    "gold_department": "Billing",
    "gold_priority": 2,
    "predicted_department": "Billing",
    "predicted_priority": 2
  }'

# 7. Baseline — runs GPT-4o-mini (needs API key set)
curl -X POST $BASE/baseline \
  -H "Content-Type: application/json" \
  -d '{"task_ids": ["task1","task2","task3"], "max_tickets": 5}'
```

---

## Step 6 — Submit to the Hackathon

1. Go to the hackathon portal
2. Click **Submit your Assessment**
3. Paste your Space URL:
   `https://YOUR_USERNAME-support-ticket-agent.hf.space`
4. Also paste your GitHub/HuggingFace repo link
5. Submit before **April 7, 11:59 PM**

---

## Pre-Submission Checklist

Go through this before submitting — all must pass:

- [ ] HuggingFace Space URL returns 200 on `/health`
- [ ] `/reset` returns an observation with ticket data
- [ ] `/step` returns reward with score between 0.0 and 1.0
- [ ] `/tasks` returns all 3 tasks with action schemas
- [ ] `/grader` returns a score for a test action
- [ ] `/baseline` returns scores (even mock scores without API key)
- [ ] `docker build` works locally without errors
- [ ] `openenv.yaml` has name, tasks, endpoints, reward_range fields
- [ ] README has environment description, action space, setup instructions
- [ ] Baseline scores span easy (~0.8) → hard (~0.4) — shows difficulty range

---

## Testing Docker Locally (Before Uploading)

```bash
cd support_ticket_env/

# Build
docker build -t support-ticket-agent .

# Run
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... support-ticket-agent

# Test
curl http://localhost:7860/health
curl http://localhost:7860/tasks
```

---

## Common Issues and Fixes

| Problem | Fix |
|---------|-----|
| Space stuck on "Building" | Check Logs tab for errors |
| `ModuleNotFoundError` | Check requirements.txt has all packages |
| Dataset load fails | HuggingFace may be rate-limiting — retry |
| `/baseline` returns no_api_key | Set OPENAI_API_KEY secret in Space settings |
| Port 7860 not responding | Make sure Dockerfile EXPOSE 7860 is there |
| `openenv validate` fails | Check openenv.yaml has all required fields |

---

## What Judges Check (Automated)

The judging system will automatically:

1. Ping `YOUR_SPACE_URL/health` → must return `{"status": "ok"}`
2. POST to `/reset` → must return observation with ticket data
3. POST to `/step` with an action → must return score in [0.0, 1.0]
4. GET `/tasks` → must list 3 tasks with action schemas
5. Run `docker build` on your repo
6. POST to `/baseline` → must return scores without crashing

**All 6 must pass or you are disqualified.**
Make sure to test every single one before submitting.