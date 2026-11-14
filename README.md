# AI Onboarding Engine

> An adaptive, AI-powered employee onboarding platform that analyzes skill gaps, generates personalized assessments, and creates customized learning pathways — powered by any OpenAI-compatible LLM — run it fully offline with Ollama (Qwen2.5:7b), or point it at a hosted API (Groq, Gemini, OpenRouter) for cloud deployment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (SPA)                            │
│    Tailwind CSS + Vanilla JS + Font Awesome + Firebase Auth     │
│      Views: Welcome → Setup → Analysis → Questionnaire →        │
│                Results → Pathway → Skill Tests                  │
│                                                                 │
│  /manager.html  — Manager dashboard (Firebase auth)             │
│  /admin.html    — Admin panel (password protected)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP / JSON
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (app.py)                     │
│                                                                 │
│  Employee endpoints  /api/sessions/...                          │
│  Manager endpoints   /api/manager/...                           │
│  Admin endpoints     /api/admin/...                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
             ┌─────────────┴──────────────┐
             ▼                            ▼
┌────────────────────┐   ┌────────────────────────────────────────┐
│   SQLite DB        │   │   LLM via OpenAI API (ai_engine.py)    │
│   (database.py)    │   │   Ollama (default) or hosted provider  │
│                    │   │                                        │
│  sessions          │   │  parse_resume()                        │
│  skill_tests       │   │  parse_job_description()               │
│  managers          │   │  analyze_skill_gap()                   │
│  manager_sessions  │
│  question_bank     │   │  generate_questionnaire()              │
│                    │   │  calculate_proficiency_scores()        │
│                    │   │  generate_learning_pathway()           │
│                    │   │  generate_skill_test()                 │
│                    │   │  evaluate_skill_test()                 │
└────────────────────┘   └────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | **Qwen2.5:7b** via **Ollama** by default (fully offline), or any OpenAI-compatible API (Groq, Gemini, OpenRouter…) |
| Backend | **FastAPI** (Python 3.11+) |
| Database | **SQLite** via Python's built-in `sqlite3` |
| PDF Parsing | **pdfplumber** (column-aware extraction) |
| Auth | **Firebase Authentication** (Google sign-in + email/password) |
| Frontend | **Vanilla JavaScript** + **Tailwind CSS** (CDN) + **Font Awesome 6** |
| Serving | **Uvicorn** ASGI server |
| Containerization | **Docker** + **Docker Compose** |

---

## Setup

### Option 1 — start.sh (recommended for local use)

```bash
# 1. Clone the repository
git clone https://github.com/Rejul007/hecklers.git
cd hecklers

# 2. Configure environment
cp .env.example .env
# Edit .env and set your ADMIN_PASSWORD

# 3. Start everything (installs deps, pulls model, starts server)
bash start.sh
```

`start.sh` will automatically:
- Create a Python virtual environment
- Install all Python dependencies
- Install Ollama if not present
- Pull the qwen2.5:7b model if not already downloaded (~4.7 GB, first run only)
- Start the FastAPI server on port 8000

### Option 2 — Docker Compose (recommended for servers)

```bash
# 1. Clone the repository
git clone https://github.com/Rejul007/hecklers.git
cd hecklers

# 2. Configure environment
cp .env.example .env
# Edit .env and set your ADMIN_PASSWORD

# 3. Build and start
docker compose up --build

# Subsequent starts (no rebuild needed)
docker compose up
```

Docker Compose persists:
- The Ollama model cache (no re-downloading on restart)
- The SQLite database (no data loss on restart)

App will be available at `http://localhost:8000`.

---

## Pages

| URL | Description | Access |
|-----|-------------|--------|
| `/` | Employee onboarding app | Firebase auth (Google / email) |
| `/manager.html` | Manager dashboard — view all employees, skill gaps, progress | Firebase auth + manager whitelist |
| `/admin.html` | Admin panel — add/remove managers | Admin password from `.env` |

---

## How It Works

### 1. Skill Gap Analysis

When a candidate submits their resume and a job description:

1. **Resume Verification** — before anything is sent to the LLM, the file is checked (PDF or .txt, ≤ 5 MB, ≤ 10 pages, readable text) and scored for résumé signals: contact details, section headings, employment dates; invoices, papers and data tables are rejected with a clear message. Only then is the text trimmed (first 5 pages, ~12k characters). The parse step makes a final "is this a résumé?" check.
2. **Resume Parsing** — extracts skills with proficiency levels, work experience, education
3. **JD Parsing** — extracts required skills with importance levels (required / preferred / nice_to_have)
4. **Gap Analysis** — the LLM only matches résumé skills to job skills (e.g. *Postgres* → *PostgreSQL*); severity, priority and the fit score are computed in Python, so results are consistent. It produces:
   - **Overall Fit Score** (0–100)
   - **Strong Matches** — skills where candidate meets or exceeds requirements
   - **Skill Gaps** — per-skill severity (critical / high / medium / low) with reasoning
   - **Missing Skills** — skills with zero evidence in the resume

### 2. Adaptive Questionnaire

- **10–50 MCQ questions** (default 30) across all gap skills, at basic, intermediate and advanced levels
- Questions are allotted in Python by gap severity (critical/high gaps get more; every skill gets at least 3)
- Questions come from a shared **question bank**: each (skill, level) is generated once — one LLM call per skill, in parallel — and reused for later candidates
- Option order is shuffled per candidate; answer keys never reach the browser before submission
- Scoring: basic=1x, intermediate=2x, advanced=3x weight

### 3. Proficiency Score Calculation

```
weighted_score = (basic_acc×1 + inter_acc×2 + adv_acc×3) / 6

Level:  ≥85% → advanced
        ≥65% → intermediate
        ≥40% → basic
        <40% → beginner
```

### 4. Personalized Learning Pathway

- Learning order optimized for prerequisites (foundational skills first)
- Week-by-week breakdown with specific tasks and milestones
- Practice project and 3 resources per skill (links are search URLs, so they never 404)
- Estimated timeline based on gap severity

### 5. Skill Verification Tests

- 10 scenario-based MCQs per skill at the target proficiency level, drawn from the question bank
- Pass threshold: 75% (8/10 correct)
- Explanations for every wrong answer, plus feedback on the concepts to review
- Failed skills enter "needs_revision" and can be retested with a fresh set of questions

---

## API Endpoints

### Sessions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions` | Create new session |
| GET | `/api/sessions/{id}` | Get full session data |
| POST | `/api/sessions/{id}/resume/text` | Upload resume as text |
| POST | `/api/sessions/{id}/resume/file` | Upload PDF/txt file |
| POST | `/api/sessions/{id}/jd` | Submit job description |
| POST | `/api/sessions/{id}/analyze` | Run skill gap analysis |
| POST | `/api/sessions/{id}/questionnaire/generate` | Generate MCQs |
| POST | `/api/sessions/{id}/questionnaire/answer` | Submit one answer |
| POST | `/api/sessions/{id}/questionnaire/answers` | Submit all answers at once |
| POST | `/api/sessions/{id}/questionnaire/finish` | Calculate proficiency scores |
| POST | `/api/sessions/{id}/pathway/generate` | Generate learning pathway |
| GET | `/api/sessions/{id}/pathway` | Get pathway + skill progress |
| POST | `/api/sessions/{id}/skills/{name}/test/generate` | Generate skill test |
| POST | `/api/sessions/{id}/skills/{name}/test/submit` | Submit + evaluate test |
| POST | `/api/sessions/{id}/skills/{name}/status` | Update skill status |
| GET | `/api/sessions/{id}/skills/{name}/progress` | Get skill test details |
| POST | `/api/sessions/{id}/link-user` | Link Firebase user to session |

### Manager
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/manager/login` | Verify whitelist + issue session token |
| POST | `/api/manager/logout` | Invalidate session token |
| GET | `/api/manager/dashboard` | Get all employee data |
| DELETE | `/api/manager/employees/{session_id}` | Remove an employee |

### Admin
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/login` | Verify admin password + issue token |
| POST | `/api/admin/logout` | Invalidate admin token |
| GET | `/api/admin/managers` | List all whitelisted managers |
| POST | `/api/admin/managers` | Add manager to whitelist |
| DELETE | `/api/admin/managers/{email}` | Remove manager from whitelist |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ADMIN_PASSWORD` | Yes | Password for the `/admin.html` page |
| `DB_PATH` | No | SQLite file path (default: `onboarding.db`, set to `/app/data/onboarding.db` in Docker) |
| `LLM_BASE_URL` | No | OpenAI-compatible API base URL (default: `http://localhost:11434/v1`, i.e. local Ollama) |
| `LLM_API_KEY` | No | API key for the provider (ignored by Ollama) |
| `LLM_MODEL` | No | Model name (default: `qwen2.5:7b`) |
| `LLM_MAX_TOKENS` | No | Upper bound on `max_tokens` per request, for providers with lower output limits |
| `LLM_REASONING_BUDGET` | No | Extra tokens added to every request when using a reasoning model, whose hidden thinking counts against `max_tokens` (default `0`). A non-reasoning model is cheaper and faster for this app. |
| `LLM_CONCURRENCY` | No | Max LLM requests in flight at once (default: `4`; lower it for strict free tiers) |
| `LLM_JSON_SCHEMA` | No | Set `false` if the provider rejects `response_format` JSON schemas (detected automatically otherwise) |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | No | Request timeout in seconds (default `600`) / retries on rate limits and server errors, with 2s → 8s → 20s → 45s backoff or the provider's `retry-after` (default `4`) |
| `PORT` | No | Port to listen on (default: `8000`; most hosts set this automatically) |

LLM token usage is logged per call (`LLM <step>: N prompt + M completion tokens`).

---

## Deploying online (hosted LLM)

Ollama needs several GB of RAM and ideally a GPU, which free hosting tiers don't provide. For a public deployment, use a hosted LLM API instead — no code changes needed:

1. Get an API key from an OpenAI-compatible provider (see the examples in `.env.example`).
2. Deploy the repo's `Dockerfile` to a container host. The default build skips Ollama, so the image stays small.
3. Set `ADMIN_PASSWORD`, `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` as environment variables on the host.
4. Add the deployed domain to **Firebase → Authentication → Settings → Authorized domains**, or sign-in will fail.

`start.sh` skips Ollama automatically whenever `LLM_BASE_URL` isn't a localhost address.

> **Note:** SQLite lives on the container's disk. On hosts without a persistent volume, data (sessions, the manager whitelist and the question bank) resets when the container restarts or redeploys.

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

The tests use a fake LLM, so they run offline in about a second.

---

## Project Structure

```
hecklers/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application + all API endpoints
│   ├── ai_engine.py      # LLM integration, question bank, gap scoring
│   ├── resume_intake.py  # Résumé file checks, extraction, verification, trimming
│   ├── schemas.py        # Pydantic models for structured LLM output
│   └── database.py       # SQLite operations (init, CRUD, question bank)
├── static/
│   ├── index.html        # Employee onboarding SPA
│   ├── manager.html      # Manager dashboard
│   └── admin.html        # Admin panel
├── tests/                # pytest suite (fake LLM, no network)
├── requirements.txt      # Python dependencies
├── requirements-dev.txt  # + test dependencies
├── start.sh              # One-command local startup script
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker Compose with persistent volumes
├── .env.example          # Environment variable template
└── README.md             # This file
```
