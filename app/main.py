"""
main.py - FastAPI backend for the AI Onboarding Engine.

Endpoints:
  POST /api/sessions                          - create a new session
  GET  /api/sessions/{id}                     - get session data
  POST /api/sessions/{id}/resume/text         - upload resume as text
  POST /api/sessions/{id}/resume/file         - upload resume as PDF/text file
  POST /api/sessions/{id}/jd                  - submit job description
  POST /api/sessions/{id}/analyze             - run skill gap analysis
  POST /api/sessions/{id}/questionnaire/generate   - generate MCQ questionnaire
  POST /api/sessions/{id}/questionnaire/answer     - submit one answer
  POST /api/sessions/{id}/questionnaire/answers    - submit all answers at once
  POST /api/sessions/{id}/questionnaire/finish     - finish and score questionnaire
  POST /api/sessions/{id}/pathway/generate    - generate learning pathway
  GET  /api/sessions/{id}/pathway             - get pathway + skill progress
  POST /api/sessions/{id}/skills/{skill}/test/generate - generate skill test
  POST /api/sessions/{id}/skills/{skill}/test/submit   - submit skill test answers
  POST /api/sessions/{id}/skills/{skill}/status        - update skill status
  GET  /api/sessions/{id}/skills/{skill}/progress      - get skill test details
  POST /api/sessions/{id}/link-user           - link Firebase user to session
  POST /api/manager/login                     - manager authentication
  POST /api/manager/logout                    - manager sign out
  GET  /api/manager/dashboard                 - get all employee data (manager only)
"""

import hashlib
import json
import logging
import os
import secrets
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import ai_engine, database
from app.resume_intake import ResumeRejected, prepare_from_text, prepare_from_upload, MAX_FILE_BYTES

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_JD_CHARS = 15_000  # job descriptions longer than this are trimmed before parsing

# ---------------------------------------------------------------------------
# Admin session store (in-memory — single admin, no DB needed)
# ---------------------------------------------------------------------------
_admin_sessions: set = set()


def _require_admin(request: Request):
    token = request.headers.get("X-Admin-Token", "")
    if not token or token not in _admin_sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _public_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """Question as sent to the browser: no answer key or explanation before submission."""
    return {k: v for k, v in q.items() if k not in ("correct_answer", "explanation")}


def _fingerprint(*parts: Any) -> str:
    """Short hash of the inputs a generated result depends on."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _questionnaire_correct(session: Dict[str, Any]) -> int:
    answers = (session.get("questionnaire_state") or {}).get("answers", {})
    return sum(
        1 for q in session.get("questionnaire_questions") or []
        if answers.get(str(q["id"])) == q.get("correct_answer")
    )


def _record_answers(state: Dict[str, Any], questions: List[Dict], answers: Dict[str, str]):
    """Validate and store answers. An answer can't be changed once recorded."""
    ids = {str(q["id"]) for q in questions}
    for qid, letter in answers.items():
        if qid not in ids:
            raise HTTPException(status_code=404, detail=f"Question {qid} not found")
        if letter not in ("A", "B", "C", "D", "E"):
            raise HTTPException(status_code=400, detail="Answer must be A, B, C, D, or E")
        previous = state["answers"].get(qid)
        if previous is not None and previous != letter:
            raise HTTPException(status_code=409, detail=f"Question {qid} has already been answered")
        state["answers"][qid] = letter


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Onboarding Engine",
    description="Adaptive AI-powered employee onboarding with skill gap analysis and personalized learning pathways",
    version="1.0.0"
)

# CORS: allow all origins for development/demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the database on application startup."""
    database.init_db()


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    text: str


class AnswerInput(BaseModel):
    question_id: int
    answer: str  # "A", "B", "C", "D" or "E"


class AnswersBatchInput(BaseModel):
    answers: Dict[str, str]  # {str(question_id): letter}


class PathwayGenerateRequest(BaseModel):
    regenerate: bool = False


class TestAnswersInput(BaseModel):
    answers: Dict[str, str]  # {str(question_id): letter}


class StatusInput(BaseModel):
    status: str


class ManagerLoginInput(BaseModel):
    email: str


class AdminLoginInput(BaseModel):
    password: str


class AdminAddManagerInput(BaseModel):
    email: str


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sessions")
async def create_session():
    """Create a new onboarding session."""
    session_id = database.create_session()
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get all session data with all JSON fields parsed."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session["questionnaire_correct"] = _questionnaire_correct(session)
    if session.get("questionnaire_questions"):
        session["questionnaire_questions"] = [_public_question(q) for q in session["questionnaire_questions"]]
    return session


# ---------------------------------------------------------------------------
# Resume endpoints
# ---------------------------------------------------------------------------

async def _parse_resume_or_reject(resume_text: str) -> Dict[str, Any]:
    """Parse with the LLM; reject (422) if the model judges it isn't a résumé."""
    try:
        resume_data = await ai_engine.parse_resume(resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {str(e)}")
    if not resume_data.get("is_resume", True):
        reason = resume_data.get("not_resume_reason") or "it doesn't contain a work history, education or skills"
        raise HTTPException(status_code=422, detail=f"This doesn't look like a résumé ({reason.rstrip('.')}).")
    return resume_data


@app.post("/api/sessions/{session_id}/resume/text")
async def upload_resume_text(session_id: str, body: TextInput):
    """Verify, trim and parse a résumé pasted as text."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty")

    try:
        resume_text = prepare_from_text(body.text)
    except ResumeRejected as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    resume_data = await _parse_resume_or_reject(resume_text)

    database.update_session(session_id, {
        "resume_text": resume_text,
        "resume_data": resume_data,
        "candidate_name": resume_data.get("candidate_name", ""),
        "status": "resume_uploaded"
    })

    return {"resume_data": resume_data, "candidate_name": resume_data.get("candidate_name", "")}


@app.post("/api/sessions/{session_id}/resume/file")
async def upload_resume_file(session_id: str, file: UploadFile = File(...)):
    """Verify, trim and parse an uploaded résumé (PDF or .txt)."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Read one byte past the limit so oversized uploads are detected without reading them fully
    content = await file.read(MAX_FILE_BYTES + 1)
    try:
        resume_text = prepare_from_upload(content, file.filename or "")
    except ResumeRejected as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    resume_data = await _parse_resume_or_reject(resume_text)

    database.update_session(session_id, {
        "resume_text": resume_text,
        "resume_data": resume_data,
        "candidate_name": resume_data.get("candidate_name", ""),
        "status": "resume_uploaded"
    })

    return {"resume_data": resume_data, "candidate_name": resume_data.get("candidate_name", "")}


# ---------------------------------------------------------------------------
# Job Description endpoint
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/jd")
async def upload_jd(session_id: str, body: TextInput):
    """Parse a job description."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")

    try:
        jd_data = await ai_engine.parse_job_description(body.text[:MAX_JD_CHARS])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse job description: {str(e)}")

    database.update_session(session_id, {
        "jd_text": body.text,
        "jd_data": jd_data,
        "status": "jd_uploaded"
    })

    return {"jd_data": jd_data}


# ---------------------------------------------------------------------------
# Skill gap analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/analyze")
async def analyze_skill_gap(session_id: str):
    """Run skill gap analysis comparing resume vs job description."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.get("resume_data"):
        raise HTTPException(status_code=400, detail="Resume not uploaded yet")
    if not session.get("jd_data"):
        raise HTTPException(status_code=400, detail="Job description not uploaded yet")

    try:
        skill_gaps = await ai_engine.analyze_skill_gap(
            session["resume_data"],
            session["jd_data"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze skill gap: {str(e)}")

    database.update_session(session_id, {
        "skill_gaps": skill_gaps,
        "status": "analyzed"
    })

    return {"skill_gaps": skill_gaps}


# ---------------------------------------------------------------------------
# Questionnaire endpoints
# ---------------------------------------------------------------------------

class QuestionnaireGenerateRequest(BaseModel):
    num_questions: int = 30
    regenerate: bool = False

@app.post("/api/sessions/{session_id}/questionnaire/generate")
async def generate_questionnaire(session_id: str, body: QuestionnaireGenerateRequest = QuestionnaireGenerateRequest()):
    """Generate adaptive MCQ questions covering all skill gaps."""
    num_questions = max(10, min(50, body.num_questions))  # clamp to 10-50
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.get("skill_gaps"):
        raise HTTPException(status_code=400, detail="Skill gap analysis not completed yet")

    # Reuse an unfinished questionnaire built from the same analysis and size
    source = _fingerprint(session["skill_gaps"], num_questions)
    existing = session.get("questionnaire_questions")
    state = session.get("questionnaire_state") or {}
    if existing and not body.regenerate and state.get("source") == source and not state.get("completed"):
        questions = existing
    else:
        try:
            questions = await ai_engine.generate_questionnaire(
                session["skill_gaps"],
                session["resume_data"],
                session["jd_data"],
                num_questions=num_questions
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate questionnaire: {str(e)}")
        state = {
            "answers": {},        # {str(question_id): letter}
            "current_index": 0,   # index into questions list
            "completed": False,
            "source": source,     # fingerprint of the inputs, for reuse
        }

    database.update_session(session_id, {
        "questionnaire_questions": questions,
        "questionnaire_state": state,
        "status": "questionnaire"
    })

    # Identify all unique skills being assessed
    skills_assessed = list(set(q["skill"] for q in questions))

    public = [_public_question(q) for q in questions]
    return {
        "questions": public,
        "first_question": public[0] if public else None,
        "total_questions": len(questions),
        "skills_being_assessed": skills_assessed
    }


@app.post("/api/sessions/{session_id}/questionnaire/answer")
async def submit_answer(session_id: str, body: AnswerInput):
    """
    Submit an answer to a questionnaire question.
    Returns immediate feedback and the next question.
    """
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    questions = session.get("questionnaire_questions")
    state = session.get("questionnaire_state")

    if not questions or not state:
        raise HTTPException(status_code=400, detail="Questionnaire not started yet")

    # Find the question being answered
    q_id = body.question_id
    question = next((q for q in questions if q["id"] == q_id), None)
    if not question:
        raise HTTPException(status_code=404, detail=f"Question {q_id} not found")

    # Validate and record the answer (answers can't be changed once recorded)
    _record_answers(state, questions, {str(q_id): body.answer})

    # Determine correctness
    correct_answer = question.get("correct_answer", "")
    is_correct = body.answer == correct_answer

    # Find next unanswered question
    answered_ids = set(state["answers"].keys())
    next_question = None
    next_index = None
    for i, q in enumerate(questions):
        if str(q["id"]) not in answered_ids:
            next_question = q
            next_index = i
            break

    if next_question:
        state["current_index"] = next_index
    else:
        state["completed"] = True

    # Save updated state
    database.update_session(session_id, {"questionnaire_state": state})

    # Calculate progress
    answered_count = len(state["answers"])
    total_count = len(questions)
    progress_pct = round((answered_count / total_count) * 100) if total_count > 0 else 0

    return {
        "is_correct": is_correct,
        "correct_answer": correct_answer,
        "explanation": question.get("explanation", ""),
        "next_question": _public_question(next_question) if next_question else None,
        "progress": {
            "answered": answered_count,
            "total": total_count,
            "percentage": progress_pct,
            "completed": state["completed"]
        }
    }


@app.post("/api/sessions/{session_id}/questionnaire/answers")
async def submit_answers(session_id: str, body: AnswersBatchInput):
    """Submit all questionnaire answers in one request."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    questions = session.get("questionnaire_questions")
    state = session.get("questionnaire_state")
    if not questions or not state:
        raise HTTPException(status_code=400, detail="Questionnaire not started yet")

    _record_answers(state, questions, body.answers)
    state["completed"] = True
    database.update_session(session_id, {"questionnaire_state": state})

    return {"answered": len(state["answers"]), "total": len(questions)}


@app.post("/api/sessions/{session_id}/questionnaire/finish")
async def finish_questionnaire(session_id: str):
    """Calculate proficiency scores after completing the questionnaire."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    questions = session.get("questionnaire_questions")
    state = session.get("questionnaire_state")

    if not questions or not state:
        raise HTTPException(status_code=400, detail="Questionnaire not started")

    answers = state.get("answers", {})

    proficiency_scores = ai_engine.calculate_proficiency_scores(questions, answers)

    database.update_session(session_id, {
        "proficiency_scores": proficiency_scores,
        "status": "scored"
    })

    return {"proficiency_scores": proficiency_scores}


# ---------------------------------------------------------------------------
# Learning pathway endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/pathway/generate")
async def generate_pathway(session_id: str, body: PathwayGenerateRequest = PathwayGenerateRequest()):
    """Generate personalized learning pathway and initialize skill tests."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.get("proficiency_scores"):
        raise HTTPException(status_code=400, detail="Proficiency scores not calculated yet")

    # Reuse the stored pathway if its inputs haven't changed
    source = _fingerprint(session["proficiency_scores"], session["skill_gaps"])
    existing = session.get("pathway")
    if existing and not body.regenerate and existing.get("_source") == source:
        return {"pathway": existing}

    try:
        pathway = await ai_engine.generate_learning_pathway(
            session["proficiency_scores"],
            session["skill_gaps"],
            session["resume_data"],
            session["jd_data"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate pathway: {str(e)}")
    pathway["_source"] = source

    # Create skill test records for new skills; keep progress on existing ones
    for skill_entry in pathway.get("skills", []):
        skill_name = skill_entry.get("skill_name", "")
        if skill_name and not database.get_skill_test(session_id, skill_name):
            database.upsert_skill_test(session_id, skill_name, {
                "status": "not_started",
                "questions": None,
                "answers": None,
                "score": 0,
                "passed": 0,
                "attempts": 0,
                "feedback": None
            })

    database.update_session(session_id, {
        "pathway": pathway,
        "status": "pathway"
    })

    return {"pathway": pathway}


@app.get("/api/sessions/{session_id}/pathway")
async def get_pathway(session_id: str):
    """Get the learning pathway along with per-skill test progress."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.get("pathway"):
        raise HTTPException(status_code=404, detail="Pathway not generated yet")

    skill_tests = database.get_all_skill_tests(session_id)

    # Build a progress summary per skill
    skill_progress = {
        test["skill_name"]: {
            "status": test["status"],
            "score": test["score"],
            "passed": test["passed"],
            "attempts": test["attempts"]
        }
        for test in skill_tests
    }

    return {
        "pathway": session["pathway"],
        "skill_progress": skill_progress
    }


# ---------------------------------------------------------------------------
# Skill test endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/skills/{skill_name}/test/generate")
async def generate_skill_test(session_id: str, skill_name: str):
    """Generate 10 verification MCQs for a specific skill."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find target level from pathway
    pathway = session.get("pathway", {})
    target_level = "intermediate"  # default
    for skill_entry in pathway.get("skills", []):
        if skill_entry.get("skill_name", "").lower() == skill_name.lower():
            target_level = skill_entry.get("target_level", "intermediate")
            break

    # An unanswered test that's already been generated is returned as-is
    skill_test = database.get_skill_test(session_id, skill_name) or {}
    if skill_test.get("status") == "testing" and skill_test.get("questions") and not skill_test.get("answers"):
        questions = skill_test["questions"]
    else:
        # Skip questions this session has already seen, so a retest is fresh
        seen = list(skill_test.get("seen_question_ids") or [])
        seen += [q.get("bank_id") for q in skill_test.get("questions") or []]
        seen += [q.get("bank_id") for q in session.get("questionnaire_questions") or []]
        seen = list(dict.fromkeys(i for i in seen if i))
        try:
            questions = await ai_engine.generate_skill_test(skill_name, target_level, exclude_ids=seen)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate skill test: {str(e)}")

        database.upsert_skill_test(session_id, skill_name, {
            "status": "testing",
            "questions": questions,
            "answers": {},
            "score": 0,
            "passed": 0,
            "seen_question_ids": seen + [q["bank_id"] for q in questions if q.get("bank_id")],
        })

    return {
        "questions": [_public_question(q) for q in questions],
        "target_level": target_level,
        "total_questions": len(questions)
    }


@app.post("/api/sessions/{session_id}/skills/{skill_name}/test/submit")
async def submit_skill_test(session_id: str, skill_name: str, body: TestAnswersInput):
    """Submit all answers for a skill test and get evaluation + feedback."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    skill_test = database.get_skill_test(session_id, skill_name)
    if not skill_test or not skill_test.get("questions"):
        raise HTTPException(status_code=400, detail="Skill test not generated yet")

    questions = skill_test["questions"]

    # Find target level from pathway
    pathway = session.get("pathway", {})
    target_level = "intermediate"
    for skill_entry in pathway.get("skills", []):
        if skill_entry.get("skill_name", "").lower() == skill_name.lower():
            target_level = skill_entry.get("target_level", "intermediate")
            break

    try:
        result = ai_engine.evaluate_skill_test(
            skill_name,
            questions,
            body.answers,
            target_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate test: {str(e)}")

    # Determine new status
    new_status = "completed" if result["passed"] else "needs_revision"
    attempts = (skill_test.get("attempts") or 0) + 1

    database.upsert_skill_test(session_id, skill_name, {
        "status": new_status,
        "answers": body.answers,
        "score": result["score"],
        "passed": 1 if result["passed"] else 0,
        "attempts": attempts,
        "feedback": result["feedback"]
    })

    return {
        "result": result,
        "new_status": new_status,
        "attempts": attempts
    }


@app.post("/api/sessions/{session_id}/skills/{skill_name}/status")
async def update_skill_status(session_id: str, skill_name: str, body: StatusInput):
    """Manually update the status of a skill (e.g., mark as 'learning')."""
    valid_statuses = {"not_started", "learning", "testing", "completed", "needs_revision"}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    database.upsert_skill_test(session_id, skill_name, {"status": body.status})

    return {"skill_name": skill_name, "status": body.status}


@app.get("/api/sessions/{session_id}/skills/{skill_name}/progress")
async def get_skill_progress(session_id: str, skill_name: str):
    """Get detailed skill test progress including questions and results."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    skill_test = database.get_skill_test(session_id, skill_name)
    if not skill_test:
        raise HTTPException(status_code=404, detail="Skill test record not found")

    if skill_test.get("questions") and not skill_test.get("answers"):
        skill_test["questions"] = [_public_question(q) for q in skill_test["questions"]]
    skill_test.pop("seen_question_ids", None)
    return skill_test


# ---------------------------------------------------------------------------
# Manager endpoints
# ---------------------------------------------------------------------------

@app.post("/api/manager/login")
async def manager_login(body: ManagerLoginInput):
    """Check manager email against whitelist and return a session token."""
    manager = database.verify_manager_email(body.email)
    if not manager:
        raise HTTPException(status_code=401, detail="Not authorized as a manager")
    token = database.create_manager_session(manager["id"], manager["email"])
    return {"token": token, "email": manager["email"]}


@app.post("/api/manager/logout")
async def manager_logout(request: Request):
    token = request.headers.get("X-Manager-Token", "")
    database.delete_manager_session(token)
    return {"ok": True}


@app.get("/api/manager/dashboard")
async def manager_dashboard(request: Request):
    """Return all employee data. Requires valid manager session token."""
    token = request.headers.get("X-Manager-Token", "")
    session = database.verify_manager_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")
    employees = database.get_all_employees()
    return {"employees": employees}


@app.delete("/api/manager/employees/{session_id}")
async def manager_delete_employee(session_id: str, request: Request):
    """Delete an employee session. Requires valid manager session token."""
    token = request.headers.get("X-Manager-Token", "")
    if not database.verify_manager_session(token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    deleted = database.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Employee not found")
    return {"ok": True}


@app.post("/api/sessions/{session_id}/link-user")
async def link_user_to_session(session_id: str, request: Request):
    """Link a Firebase user's UID and email to a session."""
    body = await request.json()
    firebase_uid = body.get("firebase_uid", "")
    user_email = body.get("user_email", "")
    display_name = body.get("display_name", "")
    if firebase_uid:
        database.update_session_user(session_id, firebase_uid, user_email)
    # Set candidate_name from Google display name if not already set by resume parsing
    if display_name:
        session = database.get_session(session_id)
        if session and not session.get("candidate_name"):
            database.update_session(session_id, {"candidate_name": display_name})
    return {"ok": True}


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@app.post("/api/admin/login")
async def admin_login(body: AdminLoginInput):
    """Verify the admin password from .env and issue a session token."""
    admin_password = os.getenv("ADMIN_PASSWORD", "")
    if not admin_password or body.password != admin_password:
        raise HTTPException(status_code=401, detail="Invalid admin password")
    token = secrets.token_hex(32)
    _admin_sessions.add(token)
    return {"token": token}


@app.post("/api/admin/logout")
async def admin_logout(request: Request):
    token = request.headers.get("X-Admin-Token", "")
    _admin_sessions.discard(token)
    return {"ok": True}


@app.get("/api/admin/managers")
async def admin_list_managers(request: Request):
    _require_admin(request)
    return {"managers": database.list_managers()}


@app.post("/api/admin/managers")
async def admin_add_manager(request: Request, body: AdminAddManagerInput):
    _require_admin(request)
    added = database.add_manager(body.email)
    if not added:
        raise HTTPException(status_code=409, detail="Manager already exists")
    return {"ok": True, "email": body.email.lower().strip()}


@app.delete("/api/admin/managers/{email}")
async def admin_remove_manager(email: str, request: Request):
    _require_admin(request)
    removed = database.remove_manager(email)
    if not removed:
        raise HTTPException(status_code=404, detail="Manager not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Static file serving (SPA)
# ---------------------------------------------------------------------------

# Serve the frontend SPA from the static/ directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
