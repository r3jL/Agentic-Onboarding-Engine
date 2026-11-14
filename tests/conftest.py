"""Shared fixtures: an isolated SQLite DB per test and a fake LLM that counts calls."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import ai_engine, database  # noqa: E402

RESUME_TEXT = """Jane Doe
jane.doe@example.com | +1 415 555 0134 | linkedin.com/in/janedoe

SUMMARY
Backend engineer with six years of experience building data-heavy web services in Python.

EXPERIENCE
Senior Software Engineer, Acme Corp            Jan 2021 - Present
- Led the migration of a monolith to FastAPI microservices serving 2M requests a day.
- Designed PostgreSQL schemas and tuned slow queries, cutting p95 latency by 40%.
Software Engineer, Widget Labs                  Jun 2018 - Dec 2020
- Built ETL pipelines in Python and Airflow; maintained Docker-based deployments.
- Wrote integration tests and introduced code review guidelines for the team.

EDUCATION
B.Sc. Computer Science, State University, 2018

SKILLS
Python, FastAPI, Django, PostgreSQL, Docker, Airflow, Git, REST APIs, Linux
"""

JD_TEXT = "Senior ML Engineer. Requires Python (advanced), Kubernetes (intermediate), Machine Learning (advanced). 5+ years."


class FakeLLM:
    """Stands in for ai_engine._complete_json; answers by call label and records calls."""

    def __init__(self):
        self.calls = []
        self.counter = 0
        self.is_resume = True

    async def __call__(self, label, prompt, schema, max_tokens):
        self.calls.append(label)
        if label == "parse_resume":
            if not self.is_resume:
                return {"is_resume": False, "not_resume_reason": "a restaurant menu"}
            return {
                "is_resume": True,
                "candidate_name": "Jane Doe",
                "skills": [
                    {"name": "Python", "proficiency": "Advanced", "years": 6},
                    {"name": "Docker", "proficiency": "basic", "years": 3},
                ],
                "work_experience": [{"company": "Acme", "title": "Senior SWE", "duration": "2021-"}],
                "education": [{"degree": "B.Sc.", "institution": "State", "year": 2018}],
                "total_experience_years": "6",
            }
        if label == "parse_jd":
            return {
                "role_title": "Senior ML Engineer",
                "required_skills": [
                    {"name": "Python", "level": "advanced", "importance": "required"},
                    {"name": "Kubernetes", "level": "intermediate", "importance": "required"},
                    {"name": "Machine Learning", "level": "advanced", "importance": "required"},
                    {"name": "Container orchestration basics", "level": "basic", "importance": "nice_to_have"},
                ],
                "responsibilities": ["Ship models"],
                "required_experience_years": 5,
            }
        if label == "match_skills":
            return {
                "matches": [
                    {"jd_skill": "Python", "resume_skill": "Python", "reason": "Six years of Python."},
                    {"jd_skill": "Kubernetes", "resume_skill": "Docker", "reason": "Containers only, no orchestration."},
                    {"jd_skill": "Machine Learning", "resume_skill": None, "reason": ""},
                ],
                "summary": "Strong backend engineer; needs ML.",
            }
        if label.startswith("mcq:"):
            questions = []
            for n, level in re.findall(r"(\d+) (basic|intermediate|advanced|expert)", prompt.split("\n")[0]):
                for _ in range(int(n)):
                    self.counter += 1
                    questions.append({
                        "level": level,
                        "question": f"Scenario question number {self.counter} about {label[4:]}?",
                        "options": {"A": f"right {self.counter}", "B": "wrong b", "C": "wrong c", "D": "wrong d"},
                        "correct_answer": "a",
                        "explanation": "Because.",
                        "concept": f"concept {self.counter}",
                    })
            return {"questions": questions}
        if label == "pathway":
            return {
                "total_estimated_weeks": 10,
                "overview": "Plan.",
                "skills": [{
                    "skill_name": "Machine Learning",
                    "current_level": "none",
                    "target_level": "Advanced",
                    "priority": "critical",
                    "estimated_weeks": 6,
                    "resources": [
                        {"type": "course", "title": "Machine Learning Specialization", "provider": "Coursera"},
                        {"type": "video", "title": "Neural networks explained", "provider": "YouTube"},
                    ],
                    "learning_steps": [{"week": 1, "focus": "Basics", "tasks": ["Read"], "milestone": "Done"}],
                }],
            }
        raise AssertionError(f"unexpected LLM call {label}")


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    database.init_db()
    return database


@pytest.fixture
def llm(monkeypatch, db):
    fake = FakeLLM()
    monkeypatch.setattr(ai_engine, "_complete_json", fake)
    return fake
