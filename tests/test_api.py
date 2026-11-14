"""End-to-end API flow with the fake LLM."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from conftest import JD_TEXT, RESUME_TEXT


@pytest.fixture
def client(llm):
    with TestClient(app) as c:
        yield c


def new_session(client):
    return client.post("/api/sessions").json()["session_id"]


def test_non_resume_text_rejected_without_llm_call(client, llm):
    sid = new_session(client)
    r = client.post(f"/api/sessions/{sid}/resume/text", json={"text": "Lorem ipsum dolor sit amet. " * 30})
    assert r.status_code == 422
    assert "résumé" in r.json()["detail"]
    assert llm.calls == []
    assert client.get(f"/api/sessions/{sid}").json()["resume_data"] is None


def test_llm_guard_rejects_borderline_document(client, llm):
    llm.is_resume = False
    sid = new_session(client)
    r = client.post(f"/api/sessions/{sid}/resume/text", json={"text": RESUME_TEXT})
    assert r.status_code == 422
    assert "restaurant menu" in r.json()["detail"]
    assert client.get(f"/api/sessions/{sid}").json()["resume_data"] is None


def test_unsupported_upload_rejected(client):
    sid = new_session(client)
    r = client.post(f"/api/sessions/{sid}/resume/file", files={"file": ("cv.docx", b"PK\x03\x04", "application/octet-stream")})
    assert r.status_code == 415


def test_full_flow(client, llm):
    sid = new_session(client)
    assert client.post(f"/api/sessions/{sid}/resume/text", json={"text": RESUME_TEXT}).status_code == 200
    assert client.post(f"/api/sessions/{sid}/jd", json={"text": JD_TEXT}).status_code == 200
    gaps = client.post(f"/api/sessions/{sid}/analyze").json()["skill_gaps"]
    assert gaps["strong_matches"] == ["Python"]

    # Questionnaire: answers hidden, repeat call reuses it
    r = client.post(f"/api/sessions/{sid}/questionnaire/generate", json={"num_questions": 12}).json()
    questions = r["questions"]
    assert len(questions) == 12
    assert all("correct_answer" not in q and "explanation" not in q for q in questions)
    calls = len(llm.calls)
    again = client.post(f"/api/sessions/{sid}/questionnaire/generate", json={"num_questions": 12}).json()
    assert [q["bank_id"] for q in again["questions"]] == [q["bank_id"] for q in questions]
    assert len(llm.calls) == calls
    session = client.get(f"/api/sessions/{sid}").json()
    assert all("correct_answer" not in q for q in session["questionnaire_questions"])

    # Batch answers; an answer can't be changed afterwards
    answers = {str(q["id"]): "A" for q in questions}
    r = client.post(f"/api/sessions/{sid}/questionnaire/answers", json={"answers": answers})
    assert r.json() == {"answered": 12, "total": 12}
    changed = dict(answers, **{"1": "B" if answers["1"] != "B" else "C"})
    assert client.post(f"/api/sessions/{sid}/questionnaire/answers", json={"answers": changed}).status_code == 409
    scores = client.post(f"/api/sessions/{sid}/questionnaire/finish").json()["proficiency_scores"]
    assert set(scores) == {"Machine Learning", "Kubernetes", "Container orchestration basics"}
    assert isinstance(client.get(f"/api/sessions/{sid}").json()["questionnaire_correct"], int)

    # Pathway: generated once, resources get search links
    pathway = client.post(f"/api/sessions/{sid}/pathway/generate").json()["pathway"]
    assert pathway["skills"][0]["resources"][0]["url"].startswith("https://www.google.com/search?q=")
    assert pathway["learning_order"] == ["Machine Learning"]
    client.post(f"/api/sessions/{sid}/pathway/generate")
    assert llm.calls.count("pathway") == 1

    # Skill test: hidden answers, idempotent until answered, fresh on retest
    test = client.post(f"/api/sessions/{sid}/skills/Machine%20Learning/test/generate").json()
    assert test["total_questions"] == 10 and test["target_level"] == "advanced"
    assert all("correct_answer" not in q for q in test["questions"])
    same = client.post(f"/api/sessions/{sid}/skills/Machine%20Learning/test/generate").json()
    assert [q["bank_id"] for q in same["questions"]] == [q["bank_id"] for q in test["questions"]]

    result = client.post(f"/api/sessions/{sid}/skills/Machine%20Learning/test/submit",
                         json={"answers": {str(q["id"]): "D" for q in test["questions"]}}).json()
    assert result["new_status"] in ("completed", "needs_revision")
    assert result["result"]["feedback"]

    retest = client.post(f"/api/sessions/{sid}/skills/Machine%20Learning/test/generate").json()
    assert not {q["bank_id"] for q in retest["questions"]} & {q["bank_id"] for q in test["questions"]}

    # Regenerating the pathway keeps skill test progress
    client.post(f"/api/sessions/{sid}/pathway/generate", json={"regenerate": True})
    progress = client.get(f"/api/sessions/{sid}/pathway").json()["skill_progress"]
    assert progress["Machine Learning"]["attempts"] == 1
