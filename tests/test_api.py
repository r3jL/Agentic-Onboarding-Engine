"""End-to-end API flow with the fake LLM."""

import pytest
from fastapi.testclient import TestClient

from app import ai_engine
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

    # Pathway: one parallel call per gap skill, resources get search links, generated once
    pathway = client.post(f"/api/sessions/{sid}/pathway/generate").json()["pathway"]
    gap_skills = {g["skill"] for g in gaps["skill_gaps"]}
    assert {s["skill_name"] for s in pathway["skills"]} == gap_skills
    assert pathway["skills"][0]["resources"][0]["url"].startswith("https://www.google.com/search?q=")
    order = pathway["learning_order"]
    assert order.index("Machine Learning") < order.index("Kubernetes")   # prerequisite first
    # weeks are capped at 2 per level climbed, and the total is their sum
    for s in pathway["skills"]:
        levels = ai_engine.LEVEL_RANK[s["target_level"]] - ai_engine.LEVEL_RANK[s["current_level"]]
        assert s["estimated_weeks"] <= max(1, 2 * levels)
    assert pathway["total_estimated_weeks"] == sum(s["estimated_weeks"] for s in pathway["skills"])
    ml = next(s for s in pathway["skills"] if s["skill_name"] == "Machine Learning")
    assert ml["target_level"] == "advanced" and ml["priority"] == "critical"
    assert "your assessment placed you at" in ml["reasoning"]
    # one call per skill that needs a plan (skills already at target are verify-only)
    to_study = sum(1 for s in pathway["skills"] if not s["verify_only"])
    assert len([c for c in llm.calls if c.startswith("pathway:")]) == to_study
    client.post(f"/api/sessions/{sid}/pathway/generate")
    assert len([c for c in llm.calls if c.startswith("pathway:")]) == to_study

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


def test_pathway_entries_are_reused_across_candidates(llm):
    import asyncio
    from app import ai_engine
    gaps = {"skill_gaps": [
        {"skill": "Kubernetes", "candidate_level": "none", "required_level": "advanced", "gap_severity": "critical", "priority": 1, "reason": "No evidence."},
        {"skill": "Machine Learning", "candidate_level": "none", "required_level": "advanced", "gap_severity": "critical", "priority": 2, "reason": "No evidence."},
    ]}
    scores = {"Kubernetes": {"level": "beginner", "score": 20}, "Machine Learning": {"level": "basic", "score": 45}}
    first = asyncio.run(ai_engine.generate_learning_pathway(scores, gaps, {}, {"role_title": "ML Engineer"}))
    calls = len(llm.calls)
    assert calls == 2
    second = asyncio.run(ai_engine.generate_learning_pathway(scores, gaps, {}, {"role_title": "Other role"}))
    assert len(llm.calls) == calls                                  # same skills and levels: no LLM calls
    assert second["learning_order"] == first["learning_order"] == ["Machine Learning", "Kubernetes"]
    assert "Other role" in second["overview"]                        # personal parts still per candidate
    other = dict(scores, Kubernetes={"level": "basic", "score": 50})
    asyncio.run(ai_engine.generate_learning_pathway(other, gaps, {}, {}))
    assert llm.calls[calls:] == ["pathway:Kubernetes"]               # different level: one new entry


def test_already_met_skills_are_verify_only(llm):
    import asyncio
    from app import ai_engine
    gaps = {"skill_gaps": [
        {"skill": "Kubernetes", "candidate_level": "none", "required_level": "intermediate", "gap_severity": "high", "priority": 1, "reason": "No evidence."},
        {"skill": "Machine Learning", "candidate_level": "none", "required_level": "advanced", "gap_severity": "critical", "priority": 2, "reason": "No evidence."},
    ]}
    # Kubernetes assessed at intermediate = the required level; Machine Learning still below
    scores = {"Kubernetes": {"level": "intermediate", "score": 70}, "Machine Learning": {"level": "basic", "score": 45}}
    p = asyncio.run(ai_engine.generate_learning_pathway(scores, gaps, {}, {"role_title": "ML Engineer"}))
    assert llm.calls == ["pathway:Machine Learning"]                 # no plan generated for Kubernetes
    assert p["learning_order"] == ["Machine Learning", "Kubernetes"]  # verify-only goes last
    k8s = p["skills"][1]
    assert k8s["verify_only"] and k8s["estimated_weeks"] == 0 and k8s["learning_steps"] == []
    assert "meets the intermediate level" in k8s["reasoning"]
    assert p["total_estimated_weeks"] == p["skills"][0]["estimated_weeks"]
    assert "just take the verification test" in p["overview"]

    all_met = {"Kubernetes": {"level": "intermediate", "score": 70}, "Machine Learning": {"level": "advanced", "score": 90}}
    p2 = asyncio.run(ai_engine.generate_learning_pathway(all_met, gaps, {}, {"role_title": "ML Engineer"}))
    assert all(s["verify_only"] for s in p2["skills"]) and p2["total_estimated_weeks"] == 0
    assert llm.calls == ["pathway:Machine Learning"]                 # still no extra calls
