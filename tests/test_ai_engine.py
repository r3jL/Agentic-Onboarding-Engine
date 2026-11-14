import asyncio
import json

from app import ai_engine
from app.schemas import Pathway, ResumeData, response_format


# --- schemas ---------------------------------------------------------------

def test_schemas_are_lenient():
    resume = ResumeData.model_validate({
        "is_resume": True,
        "candidate_name": None,
        "skills": [{"name": "Python", "proficiency": "Advanced", "years": "5"},
                   {"name": "Go", "proficiency": "guru"}],
        "education": [{"degree": "BSc", "year": 2018}],
    })
    assert resume.candidate_name == ""
    assert resume.skills[0].proficiency == "advanced" and resume.skills[0].years == 5
    assert resume.skills[1].proficiency == "intermediate"
    assert resume.education[0].year == "2018"

    pathway = Pathway.model_validate({"skills": [{"skill_name": "ML", "learning_steps": [{"week": 3}]}]})
    assert pathway.skills[0].learning_steps[0].week == "3"


def test_response_format_inlines_refs_and_drops_title_annotations():
    schema = response_format(Pathway)["json_schema"]["schema"]
    assert "$ref" not in json.dumps(schema) and "$defs" not in schema

    def annotations(node):
        if isinstance(node, dict):
            yield from (v for k, v in node.items() if k == "title" and isinstance(v, str))
            for v in node.values():
                yield from annotations(v)
        elif isinstance(node, list):
            for v in node:
                yield from annotations(v)

    assert list(annotations(schema)) == []
    resource = schema["properties"]["skills"]["items"]["properties"]["resources"]["items"]
    assert "title" in resource["properties"]          # a real field named "title" survives


# --- retries ---------------------------------------------------------------

def test_rate_limits_retry_with_backoff(monkeypatch):
    import httpx
    import openai

    def rate_limited(retry_after=None):
        headers = {"retry-after": retry_after} if retry_after else {}
        response = httpx.Response(429, request=httpx.Request("POST", "http://llm"), headers=headers)
        return openai.RateLimitError("rate limited", response=response, body=None)

    errors = [rate_limited(), rate_limited("30")]
    sleeps = []

    class FakeCompletions:
        async def create(self, **kwargs):
            if errors:
                raise errors.pop(0)
            message = type("M", (), {"content": "{}"})
            return type("R", (), {"choices": [type("C", (), {"message": message})], "usage": None})

    fake_client = type("Client", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})})

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(ai_engine, "_get_client", lambda: fake_client)
    monkeypatch.setattr(ai_engine.asyncio, "sleep", fake_sleep)
    asyncio.run(ai_engine._complete("t", 10, []))
    assert len(sleeps) == 2
    assert 2 <= sleeps[0] <= 2.5          # first backoff step
    assert 30 <= sleeps[1] <= 37.5        # retry-after (30s) beats the 8s step


def test_daily_quota_is_not_retried(monkeypatch):
    import httpx
    import openai
    import pytest

    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(1)
            response = httpx.Response(429, request=httpx.Request("POST", "http://llm"))
            raise openai.RateLimitError("Rate limit exceeded: free-models-per-day", response=response, body=None)

    fake_client = type("Client", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})})
    monkeypatch.setattr(ai_engine, "_get_client", lambda: fake_client)
    with pytest.raises(openai.RateLimitError):
        asyncio.run(ai_engine._complete("t", 10, []))
    assert len(calls) == 1


# --- gap analysis ----------------------------------------------------------

RESUME = {
    "skills": [{"name": "Python", "proficiency": "advanced"}, {"name": "Docker", "proficiency": "basic"}],
    "total_experience_years": 6,
}
JD = {
    "required_skills": [
        {"name": "Python", "level": "advanced", "importance": "required"},
        {"name": "Kubernetes", "level": "intermediate", "importance": "required"},
        {"name": "Machine Learning", "level": "advanced", "importance": "required"},
        {"name": "Terraform", "level": "basic", "importance": "nice_to_have"},
    ],
    "required_experience_years": 5,
}
MATCHES = {"matches": [
    {"jd_skill": "python", "resume_skill": "Python", "reason": "ok"},
    {"jd_skill": "Kubernetes", "resume_skill": "Docker", "reason": "containers only"},
    {"jd_skill": "Machine Learning", "resume_skill": None, "reason": ""},
], "summary": ""}


def test_gap_analysis_is_deterministic_and_correct():
    a = ai_engine.compute_gap_analysis(RESUME, JD, MATCHES)
    b = ai_engine.compute_gap_analysis(RESUME, JD, MATCHES)
    assert a == b

    gaps = {g["skill"]: g for g in a["skill_gaps"]}
    assert a["strong_matches"] == ["Python"]
    assert gaps["Machine Learning"]["gap_severity"] == "critical"     # none -> advanced, required
    assert gaps["Kubernetes"]["gap_severity"] == "medium"             # basic -> intermediate, required
    assert gaps["Kubernetes"]["candidate_level"] == "basic"
    assert gaps["Terraform"]["gap_severity"] == "low"                 # none -> basic, nice to have

    assert gaps["Machine Learning"]["priority"] == 1
    assert a["missing_skills"] == ["Machine Learning"]                # Terraform isn't required
    assert gaps["Machine Learning"]["reason"]                          # fallback reason filled in
    assert 0 < a["overall_fit_score"] < 100
    assert a["summary"]


def test_severity_scale():
    jd = {"required_skills": [
        {"name": "A", "level": "advanced", "importance": "required"},       # none -> advanced
        {"name": "B", "level": "intermediate", "importance": "required"},   # none -> intermediate
        {"name": "C", "level": "intermediate", "importance": "preferred"},  # none -> intermediate
        {"name": "D", "level": "intermediate", "importance": "nice_to_have"},
    ], "required_experience_years": 0}
    result = ai_engine.compute_gap_analysis({"skills": []}, jd, {"matches": []})
    severities = {g["skill"]: g["gap_severity"] for g in result["skill_gaps"]}
    assert severities == {"A": "critical", "B": "high", "C": "medium", "D": "low"}


# --- questionnaire planning ------------------------------------------------

def test_plan_sums_to_total_with_minimum_three():
    skills = [{"skill": f"S{i}", "gap_severity": sev}
              for i, sev in enumerate(["critical", "high", "medium", "low", "low"])]
    for n in (15, 30, 37, 50):
        plan = ai_engine.plan_questionnaire(skills, n)
        totals = [sum(p["counts"].values()) for p in plan]
        assert sum(totals) == n
        assert min(totals) >= 3
        assert all(min(p["counts"].values()) >= 1 for p in plan)
    plan = ai_engine.plan_questionnaire(skills, 30)
    assert sum(plan[0]["counts"].values()) > sum(plan[-1]["counts"].values())


def test_plan_drops_lowest_priority_when_too_many_skills():
    skills = [{"skill": f"S{i}", "gap_severity": "medium"} for i in range(8)]
    plan = ai_engine.plan_questionnaire(skills, 12)
    assert [p["skill"] for p in plan] == ["S0", "S1", "S2", "S3"]


def test_shuffle_keeps_correct_answer_and_adds_idk():
    q = {"options": {"A": "right", "B": "x", "C": "y", "D": "z"}, "correct_answer": "A"}
    positions = set()
    for _ in range(200):
        options, correct = ai_engine._shuffle_options(q, include_idk=True)
        assert options[correct] == "right"
        assert options["E"] == ai_engine.IDK_OPTION
        positions.add(correct)
    assert positions == {"A", "B", "C", "D"}


# --- question bank ---------------------------------------------------------

GAPS = {"skill_gaps": [
    {"skill": "Kubernetes", "gap_severity": "high", "priority": 1},
    {"skill": "Machine Learning", "gap_severity": "critical", "priority": 2},
]}


def test_questionnaire_reuses_bank(llm):
    first = asyncio.run(ai_engine.generate_questionnaire(GAPS, {}, {}, num_questions=12))
    calls_after_first = len(llm.calls)
    assert len(first) == 12
    assert calls_after_first == 2                           # one call per skill, not per question

    second = asyncio.run(ai_engine.generate_questionnaire(GAPS, {}, {}, num_questions=12))
    assert len(second) == 12
    assert len(llm.calls) == calls_after_first              # served entirely from the bank
    assert {q["bank_id"] for q in second} == {q["bank_id"] for q in first}
    for q in second:
        assert q["options"]["E"] == ai_engine.IDK_OPTION
        assert q["options"][q["correct_answer"]].startswith("right")


def test_skill_test_retest_gets_fresh_questions(llm):
    first = asyncio.run(ai_engine.generate_skill_test("Kubernetes", "intermediate"))
    assert len(first) == 10 and "E" not in first[0]["options"]
    seen = [q["bank_id"] for q in first]
    retest = asyncio.run(ai_engine.generate_skill_test("Kubernetes", "intermediate", exclude_ids=seen))
    assert len(retest) == 10
    assert not set(seen) & {q["bank_id"] for q in retest}


def test_skill_key_normalises():
    assert ai_engine.skill_key("  Node.JS ") == "node.js"
    assert ai_engine.skill_key("C++") == "c++"
    assert ai_engine.skill_key("machine-learning") == ai_engine.skill_key("Machine Learning")


def test_resource_urls_are_search_links():
    assert ai_engine.resource_url({"type": "video", "title": "K8s in 100s", "provider": "YouTube"}) \
        .startswith("https://www.youtube.com/results?search_query=")
    url = ai_engine.resource_url({"type": "course", "title": "ML Specialization", "provider": "Coursera"})
    assert "site%3Acoursera.org" in url


def test_feedback_is_templated():
    questions = [{"id": i, "question": "q", "options": {"A": "a", "B": "b"}, "correct_answer": "A",
                  "concept_tested": f"topic {i}"} for i in range(1, 11)]
    result = ai_engine.evaluate_skill_test("Kubernetes", questions, {str(i): "A" for i in range(1, 8)}, "intermediate")
    assert result["score"] == 70 and not result["passed"]
    assert "topic 8" in result["feedback"] and "fresh set" in result["feedback"]


def test_empty_parse_retries_then_fails(llm, monkeypatch):
    import pytest

    async def empty(label, prompt, schema, max_tokens):
        llm.calls.append(label)
        return {"is_resume": True, "skills": []} if label == "parse_resume" else {"required_skills": []}

    monkeypatch.setattr(ai_engine, "_complete_json", empty)
    with pytest.raises(ValueError, match="no skills"):
        asyncio.run(ai_engine.parse_resume("text"))
    with pytest.raises(ValueError, match="no skills"):
        asyncio.run(ai_engine.parse_job_description("text"))
    assert llm.calls == ["parse_resume", "parse_resume", "parse_jd", "parse_jd"]
