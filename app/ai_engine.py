"""
ai_engine.py - AI integration for the AI Onboarding Engine.

Talks to any OpenAI-compatible chat completions API. By default this is a local
Ollama server (localhost:11434) serving Qwen2.5-7B, with no API key required.
Set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL to use a hosted provider instead
(e.g. Groq, Google Gemini, OpenRouter) — see .env.example.

Token-saving design:
- Structured output (JSON schema) where the provider supports it, with a
  plain-JSON fallback; compact prompts with instructions first, data last.
- Gap analysis is computed in Python; the LLM only matches skill names.
- Questions live in a shared bank (database.question_bank) and are reused
  across sessions; only shortfalls are generated, one call per skill, in parallel.
- Option order and the "I don't know" option are handled in Python.
- Test feedback is templated rather than generated.
"""

import asyncio
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import ValidationError

from app import database
from app.schemas import (
    GeneratedMCQ,
    JDData,
    MatchResult,
    MCQBatch,
    ResumeData,
    SkillPlan,
    response_format,
)

load_dotenv()
log = logging.getLogger(__name__)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")  # Ollama ignores the key
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
# Optional cap on max_tokens, for providers with a lower output limit
_MAX_TOKENS_CAP = int(os.environ["LLM_MAX_TOKENS"]) if os.environ.get("LLM_MAX_TOKENS") else None
# Extra tokens per request for reasoning models, whose hidden "thinking" counts
# against max_tokens (budgets below are sized for the visible answer only)
_REASONING_BUDGET = int(os.environ.get("LLM_REASONING_BUDGET", "0"))
# Max LLM requests in flight at once (parallel question generation respects this)
_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("LLM_CONCURRENCY", "8")))
# Structured output is switched off automatically if the provider rejects it
_use_json_schema = os.environ.get("LLM_JSON_SCHEMA", "true").lower() != "false"

IDK_OPTION = "I don't know"
SKILL_TEST_SIZE = 10
SKILL_TEST_PASS_MARK = 75

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=float(os.environ.get("LLM_TIMEOUT", "600")),
            # Retries are handled in _complete with longer backoff than the SDK's
            max_retries=0,
        )
    return _client


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

# Free tiers often return 429 with no retry-after; the SDK's sub-second backoff
# just burns quota, so wait longer between attempts.
_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "4"))
_BACKOFF_SECONDS = (2, 8, 20, 45)
_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)


def _retry_delay(error: Exception, attempt: int) -> float:
    delay = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
    response = getattr(error, "response", None)
    try:
        delay = max(delay, float(response.headers.get("retry-after")))
    except (AttributeError, TypeError, ValueError):
        pass
    return min(delay, 120) * random.uniform(1.0, 1.25)


async def _complete(label: str, max_tokens: int, messages: List[Dict[str, str]], **extra):
    """Run one chat completion, bounded by the concurrency limit, and log token usage."""
    max_tokens += _REASONING_BUDGET
    if _MAX_TOKENS_CAP is not None:
        max_tokens = min(max_tokens, _MAX_TOKENS_CAP)
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with _SEMAPHORE:
                response = await _get_client().chat.completions.create(
                    model=MODEL, max_tokens=max_tokens, messages=messages, **extra
                )
            break
        except _RETRYABLE as e:
            # A daily quota won't reset within any backoff window; fail fast
            if attempt == _MAX_RETRIES or re.search(r"per[- ]day|daily", str(e), re.I):
                raise
            delay = _retry_delay(e, attempt)
            log.warning("LLM %s: %s, retrying in %.0fs (%d/%d)",
                        label, type(e).__name__, delay, attempt + 1, _MAX_RETRIES)
            await asyncio.sleep(delay)  # outside the semaphore, so other calls can proceed
    usage = getattr(response, "usage", None)
    if usage:
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None) or 0
        log.info("LLM %s: %s prompt + %s completion tokens%s", label, usage.prompt_tokens,
                 usage.completion_tokens, f" ({reasoning} reasoning)" if reasoning else "")
    if response.choices and getattr(response.choices[0], "finish_reason", None) == "length":
        log.warning("LLM %s: reply cut off at max_tokens=%s; raise LLM_MAX_TOKENS or use a "
                    "non-reasoning model", label, max_tokens)
    return response


def _get_text(response) -> str:
    """Extract text content from a chat completion response."""
    return response.choices[0].message.content or ""


async def _complete_json(label: str, prompt: str, schema: type, max_tokens: int) -> Any:
    """
    Ask for JSON matching `schema`. Uses `response_format=json_schema` when the
    provider supports it; if the provider rejects that, falls back (once, for the
    rest of the process) to plain prompting + tolerant parsing.
    """
    global _use_json_schema
    messages = [
        {"role": "system", "content": "You reply with a single valid JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(2):  # one retry if the reply isn't valid JSON
        if attempt:
            max_tokens *= 2  # the usual cause is a reply cut off at the token limit
        try:
            if _use_json_schema:
                try:
                    response = await _complete(label, max_tokens, messages,
                                               response_format=response_format(schema))
                except openai.BadRequestError as e:
                    if not re.search(r"response_format|schema|json", str(e), re.I):
                        raise
                    log.warning("Provider rejected json_schema output; using plain JSON prompting. (%s)", e)
                    _use_json_schema = False
                    response = await _complete(label, max_tokens, messages)
            else:
                response = await _complete(label, max_tokens, messages)
            return _parse_json_response(_get_text(response))
        except ValueError as e:
            if attempt == 1:
                raise
            log.warning("LLM %s: invalid JSON, retrying once with double the token budget (%s)",
                        label, str(e).splitlines()[0])


def _parse_json_response(text: str) -> Any:
    """
    Parse a JSON response from the model, stripping markdown code fences if present.
    Models sometimes wrap output in ```json ... ``` blocks.
    """
    text = text.strip()

    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to find a complete JSON object/array within the text
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Truncated array recovery: find the last complete object and close the array
        if text.lstrip().startswith('['):
            last_close = text.rfind('}')
            if last_close != -1:
                truncated = text[:last_close + 1].rstrip().rstrip(',') + '\n]'
                try:
                    return json.loads(truncated)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Could not parse JSON from model response: {e}\nText was: {text[:500]}")


def _compact(data: Any) -> str:
    """JSON without whitespace — pretty-printing only costs tokens."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def skill_key(name: str) -> str:
    """Normalised skill name used as the question-bank key ("Node.JS " -> "node.js")."""
    return re.sub(r"[^a-z0-9+#.]+", " ", (name or "").lower()).strip()


def _question_norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


# ---------------------------------------------------------------------------
# Resume and job description parsing
# ---------------------------------------------------------------------------

async def parse_resume(text: str) -> Dict[str, Any]:
    """
    Extract structured information from résumé text.

    Returns ResumeData as a dict. `is_resume` is False (with `not_resume_reason`)
    when the model judges the document is not a résumé — the caller rejects it.
    """
    prompt = f"""You parse résumés. First decide whether the document below is a résumé/CV (a person's professional profile: work history, education, skills).
If it is NOT a résumé, return {{"is_resume":false,"not_resume_reason":"<what it appears to be, max 12 words>"}}.
If it is, return:
{{"is_resume":true,"candidate_name":str,"skills":[{{"name":str,"proficiency":"beginner|basic|intermediate|advanced|expert","years":number}}],"work_experience":[{{"company":str,"title":str,"duration":str}}],"education":[{{"degree":str,"institution":str,"year":str}}],"total_experience_years":number}}
Rules:
- skills: technical skills, tools, frameworks and domain knowledge; at most 5 soft skills.
- proficiency: judge from years, role seniority and how the skill is used.
- years: 0 if unknown. total_experience_years: total professional experience.

DOCUMENT:
<<<
{text}
>>>"""
    for attempt in range(2):
        raw = await _complete_json("parse_resume", prompt, ResumeData, max_tokens=2500)
        resume = ResumeData.model_validate(raw)
        if not resume.is_resume or resume.skills:
            return resume.model_dump()
        log.warning("LLM parse_resume: no skills extracted (attempt %d)", attempt + 1)
    raise ValueError("the AI model returned no skills for this résumé; please try again")


async def parse_job_description(text: str) -> Dict[str, Any]:
    """Extract role title, required skills (level + importance), responsibilities, years."""
    prompt = f"""You parse job descriptions. Return:
{{"role_title":str,"required_skills":[{{"name":str,"level":"basic|intermediate|advanced|expert","importance":"required|preferred|nice_to_have"}}],"responsibilities":[str],"required_experience_years":number}}
Rules:
- required_skills: at most 12 distinct skills, each a technology, tool or discipline under its common industry name (e.g. "Machine Learning", "Kubernetes", "Apache Kafka"), with the level the role needs.
- Fold sub-topics into their parent skill (e.g. evaluation metrics and feature engineering belong to Machine Learning) and merge overlapping items (e.g. "streaming data" and "Kafka").
- responsibilities: at most 6, each under 12 words.
- required_experience_years: 0 if not stated.

JOB DESCRIPTION:
<<<
{text}
>>>"""
    for attempt in range(2):
        raw = await _complete_json("parse_jd", prompt, JDData, max_tokens=1500)
        jd = JDData.model_validate(raw)
        if jd.required_skills:
            return jd.model_dump()
        log.warning("LLM parse_jd: no skills extracted (attempt %d)", attempt + 1)
    raise ValueError("no skills could be found in this job description; check it lists the role's requirements")


# ---------------------------------------------------------------------------
# Skill gap analysis (LLM matches names; Python does the scoring)
# ---------------------------------------------------------------------------

LEVEL_RANK = {"none": 0, "beginner": 1, "basic": 2, "intermediate": 3, "advanced": 4, "expert": 5}
IMPORTANCE_WEIGHT = {"required": 3, "preferred": 2, "nice_to_have": 1}
_IMPORTANCE_BONUS = {"required": 1, "preferred": -1, "nice_to_have": -2}
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


async def analyze_skill_gap(resume_data: Dict, jd_data: Dict) -> Dict[str, Any]:
    """
    Compare the résumé with the job requirements.

    Returns:
        {overall_fit_score, strong_matches, skill_gaps: [{skill, candidate_level,
         required_level, gap_severity, priority, reason, importance}],
         missing_skills, summary}
    """
    jd_skills = [
        {"name": s["name"], "level": s["level"], "importance": s["importance"]}
        for s in jd_data.get("required_skills", [])
    ]
    resume_skills = [
        {"name": s["name"], "proficiency": s["proficiency"], "years": s.get("years", 0)}
        for s in resume_data.get("skills", [])
    ]
    match_result: Dict[str, Any] = {"matches": [], "summary": ""}
    if jd_skills:
        prompt = f"""Match each job requirement to the candidate's résumé skills.
For each job skill return resume_skill = the résumé skill that is the same or directly demonstrates it (e.g. "Postgres" for "PostgreSQL", "Django" for "Python"), or null if there is no evidence.
reason: max 15 words on how the candidate's evidence compares with what the role needs.
summary: 2 sentences on the candidate's overall fit.
Return: {{"matches":[{{"jd_skill":str,"resume_skill":str|null,"reason":str}}],"summary":str}}

ROLE: {jd_data.get('role_title', '')} ({jd_data.get('required_experience_years', 0)}+ years)
JOB SKILLS: {_compact(jd_skills)}
CANDIDATE: {resume_data.get('total_experience_years', 0)} years; SKILLS: {_compact(resume_skills)}"""
        raw = await _complete_json("match_skills", prompt, MatchResult, max_tokens=60 * len(jd_skills) + 300)
        match_result = MatchResult.model_validate(raw).model_dump()

    return compute_gap_analysis(resume_data, jd_data, match_result)


def compute_gap_analysis(resume_data: Dict, jd_data: Dict, match_result: Dict) -> Dict[str, Any]:
    """Deterministic gap scoring from parsed data plus the LLM's skill-name matches."""
    resume_by_name = {s["name"].strip().lower(): s for s in resume_data.get("skills", [])}
    matches = {
        (m.get("jd_skill") or "").strip().lower(): m for m in match_result.get("matches", [])
    }

    strong_matches, gaps, missing = [], [], []
    weighted_fit = total_weight = 0.0

    for req in jd_data.get("required_skills", []):
        name = req["name"]
        importance = req.get("importance", "required")
        required_level = req.get("level", "intermediate")
        weight = IMPORTANCE_WEIGHT.get(importance, 3)
        match = matches.get(name.strip().lower(), {})

        matched = resume_by_name.get((match.get("resume_skill") or "").strip().lower()) \
            or resume_by_name.get(name.strip().lower())
        candidate_level = matched["proficiency"] if matched else "none"

        req_rank = LEVEL_RANK.get(required_level, 3)
        cand_rank = LEVEL_RANK.get(candidate_level, 0)
        total_weight += weight
        weighted_fit += weight * min(cand_rank / req_rank, 1.0) if req_rank else weight

        diff = req_rank - cand_rank
        if diff <= 0:
            strong_matches.append(name)
            continue

        # e.g. required none->advanced = 5 (critical); required none->intermediate = 4 (high);
        # required basic->intermediate = 2 (medium); nice-to-have none->intermediate = 1 (low)
        severity_score = diff + _IMPORTANCE_BONUS.get(importance, 0)
        severity = ("critical" if severity_score >= 5 else "high" if severity_score >= 3
                    else "medium" if severity_score == 2 else "low")
        reason = (match.get("reason") or "").strip() or (
            f"Résumé shows {candidate_level} level; the role needs {required_level}."
            if matched else f"No evidence of {name} in the résumé."
        )
        gaps.append({
            "skill": name,
            "candidate_level": candidate_level,
            "required_level": required_level,
            "gap_severity": severity,
            "importance": importance,
            "reason": reason,
            "_sort": (SEVERITY_RANK[severity], -weight, -diff),
        })
        if not matched and importance == "required":
            missing.append(name)

    gaps.sort(key=lambda g: g["_sort"])
    for priority, gap in enumerate(gaps, start=1):
        gap["priority"] = priority
        del gap["_sort"]

    skill_fit = weighted_fit / total_weight if total_weight else 0.5
    required_years = float(jd_data.get("required_experience_years") or 0)
    candidate_years = float(resume_data.get("total_experience_years") or 0)
    years_fit = min(candidate_years / required_years, 1.0) if required_years > 0 else 1.0
    fit_score = round(100 * (0.85 * skill_fit + 0.15 * years_fit))

    summary = (match_result.get("summary") or "").strip() or (
        f"The candidate meets {len(strong_matches)} of {len(strong_matches) + len(gaps)} "
        f"requirements, with {len(gaps)} skill gaps to close."
    )
    return {
        "overall_fit_score": fit_score,
        "strong_matches": strong_matches,
        "skill_gaps": gaps,
        "missing_skills": missing,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Question generation and the shared question bank
# ---------------------------------------------------------------------------

QUESTION_LEVELS = ("basic", "intermediate", "advanced")
SEVERITY_WEIGHT = {"critical": 6, "high": 5, "medium": 4, "low": 3}
_LEVEL_MEANING = {
    "basic": "core concepts used day to day",
    "intermediate": "applying them in non-trivial situations",
    "advanced": "trade-offs, edge cases and internals",
    "expert": "deep internals and design decisions at scale",
}


def plan_questionnaire(skills: List[Dict], num_questions: int) -> List[Dict]:
    """
    Decide how many questions each skill gets, and how they split across levels.

    `skills` are gap dicts ({skill, gap_severity, priority...}) in priority order.
    Every assessed skill gets at least 3 questions (one per level); if there are
    too many skills for that, the lowest-priority ones are left out.
    Returns [{skill, gap_severity, counts: {basic, intermediate, advanced}}].
    """
    if not skills or num_questions <= 0:
        return []
    skills = skills[: max(1, num_questions // 3)]
    weights = [SEVERITY_WEIGHT.get(s.get("gap_severity", "medium"), 4) for s in skills]

    totals = [3] * len(skills)
    remaining = num_questions - sum(totals)
    if remaining > 0:
        shares = [remaining * w / sum(weights) for w in weights]
        extra = [int(x) for x in shares]
        # largest remainder, ties broken by priority order
        order = sorted(range(len(skills)), key=lambda i: shares[i] - extra[i], reverse=True)
        for i in order[: remaining - sum(extra)]:
            extra[i] += 1
        totals = [t + e for t, e in zip(totals, extra)]

    plan = []
    for skill, total in zip(skills, totals):
        base, rem = divmod(total, 3)
        counts = {level: base for level in QUESTION_LEVELS}
        for level in ("intermediate", "basic", "advanced")[:rem]:
            counts[level] += 1
        plan.append({"skill": skill["skill"], "gap_severity": skill.get("gap_severity"), "counts": counts})
    return plan


async def _generate_mcqs(skill: str, counts: Dict[str, int], avoid: List[str]) -> List[Tuple[str, Dict]]:
    """One LLM call for one skill. Returns [(level, question dict)] for valid questions."""
    wanted = {lvl: n for lvl, n in counts.items() if n > 0}
    total = sum(wanted.values())
    if total == 0:
        return []
    single_level = len(wanted) == 1
    breakdown = ", ".join(f"{n} {lvl}" for lvl, n in wanted.items())
    levels_text = "; ".join(f"{lvl} = {_LEVEL_MEANING[lvl]}" for lvl in wanted)
    avoid_text = f"\nAlready covered, choose different concepts: {'; '.join(avoid)}" if avoid else ""

    prompt = f"""Write exactly {total} multiple-choice questions on {skill}: {breakdown}.
Levels: {levels_text}.
Rules:
- At least 60% scenario-based: a realistic situation where the reader must choose, debug or decide. Avoid pure definitions.
- Four options A-D, all plausible, exactly one correct.
- explanation: one sentence on why the correct answer is right.
- concept: the sub-topic tested, 2-6 words.{avoid_text}
Return: {{"questions":[{{"level":"basic|intermediate|advanced","question":str,"options":{{"A":str,"B":str,"C":str,"D":str}},"correct_answer":"A|B|C|D","explanation":str,"concept":str}}]}}"""

    raw = await _complete_json(f"mcq:{skill}", prompt, MCQBatch, max_tokens=230 * total + 200)
    items = raw.get("questions", []) if isinstance(raw, dict) else raw if isinstance(raw, list) else []

    valid = []
    for item in items:
        try:
            mcq = GeneratedMCQ.model_validate(item)
        except ValidationError:
            continue
        options = mcq.options.model_dump()
        if not mcq.question.strip() or len({o.strip().lower() for o in options.values()}) < 4:
            continue
        level = next(iter(wanted)) if single_level else mcq.level
        valid.append((level, {
            "question": mcq.question.strip(),
            "options": options,
            "correct_answer": mcq.correct_answer,
            "explanation": mcq.explanation.strip(),
            "concept": mcq.concept.strip(),
        }))
    return valid


async def _questions_for_skill(
    skill: str,
    counts: Dict[str, int],
    exclude_ids: Optional[List[str]] = None,
) -> Dict[str, List[Dict]]:
    """
    Fill `counts` ({level: n}) for one skill: bank first, then generate only the
    shortfall (one retry for anything still missing). New questions are saved
    to the bank. Returns {level: [question dicts with bank_id]}.
    """
    key = skill_key(skill)
    result: Dict[str, List[Dict]] = {}
    seen_norms = set()
    for level, n in counts.items():
        result[level] = database.get_bank_questions(key, level, n, exclude_ids)
        seen_norms.update(_question_norm(q["question"]) for q in result[level])

    for attempt in range(2):
        shortfall = {lvl: n - len(result[lvl]) for lvl, n in counts.items() if n > len(result[lvl])}
        if not shortfall:
            break
        avoid = []
        for lvl in shortfall:
            avoid += database.get_bank_concepts(key, lvl, limit=15)
        generated = await _generate_mcqs(skill, shortfall, avoid[:30])

        by_level: Dict[str, List[Dict]] = {}
        for level, q in generated:
            norm = _question_norm(q["question"])
            if norm in seen_norms:
                continue
            seen_norms.add(norm)
            by_level.setdefault(level, []).append(q)

        for level, qs in by_level.items():
            ids = database.add_bank_questions(key, skill, level, qs, [_question_norm(q["question"]) for q in qs], MODEL)
            for q, bank_id in zip(qs, ids):
                q["bank_id"] = bank_id
            if level in shortfall:
                result[level].extend(qs[: shortfall[level]])
        if attempt == 0 and shortfall:
            log.info("Generated questions for %s (%s)", skill, shortfall)
    return result


def _shuffle_options(q: Dict, include_idk: bool) -> Tuple[Dict[str, str], str]:
    """Randomise option order (fixing the model's bias toward B/C) and remap the answer."""
    letters = ["A", "B", "C", "D"]
    order = letters[:]
    random.shuffle(order)
    options = {new: q["options"][old] for new, old in zip(letters, order)}
    correct = letters[order.index(q["correct_answer"])]
    if include_idk:
        options["E"] = IDK_OPTION
    return options, correct


async def generate_questionnaire(
    skill_gaps: Dict,
    resume_data: Dict,
    jd_data: Dict,
    num_questions: int = 30
) -> List[Dict]:
    """
    Build the assessment for all skill gaps, reusing bank questions where possible.

    Returns list of MCQ dicts:
        [{id, bank_id, skill, level, question, options {A-E}, correct_answer, explanation, concept}]
    """
    skills, seen = [], set()
    for gap in sorted(skill_gaps.get("skill_gaps", []), key=lambda g: g.get("priority", 99)):
        if gap["skill"].lower() not in seen:
            seen.add(gap["skill"].lower())
            skills.append(gap)
    for name in skill_gaps.get("missing_skills", []):  # older analyses listed these separately
        if name.lower() not in seen:
            seen.add(name.lower())
            skills.append({"skill": name, "gap_severity": "high"})

    plan = plan_questionnaire(skills, num_questions)
    results = await asyncio.gather(
        *(_questions_for_skill(p["skill"], p["counts"]) for p in plan),
        return_exceptions=True,
    )

    questions: List[Dict] = []
    for entry, result in zip(plan, results):
        if isinstance(result, Exception):
            log.warning("Question generation failed for %s: %s", entry["skill"], result)
            continue
        for level in QUESTION_LEVELS:
            for q in result.get(level, []):
                options, correct = _shuffle_options(q, include_idk=True)
                questions.append({
                    "id": len(questions) + 1,
                    "bank_id": q.get("bank_id"),
                    "skill": entry["skill"],
                    "level": level,
                    "question": q["question"],
                    "options": options,
                    "correct_answer": correct,
                    "explanation": q.get("explanation", ""),
                    "concept": q.get("concept", ""),
                })

    if not questions:
        errors = [r for r in results if isinstance(r, Exception)]
        raise ValueError(f"Could not generate any questions: {errors[0] if errors else 'no skill gaps'}")
    return questions


async def generate_skill_test(
    skill_name: str,
    target_level: str,
    exclude_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """
    10 verification MCQs at the target level, skipping questions this session has
    already seen (so a retest gets a fresh set).

    Returns [{id, bank_id, question, options {A-D}, correct_answer, explanation, concept_tested}]
    """
    level = target_level if target_level in _LEVEL_MEANING else "intermediate"
    result = await _questions_for_skill(skill_name, {level: SKILL_TEST_SIZE}, exclude_ids)
    pool = result.get(level, [])
    if not pool:
        raise ValueError("The model failed to return any questions.")

    questions = []
    for q in pool[:SKILL_TEST_SIZE]:
        options, correct = _shuffle_options(q, include_idk=False)
        questions.append({
            "id": len(questions) + 1,
            "bank_id": q.get("bank_id"),
            "question": q["question"],
            "options": options,
            "correct_answer": correct,
            "explanation": q.get("explanation", ""),
            "concept_tested": q.get("concept", ""),
        })
    return questions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def calculate_proficiency_scores(
    questions: List[Dict],
    answers: Dict[str, str]
) -> Dict[str, Dict]:
    """
    Calculate per-skill proficiency scores based on questionnaire answers.
    Pure Python calculation - no AI call needed here.

    Args:
        questions: list of MCQ dicts with skill, level, correct_answer
        answers: dict mapping str(question_id) to chosen letter (A/B/C/D)

    Returns:
        {
            "Python": {
                "level": "intermediate",
                "score": 67,
                "details": {
                    "basic_accuracy": 100,
                    "intermediate_accuracy": 50,
                    "advanced_accuracy": 33
                }
            }
        }
    """
    # Group questions by skill and level
    skill_data = {}
    for q in questions:
        skill = q["skill"]
        level = q["level"]
        q_id = str(q["id"])

        if skill not in skill_data:
            skill_data[skill] = {
                "basic": {"correct": 0, "total": 0},
                "intermediate": {"correct": 0, "total": 0},
                "advanced": {"correct": 0, "total": 0}
            }

        skill_data[skill][level]["total"] += 1
        if answers.get(q_id) == q.get("correct_answer"):
            skill_data[skill][level]["correct"] += 1

    # Calculate scores per skill
    proficiency_scores = {}
    for skill, levels in skill_data.items():
        basic = levels["basic"]
        inter = levels["intermediate"]
        adv = levels["advanced"]

        # Calculate accuracy per level (None if no questions at that level)
        def accuracy(data):
            if data["total"] == 0:
                return None
            return round((data["correct"] / data["total"]) * 100)

        basic_acc = accuracy(basic)
        inter_acc = accuracy(inter)
        adv_acc = accuracy(adv)

        # Overall score: weighted average (basic=1x, intermediate=2x, advanced=3x)
        total_weight = 0
        weighted_sum = 0
        if basic_acc is not None:
            weighted_sum += basic_acc * 1
            total_weight += 1
        if inter_acc is not None:
            weighted_sum += inter_acc * 2
            total_weight += 2
        if adv_acc is not None:
            weighted_sum += adv_acc * 3
            total_weight += 3

        overall_score = round(weighted_sum / total_weight) if total_weight > 0 else 0

        # Determine proficiency level from score
        if overall_score >= 85:
            level = "advanced"
        elif overall_score >= 65:
            level = "intermediate"
        elif overall_score >= 40:
            level = "basic"
        else:
            level = "beginner"

        proficiency_scores[skill] = {
            "level": level,
            "score": overall_score,
            "details": {
                "basic_accuracy": basic_acc if basic_acc is not None else 0,
                "intermediate_accuracy": inter_acc if inter_acc is not None else 0,
                "advanced_accuracy": adv_acc if adv_acc is not None else 0,
                "questions_attempted": basic["total"] + inter["total"] + adv["total"],
                "correct_answers": basic["correct"] + inter["correct"] + adv["correct"]
            }
        }

    return proficiency_scores


# ---------------------------------------------------------------------------
# Learning pathway
# ---------------------------------------------------------------------------

_PROVIDER_SITES = {
    "coursera": "coursera.org", "udemy": "udemy.com", "edx": "edx.org",
    "freecodecamp": "freecodecamp.org", "khan academy": "khanacademy.org",
    "pluralsight": "pluralsight.com", "linkedin learning": "linkedin.com/learning",
    "o'reilly": "oreilly.com", "oreilly": "oreilly.com", "manning": "manning.com",
    "mdn": "developer.mozilla.org", "kaggle": "kaggle.com", "leetcode": "leetcode.com",
    "hackerrank": "hackerrank.com", "exercism": "exercism.org", "datacamp": "datacamp.com",
    "aws": "aws.amazon.com", "microsoft learn": "learn.microsoft.com", "google cloud": "cloud.google.com",
}


def resource_url(resource: Dict[str, str]) -> str:
    """A search link for a named resource — always resolves, unlike model-written URLs."""
    title, provider = resource.get("title", ""), resource.get("provider", "")
    query = f"{title} {provider}".strip()
    if resource.get("type") == "video" or "youtube" in provider.lower():
        return "https://www.youtube.com/results?search_query=" + quote_plus(title)
    site = next((domain for name, domain in _PROVIDER_SITES.items() if name in provider.lower()), None)
    if site:
        query = f"{title} site:{site}"
    return "https://www.google.com/search?q=" + quote_plus(query)


async def _skill_plan(skill: str, from_level: str, to_level: str) -> Dict[str, Any]:
    """One skill's learning plan: from the pathway bank, or generated and cached."""
    key = skill_key(skill)
    cached = database.get_pathway_entry(key, from_level, to_level)
    if cached:
        return cached

    start = "no prior knowledge" if from_level in ("none", "", None) else f"{from_level} level"
    # A focused onboarding plan: about 2 weeks per proficiency level to climb
    max_weeks = max(1, 2 * (LEVEL_RANK.get(to_level, 3) - LEVEL_RANK.get(from_level or "none", 0)))
    prompt = f"""Design a focused onboarding plan that takes a working software professional from {start} to {to_level} level in {skill}.
Return: {{"estimated_weeks":number,"prerequisites":[str],"resources":[{{"type":"course|book|docs|video|practice|article","title":str,"provider":str}}],"learning_steps":[{{"week":"1 or 2-3","focus":str,"tasks":[str],"milestone":str}}],"practice_project":"1 sentence"}}
Rules:
- estimated_weeks: at most {max_weeks}; cover what the job needs rather than the whole field.
- prerequisites: other skills to learn first (short names), or [] if none.
- resources: 3 well-known, real resources (title and provider only, no URLs).
- learning_steps: at most 4, covering all the weeks; group weeks (e.g. "3-4"); 2-3 short tasks each."""
    raw = await _complete_json(f"pathway:{skill}", prompt, SkillPlan, max_tokens=1200)
    plan = SkillPlan.model_validate(raw).model_dump()
    plan["estimated_weeks"] = min(max(plan["estimated_weeks"], 1), max_weeks)
    database.add_pathway_entry(key, from_level, to_level, plan, MODEL)
    return plan


def order_pathway(skills: List[Dict]) -> List[Dict]:
    """
    Order skills so that any gap skill listed as another's prerequisite comes
    first; otherwise keep the given (priority) order. Cycles fall back to priority.
    """
    keys = [skill_key(s["skill_name"]) for s in skills]

    def depends_on(skill: Dict) -> set:
        own = skill_key(skill["skill_name"])
        found = set()
        for prereq in skill.get("prerequisites", []):
            pk = skill_key(prereq)
            found.update(k for k in keys if k != own and re.search(rf"(^|\s){re.escape(k)}(\s|$)", pk))
        return found

    deps = {skill_key(s["skill_name"]): depends_on(s) for s in skills}
    ordered, placed, remaining = [], set(), list(skills)
    while remaining:
        ready = next((s for s in remaining if deps[skill_key(s["skill_name"])] <= placed), remaining[0])
        ordered.append(ready)
        placed.add(skill_key(ready["skill_name"]))
        remaining.remove(ready)
    return ordered


async def generate_learning_pathway(
    proficiency_scores: Dict,
    skill_gaps: Dict,
    resume_data: Dict,
    jd_data: Dict
) -> Dict[str, Any]:
    """
    Personalised learning pathway covering every skill gap.

    Each skill's plan is generated in its own request, all in parallel, and cached
    by (skill, current level, target level) so later candidates reuse it. The
    personal parts — levels, priority, the "why" line, order and totals — are
    filled in here per candidate.
    """
    gaps = sorted(skill_gaps.get("skill_gaps", []), key=lambda g: g.get("priority", 99))
    targets = []
    for gap in gaps:
        assessed = proficiency_scores.get(gap["skill"], {})
        targets.append((gap, assessed.get("level") or gap.get("candidate_level") or "none",
                        gap.get("required_level") or "intermediate", assessed.get("score")))

    plans = await asyncio.gather(
        *(_skill_plan(gap["skill"], current, target) for gap, current, target, _ in targets),
        return_exceptions=True,
    )

    skills = []
    for (gap, current, target, score), plan in zip(targets, plans):
        if isinstance(plan, Exception):
            log.warning("Pathway generation failed for %s: %s", gap["skill"], plan)
            continue
        assessed = f"your assessment placed you at {current} ({score}%)" if score is not None \
            else f"your résumé shows {current} level"
        entry = {
            "skill_name": gap["skill"],
            "current_level": current,
            "target_level": target,
            "priority": gap.get("gap_severity", "medium"),
            "reasoning": f"The role needs {target} {gap['skill']}; {assessed}. {gap.get('reason', '')}".strip(),
            **plan,
        }
        entry["resources"] = [dict(r, url=resource_url(r)) for r in plan.get("resources", [])]
        skills.append(entry)

    if not skills:
        errors = [p for p in plans if isinstance(p, Exception)]
        raise ValueError(f"Could not generate a pathway: {errors[0] if errors else 'no skill gaps'}")

    skills = order_pathway(skills)
    total_weeks = round(sum(float(s.get("estimated_weeks") or 0) for s in skills))
    names = [s["skill_name"] for s in skills]
    critical = sum(1 for s in skills if s["priority"] in ("critical", "high"))
    overview = (f"A {total_weeks}-week plan across {len(skills)} skills for the "
                f"{jd_data.get('role_title') or 'target'} role, starting with {', '.join(names[:2])}. "
                f"{critical} of the gaps are high priority, so they come early in the plan.")
    return {
        "total_estimated_weeks": total_weeks,
        "overview": overview,
        "skills": skills,
        "learning_order": names,
    }


# ---------------------------------------------------------------------------
# Skill test evaluation
# ---------------------------------------------------------------------------

def evaluate_skill_test(
    skill_name: str,
    questions: List[Dict],
    answers: Dict[str, str],
    target_level: str
) -> Dict[str, Any]:
    """
    Score a completed skill test and build feedback (no LLM call).

    Returns:
        {score, correct, total, passed, wrong_answers: [...], feedback, areas_to_review}
    """
    correct_count = 0
    wrong_answers = []

    for q in questions:
        q_id = str(q["id"])
        user_answer = answers.get(q_id, "")
        correct = q.get("correct_answer", "")

        if user_answer == correct:
            correct_count += 1
        else:
            wrong_answers.append({
                "question": q["question"],
                "your_answer": f"{user_answer}. {q['options'].get(user_answer, 'No answer')}",
                "correct_answer": f"{correct}. {q['options'].get(correct, '')}",
                "explanation": q.get("explanation", ""),
                "concept": q.get("concept_tested", "")
            })

    total = len(questions)
    score = round((correct_count / total) * 100) if total > 0 else 0
    passed = score >= SKILL_TEST_PASS_MARK

    areas_to_review = list(dict.fromkeys(
        wa["concept"] for wa in wrong_answers if wa.get("concept")
    ))

    return {
        "score": score,
        "correct": correct_count,
        "total": total,
        "passed": passed,
        "wrong_answers": wrong_answers,
        "feedback": _feedback(skill_name, target_level, score, correct_count, total, passed, areas_to_review),
        "areas_to_review": areas_to_review
    }


def _feedback(skill: str, level: str, score: int, correct: int, total: int,
              passed: bool, areas: List[str]) -> str:
    focus = ", ".join(areas[:3])
    if passed:
        text = f"Well done: you scored {score}% ({correct}/{total}) and passed the {level} {skill} check. "
        text += (f"To round things off, revisit {focus}. " if focus
                 else "You answered every question correctly. ")
        return text + "You're ready to move on to the next skill in your pathway."
    text = f"You scored {score}% ({correct}/{total}); the pass mark is {SKILL_TEST_PASS_MARK}%. "
    if focus:
        text += f"Focus on {focus}. "
    return text + ("Read the explanations below, work through this skill's resources in your pathway, "
                   "then retake the test. You'll get a fresh set of questions.")
