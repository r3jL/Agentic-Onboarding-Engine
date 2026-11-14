"""
schemas.py - Pydantic models for every structured LLM response.

They serve two purposes:
  1. Their JSON schema is sent as `response_format` so providers that support
     structured output return valid JSON of the right shape.
  2. Responses are validated with them. Validators are lenient: enum-like fields
     are lower-cased and unknown values fall back to a sensible default, so a
     slightly-off model answer is still usable.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field
from typing_extensions import Annotated


def _choice(allowed: tuple, default: str):
    """Literal type that normalises case/whitespace and falls back to `default`."""

    def normalise(value: Any) -> str:
        text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        return text if text in allowed else default

    return Annotated[Literal[allowed], BeforeValidator(normalise)]


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


Number = Annotated[float, BeforeValidator(_number)]
# Text that tolerates null and numbers (e.g. "week": 3)
Text = Annotated[str, BeforeValidator(lambda v: "" if v is None else str(v))]
Proficiency = _choice(("beginner", "basic", "intermediate", "advanced", "expert"), "intermediate")
RequiredLevel = _choice(("basic", "intermediate", "advanced", "expert"), "intermediate")
Importance = _choice(("required", "preferred", "nice_to_have"), "required")
QuestionLevel = _choice(("basic", "intermediate", "advanced"), "intermediate")
ResourceType = _choice(("course", "book", "docs", "video", "practice", "article"), "course")
AnswerLetter = Annotated[
    Literal["A", "B", "C", "D"],
    BeforeValidator(lambda v: str(v or "").strip().upper()[:1]),
]


# ---------------------------------------------------------------------------
# Resume / job description
# ---------------------------------------------------------------------------

class ResumeSkill(BaseModel):
    name: str
    proficiency: Proficiency = "intermediate"
    years: Number = 0


class WorkExperience(BaseModel):
    company: Text = ""
    title: Text = ""
    duration: Text = ""


class Education(BaseModel):
    degree: Text = ""
    institution: Text = ""
    year: Text = ""


class ResumeData(BaseModel):
    is_resume: bool
    not_resume_reason: Text = ""
    candidate_name: Text = ""
    skills: List[ResumeSkill] = Field(default_factory=list)
    work_experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    total_experience_years: Number = 0


class JDSkill(BaseModel):
    name: str
    level: RequiredLevel = "intermediate"
    importance: Importance = "required"


class JDData(BaseModel):
    role_title: Text = ""
    required_skills: List[JDSkill] = Field(default_factory=list)
    responsibilities: List[Text] = Field(default_factory=list)
    required_experience_years: Number = 0


# ---------------------------------------------------------------------------
# Skill matching (the only LLM part of gap analysis)
# ---------------------------------------------------------------------------

class SkillMatch(BaseModel):
    jd_skill: str
    resume_skill: Optional[str] = None
    reason: Text = ""


class MatchResult(BaseModel):
    matches: List[SkillMatch] = Field(default_factory=list)
    summary: Text = ""


# ---------------------------------------------------------------------------
# Multiple-choice questions
# ---------------------------------------------------------------------------

class MCQOptions(BaseModel):
    A: str
    B: str
    C: str
    D: str


class GeneratedMCQ(BaseModel):
    level: QuestionLevel = "intermediate"
    question: str
    options: MCQOptions
    correct_answer: AnswerLetter
    explanation: Text = ""
    concept: Text = ""


class MCQBatch(BaseModel):
    questions: List[GeneratedMCQ] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Learning pathway
# ---------------------------------------------------------------------------

class PathwayResource(BaseModel):
    type: ResourceType = "course"
    title: str
    provider: Text = ""


class LearningStep(BaseModel):
    week: Text
    focus: Text = ""
    tasks: List[Text] = Field(default_factory=list)
    milestone: Text = ""


class PathwaySkill(BaseModel):
    skill_name: str
    current_level: Text = ""
    target_level: RequiredLevel = "intermediate"
    priority: _choice(("critical", "high", "medium", "low"), "medium") = "medium"
    estimated_weeks: Number = 1
    reasoning: Text = ""
    prerequisites: List[Text] = Field(default_factory=list)
    resources: List[PathwayResource] = Field(default_factory=list)
    learning_steps: List[LearningStep] = Field(default_factory=list)
    practice_project: Text = ""


class Pathway(BaseModel):
    total_estimated_weeks: Number = 0
    overview: Text = ""
    skills: List[PathwaySkill] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON-schema helper
# ---------------------------------------------------------------------------

def response_format(model: type) -> Dict[str, Any]:
    """
    OpenAI-style `response_format` for a model, with `$ref`s inlined and titles
    removed. Some providers don't resolve `$defs`, and titles only add tokens.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            # drop "title" annotations (strings), but keep properties named "title" (dicts)
            return {k: resolve(v) for k, v in node.items() if not (k == "title" and isinstance(v, str))}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return {
        "type": "json_schema",
        "json_schema": {"name": model.__name__, "schema": resolve(schema), "strict": False},
    }
