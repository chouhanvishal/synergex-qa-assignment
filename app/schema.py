from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


def _sentence_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    chunks = re.split(r"(?<=[.!?])\s+", text)
    return len([c for c in chunks if c.strip()])


class CallTranscript(BaseModel):
    call_id: str = Field(..., description="Unique identifier for the call")
    agent_name: str = Field(..., description="Name of the virtual assistant")
    call_date: str = Field(..., description="Date of the call (YYYY-MM-DD)")
    call_duration_seconds: int = Field(..., ge=0, description="Duration of the call in seconds")
    department: str = Field(
        ...,
        description="Department handling the call (e.g. Scheduling, Onboarding, Helpdesk, Follow-Ups, Records)",
    )
    transcript: str = Field(..., description="Multi-turn conversation formatted as 'Agent: ...' / 'Caller: ...' lines")

    @field_validator("transcript")
    @classmethod
    def transcript_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("transcript must not be empty")
        return v

    @field_validator("call_date")
    @classmethod
    def call_date_yyyy_mm_dd(cls, v: str) -> str:
        try:
            datetime.strptime(v.strip(), "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("call_date must be a valid date in YYYY-MM-DD format") from exc
        return v.strip()


class OverallAssessment(str, Enum):
    PASS = "pass"
    NEEDS_REVIEW = "needs_review"
    ESCALATE = "escalate"


class FlagType(str, Enum):
    HIPAA_CONCERN = "hipaa_concern"
    MISINFORMATION = "misinformation"
    RUDENESS = "rudeness"
    PROTOCOL_VIOLATION = "protocol_violation"
    POSITIVE_INTERACTION = "positive_interaction"


class Severity(str, Enum):
    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"
    POSITIVE = "positive"


class ComplianceFlag(BaseModel):
    type: FlagType
    severity: Severity
    description: str = Field(..., description="1–2 sentence description of the specific issue or positive behaviour")
    transcript_excerpt: str = Field(..., description="The relevant portion of the transcript")


class AgentPerformance(BaseModel):
    professionalism_score: float = Field(..., ge=0.0, le=1.0, description="Tone, language, empathy (0–1)")
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Correctness of information provided (0–1)")
    resolution_score: float = Field(..., ge=0.0, le=1.0, description="Whether caller's issue was addressed (0–1)")
    strengths: List[str] = Field(..., min_length=1, max_length=3, description="1–3 specific things the agent did well")
    improvements: List[str] = Field(
        ..., min_length=1, max_length=3, description="1–3 specific areas for improvement"
    )


class QAAnalysis(BaseModel):
    call_id: str
    overall_assessment: OverallAssessment
    assessment_reasoning: str = Field(..., description="2–4 sentences explaining the overall assessment")
    compliance_flags: List[ComplianceFlag] = Field(default_factory=list)
    agent_performance: AgentPerformance
    escalation_required: bool
    escalation_reason: Optional[str] = None

    @field_validator("assessment_reasoning")
    @classmethod
    def assessment_reasoning_sentence_count(cls, v: str) -> str:
        n = _sentence_count(v)
        if not 2 <= n <= 4:
            raise ValueError(
                f"assessment_reasoning must contain 2–4 sentences (found {n}); split ideas with clear sentence boundaries."
            )
        return v


class BatchRequest(BaseModel):
    calls: List[CallTranscript] = Field(..., min_length=1)


class BatchResponse(BaseModel):
    results: List[QAAnalysis]
    errors: List[dict] = Field(default_factory=list)
