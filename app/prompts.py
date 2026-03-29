"""
Prompt engineering for the Synergex Med QA system.

Design goals:
1. Avoid false positives — only flag what is genuinely visible in the transcript.
2. Separate factual observation from AI assessment.
3. Apply department-specific rules where relevant.
4. Return deterministic JSON that Pydantic can parse.
"""

from __future__ import annotations

from app.schema import CallTranscript

# ─── Department-specific guidance ─────────────────────────────────────────────

DEPARTMENT_RULES: dict[str, str] = {
    "Scheduling": (
        "For Scheduling calls, verify: "
        "(a) the agent confirmed appointment date, time, and location with the caller; "
        "(b) the agent asked for or confirmed the patient's date of birth or other identifier to locate the correct record; "
        "(c) if an appointment was changed or cancelled, the agent confirmed the new details clearly."
    ),
    "Onboarding": (
        "For Onboarding calls, verify: "
        "(a) the agent discussed or referenced the lien agreement where applicable; "
        "(b) the agent explained next steps clearly; "
        "(c) the agent captured all required intake information."
    ),
    "Records": (
        "For Records calls, verify: "
        "(a) the agent verified patient identity before discussing or releasing any records (HIPAA); "
        "(b) the agent did not disclose PHI to unauthorised parties; "
        "(c) record request timelines were communicated accurately."
    ),
    "Helpdesk": (
        "For Helpdesk calls, verify: "
        "(a) the agent understood and accurately restated the caller's issue; "
        "(b) the agent provided correct information or escalated appropriately; "
        "(c) the caller's issue was resolved or a clear follow-up path was established."
    ),
    "Follow-Ups": (
        "For Follow-Up calls, verify: "
        "(a) the agent confirmed the reason for the follow-up and referenced the correct case; "
        "(b) the agent provided accurate status updates; "
        "(c) the agent documented next steps or commitments."
    ),
}

DEFAULT_DEPT_RULE = (
    "Apply general quality standards: identity verification, accuracy, professionalism, and resolution."
)


def get_department_guidance(department: str) -> str:
    return DEPARTMENT_RULES.get(department, DEFAULT_DEPT_RULE)


# ─── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior clinical quality assurance specialist at Synergex Med, a pain management and neurology clinic operating 39+ locations across the US. You are reviewing phone call transcripts made by virtual assistants who interact with patients, law offices, and insurance companies.

Your job is to produce a structured, fair, evidence-based quality analysis. The goal of this QA system is NOT punitive scoring — it is to detect genuine problems (HIPAA violations, dangerous misinformation, clear rudeness) and recognise good work. Staff morale depends on you being accurate and fair.

CRITICAL RULES YOU MUST FOLLOW:
1. DO NOT invent issues that are not present in the transcript. Only flag what you can directly observe.
2. If something is ambiguous, note the ambiguity — do not assume the worst.
3. Reserve "escalate" only for: confirmed HIPAA violations, clear rudeness/hostility, or dangerous medical misinformation. Minor mistakes or awkward phrasing are NOT grounds for escalation.
4. Separate what you observed (transcript evidence) from your assessment (your interpretation).
5. Short calls, calls with few turns, or routine calls with no issues should be assessed honestly — not inflated.

SEVERITY GUIDE:
- critical: HIPAA violation, dangerous misinformation, explicit rudeness or aggression
- moderate: Incorrect information that could cause scheduling or administrative problems; unprofessional but not hostile language
- minor: Small process deviation, slight awkwardness, minor omission that had no impact
- positive: Genuine good practice worth noting

OUTPUT REQUIREMENTS:
You must respond with ONLY valid JSON, no markdown, no code fences, no commentary — just the raw JSON object.

Return exactly this structure:
{
  "overall_assessment": "pass" | "needs_review" | "escalate",
  "assessment_reasoning": "<2–4 sentences>",
  "compliance_flags": [
    {
      "type": "hipaa_concern" | "misinformation" | "rudeness" | "protocol_violation" | "positive_interaction",
      "severity": "critical" | "moderate" | "minor" | "positive",
      "description": "<1–2 sentence description>",
      "transcript_excerpt": "<exact excerpt from the transcript>"
    }
  ],
  "agent_performance": {
    "professionalism_score": <float 0-1>,
    "accuracy_score": <float 0-1>,
    "resolution_score": <float 0-1>,
    "strengths": ["<strength 1>", "<strength 2>"],
    "improvements": ["<improvement 1>"]
  },
  "escalation_required": true | false,
  "escalation_reason": "<string or null>"
}
"""


def build_user_prompt(call: CallTranscript) -> str:
    dept_guidance = get_department_guidance(call.department)
    duration_note = _duration_note(call.call_duration_seconds)

    return f"""CALL METADATA
Call ID: {call.call_id}
Agent: {call.agent_name}
Date: {call.call_date}
Duration: {call.call_duration_seconds} seconds{duration_note}
Department: {call.department}

DEPARTMENT-SPECIFIC RULES
{dept_guidance}

TRANSCRIPT
{call.transcript.strip()}

TASK
Analyse this transcript using the criteria above. Remember: only flag genuine issues. Be fair and evidence-based. Return only the JSON object as specified."""


def _duration_note(seconds: int) -> str:
    if seconds < 30:
        return " (very short call — limited information available; keep assessments appropriately tentative)"
    if seconds < 90:
        return " (short call — assess only what is observable)"
    return ""
