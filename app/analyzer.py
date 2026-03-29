"""
Core analysis pipeline:
  1. Build prompt
  2. Call LLM via provider abstraction (with retries)
  3. Parse JSON response into QAAnalysis Pydantic model
  4. Log prompt, response, latency, token usage
  5. Validate and enforce business rules (e.g., escalation consistency)
"""


import json
import logging

from pydantic import ValidationError

from app.llm_client import llm_complete
from app.logger import get_logger
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.schema import CallTranscript, OverallAssessment, QAAnalysis

logger = get_logger(__name__)


async def analyze_transcript(call: CallTranscript) -> QAAnalysis:
    user_prompt = build_user_prompt(call)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("LLM user prompt call_id=%s\n%s", call.call_id, user_prompt)

    raw_text, usage = await llm_complete(SYSTEM_PROMPT, user_prompt)

    logger.info(
        "LLM response received",
        extra={
            "call_id": call.call_id,
            "provider": usage.get("provider"),
            "model": usage.get("model"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency_seconds": usage.get("latency_seconds"),
        },
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("LLM raw response call_id=%s\n%s", call.call_id, raw_text)

    parsed = _parse_llm_response(raw_text, call.call_id)
    result = _enforce_business_rules(parsed, call)
    return result


def _parse_llm_response(raw: str, call_id: str) -> QAAnalysis:
    """
    Parse the LLM response into a QAAnalysis model.

    Handles:
    - Raw JSON (ideal)
    - JSON wrapped in markdown code fences (model sometimes adds these despite instructions)
    - Partial / malformed JSON → raises with clear message
    """
    text = raw.strip()

    # Strip optional markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence (```json or ```)
        lines = lines[1:] if lines[0].startswith("```") else lines
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-JSON response for call_id={call_id}: {exc}\nRaw (first 400 chars): {raw[:400]}"
        ) from exc

    # Inject call_id so Pydantic validation passes (LLM does not produce it)
    data["call_id"] = call_id

    try:
        return QAAnalysis(**data)
    except ValidationError as exc:
        raise ValueError(
            f"LLM JSON did not match expected schema for call_id={call_id}: {exc}"
        ) from exc


def _enforce_business_rules(analysis: QAAnalysis, call: CallTranscript) -> QAAnalysis:
    """
    Post-processing to enforce consistency rules:
    - If any compliance flag has severity=critical, escalation_required must be True.
    - If overall_assessment is 'escalate', escalation_required must be True.
    - If escalation_required is True, escalation_reason must not be None.
    """
    has_critical = any(f.severity.value == "critical" for f in analysis.compliance_flags)

    escalation_required = analysis.escalation_required or has_critical or (
        analysis.overall_assessment == OverallAssessment.ESCALATE
    )

    # Sync overall_assessment with escalation
    overall = analysis.overall_assessment
    if escalation_required and overall != OverallAssessment.ESCALATE:
        overall = OverallAssessment.ESCALATE

    escalation_reason = analysis.escalation_reason
    if escalation_required and not escalation_reason:
        critical_flags = [f for f in analysis.compliance_flags if f.severity.value == "critical"]
        if critical_flags:
            escalation_reason = f"Critical flag detected: {critical_flags[0].description}"
        else:
            escalation_reason = "Critical issue detected — manual review required."

    return analysis.model_copy(
        update={
            "escalation_required": escalation_required,
            "overall_assessment": overall,
            "escalation_reason": escalation_reason if escalation_required else None,
        }
    )
