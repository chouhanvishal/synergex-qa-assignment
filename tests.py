#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import httpx

from app.logger import get_logger
from app.schema import QAAnalysis

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent
SAMPLES_DIR = BASE_DIR / "sample_transcripts"

EXPECTED: dict[str, dict] = {
    "CALL-001": {
        "overall_assessment": "pass",
        "escalation_required": False,
        "description": "Clean scheduling call — should pass with no critical flags",
    },
    "CALL-002": {
        "overall_assessment": "escalate",
        "escalation_required": True,
        "description": "HIPAA violation (disclosing PHI without verifying release) + rudeness",
    },
    "CALL-003": {
        "overall_assessment": "pass",
        "escalation_required": False,
        "description": "Very short transfer call — should pass; no issues observable",
    },
}


def load_samples() -> list[dict]:
    samples = []
    for path in sorted(SAMPLES_DIR.glob("*.json")):
        with open(path) as f:
            samples.append(json.load(f))
    return samples


def run_eval(base_url: str) -> None:
    samples = load_samples()
    logger.info("Loaded %d sample transcript(s) from %s", len(samples), SAMPLES_DIR)

    passed = 0
    failed = 0

    with httpx.Client(base_url=base_url, timeout=60) as client:
        for sample in samples:
            call_id = sample["call_id"]
            logger.info("%s", "=" * 60)
            logger.info("Testing call_id: %s  (%s)", call_id, sample["department"])

            resp = client.post("/analyze-call", json=sample)

            if resp.status_code != 200:
                logger.error("HTTP %s: %s", resp.status_code, resp.text[:300])
                failed += 1
                continue

            try:
                analysis = QAAnalysis(**resp.json())
            except Exception as exc:
                logger.error("Schema validation failed: %s", exc)
                failed += 1
                continue

            logger.info("Schema valid")
            logger.info("Overall assessment : %s", analysis.overall_assessment.value)
            logger.info("Escalation required: %s", analysis.escalation_required)
            logger.info("Compliance flags   : %d", len(analysis.compliance_flags))
            for flag in analysis.compliance_flags:
                marker = (
                    "🚨"
                    if flag.severity.value == "critical"
                    else "⚠️"
                    if flag.severity.value in ("moderate",)
                    else "✨"
                    if flag.severity.value == "positive"
                    else "ℹ️"
                )
                logger.info(
                    "    %s [%s] %s: %s",
                    marker,
                    flag.severity.value,
                    flag.type.value,
                    flag.description[:80],
                )
            logger.info(
                "Scores: professionalism=%.2f  accuracy=%.2f  resolution=%.2f",
                analysis.agent_performance.professionalism_score,
                analysis.agent_performance.accuracy_score,
                analysis.agent_performance.resolution_score,
            )

            if call_id in EXPECTED:
                exp = EXPECTED[call_id]
                errors = []
                if analysis.overall_assessment.value != exp["overall_assessment"]:
                    errors.append(
                        f"expected overall_assessment={exp['overall_assessment']!r}, got {analysis.overall_assessment.value!r}"
                    )
                if analysis.escalation_required != exp["escalation_required"]:
                    errors.append(
                        f"expected escalation_required={exp['escalation_required']}, got {analysis.escalation_required}"
                    )
                if errors:
                    for e in errors:
                        logger.error("Assertion failed: %s", e)
                    failed += 1
                else:
                    logger.info("All assertions passed (%s)", exp["description"])
                    passed += 1
            else:
                logger.info(
                    "No expected outcome defined for %s — schema check only", call_id
                )
                passed += 1

    logger.info("%s", "=" * 60)
    logger.info(
        "Results: %d passed, %d failed out of %d calls", passed, failed, len(samples)
    )
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sample transcripts against the Synergex Med QA API")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base URL of the running API"
    )
    args = parser.parse_args()
    run_eval(args.url)
