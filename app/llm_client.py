import asyncio
import time
from typing import Any

from app.config import Settings, get_settings
from app.logger import get_logger

logger = get_logger(__name__)


async def _call_openai(
    system: str, user: str, response_format: dict | None, settings: Settings
) -> tuple[str, dict]:
    import openai

    model = settings.openai_model
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    if response_format:
        kwargs["response_format"] = response_format

    t0 = time.monotonic()
    response = await client.chat.completions.create(**kwargs)
    latency = time.monotonic() - t0

    content = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
        "latency_seconds": round(latency, 3),
        "model": model,
        "provider": "openai",
    }
    return content, usage


async def _call_anthropic(system: str, user: str, settings: Settings) -> tuple[str, dict]:
    import anthropic

    model = settings.anthropic_model
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    t0 = time.monotonic()
    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0,
    )
    latency = time.monotonic() - t0

    content = response.content[0].text if response.content else ""
    usage = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        "latency_seconds": round(latency, 3),
        "model": model,
        "provider": "anthropic",
    }
    return content, usage


async def llm_complete(system: str, user: str) -> tuple[str, dict]:
    settings = get_settings()
    provider = settings.llm_provider.lower()
    last_exc: Exception | None = None

    for attempt in range(1, settings.llm_max_retries + 1):
        try:
            if provider == "openai":
                content, usage = await _call_openai(
                    system, user, response_format={"type": "json_object"}, settings=settings
                )
            elif provider == "anthropic":
                content, usage = await _call_anthropic(system, user, settings)
            else:
                raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'openai' or 'anthropic'.")

            logger.info(
                "LLM call succeeded",
                extra={
                    "attempt": attempt,
                    "provider": usage["provider"],
                    "model": usage["model"],
                    "total_tokens": usage["total_tokens"],
                    "latency_seconds": usage["latency_seconds"],
                },
            )
            return content, usage

        except Exception as exc:
            last_exc = exc
            delay = settings.llm_retry_base_delay * (2 ** (attempt - 1))
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                settings.llm_max_retries,
                exc,
                delay,
            )
            if attempt < settings.llm_max_retries:
                await asyncio.sleep(delay)

    raise RuntimeError(
        f"LLM call failed after {settings.llm_max_retries} attempts: {last_exc}"
    ) from last_exc
