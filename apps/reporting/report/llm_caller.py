"""
Thin litellm wrapper with per-call token and latency tracking.

All LLM calls in the report pipeline go through this module.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import litellm

from apps.reporting.report.models import LLMResult
from utils.logger import setup_logger

logger = setup_logger("report.llm_caller")

litellm.suppress_debug_info = True


async def drain_litellm_async_logging() -> None:
    """Drain LiteLLM's background logging worker before the event loop closes.

    LiteLLM schedules ``asyncio.create_task(_client_async_logging_helper(...))``
    after each ``acompletion``; that helper enqueues ``async_success_handler``
    coroutines on ``GLOBAL_LOGGING_WORKER``. If the loop shuts down before those
    tasks run or before the worker finishes, Python may emit
    ``RuntimeWarning: coroutine '...async_success_handler' was never awaited``.
    """
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    except ImportError:
        return

    # Yield repeatedly so pending create_task(helper) callbacks run and enqueue.
    for _ in range(64):
        await asyncio.sleep(0)
    await GLOBAL_LOGGING_WORKER.flush()
    await GLOBAL_LOGGING_WORKER.clear_queue()

# LiteLLM requires "provider/model". Bare names (e.g. minimax-m2.5) raise BadRequestError.
# OpenAI-compatible gateways (e.g. SJTU /v1) use openai/<id>. Native MiniMax API uses minimax/MiniMax-*.
_BARE_MODEL_ALIASES = {
    "minimax-m2.5": "openai/minimax-m2.5",
    "minimax-m2": "openai/minimax-m2",
    "minimax-m2.1": "openai/minimax-m2.1",
    "qwen3coder": "openai/qwen3coder",
}


def normalize_litellm_model(model: str) -> str:
    """Map common bare model ids to provider/model form for litellm."""
    if not model or "/" in model:
        return model
    key = model.strip().lower()
    resolved = _BARE_MODEL_ALIASES.get(key)
    if resolved:
        logger.debug(
            "Normalized model id: %r -> %r (litellm requires provider/model)",
            model,
            resolved,
        )
        return resolved
    return model


class LLMCaller:
    """Stateless LLM caller backed by litellm."""

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
    ):
        self.api_key = api_key
        self.api_base = api_base or None

    async def call(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> LLMResult:
        """Make a single chat completion call and track usage.

        Args:
            model: litellm model_id (e.g. "deepseek/deepseek-chat")
            system_prompt: system message
            user_prompt: user message
            temperature: sampling temperature
            max_tokens: max output tokens
            response_format: optional (e.g. {"type": "json_object"})

        Returns:
            LLMResult with content + usage stats
        """
        model = normalize_litellm_model(model)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": self.api_key or None,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if response_format:
            kwargs["response_format"] = response_format

        sys_chars = len(system_prompt)
        usr_chars = len(user_prompt)
        logger.debug(
            "LLM request: model=%s temp=%s max_tokens=%s json=%s "
            "system_chars=%d user_chars=%d api_base=%s",
            model,
            temperature,
            max_tokens,
            bool(response_format),
            sys_chars,
            usr_chars,
            self.api_base or "(default)",
        )

        t0 = time.monotonic()
        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as e:
            latency = int((time.monotonic() - t0) * 1000)
            logger.error(f"LLM call failed ({model}, {latency}ms): {e}")
            return LLMResult(content="", model=model, latency_ms=latency)

        latency = int((time.monotonic() - t0) * 1000)

        # Let LiteLLM's post-call logging task enqueue before concurrent callers continue.
        await asyncio.sleep(0)

        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        # Per-call token/latency at INFO; do not log prompt/response body here.
        logger.info(
            "LLM call: model=%s in=%d out=%d latency=%dms",
            model,
            input_tokens,
            output_tokens,
            latency,
        )

        return LLMResult(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
        )
