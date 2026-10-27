"""Thin wrapper over the OpenAI-compatible Portkey gateway.

Model-agnostic by design: the model comes from LLM_MODEL and only widely
supported features are used (chat messages, image parts, tool calls).
Structured outputs are requested through a forced tool call rather than a
provider-specific response_format.
"""
import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field

import openai
from PIL import Image

from .config import Settings

log = logging.getLogger("claim-triage.llm")

MAX_IMAGE_SIDE = 1600  # downscale larger photos; detail beyond this costs tokens without helping triage


class LLMError(RuntimeError):
    """Provider failure after retries; surfaced to the client as 502."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, usage) -> None:
        if usage is None:
            return
        self.calls += 1
        self.input_tokens += usage.prompt_tokens or 0
        self.output_tokens += usage.completion_tokens or 0
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        self.cached_tokens += getattr(prompt_details, "cached_tokens", 0) or 0
        self.reasoning_tokens += getattr(completion_details, "reasoning_tokens", 0) or 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict = field(default_factory=dict)  # assistant message to append to the history


def image_part(data: bytes, detail: str) -> dict:
    """Encode an uploaded photo as an image content part, downscaling very large images."""
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        if max(image.size) > MAX_IMAGE_SIDE:
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": detail}}


class LLMClient:
    def __init__(self, settings: Settings):
        self.model = settings.llm_model
        self.reasoning_effort = settings.llm_reasoning_effort
        self.image_detail = settings.llm_image_detail
        self.max_output_tokens = settings.llm_max_output_tokens
        model_id = self.model.rsplit("/", 1)[-1].lower()
        # Reasoning models take max_completion_tokens; others take max_tokens.
        self.reasoning_model = model_id.startswith(("gpt-5", "o1", "o3", "o4"))
        # Strict schemas make the provider enforce enums and required fields while generating.
        # Disabled automatically if the configured model rejects them.
        self.strict_tools = settings.llm_strict_tools
        # SDK retries are disabled: it would also retry 429s, which here means an exhausted budget.
        self.client = openai.OpenAI(
            base_url=settings.portkey_base_url,
            api_key=settings.portkey_api_key,
            default_headers={"x-portkey-api-key": settings.portkey_api_key},
            max_retries=0,
            timeout=settings.llm_timeout_seconds,
        )

    def _with_strict(self, tool: dict) -> dict:
        function = {k: v for k, v in tool["function"].items() if k != "strict"}
        if self.strict_tools:
            function["strict"] = True
        return {"type": "function", "function": function}

    def chat(self, messages: list[dict], usage: Usage, tools: list[dict] | None = None,
             force_tool: str | None = None, attempts: int = 2) -> LLMReply:
        params: dict = {"model": self.model, "messages": messages}
        if self.reasoning_model:
            params["max_completion_tokens"] = self.max_output_tokens
            if self.reasoning_effort:
                params["reasoning_effort"] = self.reasoning_effort
        else:
            params["max_tokens"] = self.max_output_tokens
        if tools:
            params["tools"] = [self._with_strict(tool) for tool in tools]
            params["tool_choice"] = ({"type": "function", "function": {"name": force_tool}}
                                     if force_tool else "auto")

        for attempt in range(1, attempts + 2):  # +1 allows the one strict-schema fallback
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**params)
                break
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
                log.warning("LLM transient error (attempt %d/%d): %s", attempt, attempts, type(exc).__name__)
                if attempt >= attempts:
                    raise LLMError(f"Model provider unavailable: {type(exc).__name__}") from exc
                time.sleep(2 * attempt)
            except openai.RateLimitError as exc:
                raise LLMError("Model provider rate limit or budget exhausted; not retrying") from exc
            except openai.BadRequestError as exc:
                if tools and self.strict_tools and "strict" in str(exc.message).lower():
                    log.warning("Model rejected strict tool schemas; continuing without them")
                    self.strict_tools = False
                    params["tools"] = [self._with_strict(tool) for tool in tools]
                    continue
                log.error("LLM request rejected (HTTP 400): %s", str(exc.message)[:500])
                raise LLMError("Model provider rejected the request (HTTP 400)") from exc
            except openai.APIStatusError as exc:
                # The provider's message explains the rejection; it never contains our credentials.
                log.error("LLM request rejected (HTTP %s): %s", exc.status_code, str(exc.message)[:500])
                raise LLMError(f"Model provider rejected the request (HTTP {exc.status_code})") from exc

        usage.add(response.usage)
        message = response.choices[0].message
        log.info("LLM call %.1fs in=%s out=%s", time.perf_counter() - started,
                 getattr(response.usage, "prompt_tokens", "?"), getattr(response.usage, "completion_tokens", "?"))
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_invalid_json": call.function.arguments}
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        return LLMReply(content=message.content or "", tool_calls=calls,
                        raw_message=message.model_dump(exclude_none=True))
