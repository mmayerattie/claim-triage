"""LLM client behaviour that does not need a live provider."""
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.config import load_settings
from app.llm import LLMClient, LLMError, Usage

TOOL = {"type": "function", "function": {"name": "noop", "parameters": {
    "type": "object", "properties": {}, "required": [], "additionalProperties": False}}}


def completion():
    message = SimpleNamespace(content="ok", tool_calls=None, model_dump=lambda **_: {"role": "assistant"})
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2,
                            prompt_tokens_details=None, completion_tokens_details=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def error(cls, status, message):
    request = httpx.Request("POST", "https://gateway.test/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=None)


def client_with(responses):
    llm = LLMClient(load_settings())
    sent = []

    def create(**params):
        sent.append(params)
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return llm, sent


def test_strict_schemas_sent_by_default():
    llm, sent = client_with([completion()])
    llm.chat([{"role": "user", "content": "hi"}], Usage(), tools=[TOOL])
    assert sent[0]["tools"][0]["function"]["strict"] is True


def test_strict_rejection_falls_back_once():
    llm, sent = client_with([error(openai.BadRequestError, 400, "strict mode is not supported"), completion()])
    reply = llm.chat([{"role": "user", "content": "hi"}], Usage(), tools=[TOOL])
    assert reply.content == "ok"
    assert "strict" not in sent[1]["tools"][0]["function"]
    assert llm.strict_tools is False  # later calls skip strict entirely


def test_rate_limit_is_not_retried():
    llm, sent = client_with([error(openai.RateLimitError, 429, "budget exhausted")])
    with pytest.raises(LLMError, match="budget"):
        llm.chat([{"role": "user", "content": "hi"}], Usage())
    assert len(sent) == 1


def test_usage_is_accumulated():
    llm, _ = client_with([completion(), completion()])
    usage = Usage()
    llm.chat([], usage)
    llm.chat([], usage)
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 20, 4)
