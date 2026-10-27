"""Scripted stand-in for LLMClient: returns queued replies and records requests."""
import io

from PIL import Image

from app.llm import LLMReply, ToolCall


class FakeLLM:
    image_detail = "low"

    def __init__(self, replies: list[LLMReply] | None = None):
        self.replies = list(replies or [])
        self.requests: list[dict] = []

    def queue(self, *replies: LLMReply) -> None:
        self.replies.extend(replies)

    def chat(self, messages, usage, tools=None, force_tool=None, attempts=2):
        self.requests.append({"messages": list(messages), "tools": tools, "force_tool": force_tool})
        if not self.replies:
            raise AssertionError("FakeLLM received an unexpected call")
        return self.replies.pop(0)


def tool_reply(name: str, arguments: dict, call_id: str = "call_1") -> LLMReply:
    return LLMReply(content="", tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
                    raw_message={"role": "assistant", "content": None, "tool_calls": [
                        {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}]})


def intake_reply(**overrides) -> LLMReply:
    verdict = {"is_property_damage": True, "is_valid_image": True, "no_pii_detected": True,
               "no_offensive_content": True, "reason": "Clear photo of property damage."}
    verdict.update(overrides)
    return tool_reply("submit_intake_validation", verdict)


def jpeg_bytes(color=(120, 90, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, format="JPEG")
    return buffer.getvalue()
