"""ReAct execution trace: reason -> act -> observe events, exported per claim.

`reason` events hold the agent's one-line action purpose (a field the model fills
in on every tool call), never private model reasoning. Every `act` is paired
with the `observe` that immediately follows it.
"""
import time
from datetime import datetime, timezone


class Trace:
    def __init__(self):
        self.steps: list[dict] = []
        self._started = time.perf_counter()
        self._open_action: str | None = None

    def _add(self, kind: str, content: str, **extra) -> None:
        if self._open_action is not None and kind != "observe":
            raise RuntimeError(f"Action '{self._open_action}' has no observation")
        self.steps.append({
            "step_number": len(self.steps) + 1,
            "type": kind,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            **extra,
        })

    def reason(self, content: str) -> None:
        self._add("reason", content)

    def act(self, tool_name: str, tool_input: dict, content: str) -> None:
        self._add("act", content, tool_name=tool_name, tool_input=tool_input)
        self._open_action = tool_name

    def observe(self, tool_output: dict | list, content: str) -> None:
        if self._open_action is None:
            raise RuntimeError("Observation without a preceding action")
        self._open_action = None
        self._add("observe", content, tool_output=tool_output)

    def export(self, claim_id: str, **extra) -> dict:
        return {
            "claim_id": claim_id,
            "steps": self.steps,
            "total_duration_ms": int((time.perf_counter() - self._started) * 1000),
            **extra,
        }
