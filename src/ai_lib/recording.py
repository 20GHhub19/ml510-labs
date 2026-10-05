"""Record local execution evidence without client objects or credentials."""
from datetime import datetime, timezone
import os
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler
from ai_lib.models import KEY_NAMES, api_error_guidance


class LocalTrace(BaseCallbackHandler):
    def __init__(self, run):
        self.run = run
        self.events = []
        self.started = {}

    def _safe(self, value):
        if isinstance(value, dict):
            return {str(k): self._safe(v) for k, v in value.items()
                    if not any(word in str(k).lower() for word in ("api_key", "authorization", "secret", "password"))}
        if isinstance(value, (list, tuple)):
            return [self._safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str):
                for name in [*KEY_NAMES.values(), "TAVILY_API_KEY"]:
                    key = os.getenv(name)
                    if key:
                        value = value.replace(key, "[REDACTED]")
            return value
        return f"<{type(value).__name__}>"

    def event(self, kind, **details):
        self.events.append({"kind": kind, "at_utc": datetime.now(timezone.utc).isoformat(),
                            **self._safe(details)})
        self.run.record("trace", {"execution": "live", "events": self.events})

    @staticmethod
    def message(message):
        return {"type": message.type, "content": message.content,
                "tool_calls": getattr(message, "tool_calls", []),
                "tool_call_id": getattr(message, "tool_call_id", None),
                "usage": getattr(message, "usage_metadata", None),
                "completion": {key: value for key, value in getattr(message, "response_metadata", {}).items()
                               if key in {"finish_reason", "stop_reason", "model_name", "model", "model_provider"}}}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.started[str(run_id)] = perf_counter()
        self.event("model_input", run_id=str(run_id), messages=[[self.message(m) for m in batch] for batch in messages])

    def on_llm_end(self, response, *, run_id, **kwargs):
        elapsed = perf_counter() - self.started.pop(str(run_id), perf_counter())
        outputs = []
        for batch in response.generations:
            for generation in batch:
                outputs.append(self.message(generation.message))
        self.event("model_output", run_id=str(run_id), seconds=elapsed, responses=outputs)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.event("model_error", run_id=str(run_id), guidance=api_error_guidance(error))

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.started[str(run_id)] = perf_counter()
        self.event("tool_input", run_id=str(run_id), name=serialized.get("name"), arguments=input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        if hasattr(output, "content"):
            output = self.message(output)
        elapsed = perf_counter() - self.started.pop(str(run_id), perf_counter())
        self.event("tool_output", run_id=str(run_id), seconds=elapsed, result=output)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.event("tool_error", run_id=str(run_id), guidance=api_error_guidance(error))
