"""Chat-with-tools clients for a local Ollama model or an OpenAI-compatible host.

Both return the same Reply, so the conversation loop does not care which one
it is talking to:

    reply = client.chat(messages, tool_schemas)
    messages.append(reply.assistant_message)
    for call in reply.tool_calls:
        messages.append(client.tool_result_message(call, "result text"))

OpenAI-compatible covers NRP (https://ellm.nrp-nautilus.io/v1/chat/completions,
the endpoint GASolver uses for deepseek-v4-flash and friends), OpenAI itself,
DeepSeek's own API and Ollama's /v1 endpoint. Standard library only.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .tools import parse_arguments

RETRY_STATUSES = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | None   # None when they could not be parsed
    raw_arguments: Any = None
    parse_error: str | None = None
    from_text: bool = False            # recovered from plain text, not a real tool call


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall]
    assistant_message: dict[str, Any]  # provider-native, to append to the history
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str],
               timeout: float, attempts: int = 5) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    last_error = ""
    for attempt in range(attempts):
        request = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:600]
            if exc.code not in RETRY_STATUSES:
                raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from None
            last_error = f"HTTP {exc.code}: {detail}"
        except (TimeoutError, socket.timeout, urllib.error.URLError, ConnectionError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts - 1:
            delay = 2 ** (attempt + 1)
            print(f"  [retry] attempt {attempt + 1} failed ({last_error}); waiting {delay}s")
            time.sleep(delay)
    raise LLMError(f"Giving up on {url} after {attempts} attempts: {last_error}")


def extract_text_tool_calls(text: str, tool_names: set[str]) -> list[ToolCall]:
    """Find tool calls a model wrote as JSON in its text instead of calling.

    Some models (or some hosts) do not support native tool calling and
    write {"name": "read_voltage", "arguments": {...}} in the reply instead.
    """
    calls: list[ToolCall] = []
    decoder = json.JSONDecoder()
    index = 0
    while len(calls) < 8:
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            index = start + 1
            continue
        index = end
        if isinstance(value, dict) and value.get("name") in tool_names:
            raw = value.get("arguments", value.get("parameters", {}))
            args, error = parse_arguments(raw)
            calls.append(ToolCall(id=f"text_{len(calls) + 1}", name=value["name"],
                                  arguments=args, raw_arguments=raw,
                                  parse_error=error, from_text=True))
    return calls


class ChatClient:
    provider = "base"

    def __init__(self, url: str, model: str, temperature: float | None = None,
                 timeout: float = 300.0):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        raise NotImplementedError

    def tool_result_message(self, call: ToolCall, content: str) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleClient(ChatClient):
    """/v1/chat/completions with tools (NRP, OpenAI, DeepSeek, vLLM...)."""

    provider = "openai"

    def __init__(self, url: str, model: str, api_key: str | None = None, **kw: Any):
        super().__init__(url, model, **kw)
        self.api_key = api_key

    def chat(self, messages, tools):
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post_json(self.url, payload, headers, self.timeout)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected response: {json.dumps(data)[:600]}") from None
        text = message.get("content") or ""
        calls = []
        for n, raw_call in enumerate(message.get("tool_calls") or [], start=1):
            function = raw_call.get("function") or {}
            args, error = parse_arguments(function.get("arguments"))
            calls.append(ToolCall(
                id=raw_call.get("id") or f"call_{n}",
                name=function.get("name") or "",
                arguments=args, raw_arguments=function.get("arguments"),
                parse_error=error,
            ))
        # Only content and tool calls go back into the history: a reasoning
        # model's reasoning_content must not be sent back (DeepSeek rejects it).
        assistant: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            assistant["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name,
                              "arguments": c.raw_arguments if isinstance(c.raw_arguments, str)
                              else json.dumps(c.raw_arguments or {})}}
                for c in calls
            ]
        else:
            assistant["content"] = text
        usage = data.get("usage") or {}
        return Reply(text=text, tool_calls=calls, assistant_message=assistant,
                     usage={"prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0)},
                     raw=data)

    def tool_result_message(self, call, content):
        return {"role": "tool", "tool_call_id": call.id, "content": content}


class OllamaClient(ChatClient):
    """Ollama's native /api/chat with tools."""

    provider = "ollama"

    def __init__(self, url: str, model: str, num_ctx: int | None = 16384,
                 think: bool | None = None, **kw: Any):
        super().__init__(url, model, **kw)
        self.num_ctx = num_ctx
        self.think = think

    def chat(self, messages, tools):
        options: dict[str, Any] = {}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if self.temperature is not None:
            options["temperature"] = self.temperature
        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "stream": False}
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options
        if self.think is not None:
            payload["think"] = self.think
        data = _post_json(self.url, payload, {}, self.timeout)
        message = data.get("message") or {}
        text = message.get("content") or ""
        calls = []
        for n, raw_call in enumerate(message.get("tool_calls") or [], start=1):
            function = raw_call.get("function") or {}
            args, error = parse_arguments(function.get("arguments"))
            calls.append(ToolCall(
                id=raw_call.get("id") or f"call_{n}",
                name=function.get("name") or "",
                arguments=args, raw_arguments=function.get("arguments"),
                parse_error=error,
            ))
        assistant: dict[str, Any] = {"role": "assistant", "content": text}
        if message.get("tool_calls"):
            assistant["tool_calls"] = message["tool_calls"]
        return Reply(text=text, tool_calls=calls, assistant_message=assistant,
                     usage={"prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0)},
                     raw=data)

    def tool_result_message(self, call, content):
        return {"role": "tool", "content": content, "tool_name": call.name}


DEFAULT_URLS = {
    "ollama": "http://localhost:11434/api/chat",
    "nrp": "https://ellm.nrp-nautilus.io/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
}


def make_client(provider: str, model: str, url: str | None = None,
                api_key: str | None = None, temperature: float | None = None,
                num_ctx: int | None = 16384, think: bool | None = None,
                timeout: float = 300.0) -> ChatClient:
    provider = provider.lower()
    if provider == "ollama":
        return OllamaClient(url or DEFAULT_URLS["ollama"], model, num_ctx=num_ctx,
                            think=think, temperature=temperature, timeout=timeout)
    if provider in {"nrp", "openai", "deepseek"}:
        return OpenAICompatibleClient(url or DEFAULT_URLS.get(provider, DEFAULT_URLS["nrp"]),
                                      model, api_key=api_key,
                                      temperature=temperature, timeout=timeout)
    raise ValueError(f"Unknown provider {provider!r}: use ollama, nrp or openai")
