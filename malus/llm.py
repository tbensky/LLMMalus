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
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .tools import parse_arguments

RETRY_STATUSES = {429, 500, 502, 503, 504}
# NRP's qwen3 endpoint answered HTTP 500 about 30 times in one 57-call run,
# once four times in a row; 5 attempts nearly ended that run. 8 attempts with
# waits of 2, 4, 8, 16, 30, 30, 30 s ride out about two minutes of trouble.
DEFAULT_ATTEMPTS = 8
MAX_RETRY_DELAY = 30


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
               timeout: float, attempts: int = DEFAULT_ATTEMPTS) -> dict[str, Any]:
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
            delay = min(MAX_RETRY_DELAY, 2 ** (attempt + 1))
            print(f"  [retry] attempt {attempt + 1} failed ({last_error}); waiting {delay}s")
            time.sleep(delay)
    raise LLMError(f"Giving up on {url} after {attempts} attempts: {last_error}")


# gpt-oss's own tool-call syntax, when the host fails to parse it and hands
# it back as text: "<|channel|>commentary to=functions.read_voltage ... <|call|>"
_LEAKED_CALL_RE = re.compile(r"to=functions\.([A-Za-z_]\w*)")


def _tool_specs(tools: list[dict[str, Any]]) -> dict[str, tuple[set[str], set[str]]]:
    """name -> (required argument names, all argument names)."""
    specs = {}
    for tool in tools:
        function = tool.get("function", tool)
        params = function.get("parameters") or {}
        specs[function["name"]] = (set(params.get("required") or []),
                                   set((params.get("properties") or {}).keys()))
    return specs


def _json_objects(text: str, start: int = 0, limit: int = 8):
    decoder = json.JSONDecoder()
    index, found = start, 0
    while found < limit:
        begin = text.find("{", index)
        if begin < 0:
            return
        try:
            value, end = decoder.raw_decode(text, begin)
        except ValueError:
            index = begin + 1
            continue
        index = end
        found += 1
        yield begin, value


def extract_text_tool_calls(text: str, tools: list[dict[str, Any]]) -> list[ToolCall]:
    """Find tool calls a model wrote as text instead of calling a tool.

    Three shapes are recognized, seen from real models on real hosts:

    1. {"name": "read_voltage", "arguments": {...}}  (a call written out)
    2. {"conclusion": "...", "formula": "..."}        (just the arguments: gpt-oss
       did this seven times with finish's arguments). Matched to a tool only
       when the keys include all of that tool's required arguments, use no
       others, and fit exactly one tool. Tools without required arguments are
       never matched this way, so a stray {"A": 50} in prose moves nothing.
    3. "to=functions.read_voltage ... <|call|>"       (gpt-oss's own syntax
       leaking through). Arguments are taken from the next JSON object if
       there is one; a tool that needs arguments is skipped without them.
    """
    specs = _tool_specs(tools)
    calls: list[ToolCall] = []

    def add(name: str, raw: Any) -> None:
        args, error = parse_arguments(raw)
        calls.append(ToolCall(id=f"text_{len(calls) + 1}", name=name, arguments=args,
                              raw_arguments=raw, parse_error=error, from_text=True))

    leaks = [m for m in _LEAKED_CALL_RE.finditer(text) if m.group(1) in specs]
    if leaks:
        for match in leaks[:8]:
            name = match.group(1)
            following = next(_json_objects(text, match.end(), limit=1), None)
            if following and following[0] - match.end() < 200 and isinstance(following[1], dict):
                add(name, following[1])
            elif not specs[name][0]:
                add(name, {})
        return calls

    for _, value in _json_objects(text):
        if not isinstance(value, dict) or not value:
            continue
        if value.get("name") in specs:
            add(value["name"], value.get("arguments", value.get("parameters", {})))
            continue
        keys = set(value)
        matches = [name for name, (required, allowed) in specs.items()
                   if required and required <= keys <= allowed]
        if len(matches) == 1:
            add(matches[0], value)
    return calls


# DeepSeek's own tool-call markup, when the host fails to parse a second call
# and folds it into the first one's arguments. deepseek-v4-flash run 005 sent
# write_down with {"text": "...<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"set_knob\">\n",
# "knob": "B", "value": 0}: the notebook got the markup and the knob never moved.
_DSML_MARK_RE = re.compile(r"<[｜|]\s*DSML\s*[｜|]")
_DSML_INVOKE_RE = re.compile(r'<[｜|]\s*DSML\s*[｜|]\s*invoke\s+name=\\?"([A-Za-z_]\w*)')


def split_leaked_calls(calls: list[ToolCall], tools: list[dict[str, Any]]) -> list[ToolCall]:
    """Separate a tool call that leaked into another call's arguments.

    Recognized when a call has arguments its tool does not take, a string
    argument contains DeepSeek's call markup naming another tool, and the
    extra arguments are exactly what that tool takes (all its required ones,
    nothing it does not know). The first call keeps its own arguments, its
    text cut where the markup starts; the leaked call follows it, marked
    from_text. Anything else is left alone.
    """
    specs = _tool_specs(tools)
    out: list[ToolCall] = []
    for call in calls:
        args = call.arguments
        if not isinstance(args, dict) or call.name not in specs:
            out.append(call)
            continue
        own_names = specs[call.name][1]
        extra = {k: v for k, v in args.items() if k not in own_names}
        marked = [v for v in args.values() if isinstance(v, str) and _DSML_MARK_RE.search(v)]
        names = [m.group(1) for text in marked for m in _DSML_INVOKE_RE.finditer(text)]
        target = names[-1] if names else None
        if not extra or target not in specs or not (
                specs[target][0] <= set(extra) <= specs[target][1]):
            out.append(call)
            continue
        own = {k: (_DSML_MARK_RE.split(v, 1)[0].rstrip() if isinstance(v, str) else v)
               for k, v in args.items() if k in own_names}
        out.append(ToolCall(id=call.id, name=call.name, arguments=own,
                            raw_arguments=call.raw_arguments))
        out.append(ToolCall(id=f"{call.id}_leaked", name=target, arguments=extra,
                            raw_arguments=extra, from_text=True))
    return out


class ChatClient:
    provider = "base"

    def __init__(self, url: str, model: str, temperature: float | None = None,
                 timeout: float = 300.0, attempts: int = DEFAULT_ATTEMPTS):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.attempts = max(1, attempts)

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
        data = _post_json(self.url, payload, headers, self.timeout, self.attempts)
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
        data = _post_json(self.url, payload, {}, self.timeout, self.attempts)
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
                timeout: float = 300.0, attempts: int = DEFAULT_ATTEMPTS) -> ChatClient:
    provider = provider.lower()
    if provider == "ollama":
        return OllamaClient(url or DEFAULT_URLS["ollama"], model, num_ctx=num_ctx,
                            think=think, temperature=temperature, timeout=timeout,
                            attempts=attempts)
    if provider in {"nrp", "openai", "deepseek"}:
        return OpenAICompatibleClient(url or DEFAULT_URLS.get(provider, DEFAULT_URLS["nrp"]),
                                      model, api_key=api_key, temperature=temperature,
                                      timeout=timeout, attempts=attempts)
    raise ValueError(f"Unknown provider {provider!r}: use ollama, nrp or openai")
