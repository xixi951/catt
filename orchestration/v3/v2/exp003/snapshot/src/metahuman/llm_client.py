"""Streaming chat-completion client abstraction.

Both ``Talker`` and ``Thinker`` need to call an LLM with: ``model``,
``messages``, optional thinking-mode toggle, and a streaming response of
content deltas. Two backends are currently supported:

* ``DashScopeClient`` -- the original cloud API (``dashscope.Generation.call``)
  with ``enable_thinking`` / ``thinking_budget`` as native kwargs.
* ``OpenAIClient`` -- any OpenAI-compatible chat-completions endpoint,
  including local vLLM 0.11+ serving Qwen3 family. Thinking mode is
  controlled via ``chat_template_kwargs={"enable_thinking": ...}`` in
  ``extra_body``; ``--reasoning-parser qwen3`` on the server keeps the
  ``<think>...</think>`` block out of ``delta.content`` so we can safely
  forward it to TTS without leaking reasoning.

The interface yields **content text only**. Reasoning content (DashScope's
``reasoning_content`` field, or vLLM's separate ``delta.reasoning_content``
when reasoning_parser is set) is dropped at the implementation boundary so
all downstream code (sentence-splitting, segment-parsing, TTS) sees a
single canonical text stream.
"""
from __future__ import annotations

import abc
import asyncio
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Mapping, Optional


@dataclass
class LLMStreamDelta:
    """One streaming chunk carrying content / reasoning_content text.

    Used by ``LLMClient.stream_chat_with_reasoning(...)`` for callers that
    need both the visible content stream and the hidden reasoning_content
    stream (e.g. the streaming Thinker, which consumes raw reasoning_content
    as its working notes and emits the final visible content as the EoT
    summary). Either field may be empty on any given delta; the client is
    expected to skip emitting deltas where both are empty.
    """
    content: str = ""
    reasoning_content: str = ""


class LLMClient(abc.ABC):
    """Backend-agnostic streaming chat-completion client."""

    @abc.abstractmethod
    def stream_chat(
        self,
        *,
        model: str,
        messages: List[Mapping[str, str]],
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Async-iterate text content deltas. Reasoning content is dropped."""
        ...

    @abc.abstractmethod
    def stream_chat_with_reasoning(
        self,
        *,
        model: str,
        messages: List[Mapping[str, str]],
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> AsyncIterator[LLMStreamDelta]:
        """Async-iterate ``LLMStreamDelta(content, reasoning_content)``.

        The streaming Thinker uses this to read both the model's hidden
        chain-of-thought (reasoning_content) and the post-thinking
        visible answer (content). Talker / baseline drivers continue
        to call ``stream_chat`` (content-only) for backward compatibility.
        Implementations should NOT yield deltas where both fields are
        empty.
        """
        ...


class DashScopeClient(LLMClient):
    """Wraps ``dashscope.Generation.call`` (today's behavior).

    Lazy-imports ``dashscope`` so users without the SDK can still use
    ``OpenAIClient``. Mirrors the ``stream=True``, ``result_format='message'``,
    ``incremental_output=True`` semantics from the original
    ``Talker._stream_generation`` / ``Thinker._stream_generation``.
    """

    def __init__(self) -> None:
        self._dashscope: Optional[Any] = None

    async def stream_chat(
        self,
        *,
        model: str,
        messages: List[Mapping[str, str]],
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> AsyncIterator[str]:
        if self._dashscope is None:
            import dashscope
            self._dashscope = dashscope

        kwargs: dict = {
            "model": model,
            "messages": list(messages),
            "stream": True,
            "result_format": "message",
            "incremental_output": True,
        }
        if enable_thinking:
            kwargs["enable_thinking"] = True
            if thinking_budget is not None:
                kwargs["thinking_budget"] = int(thinking_budget)

        stream = await asyncio.to_thread(
            self._dashscope.Generation.call,
            **kwargs,
        )

        iterator = iter(stream)
        sentinel = object()
        while True:
            chunk = await asyncio.to_thread(next, iterator, sentinel)
            if chunk is sentinel:
                break
            content_delta = _dashscope_extract_content(chunk)
            if content_delta:
                yield content_delta

    async def stream_chat_with_reasoning(
        self,
        *,
        model: str,
        messages: List[Mapping[str, str]],
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> AsyncIterator[LLMStreamDelta]:
        if self._dashscope is None:
            import dashscope
            self._dashscope = dashscope

        kwargs: dict = {
            "model": model,
            "messages": list(messages),
            "stream": True,
            "result_format": "message",
            "incremental_output": True,
        }
        if enable_thinking:
            kwargs["enable_thinking"] = True
            if thinking_budget is not None:
                kwargs["thinking_budget"] = int(thinking_budget)

        stream = await asyncio.to_thread(
            self._dashscope.Generation.call,
            **kwargs,
        )

        iterator = iter(stream)
        sentinel = object()
        while True:
            chunk = await asyncio.to_thread(next, iterator, sentinel)
            if chunk is sentinel:
                break
            content_delta, reasoning_delta = _dashscope_extract_both(chunk)
            if content_delta or reasoning_delta:
                yield LLMStreamDelta(
                    content=content_delta,
                    reasoning_content=reasoning_delta,
                )


class OpenAIClient(LLMClient):
    """OpenAI-compatible chat completions; works with vLLM, DashScope's
    OpenAI-compat mirror, OpenAI cloud, DeepSeek, etc.

    Two backend dialects are auto-detected from ``base_url``:

    * **DashScope** (``compatible-mode/v1`` in URL): uses *flat*
      ``extra_body`` keys ``{"enable_thinking": <bool>, "thinking_budget": N}``
      where ``thinking_budget`` caps hidden CoT only and visible answer
      is unbounded (within server max).
    * **vLLM** (everything else): uses *nested*
      ``extra_body={"chat_template_kwargs": {"enable_thinking": <bool>},
      "thinking_token_budget": N}``. The ``thinking_token_budget``
      param (vLLM 0.x late versions) is the analog of DashScope's
      ``thinking_budget``: caps reasoning tokens only, leaves answer
      tokens to the regular ``max_tokens`` cap. Server must be started
      with ``--reasoning-parser qwen3`` so the ``<think>`` block is
      split out of ``delta.content``; otherwise the raw think text
      would be forwarded to TTS.

    For Qwen3-family models, official sampling recipes are:

    * thinking on:  temperature=0.6, top_p=0.95, top_k=20
    * thinking off: temperature=0.7, top_p=0.8,  top_k=20

    The client picks the recipe based on ``enable_thinking`` and
    forwards the values as standard OpenAI params. Override by passing
    explicit values via the (future) sampling kwargs; for now the
    recipe is hardcoded since both DashScope and vLLM accept the same
    names.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAIClient requires `pip install openai>=1.40`."
            ) from exc

        resolved_base = base_url or os.environ.get("OPENAI_API_BASE")
        self._dashscope_compat = bool(
            resolved_base and "dashscope" in resolved_base.lower()
        )
        self._client = openai.AsyncOpenAI(
            base_url=resolved_base,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )

    def _build_kwargs(
        self,
        *,
        model: str,
        messages: List[Mapping[str, str]],
        enable_thinking: bool,
        thinking_budget: Optional[int],
    ) -> dict:
        """Shared kwargs builder for both stream_chat and
        stream_chat_with_reasoning. Picks Qwen3 sampling recipe and the
        right extra_body dialect (DashScope flat vs vLLM nested)."""
        # Defensive copy + Qwen3 ``/no_think`` workaround for vLLM 0.20.1
        # stream-mode bug: with ``--reasoning-parser qwen3``, the
        # ``chat_template_kwargs.enable_thinking=false`` flag is
        # silently ignored in streaming mode (model output ends up in
        # the ``reasoning`` field, leaving ``content`` empty). The
        # only reliable streaming-mode switch is the Qwen3 ``/no_think``
        # directive appended to the user message. AND including
        # ``chat_template_kwargs.enable_thinking=false`` together with
        # ``/no_think`` actually overrides the directive, so we MUST
        # omit the flag on non-thinking vLLM calls.
        prepared_messages = [dict(m) for m in messages]
        if not self._dashscope_compat and not enable_thinking:
            for i in range(len(prepared_messages) - 1, -1, -1):
                if prepared_messages[i].get("role") == "user":
                    text = prepared_messages[i].get("content", "")
                    if isinstance(text, str) and "/no_think" not in text:
                        prepared_messages[i] = dict(
                            prepared_messages[i],
                            content=text.rstrip() + " /no_think",
                        )
                    break

        kwargs: dict = {
            "model": model,
            "messages": prepared_messages,
            "stream": True,
        }
        if enable_thinking:
            kwargs["temperature"] = 0.6
            kwargs["top_p"] = 0.95
        else:
            kwargs["temperature"] = 0.7
            kwargs["top_p"] = 0.8

        extra_body: dict = {"top_k": 20}
        if self._dashscope_compat:
            extra_body["enable_thinking"] = bool(enable_thinking)
            if enable_thinking and thinking_budget is not None:
                extra_body["thinking_budget"] = int(thinking_budget)
        else:
            # vLLM: only set ``chat_template_kwargs`` when thinking is
            # ON; sending it with ``enable_thinking=False`` overrides
            # the ``/no_think`` directive (see comment in
            # ``prepared_messages`` block above).
            if enable_thinking:
                extra_body["chat_template_kwargs"] = {
                    "enable_thinking": True
                }
                if thinking_budget is not None:
                    extra_body["thinking_token_budget"] = int(thinking_budget)
        kwargs["extra_body"] = extra_body
        return kwargs

    async def stream_chat(
        self,
        *,
        model: str,
        messages: List[Mapping[str, str]],
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> AsyncIterator[str]:
        kwargs = self._build_kwargs(
            model=model, messages=messages,
            enable_thinking=enable_thinking, thinking_budget=thinking_budget,
        )
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) or ""
            # delta.reasoning_content (DashScope native or vLLM with
            # reasoning_parser=qwen3) is intentionally dropped.
            if content:
                yield content

    async def stream_chat_with_reasoning(
        self,
        *,
        model: str,
        messages: List[Mapping[str, str]],
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> AsyncIterator[LLMStreamDelta]:
        kwargs = self._build_kwargs(
            model=model, messages=messages,
            enable_thinking=enable_thinking, thinking_budget=thinking_budget,
        )
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) or ""
            # Reasoning field name varies by backend:
            # * DashScope native and vLLM <=0.18 expose ``reasoning_content``
            # * vLLM 0.19+ aligned with OpenAI's o-series schema and
            #   renamed it to ``reasoning``
            # Read both and merge; only one will be non-empty per delta.
            reasoning = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
                or ""
            )
            if content or reasoning:
                yield LLMStreamDelta(content=content, reasoning_content=reasoning)


def _dashscope_extract_content(chunk: Any) -> str:
    """Pull only ``content`` increment from one DashScope chunk.

    DashScope chunk shape: ``output.choices[0].delta.content`` (or
    sometimes ``output.choices[0].message.content`` depending on
    incremental setting). ``reasoning_content`` is dropped.
    """
    content, _ = _dashscope_extract_both(chunk)
    return content


def _dashscope_extract_both(chunk: Any) -> tuple:
    """Pull both ``content`` and ``reasoning_content`` increments from
    one DashScope chunk. Returns ``(content_str, reasoning_str)`` with
    empty strings for missing fields. Used by
    ``stream_chat_with_reasoning`` so the streaming Thinker can read
    the hidden CoT directly."""
    output = _get(chunk, "output")
    choices = _get(output, "choices") if output is not None else None
    if choices is None:
        choices = _get(chunk, "choices")
    if not choices:
        return "", ""
    choice = choices[0]
    delta = _get(choice, "delta")
    if delta is None:
        delta = _get(choice, "message")
    if delta is None:
        return "", ""
    content = _get(delta, "content")
    reasoning = _get(delta, "reasoning_content")
    return (
        content if isinstance(content, str) else "",
        reasoning if isinstance(reasoning, str) else "",
    )


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def make_llm_client_from_env() -> LLMClient:
    """Construct an ``LLMClient`` based on env vars.

    ``METAHUMAN_LLM_BACKEND``: ``dashscope`` (default) or ``openai``.
    For ``openai``, also reads ``OPENAI_API_BASE`` and ``OPENAI_API_KEY``.
    """
    backend = os.environ.get("METAHUMAN_LLM_BACKEND", "dashscope").lower()
    if backend == "openai":
        return OpenAIClient()
    if backend == "dashscope":
        return DashScopeClient()
    raise ValueError(
        f"Unknown METAHUMAN_LLM_BACKEND={backend!r}; expected 'dashscope' or 'openai'"
    )
