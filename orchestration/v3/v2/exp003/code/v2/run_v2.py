"""EXP-20260930-003 v2 matched runner: CATT vs matched single-call baselines.

Runs on newsrv next to a localhost vLLM 0.10.2 server (see launch_vllm.sh) and
the recovered router RPC server (see launch_router.sh). The recovered CATT
snapshot (``snapshot/``) is imported unmodified; every piece of instrumentation
lives in this file. BRIEF.md ("Continuation BRIEF v2") is the protocol.

Conditions
  catt             recovered Controller/Thinker/Talker/SpeakerBridge, RPC router
  non_thinking     one call, prompts_v2/baseline_matched.md, thinking off
  thinking         the same call with thinking on and the same enforced budget
  catt_force_hard  optional additional condition (router score logged, route forced hard)

Every condition runs through the same LLMBundle -> Controller -> SpeakerBridge
-> TTS path. The baselines use a router stub that returns one direct (easy)
dispatch and a talker that issues one call with the matched prompt.

Usage (on newsrv, from the run copy of this EXP directory):
  python code/v2/run_v2.py --config <cfg.json> --phase A --out <run_dir>
  python code/v2/run_v2.py --config <cfg.json> --smoke --out <run_dir>
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import math
import random
import re
import sys
import threading
import time
import traceback
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'snapshot'
sys.dont_write_bytecode = True
sys.path[:0] = [str(SNAPSHOT / 'src'), str(SNAPSHOT / 'scripts')]

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from metahuman.controller import Controller  # noqa: E402
from metahuman.llm import LLMBundle  # noqa: E402
from metahuman.llm_client import LLMStreamDelta, OpenAIClient  # noqa: E402
from metahuman.speaker_bridge import PlaybackWindow, SpeakerBridge  # noqa: E402
from metahuman.talker import DashScopeTalker  # noqa: E402
from metahuman.thinker import DashScopeThinker  # noqa: E402
from metahuman.tts import TTSInterface, TTSResult  # noqa: E402
from metahuman.types import AudioMeta, RouteDecision, UserUtterance  # noqa: E402

# Qwen3 technical report, thinking-budget recipe: the stop-thinking instruction
# inserted when the reasoning length reaches the budget (preceded here by a
# paragraph break because the cut can fall mid-sentence).
FORCE_STOP_TEXT = ("\n\nConsidering the limited time by the user, I have to give the solution "
                   "based on the thinking directly now.\n</think>.\n\n")
WARMUP_PROMPTS = ('Say hello in one short sentence.', 'Name three primary colors.')


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(Path(path).read_bytes())


def derive_seed(*parts: Any) -> int:
    return int(sha_bytes('|'.join(str(p) for p in parts).encode())[:8], 16) % (2 ** 31)


def read_jsonl(path: Path) -> List[dict]:
    # File-line iteration: str.splitlines() would also split U+2028 inside strings.
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Per-attempt event log
# ---------------------------------------------------------------------------


class AttemptLog:
    """Append-only JSONL logs for one attempt. ``t`` is seconds since the turn
    anchor (monotonic); before the anchor is set it is relative to log creation."""

    def __init__(self, cell: Path, context: Dict[str, Any]) -> None:
        self.cell = cell
        self.context = context
        self.lock = threading.Lock()
        self.anchor = time.monotonic()
        self.events = (cell / 'events.jsonl').open('a', encoding='utf-8')
        self.deltas = (cell / 'deltas.jsonl').open('a', encoding='utf-8')
        self.call_counter = 0
        self.active_calls = 0
        self.peak_calls = 0
        self.errors: List[str] = []
        self.calls: Dict[str, Dict[str, Any]] = {}

    def t(self) -> float:
        return time.monotonic() - self.anchor

    def emit(self, kind: str, **data: Any) -> None:
        row = {'t': self.t(), 'wall': time.time(), 'kind': kind, **data}
        with self.lock:
            self.events.write(json.dumps(row, ensure_ascii=False) + '\n')
            self.events.flush()

    def delta(self, call_id: str, channel: str, text: str, ntok: Optional[int]) -> None:
        row = {'t': self.t(), 'c': call_id, 'k': channel, 'x': text, 'n': ntok}
        with self.lock:
            self.deltas.write(json.dumps(row, ensure_ascii=False) + '\n')

    def begin_call(self, role: str) -> str:
        with self.lock:
            self.call_counter += 1
            self.active_calls += 1
            self.peak_calls = max(self.peak_calls, self.active_calls)
            call_id = f'{role}-{self.call_counter:03d}'
            self.calls[call_id] = {'role': role}
            return call_id

    def end_call(self) -> None:
        with self.lock:
            self.active_calls -= 1

    def close(self) -> None:
        with self.lock:
            self.events.close()
            self.deltas.close()


# ---------------------------------------------------------------------------
# LLM client with enforced reasoning budget
# ---------------------------------------------------------------------------


class Tokenizer:
    """Local rendering of the server's chat template (same weights snapshot)."""

    def __init__(self, path: str) -> None:
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)

    def render(self, messages: List[dict]) -> str:
        # The server renders non-thinking calls without chat_template_kwargs and
        # thinking calls with enable_thinking=True; both give this text.
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False).input_ids)


class BudgetedClient:
    """LLMClient for the snapshot Thinker/Talker. One instance per role per attempt.

    * Request body = recovered ``OpenAIClient._build_kwargs`` (``/no_think``
      dialect, Qwen3 sampling recipe) with ``extra_body`` flattened, the ignored
      ``thinking_token_budget`` field removed, plus explicit seed, max_tokens and
      continuous usage statistics.
    * Thinking calls: when cumulative generated tokens reach the reasoning budget
      while still inside the reasoning block, the stream is closed (vLLM aborts
      the request) and the answer is continued through /v1/completions with the
      Qwen3 stop-thinking text appended.
    """

    _dashscope_compat = False  # read by OpenAIClient._build_kwargs

    def __init__(self, cfg: dict, log: AttemptLog, role: str, tokenizer: Tokenizer,
                 http: httpx.AsyncClient, seed_parts: tuple) -> None:
        self.cfg, self.log, self.role = cfg, log, role
        self.tokenizer, self.http, self.seed_parts = tokenizer, http, seed_parts
        self.n_calls = 0

    async def stream_chat(self, *, model: str, messages: List[dict], enable_thinking: bool = False,
                          thinking_budget: Optional[int] = None) -> AsyncIterator[str]:
        async for delta in self.stream_chat_with_reasoning(
                model=model, messages=messages, enable_thinking=enable_thinking,
                thinking_budget=thinking_budget):
            if delta.content:
                yield delta.content

    def _cap(self, prompt_tokens: int, nominal: int) -> int:
        return max(1, min(nominal, self.cfg['max_model_len'] - prompt_tokens - 16))

    async def stream_chat_with_reasoning(self, *, model: str, messages: List[dict],
                                         enable_thinking: bool = False,
                                         thinking_budget: Optional[int] = None) -> AsyncIterator[LLMStreamDelta]:
        cfg = self.cfg
        if enable_thinking and thinking_budget != cfg['reasoning_budget']:
            raise ValueError(f'unexpected thinking budget {thinking_budget}; frozen value is {cfg["reasoning_budget"]}')
        self.n_calls += 1
        call_index = self.n_calls
        body = OpenAIClient._build_kwargs(self, model=model, messages=messages,
                                          enable_thinking=enable_thinking, thinking_budget=None)
        body.update(body.pop('extra_body', {}))
        body.pop('thinking_token_budget', None)
        rendered = self.tokenizer.render(body['messages'])
        prompt_tokens_local = self.tokenizer.count(rendered)
        body['max_tokens'] = self._cap(prompt_tokens_local, cfg['max_tokens_per_call'])
        body['seed'] = derive_seed(cfg['generation_seed'], *self.seed_parts, self.role, call_index)
        body['stream_options'] = {'include_usage': True, 'continuous_usage_stats': True}
        call = self.log.begin_call(self.role)
        info = self.log.calls[call]
        info.update(phase1_prompt_tokens_local=prompt_tokens_local, thinking=enable_thinking,
                    max_tokens=body['max_tokens'])
        self.log.emit('llm_request', call_id=call, role=self.role, endpoint='chat', request=body,
                      request_sha256=sha_bytes(canonical(body)), rendered_prompt_sha256=sha_bytes(rendered.encode()),
                      prompt_tokens_local=prompt_tokens_local, nominal_max_tokens=cfg['max_tokens_per_call'],
                      reasoning_budget=thinking_budget if enable_thinking else None)
        started = self.log.t()
        usage1 = usage2 = None
        finish = None
        reasoning_parts: List[str] = []
        content_seen = False
        forced = False
        first_token_t = None
        ntok = None
        status = 'ok'
        try:
            async with self.http.stream('POST', cfg['base_url'] + '/chat/completions', json=body,
                                        timeout=httpx.Timeout(cfg['request_timeout_seconds'], connect=10.0)) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode(errors='replace')
                    raise RuntimeError(f'HTTP {resp.status_code}: {text[:500]}')
                async for line in resp.aiter_lines():
                    if not line.startswith('data:'):
                        continue
                    raw = line[5:].strip()
                    if raw == '[DONE]':
                        break
                    chunk = json.loads(raw)
                    if chunk.get('usage') is not None:
                        usage1 = chunk['usage']
                        ntok = usage1.get('completion_tokens')
                    for choice in chunk.get('choices', []):
                        if choice.get('finish_reason'):
                            finish = choice['finish_reason']
                        d = choice.get('delta') or {}
                        content = d.get('content') or ''
                        reasoning = d.get('reasoning_content') or d.get('reasoning') or ''
                        if (content or reasoning) and first_token_t is None:
                            first_token_t = self.log.t()
                            self.log.emit('llm_first_token', call_id=call, role=self.role)
                        if reasoning:
                            reasoning_parts.append(reasoning)
                            self.log.delta(call, 'r', reasoning, ntok)
                        if content:
                            if not content_seen:
                                content_seen = True
                                info['reasoning_tokens_upto_first_content'] = ntok
                                self.log.emit('llm_first_content', call_id=call, role=self.role, phase=1,
                                              completion_tokens=ntok)
                            self.log.delta(call, 'c', content, ntok)
                        if content or reasoning:
                            yield LLMStreamDelta(content=content, reasoning_content=reasoning)
                    if (enable_thinking and not content_seen and ntok is not None
                            and ntok >= thinking_budget and finish is None):
                        forced = True
                        break  # leaving the context manager closes the stream; vLLM aborts
            if forced:
                info['reasoning_tokens_at_stop'] = ntok
                async for text, usage, fin in self._continue_after_budget(call, model, body, rendered,
                                                                          reasoning_parts, ntok):
                    usage2 = usage if usage is not None else usage2
                    finish = fin or finish
                    if text:
                        yield LLMStreamDelta(content=text)
            if finish is None:
                raise RuntimeError('stream ended without finish_reason')
        except asyncio.CancelledError:
            status = 'cancelled'
            raise
        except GeneratorExit:
            status = 'closed_by_consumer'
            raise
        except BaseException as exc:
            status = 'error'
            msg = f'{type(exc).__name__}: {exc}'
            self.log.errors.append(f'{call}: {msg}')
            self.log.emit('llm_error', call_id=call, role=self.role, error=msg)
            raise
        finally:
            info.update(status=status, usage_phase1=usage1, usage_phase2=usage2, budget_forced=forced,
                        reasoning_chars=sum(len(p) for p in reasoning_parts),
                        finish_reason=finish, first_token_t=first_token_t, started_t=started, ended_t=self.log.t())
            self.log.emit('llm_end', call_id=call, role=self.role, status=status, usage_phase1=usage1,
                          usage_phase2=usage2, usage_missing=usage1 is None, budget_forced=forced,
                          finish_reason=finish, elapsed_seconds=self.log.t() - started)
            self.log.end_call()

    async def _continue_after_budget(self, call: str, model: str, body: dict, rendered: str,
                                     reasoning_parts: List[str], ntok: int):
        cfg = self.cfg
        prompt = rendered + '<think>' + ''.join(reasoning_parts) + FORCE_STOP_TEXT
        prompt_tokens_local = self.tokenizer.count(prompt)
        remaining = cfg['max_tokens_per_call'] - ntok
        req = {'model': model, 'prompt': prompt, 'stream': True,
               'max_tokens': self._cap(prompt_tokens_local, remaining),
               'temperature': body['temperature'], 'top_p': body['top_p'], 'top_k': body['top_k'],
               'seed': (body['seed'] + 1) % (2 ** 31),
               'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}
        self.log.emit('llm_budget_forced', call_id=call, role=self.role, reasoning_tokens_at_stop=ntok,
                      request=req, request_sha256=sha_bytes(canonical(req)),
                      prompt_sha256=sha_bytes(prompt.encode()), prompt_tokens_local=prompt_tokens_local,
                      force_stop_text=FORCE_STOP_TEXT)
        self.log.calls[call]['phase2_prompt_tokens_local'] = prompt_tokens_local
        first = True
        async with self.http.stream('POST', cfg['base_url'] + '/completions', json=req,
                                    timeout=httpx.Timeout(cfg['request_timeout_seconds'], connect=10.0)) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode(errors='replace')
                raise RuntimeError(f'HTTP {resp.status_code} (continuation): {text[:500]}')
            async for line in resp.aiter_lines():
                if not line.startswith('data:'):
                    continue
                raw = line[5:].strip()
                if raw == '[DONE]':
                    break
                chunk = json.loads(raw)
                usage = chunk.get('usage')
                ntok2 = usage.get('completion_tokens') if usage else None
                if not chunk.get('choices'):
                    yield ('', usage, None)
                for choice in chunk.get('choices', []):
                    text = choice.get('text') or ''
                    if text:
                        if first:
                            first = False
                            self.log.emit('llm_first_content', call_id=call, role=self.role, phase=2,
                                          completion_tokens=ntok2)
                        self.log.delta(call, 'c2', text, ntok2)
                    yield (text, usage, choice.get('finish_reason'))


# ---------------------------------------------------------------------------
# TTS, bridge and routers
# ---------------------------------------------------------------------------


class RecordedTTS(TTSInterface):
    """Wraps the shared SherpaOnnxTTS; logs every synthesis and saves its WAV."""

    def __init__(self, inner: Any, log: AttemptLog, cell: Path) -> None:
        self.inner, self.log, self.cell = inner, log, cell
        self.submitted = 0
        self.lock = threading.Lock()

    async def async_synthesize_text(self, text: str, turn_anchor_monotonic: float) -> TTSResult:
        # Index assigned before the first await, i.e. in task-creation (FIFO) order.
        with self.lock:
            self.submitted += 1
            index = self.submitted
        self.log.emit('tts_submit', tts_id=index, text=text, warmup=not text.strip())
        return await asyncio.to_thread(self._synth, index, text, turn_anchor_monotonic)

    def synthesize_text(self, text: str, turn_anchor_monotonic: float) -> TTSResult:
        with self.lock:
            self.submitted += 1
            index = self.submitted
        return self._synth(index, text, turn_anchor_monotonic)

    def _synth(self, index: int, text: str, anchor: float) -> TTSResult:
        result = self.inner.synthesize_text(text, anchor)
        path = self.cell / 'audio' / f'tts-{index:04d}.wav'
        with wave.open(str(path), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(result.sample_rate)
            handle.writeframes(np.asarray(result.audio, dtype='<i2').tobytes())
        self.log.emit('tts_result', tts_id=index, synth_started_at=result.started_at,
                      synth_finished_at=result.finished_at, duration_s=result.duration_s,
                      sample_rate=result.sample_rate, wav=str(path.relative_to(self.cell)),
                      wav_sha256=sha_file(path), warmup=not text.strip())
        return result

    def backlog_seconds(self) -> float:
        return self.inner.backlog_seconds()

    def reset(self) -> None:
        self.inner.reset()


class InstrumentedBridge(SpeakerBridge):
    """Records the virtual playback window computed for each synthesized chunk."""

    last_window: Optional[PlaybackWindow] = None
    last_synth: Optional[tuple] = None

    def _compute_playback_window(self, synth_finished_at_monotonic: float, synth_duration_s: float,
                                 turn_anchor_monotonic: float) -> PlaybackWindow:
        window = super()._compute_playback_window(synth_finished_at_monotonic, synth_duration_s,
                                                  turn_anchor_monotonic)
        self.last_window = window
        self.last_synth = (synth_finished_at_monotonic - turn_anchor_monotonic, synth_duration_s)
        return window


class SingleCallRouter:
    """Baseline router stub: no routing, one direct dispatch."""

    async def classify(self, utterance: UserUtterance) -> RouteDecision:
        return RouteDecision(turn_id=utterance.turn_id, route='easy', confidence=-1.0,
                             rationale='single_call_baseline_not_routed', decided_at=utterance.received_at)


class ForceHardRouter:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def classify(self, utterance: UserUtterance) -> RouteDecision:
        decision = await self.inner.classify(utterance)
        return RouteDecision(turn_id=decision.turn_id, route='hard', confidence=decision.confidence,
                             rationale=f'forced hard (additional condition); orig route={decision.route}; '
                                       + decision.rationale, decided_at=decision.decided_at)


class MatchedBaselineTalker(DashScopeTalker):
    """One call with the matched prompt; reuses the CATT talker's dispatch,
    sentence splitter (first-chunk comma rule) and first-token event."""

    def __init__(self, ws: Any, *, template: str, thinking: bool, budget: int, model: str, client: Any) -> None:
        super().__init__(ws, prompts_dir=SNAPSHOT / 'prompts', easy_model=model, thinking_model=model,
                         max_prior_turns=0, thinking_budget=budget, hard_final_thinking_budget=budget,
                         enable_thinking=thinking, llm_client=client)
        self._template = template
        self._baseline_thinking = thinking

    def _should_enable_thinking(self, mode: Any) -> bool:
        return self._baseline_thinking

    def _build_prompt(self, turn_id: Any, mode: Any, snapshot: Any) -> str:
        return self._template.format(user_text=self._ws.current_user_text(turn_id) or '')


class UnusedThinker(DashScopeThinker):
    """Baseline placeholder: the easy path never starts a thinker."""

    async def start(self, turn_id: Any, prompt: str):
        raise RuntimeError('baseline thinker must not start')
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Server probes
# ---------------------------------------------------------------------------


METRIC_NAMES = ('vllm:prompt_tokens_total', 'vllm:generation_tokens_total', 'vllm:num_requests_running',
                'vllm:num_requests_waiting', 'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total',
                'vllm:num_preemptions_total', 'vllm:request_success_total')


async def scrape_metrics(http: httpx.AsyncClient, cfg: dict) -> Dict[str, float]:
    resp = await http.get(cfg['server_root'] + '/metrics', timeout=10.0)
    out: Dict[str, float] = {}
    for line in resp.text.splitlines():
        if line.startswith('#'):
            continue
        for name in METRIC_NAMES:
            if line.startswith(name + '{') or line.startswith(name + ' '):
                try:
                    out[name] = out.get(name, 0.0) + float(line.rsplit(' ', 1)[1])
                except ValueError:
                    pass
    return out


async def wait_server_idle(http: httpx.AsyncClient, cfg: dict, timeout: float = 120.0) -> Dict[str, float]:
    started = time.monotonic()
    while True:
        metrics = await scrape_metrics(http, cfg)
        if metrics.get('vllm:num_requests_running', 0) == 0 and metrics.get('vllm:num_requests_waiting', 0) == 0:
            return metrics
        if time.monotonic() - started > timeout:
            raise RuntimeError(f'server not idle after {timeout}s: {metrics}')
        await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# One attempt
# ---------------------------------------------------------------------------


@dataclass
class Shared:
    cfg: dict
    tokenizer: Tokenizer
    http: httpx.AsyncClient
    tts: Any
    rpc_router: Any
    baseline_template: str
    questions: Dict[str, Any]


def spoken_text_from_ws(ws_log: Path, turn_id: str) -> tuple:
    """Concatenate talker_segment texts in seq order (the scorer's definition)."""
    segments = []
    for event in read_jsonl(ws_log):
        data = event.get('data') or {}
        if event.get('kind') == 'talker_segment' and data.get('turn_id') == turn_id:
            segments.append((data.get('seq'), data.get('mode'), data.get('text') or ''))
    segments.sort(key=lambda s: s[0])
    return ''.join(s[2] for s in segments), segments


async def run_attempt(shared: Shared, entry: dict, out: Path, number: int, label: str) -> dict:
    cfg = shared.cfg
    condition = entry['condition']
    question = shared.questions[entry['question_id']]
    cell = out / f'attempt-{number:04d}'
    cell.mkdir()
    (cell / 'audio').mkdir()
    log = AttemptLog(cell, entry)
    tts = RecordedTTS(shared.tts, log, cell)
    seed_parts = (condition, question.id)
    model = cfg['model']
    log.emit('attempt_start', label=label, **entry)
    status = 'COMPLETE'
    turn_id = f'{condition}/{question.id}'
    ws_log: Optional[Path] = None
    chunks_log: List[dict] = []
    m_before = m_after = None
    wall_start = wall_end = None
    bundle = None
    bridge = None
    try:
        if condition in ('catt', 'catt_force_hard'):
            router = shared.rpc_router if condition == 'catt' else ForceHardRouter(shared.rpc_router)

            def thinker_factory(ws: Any) -> Any:
                return DashScopeThinker(ws, prompts_dir=SNAPSHOT / 'prompts', thinking_model=model,
                                        enable_thinking=True, thinking_budget=cfg['reasoning_budget'],
                                        llm_client=BudgetedClient(cfg, log, 'thinker', shared.tokenizer,
                                                                  shared.http, seed_parts))

            def talker_factory(ws: Any) -> Any:
                return DashScopeTalker(ws, prompts_dir=SNAPSHOT / 'prompts', easy_model=model, thinking_model=model,
                                       enable_thinking=True, thinking_budget=cfg['reasoning_budget'],
                                       llm_client=BudgetedClient(cfg, log, 'talker', shared.tokenizer,
                                                                 shared.http, seed_parts))
        elif condition in ('non_thinking', 'thinking'):
            router = SingleCallRouter()
            thinking = condition == 'thinking'

            def thinker_factory(ws: Any) -> Any:
                return UnusedThinker(ws, prompts_dir=SNAPSHOT / 'prompts', thinking_model=model,
                                     llm_client=BudgetedClient(cfg, log, 'unused', shared.tokenizer,
                                                               shared.http, seed_parts))

            def talker_factory(ws: Any) -> Any:
                return MatchedBaselineTalker(ws, template=shared.baseline_template, thinking=thinking,
                                             budget=cfg['reasoning_budget'], model=model,
                                             client=BudgetedClient(cfg, log, 'baseline', shared.tokenizer,
                                                                   shared.http, seed_parts))
        else:
            raise ValueError(condition)

        def controller_factory(ws: Any, router_: Any, thinker: Any, talker: Any) -> Controller:
            return Controller(ws, router_, thinker, talker,
                              mid_trigger_tts_backlog_seconds=cfg['backlog_seconds'],
                              mid_trigger_min_backlog=1, mid_trigger_cooldown=cfg['cooldown_seconds'],
                              enable_hard_first_bridge=True, enable_hard_mid=True)

        bundle = LLMBundle(router, thinker_factory, talker_factory, controller_factory, log_dir=cell)
        dialog_id = re.sub(r'[^A-Za-z0-9_.-]', '_', f'{condition}-{question.id}')
        bundle.begin_dialog(dialog_id)
        await tts.async_synthesize_text(' ', time.monotonic())  # TTS warm-up, as in the May driver
        tts.reset()
        m_before = await wait_server_idle(shared.http, cfg)
        bridge = InstrumentedBridge(bundle, tts)
        await bridge.start()
        anchor = time.monotonic()
        wall_start = time.time()
        log.anchor = anchor
        log.emit('turn_anchor', turn_id=turn_id, anchor_wall=wall_start)
        chunks = bundle.handle_user_text(turn_id=turn_id, text=question.question,
                                         audio_meta=AudioMeta(duration_s=0.0, sample_rate=16000,
                                                              started_at=0.0, finished_at=0.0),
                                         turn_anchor_monotonic=anchor, turn_anchor_walltime=wall_start)
        synth_index = 0

        async def consume() -> None:
            nonlocal synth_index
            async for chunk in bridge.run(chunks, turn_anchor_monotonic=anchor):
                row = {'seq': chunk.talker_seg_seq, 'text': chunk.text, 'is_last': chunk.is_last_in_segment,
                       'produced_at': chunk.produced_at, 'yielded_at': time.monotonic() - anchor}
                if chunk.text:
                    synth_index += 1
                    window, synth = bridge.last_window, bridge.last_synth
                    row.update(synth_order=synth_index, synth_ready_at=synth[0], duration_s=synth[1],
                               playback_start=window.started_at, playback_end=window.finished_at)
                else:
                    tail = bridge._play_until_monotonic
                    row.update(boundary=True, playback_tail=(tail - anchor) if tail is not None else None)
                chunks_log.append(row)
                log.emit('chunk', **row)

        try:
            await asyncio.wait_for(consume(), timeout=cfg['turn_watchdog_seconds'])
        except asyncio.TimeoutError:
            status = 'TIMEOUT'
            log.emit('watchdog_timeout', seconds=cfg['turn_watchdog_seconds'])
        wall_end = time.time()
        log.emit('turn_end', status=status)
    except Exception as exc:
        status = 'FAILED'
        log.errors.append(f'{type(exc).__name__}: {exc}')
        log.emit('attempt_error', error=log.errors[-1], traceback=traceback.format_exc())
        wall_end = wall_end or time.time()
    finally:
        if bridge is not None:
            await bridge.stop()
        if bundle is not None:
            ws_log = bundle.workspace_log_path()
            await bundle.end_dialog()
        try:
            m_after = await wait_server_idle(shared.http, cfg, timeout=180.0)
        except Exception as exc:  # recorded; the next attempt waits again
            log.emit('server_not_idle_after_attempt', error=str(exc))
        tts.reset()
    segment_text, segments = ('', [])
    if ws_log is not None and ws_log.exists():
        segment_text, segments = spoken_text_from_ws(ws_log, turn_id)
    # What the listener hears: played chunks in playback order. segment_text is the
    # May scorer's definition; they differ only when chunks arrive after turn close.
    spoken = ' '.join(c['text'] for c in chunks_log if c.get('text'))
    if status == 'COMPLETE':
        if not any(s[1] in ('easy', 'hard_final') for s in segments):
            log.errors.append('terminal (easy/hard_final) talker segment absent')
        if not chunks_log:
            log.errors.append('no spoken chunk')
        if log.errors:
            status = 'FAILED'
    calls = log.calls
    summary = {
        **entry, 'attempt': number, 'label': label, 'status': status, 'errors': log.errors,
        'turn_id': turn_id, 'ws_log': ws_log.name if ws_log else None,
        'wall_start': wall_start, 'wall_end': wall_end,
        'spoken_text': spoken, 'segment_text': segment_text, 'segments': [{'seq': s[0], 'mode': s[1], 'text': s[2]} for s in segments],
        'llm_calls': len(calls),
        'llm_calls_by_role': {r: sum(1 for c in calls.values() if c['role'] == r)
                              for r in sorted({c['role'] for c in calls.values()})},
        'tts_calls': max(0, tts.submitted - 1),  # minus the warm-up
        'peak_concurrent_llm_calls': log.peak_calls,
        'budget_forced_calls': sum(1 for c in calls.values() if c.get('budget_forced')),
        'calls': calls,
        'server_metrics_before': m_before, 'server_metrics_after': m_after,
        'first_playback_start': min([c['playback_start'] for c in chunks_log if 'playback_start' in c], default=None),
        'last_playback_end': max([c['playback_end'] for c in chunks_log if 'playback_end' in c], default=None),
        'n_chunks': sum(1 for c in chunks_log if c.get('text')),
    }
    log.emit('attempt_end', status=status, errors=log.errors)
    log.close()
    (cell / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
    return summary


# ---------------------------------------------------------------------------
# Schedules and phases
# ---------------------------------------------------------------------------


def build_schedule(cfg: dict, questions: List[Any], phase: str) -> List[dict]:
    ids = [q.id for q in questions]
    rng = random.Random(f"{cfg['schedule_seed']}-{phase}")
    if phase == 'A':
        order = list(ids)
        rng.shuffle(order)
        return [{'phase': 'A', 'question_id': q, 'condition': 'catt', 'repeat': 0} for q in order]
    if phase == 'B':
        order = list(ids)
        rng.shuffle(order)
        schedule = []
        for q in order:
            conds = ['non_thinking', 'thinking']
            rng.shuffle(conds)
            schedule += [{'phase': 'B', 'question_id': q, 'condition': c, 'repeat': 0} for c in conds]
        return schedule
    if phase == 'C':
        order = [q.id for q in questions if q.source == 'voicebench_alpaca_eval']
        rng.shuffle(order)
        return [{'phase': 'C', 'question_id': q, 'condition': 'catt_force_hard', 'repeat': 0} for q in order]
    raise ValueError(phase)


SMOKE_QUESTIONS = [
    ('smoke/easy', 'smoke', 'general', 'What is the capital of France? Answer briefly.', 'Paris'),
    ('smoke/mcq', 'smoke', 'mcq', 'Which planet is known as the Red Planet?\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn\n\n'
     'What is the answer to the above multiple choice question? Select one of the following: A, B, C, or D.', 'B'),
    ('smoke/math', 'smoke', 'math', 'Solve the following math problem step by step. Put your answer inside \\boxed{}.\n\n'
     'How many positive divisors does 2024 have?', '16'),
]


async def server_warmup(cfg: dict, tokenizer: Tokenizer, http: httpx.AsyncClient, out: Path, tag: str) -> None:
    """Fixed non-fixture prompts before a phase; labelled warm-up, never outcomes."""
    cell = out / f'warmup-{tag}-{int(time.time())}'
    cell.mkdir()
    log = AttemptLog(cell, {'warmup': tag})
    for i, prompt in enumerate(WARMUP_PROMPTS):
        for thinking in (False, True):
            client = BudgetedClient(cfg, log, f'warmup{i}', tokenizer, http, ('warmup', tag))
            async for _ in client.stream_chat(model=cfg['model'], messages=[{'role': 'user', 'content': prompt}],
                                              enable_thinking=thinking,
                                              thinking_budget=cfg['reasoning_budget'] if thinking else None):
                pass
    log.close()


async def main_async(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.config).read_text())
    baseline = importlib.import_module('baseline_direct_llm')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fixture = SNAPSHOT / cfg['fixture']
    if sha_file(fixture) != cfg['fixture_sha256']:
        raise ValueError('fixture hash mismatch')
    if args.smoke:
        questions = [baseline.Question(i, s, sub, q, a, 'smoke') for i, s, sub, q, a in SMOKE_QUESTIONS]
        if args.smoke_budget:
            cfg = dict(cfg, reasoning_budget=args.smoke_budget)
        schedule = [{'phase': 'SMOKE', 'question_id': q.id, 'condition': c, 'repeat': 0}
                    for q in questions for c in args.smoke_conditions.split(',')]
        label = 'SMOKE_NOT_OUTCOME'
    else:
        questions = baseline.load_fixture(fixture)
        schedule = build_schedule(cfg, questions, args.phase)
        label = f'PHASE_{args.phase}'
    if args.limit:
        schedule = schedule[:args.limit]
    qmap = {q.id: q for q in questions}
    template = (ROOT / 'prompts_v2' / 'baseline_matched.md').read_text(encoding='utf-8')
    from metahuman.adapters_impl.sherpa_onnx_tts import SherpaOnnxTTS
    from metahuman.adapters_impl.router_rpc_client import RouterRPCClient
    tts = SherpaOnnxTTS(model_dir=Path(cfg['tts_model_dir']))
    rpc = RouterRPCClient(uds_path=cfg['router_uds'], hard_threshold=cfg['router_threshold'])
    tokenizer = Tokenizer(cfg['weights_path'])
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=0)
    async with httpx.AsyncClient(limits=limits) as http:
        shared = Shared(cfg, tokenizer, http, tts, rpc, template, qmap)
        freeze_path = out / 'freeze.json'
        attempts_path = out / 'attempts.jsonl'
        done = {row['attempt']: row for row in read_jsonl(attempts_path)} if attempts_path.exists() else {}
        if not freeze_path.exists():
            models = (await http.get(cfg['base_url'] + '/models', timeout=10)).json()
            version = (await http.get(cfg['server_root'] + '/version', timeout=10)).json()
            freeze = {
                'evidence_kind': label, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
                'config': cfg, 'schedule': schedule, 'schedule_sha256': sha_bytes(canonical(schedule)),
                'runner_sha256': sha_file(Path(__file__)),
                'source_manifest_sha256': sha_file(ROOT / 'artifacts' / 'source_manifest.json'),
                'baseline_prompt_sha256': sha_bytes(template.encode()),
                'catt_prompt_sha256': {p.name: sha_file(p) for p in sorted((SNAPSHOT / 'prompts').glob('*.md'))},
                'brief_sha256': sha_file(ROOT / 'BRIEF.md'),
                'server_models': models, 'server_version': version,
                'playback': 'virtual FIFO from synthesized durations, paced in real time; no device playback',
            }
            freeze_path.write_text(json.dumps(freeze, indent=1) + '\n')
        else:
            frozen = json.loads(freeze_path.read_text())
            if frozen['schedule_sha256'] != sha_bytes(canonical(schedule)):
                raise ValueError('resume refused: schedule differs from freeze.json')
            if frozen['runner_sha256'] != sha_file(Path(__file__)):
                raise ValueError('resume refused: runner changed since freeze')
        await server_warmup(cfg, tokenizer, http, out, label)
        for number, entry in enumerate(schedule):
            if number in done:
                continue
            stale = out / f'attempt-{number:04d}'
            if stale.exists():  # interrupted attempt: preserved, rerun under the same number
                k = 1
                while (out / f'attempt-{number:04d}.interrupted-{k}').exists():
                    k += 1
                stale.rename(out / f'attempt-{number:04d}.interrupted-{k}')
                entry = dict(entry, rerun_after_interruption=k)
            summary = await run_attempt(shared, entry, out, number, label)
            row = {key: summary.get(key) for key in (
                'phase', 'question_id', 'condition', 'repeat', 'attempt', 'status', 'errors', 'llm_calls',
                'tts_calls', 'peak_concurrent_llm_calls', 'budget_forced_calls', 'wall_start', 'wall_end',
                'rerun_after_interruption')}
            with attempts_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            secs = (row['wall_end'] or 0) - (row['wall_start'] or row['wall_end'] or 0)
            print(json.dumps({'attempt': number, 'condition': row['condition'], 'status': row['status'],
                              'calls': row['llm_calls'], 'secs': round(secs, 1)}), flush=True)
    close = getattr(rpc, 'close', None)
    if close is not None:
        result = close()
        if asyncio.iscoroutine(result):
            await result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--phase', choices=['A', 'B', 'C'])
    parser.add_argument('--limit', type=int, default=0, help='first N schedule entries only (validation)')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-budget', type=int, default=0)
    parser.add_argument('--smoke-conditions', default='catt,non_thinking,thinking,catt_force_hard')
    args = parser.parse_args()
    if not args.smoke and not args.phase:
        parser.error('--phase is required unless --smoke')
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
