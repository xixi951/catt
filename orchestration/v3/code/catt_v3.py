"""EXP-20261007-001: CATT v3 as experiment-local subclasses of EXP-20261005-001's CATT v2.

EXP-20261005-001's package is copied read-only to ``pkg/v2`` and never modified. BRIEF section 3.1
lists the changes; each class names the ones it implements.

    engine_cfg       per-engine copy of the run config (thinker engine / talker engine, change 1)
    V3Client         V2Client plus the shared thinker-side budget handling (change 3): at the reasoning
                     budget, or when the loop guard fires, the thinker engine continues with the Qwen3
                     stop-thinking text (EXP-20260930-003's mechanism) and the continuation has its own
                     loop guard. Used by the CATT thinker AND the thinking baseline, so their requests,
                     streams and written answers are identical (change 2).
    V3Thinker        V2Thinker; on a forced stop it collects the thinker's written answer instead of
                     ending at its first token (change 3).
    V3Talker         per-question bridge length (change 4); final-answer prompt that states the
                     thinker's written answer after a forced stop (change 3).
    V3Controller     the low-queue exception for progress updates (change 4).
    AlwaysHardRouter allhard configuration (change 5): every turn takes the hard path, no router call.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'v2' / 'code'))
import catt_v2 as V2  # noqa: E402  (puts EXP-20260930-003's snapshot and run_v2 on sys.path)

R = V2.R
import httpx  # noqa: E402
from metahuman.llm_client import LLMStreamDelta, OpenAIClient  # noqa: E402
from metahuman.types import RouteDecision, SessionState, TalkerMode, UserUtterance  # noqa: E402

PROMPTS = PKG / 'prompts'
FORCE_STOP_TEXT = R.FORCE_STOP_TEXT


def engine_cfg(cfg: dict, role: str) -> dict:
    """Run config as seen by one engine's clients: base_url, server_root, max_model_len."""
    e = cfg['engines'][role]
    return dict(cfg, base_url=e['base_url'], server_root=e['server_root'], max_model_len=e['max_model_len'],
                engine_role=role)


# ---------------------------------------------------------------------------
# Loop guard (BRIEF 3.1, change 3; thresholds fixed in BRIEF v0)
# ---------------------------------------------------------------------------

LOOP_START_TOKENS = 2000
LOOP_EVERY_TOKENS = 256
LOOP_PROBE_CHARS, LOOP_WINDOW_CHARS, LOOP_MIN_COUNT = 400, 8000, 3
CONT_PROBE_CHARS, CONT_WINDOW_CHARS, CONT_MIN_COUNT = 200, 4000, 3


def repeated_tail(text: str, probe: int, window: int, min_count: int) -> bool:
    """True iff the last ``probe`` characters occur at least ``min_count`` times (non-overlapping)
    within the last ``window`` characters."""
    if len(text) < probe * min_count:
        return False
    tail = text[-window:]
    return tail.count(text[-probe:]) >= min_count


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class V3Client(V2.V2Client):
    """V2Client with the shared thinker-side budget handling.

    * ``on_budget='continue'`` (thinker and thinking baseline): at the reasoning budget, or when the
      reasoning loop guard fires, the stream is closed and the thinker engine continues through
      ``/v1/completions`` with the stop-thinking text (prompt, sampling and seed rule exactly as
      EXP-20260930-003's ``_continue_after_budget``). The continuation is closed early by its own
      loop guard. Continuation text is yielded as content; ``in_continuation`` is True meanwhile.
    * Records ``last_forced``, ``last_force_reason`` ('budget' | 'loop_guard'), ``last_reasoning``,
      ``last_continuation`` for the attempt summary (reasoning-identity check, BRIEF 1).
    The request body is built exactly as in V2Client, so a CATT thinker request and a thinking
    baseline request with the same seed key are byte-identical.
    """

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.in_continuation = False
        self.last_forced = False
        self.last_force_reason: Optional[str] = None
        self.last_reasoning = ''
        self.last_continuation = ''
        self.last_continuation_stop: Optional[str] = None

    async def stream_chat_with_reasoning(self, *, model: str, messages: List[dict], enable_thinking: bool = False,
                                         thinking_budget: Optional[int] = None,
                                         max_tokens: Optional[int] = None) -> AsyncIterator[LLMStreamDelta]:
        cfg = self.cfg
        if enable_thinking and thinking_budget != cfg['reasoning_budget']:
            raise ValueError(f'unexpected thinking budget {thinking_budget}; frozen value is {cfg["reasoning_budget"]}')
        self.n_calls += 1
        call_index = self.n_calls
        self.last_ntok = 0
        self.last_budget_stopped = False
        self.in_continuation = False
        self.last_forced = False
        self.last_force_reason = None
        self.last_reasoning = ''
        self.last_continuation = ''
        self.last_continuation_stop = None
        body = OpenAIClient._build_kwargs(self, model=model, messages=messages,
                                          enable_thinking=enable_thinking, thinking_budget=None)
        body.update(body.pop('extra_body', {}))
        body.pop('thinking_token_budget', None)
        rendered = self.tokenizer.render(body['messages'])
        prompt_tokens_local = self.tokenizer.count(rendered)
        nominal = cfg['max_tokens_per_call'] if max_tokens is None else int(max_tokens)
        body['max_tokens'] = self._cap(prompt_tokens_local, nominal)
        body['seed'] = R.derive_seed(cfg['generation_seed'], *self.seed_parts, self.seed_role, call_index)
        body['stream_options'] = {'include_usage': True, 'continuous_usage_stats': True}
        call = self.log.begin_call(self.role)
        info = self.log.calls[call]
        info.update(phase1_prompt_tokens_local=prompt_tokens_local, thinking=enable_thinking,
                    max_tokens=body['max_tokens'], seed_key=[*self.seed_parts, self.seed_role, call_index],
                    engine=cfg.get('engine_role'))
        self.log.emit('llm_request', call_id=call, role=self.role, endpoint='chat', engine=cfg.get('engine_role'),
                      request=body, request_sha256=R.sha_bytes(R.canonical(body)),
                      rendered_prompt_sha256=R.sha_bytes(rendered.encode()),
                      prompt_tokens_local=prompt_tokens_local, nominal_max_tokens=nominal,
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
        next_loop_check = LOOP_START_TOKENS
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
                        if ntok is not None:
                            self.last_ntok = ntok
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
                    if enable_thinking and not content_seen and ntok is not None and finish is None:
                        if ntok >= thinking_budget:
                            forced, self.last_force_reason = True, 'budget'
                            break  # leaving the context manager closes the stream; vLLM aborts
                        if ntok >= next_loop_check:
                            next_loop_check = ntok + LOOP_EVERY_TOKENS
                            if repeated_tail(''.join(reasoning_parts), LOOP_PROBE_CHARS, LOOP_WINDOW_CHARS, LOOP_MIN_COUNT):
                                forced, self.last_force_reason = True, 'loop_guard'
                                self.log.emit('llm_loop_guard', call_id=call, role=self.role, phase=1,
                                              reasoning_tokens=ntok)
                                break
            self.last_reasoning = ''.join(reasoning_parts)
            if forced:
                self.last_forced = True
                info['reasoning_tokens_at_stop'] = ntok
                info['force_reason'] = self.last_force_reason
                if self.on_budget == 'stop':
                    self.last_budget_stopped = True
                    finish = 'budget_stop'
                    self.log.emit('llm_budget_stopped', call_id=call, role=self.role, reasoning_tokens_at_stop=ntok)
                else:
                    self.in_continuation = True
                    cont_parts: List[str] = []
                    agen = self._continue_after_budget(call, model, body, rendered, reasoning_parts, ntok)
                    try:
                        async for text, usage, fin in agen:
                            usage2 = usage if usage is not None else usage2
                            finish = fin or finish
                            if text:
                                cont_parts.append(text)
                                yield LLMStreamDelta(content=text)
                                if repeated_tail(''.join(cont_parts), CONT_PROBE_CHARS, CONT_WINDOW_CHARS, CONT_MIN_COUNT):
                                    finish = 'loop_guard'
                                    self.last_continuation_stop = 'loop_guard'
                                    self.log.emit('llm_loop_guard', call_id=call, role=self.role, phase=2)
                                    break
                    finally:
                        await agen.aclose()
                        self.last_continuation = ''.join(cont_parts)
                        self.in_continuation = False
            if finish is None:
                raise RuntimeError('stream ended without finish_reason')
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                status = 'cancelled'
            elif isinstance(exc, GeneratorExit):
                status = 'closed_by_consumer'
            else:
                status = 'error'
                msg = f'{type(exc).__name__}: {exc}'
                self.log.errors.append(f'{call}: {msg}')
                self.log.emit('llm_error', call_id=call, role=self.role, error=msg)
            if not self.last_reasoning:
                self.last_reasoning = ''.join(reasoning_parts)
            raise
        finally:
            info.update(status=status, usage_phase1=usage1, usage_phase2=usage2, budget_forced=forced,
                        force_reason=self.last_force_reason, reasoning_chars=sum(len(p) for p in reasoning_parts),
                        reasoning_sha256=R.sha_bytes(''.join(reasoning_parts).encode()),
                        continuation_sha256=R.sha_bytes(self.last_continuation.encode()) if forced else None,
                        continuation_stop=self.last_continuation_stop,
                        finish_reason=finish, first_token_t=first_token_t, started_t=started, ended_t=self.log.t())
            self.log.emit('llm_end', call_id=call, role=self.role, status=status, usage_phase1=usage1,
                          usage_phase2=usage2, usage_missing=usage1 is None, budget_forced=forced,
                          force_reason=self.last_force_reason, finish_reason=finish,
                          elapsed_seconds=self.log.t() - started)
            self.log.end_call()


# ---------------------------------------------------------------------------
# Thinker (changes 2, 3)
# ---------------------------------------------------------------------------


@dataclass
class V3ReasoningState(V2.ReasoningState):
    forced_answer: str = ''       # thinker's written answer after a forced stop (continuation text)
    force_reason: str = ''


class V3Thinker(V2.V2Thinker):
    """V2Thinker whose client handles forced stops by continuation. Normal turns: the stream is
    closed at the first visible-content token (no post-reasoning visible answer, as in v2).
    Forced turns: the continuation text is collected into ``forced_answer``; EndOfThinking follows
    once it is complete."""

    async def start(self, turn_id: Any, prompt: str) -> AsyncIterator[Any]:
        from metahuman.types import EndOfThinking, ThinkerSegment
        if turn_id in self._cancelled:
            return
        self._running.add(turn_id)
        st = self._state
        client: V3Client = self._client
        seq = 0
        buffer = ''
        agen = client.stream_chat_with_reasoning(model=self._thinking_model, messages=self._build_messages(prompt),
                                                 enable_thinking=True, thinking_budget=self._thinking_budget)
        try:
            first = True
            async for delta in agen:
                if turn_id in self._cancelled:
                    return
                if first:
                    first = False
                    self._ws.record_event('thinker_first_token', {'turn_id': turn_id, 'ts': self._ws.now(turn_id)})
                if delta.reasoning_content:
                    st.text += delta.reasoning_content
                    st.ntok = max(st.ntok, client.last_ntok)
                    buffer += delta.reasoning_content
                    while True:
                        chunk_text, buffer = self._split_chunk(buffer)
                        if chunk_text is None:
                            break
                        yield ThinkerSegment(turn_id=turn_id, seq=seq, content=chunk_text, created_at=self._ws.now(turn_id))
                        seq += 1
                if delta.content:
                    if client.in_continuation:
                        st.forced_answer += delta.content
                        continue
                    st.stop_reason = 'reasoning_end'
                    break
        finally:
            await agen.aclose()
            self._running.discard(turn_id)
        if turn_id in self._cancelled:
            return
        st.ntok = max(st.ntok, client.last_ntok)
        if client.last_forced:
            st.forced = True
            st.force_reason = client.last_force_reason or ''
            st.stop_reason = client.last_force_reason or 'budget'
            st.forced_answer = client.last_continuation or st.forced_answer
        elif not st.stop_reason:
            st.stop_reason = 'stream_end'
        tail = buffer.strip()
        if tail:
            yield ThinkerSegment(turn_id=turn_id, seq=seq, content=tail, created_at=self._ws.now(turn_id))
            seq += 1
        st.ended = True
        self._ws.record_event('v2_reasoning_end', {'turn_id': turn_id, 'ts': self._ws.now(turn_id),
                                                   'stop_reason': st.stop_reason, 'reasoning_tokens': st.ntok,
                                                   'reasoning_chars': len(st.text),
                                                   'forced_answer_chars': len(st.forced_answer)})
        yield EndOfThinking(turn_id=turn_id, final_summary='', total_segments=seq, created_at=self._ws.now(turn_id))


# ---------------------------------------------------------------------------
# Talker (changes 3, 4)
# ---------------------------------------------------------------------------


@dataclass
class V3TalkerParams(V2.TalkerParams):
    # Per-question bridge length (BRIEF 3.1 change 4): W = clip(round(wps * (T_fp - T_b + m)), lo, hi)
    bridge_fp_base_s: float = 4.9          # predicted first-progress audio time at zero prompt tokens
    bridge_fp_per_ktok_s: float = 0.15     # thinker prefill time per 1,000 prompt tokens
    bridge_start_s: float = 0.5            # predicted bridge audio start
    bridge_wps: float = 2.7                # Piper Amy spoken words per second
    bridge_margin_s: float = 0.3           # m (development set)
    bridge_min_words: int = 8
    bridge_max_words: int = 24
    bridge_cap_factor: float = 1.6         # hard cap = ceil(factor * W) words
    bridge_max_sentences: int = 2
    bridge_one_sentence_max_words: int = 14  # W at or below this: one sentence; above: up to two


_WORD = re.compile(r"[A-Za-z0-9一-鿿]+(?:['’-][A-Za-z0-9]+)*")
# Bridge guard: the bridge must never state an answer (BRIEF 3.1 change 4). A bridge sentence that reads as an
# answer statement, hedged or not, is not spoken and ends the bridge.
_BRIDGE_ANSWER_RE = re.compile(r"\banswers?\s+(?:is|are|was|would|should|must|seems?|might|may|could|looks?)\b|\banswer\s*:"
                               r"|\\boxed|\boption\s+\(?[A-J]\)?\s+(?:is|seems|looks|would)\b"
                               r"|\b(?:correct|right|best)\s+(?:answer|option|choice)\b", re.IGNORECASE)


def bridge_states_answer(sentence: str) -> bool:
    return bool(_BRIDGE_ANSWER_RE.search(sentence))



class V3Talker(V2.V2Talker):
    _FILES = dict(V2.V2Talker._FILES, final_forced='talker_final_forced.md')

    def __init__(self, ws: Any, *, model: str, client: Any, state: V3ReasoningState, params: V3TalkerParams,
                 thinker_prompt_tokens: int) -> None:
        super().__init__(ws, model=model, client=client, state=state, params=params, prompts_dir=PROMPTS)
        self._v3p = params
        self._thinker_prompt_tokens = thinker_prompt_tokens
        self.bridge_plan: Optional[dict] = None
        self._bridge_words_cap = 10 ** 6
        self._bridge_turn: Any = None
        self._bridge_max_sentences = params.bridge_max_sentences
        self._bridge_sentences = 0
        self._bridge_words = 0

    def bridge_target(self) -> dict:
        p = self._v3p
        t_fp = p.bridge_fp_base_s + p.bridge_fp_per_ktok_s * self._thinker_prompt_tokens / 1000.0
        seconds = t_fp - p.bridge_start_s + p.bridge_margin_s
        words = int(min(p.bridge_max_words, max(p.bridge_min_words, round(p.bridge_wps * seconds))))
        sentences = 1 if words <= p.bridge_one_sentence_max_words else min(2, p.bridge_max_sentences)
        return {'predicted_first_progress_s': t_fp, 'target_seconds': seconds, 'target_words': words,
                'max_sentences': sentences, 'cap_words': int(math.ceil(p.bridge_cap_factor * words)),
                'thinker_prompt_tokens': self._thinker_prompt_tokens}

    def _build_v2_prompt(self, signal: Any, mode: TalkerMode) -> str:
        user_text = self._ws.current_user_text(signal.turn_id) or ''
        st = self._state
        if mode == TalkerMode.HARD_FIRST:
            plan = self.bridge_target()
            self.bridge_plan = plan
            spec = 'one short sentence' if plan['max_sentences'] == 1 else 'one or two short sentences'
            return self._v2['bridge'].format(user_text=user_text, target_words=plan['target_words'],
                                             target_seconds=round(plan['target_seconds']), sentence_spec=spec)
        if mode == TalkerMode.HARD_FINAL and st.forced and st.forced_answer.strip():
            spoken = '\n'.join(st.kept_spoken) if st.kept_spoken else self._spoken_so_far(signal.turn_id)
            return self._v2['final_forced'].format(user_text=user_text, written_answer=st.forced_answer.strip(),
                                                   spoken_so_far=spoken or '(nothing yet)')
        return super()._build_v2_prompt(signal, mode)

    async def dispatch(self, signal: Any, mode: TalkerMode) -> AsyncIterator[str]:
        if mode == TalkerMode.HARD_FIRST:
            plan = self.bridge_target()
            self._bridge_words_cap = plan['cap_words']
            self._bridge_turn = signal.turn_id
            self._bridge_max_sentences = plan['max_sentences']
            self._bridge_sentences = 0
            self._bridge_words = 0
            # character cap for v2's buffer check: generous; the word/sentence caps below bind first
            self._p = dataclasses.replace(self._p, bridge_max_chars=12 * plan['cap_words'])
            self._ws.record_event('v3_bridge_plan', {'turn_id': signal.turn_id, **plan})
        async for text in super().dispatch(signal, mode):
            yield text

    def _bridge_cap(self, sentence: str, emitted: int):
        """v3 bridge cap: up to ``bridge_max_sentences`` sentences and ``cap_words`` words; a sentence
        that would exceed the word cap is cut at the cap and ends the bridge."""
        text = sentence.strip()
        if not text:
            return '', False
        if bridge_states_answer(text):
            self._ws.record_event('v3_bridge_sentence_suppressed', {'turn_id': self._bridge_turn, 'text': text,
                                                                    'reason': 'answer_statement'})
            return '', True
        complete = text[-1:] in '.!?。！？;；\n'
        words = _WORD.findall(text)
        room = self._bridge_words_cap - self._bridge_words
        if len(words) > room:
            if room <= 0:
                return '', True
            # cut after the room-th word
            count, cut = 0, len(text)
            for m in _WORD.finditer(text):
                count += 1
                if count == room:
                    cut = m.end()
                    break
            text = text[:cut].rstrip(' ,;:') + '.'
            self._bridge_words += room
            return text, True
        self._bridge_words += len(words)
        if complete:
            self._bridge_sentences += 1
        stop = complete and self._bridge_sentences >= self._bridge_max_sentences
        return text, stop


# ---------------------------------------------------------------------------
# Controller (change 4)
# ---------------------------------------------------------------------------


@dataclass
class V3ControllerParams(V2.ControllerParams):
    lowq_lead_s: float = 2.5          # L: dispatch early when less than L seconds of audio remain queued
    lowq_floor_tokens: int = 100      # F: ... and at least F new reasoning tokens exist
    lowq_enabled: bool = True


class V3Controller(V2.V2Controller):
    """V2Controller plus the low-queue exception (BRIEF 3.1 change 4). The regular trigger (K2 new
    reasoning tokens with backlog below 5 s) is unchanged. After the first progress dispatch, an update
    is also dispatched when the queued audio is below L seconds and at least F new reasoning tokens
    exist (every other v2 gate still applies: not finalizing, state HARD_MID, no talker call in flight)."""

    def __init__(self, *a: Any, params: V3ControllerParams, **kw: Any) -> None:
        super().__init__(*a, params=params, **kw)
        self._v3cp = params
        self._trigger: Optional[str] = None
        self.trigger_counts: Dict[str, int] = {'first': 0, 'regular': 0, 'low_queue': 0}

    def _should_dispatch_mid(self) -> Optional[str]:
        reason = super()._should_dispatch_mid()
        if reason is None:
            self._trigger = 'first' if self._mid_count == 0 else 'regular'
            return None
        p = self._v3cp
        if (p.lowq_enabled and reason == 'below_k2' and self._mid_count >= 1
                and self._tts_backlog_seconds < p.lowq_lead_s
                and self._rs.ntok - self._last_mid_ntok >= p.lowq_floor_tokens):
            self._trigger = 'low_queue'
            return None
        return reason

    def _maybe_dispatch_mid(self, cause: Any) -> None:
        if self._should_dispatch_mid() is not None:
            return
        trig = self._trigger or 'regular'
        self.trigger_counts[trig] = self.trigger_counts.get(trig, 0) + 1
        turn = self._active_turn
        a, b = self._last_mid_char, len(self._rs.text)
        self._last_mid_char, self._last_mid_ntok = b, self._rs.ntok
        self._mid_count += 1
        self._send_talker(turn, cause=f'{cause}|{trig}', snapshot_seq=0, reasoning=(a, b))


class AlwaysHardRouter:
    """allhard configuration: every turn takes the hard path; the router is not consulted."""

    async def classify(self, utterance: UserUtterance) -> RouteDecision:
        return RouteDecision(turn_id=utterance.turn_id, route='hard', confidence=-1.0,
                             rationale='allhard configuration (router not consulted)', decided_at=utterance.received_at)
