"""EXP-20261005-001: CATT v2 as experiment-local subclasses of the recovered CATT snapshot.

The recovered EXP-20260930-003 snapshot (``exp003/snapshot``) is imported read-only and never
modified. BRIEF section 3.1 lists the ten design changes; each class below names the changes it
implements.

    V2Client      LLM client: EXP-20260930-003's request dialect and budget logic, plus per-call
                  token caps, an explicit seed key (common random numbers, BRIEF 4.3), live token
                  counts, and a "stop at budget" mode for the thinker (change 5).
    V2Thinker     reasoning call identical to the thinking baseline's request (change 7); the
                  stream is cut at the first visible-content token (change 5).
    V2Talker      bridge / progress / final / easy prompts (changes 1, 3, 4, 5, 6, 8).
    V2Controller  fast first progress update (change 2), incremental reasoning slices (change 8),
                  final-answer preemption and the in-flight gate over the final call (change 6).
    V2Bridge      SpeakerBridge whose virtual playback queue can drop not-yet-played progress
                  sentences (change 6); used for every condition so all share one playback path.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'exp003' / 'code' / 'v2'))
import run_v2 as R  # noqa: E402  (also puts the snapshot on sys.path)

import httpx  # noqa: E402
from metahuman.controller import Controller, TalkerFinished, TalkerFailed  # noqa: E402
from metahuman.llm_client import LLMStreamDelta, OpenAIClient  # noqa: E402
from metahuman.speaker_bridge import PlaybackWindow, SpeakerBridge, _DispatchPlayback  # noqa: E402
from metahuman.talker import DashScopeTalker  # noqa: E402
from metahuman.thinker import DashScopeThinker  # noqa: E402
from metahuman.types import (ControlSignal, EndOfThinking, SessionState, TalkerChunk, TalkerMode,  # noqa: E402
                             ThinkerSegment, TurnId)

logger = logging.getLogger('catt_v2')
PROMPTS_V3 = PKG / 'prompts_v3'
_SKIP_RE = re.compile(r'^\W*SKIP\W*$')
# Change 3 guard: a progress sentence may mention a candidate answer only as tentative. A sentence that
# states an answer without any hedge ("The answer is D.") is not spoken.
_ANSWER_CLAIM_RE = re.compile(r"\b(?:final\s+|correct\s+|right\s+)?answer\s*(?:is|would\s+be|should\s+be|must\s+be|:)"
                              r"|\boption\s+\(?[A-J]\)?\s+is\s+(?:the\s+)?(?:correct|right|answer)\b"
                              r"|\b[A-J]\s+is\s+the\s+(?:correct|right)\s+(?:answer|option|choice)\b", re.IGNORECASE)
_HEDGE_RE = re.compile(r"\b(?:tentativ\w*|likely|probabl\w*|guess\w*|may|might|seems?|looks?|appears?|could|possibl\w*|"
                       r"perhaps|suspect\w*|lean\w*|current\w*|so\s+far|for\s+now|provisional\w*|initial\w*)\b",
                       re.IGNORECASE)


def unhedged_answer_claim(sentence: str) -> bool:
    return bool(_ANSWER_CLAIM_RE.search(sentence)) and not _HEDGE_RE.search(sentence)


# ---------------------------------------------------------------------------
# Shared per-turn reasoning state
# ---------------------------------------------------------------------------


@dataclass
class ReasoningState:
    """Written by V2Thinker, read by V2Controller (triggers) and V2Talker (prompt slices)."""
    text: str = ''
    ntok: int = 0                 # server-reported completion tokens of the reasoning call so far
    ended: bool = False
    forced: bool = False          # reasoning budget reached before reasoning ended
    stop_reason: str = ''
    kept_spoken: Optional[List[str]] = None   # texts still audible after preemption (final talker input)


@dataclass(frozen=True)
class V2Signal(ControlSignal):
    reasoning_from: int = 0
    reasoning_to: int = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class V2Client(R.BudgetedClient):
    """EXP-20260930-003's BudgetedClient with three additions (behaviour otherwise identical):

    * ``seed_parts``/``seed_role``: the per-request seed is
      derive_seed(generation_seed, *seed_parts, seed_role, call_index). The CATT v2 thinker uses
      the thinking baseline's key ('thinking', item, 'baseline'), so both reasoning calls share a
      seed (BRIEF 4.3). The logged role stays ``role``.
    * ``max_tokens`` per call (talker caps, BRIEF 4.5); the context clamp still applies.
    * ``on_budget='stop'``: at the reasoning budget the stream is closed and the call ends
      without the stop-thinking continuation (the final talker answers instead, change 5).
    ``last_ntok`` exposes the running completion-token count of the current call.
    """

    def __init__(self, cfg: dict, log: Any, role: str, tokenizer: Any, http: httpx.AsyncClient,
                 seed_parts: tuple, *, seed_role: Optional[str] = None, on_budget: str = 'continue') -> None:
        super().__init__(cfg, log, role, tokenizer, http, seed_parts)
        self.seed_role = seed_role or role
        self.on_budget = on_budget
        self.last_ntok = 0
        self.last_budget_stopped = False

    async def stream_chat(self, *, model: str, messages: List[dict], enable_thinking: bool = False,
                          thinking_budget: Optional[int] = None, max_tokens: Optional[int] = None) -> AsyncIterator[str]:
        async for delta in self.stream_chat_with_reasoning(model=model, messages=messages, enable_thinking=enable_thinking,
                                                           thinking_budget=thinking_budget, max_tokens=max_tokens):
            if delta.content:
                yield delta.content

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
                    max_tokens=body['max_tokens'], seed_key=[*self.seed_parts, self.seed_role, call_index])
        self.log.emit('llm_request', call_id=call, role=self.role, endpoint='chat', request=body,
                      request_sha256=R.sha_bytes(R.canonical(body)), rendered_prompt_sha256=R.sha_bytes(rendered.encode()),
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
                    if (enable_thinking and not content_seen and ntok is not None
                            and ntok >= thinking_budget and finish is None):
                        forced = True
                        break  # leaving the context manager closes the stream; vLLM aborts
            if forced:
                info['reasoning_tokens_at_stop'] = ntok
                if self.on_budget == 'stop':
                    self.last_budget_stopped = True
                    finish = 'budget_stop'
                    self.log.emit('llm_budget_stopped', call_id=call, role=self.role, reasoning_tokens_at_stop=ntok)
                else:
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


# ---------------------------------------------------------------------------
# Thinker (changes 5, 7)
# ---------------------------------------------------------------------------


class V2Thinker(DashScopeThinker):
    """One reasoning call whose request equals the thinking baseline's request (change 7).

    Reasoning deltas are appended to the shared ReasoningState and also segmented with the
    recovered splitter into ThinkerSegments (workspace record). At the first visible-content
    token the stream is closed (vLLM aborts): no post-reasoning visible answer (change 5).
    At the budget the client stops the call. Either way EndOfThinking follows, so every hard
    turn reaches the final-answer dispatch.
    """

    def __init__(self, ws: Any, *, template: str, model: str, budget: int, client: V2Client,
                 state: ReasoningState) -> None:
        super().__init__(ws, prompts_dir=R.SNAPSHOT / 'prompts', thinking_model=model, enable_thinking=True,
                         thinking_budget=budget, llm_client=client)
        self._template = template
        self._state = state

    def _build_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        # Exactly MatchedBaselineTalker._build_prompt + DashScopeTalker._stream_generation (no strip).
        return [{'role': 'user', 'content': self._template.format(user_text=user_prompt)}]

    async def start(self, turn_id: TurnId, prompt: str) -> AsyncIterator[Any]:
        if turn_id in self._cancelled:
            return
        self._running.add(turn_id)
        st = self._state
        seq = 0
        buffer = ''
        agen = self._client.stream_chat_with_reasoning(model=self._thinking_model, messages=self._build_messages(prompt),
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
                    st.ntok = max(st.ntok, self._client.last_ntok)
                    buffer += delta.reasoning_content
                    while True:
                        chunk_text, buffer = self._split_chunk(buffer)
                        if chunk_text is None:
                            break
                        yield ThinkerSegment(turn_id=turn_id, seq=seq, content=chunk_text, created_at=self._ws.now(turn_id))
                        seq += 1
                if delta.content:
                    st.stop_reason = 'reasoning_end'
                    break
        finally:
            await agen.aclose()
            self._running.discard(turn_id)
        if turn_id in self._cancelled:
            return
        st.ntok = max(st.ntok, self._client.last_ntok)
        if self._client.last_budget_stopped:
            st.forced = True
            st.stop_reason = 'budget'
        elif not st.stop_reason:
            st.stop_reason = 'stream_end'
        tail = buffer.strip()
        if tail:
            yield ThinkerSegment(turn_id=turn_id, seq=seq, content=tail, created_at=self._ws.now(turn_id))
            seq += 1
        st.ended = True
        self._ws.record_event('v2_reasoning_end', {'turn_id': turn_id, 'ts': self._ws.now(turn_id),
                                                   'stop_reason': st.stop_reason, 'reasoning_tokens': st.ntok,
                                                   'reasoning_chars': len(st.text)})
        yield EndOfThinking(turn_id=turn_id, final_summary='', total_segments=seq, created_at=self._ws.now(turn_id))


# ---------------------------------------------------------------------------
# Talker (changes 1, 3, 4, 5, 6, 8)
# ---------------------------------------------------------------------------


@dataclass
class TalkerParams:
    easy_variant: str = 'E0'            # E0 unchanged easy prompt; E1 conclusion-first (BRIEF 3.3)
    bridge_max_tokens: int = 40
    bridge_max_chars: int = 120
    progress_max_tokens: int = 200
    final_max_tokens: int = 800
    progress_overlap_chars: int = 400
    final_tail_chars: int = 6000


class V2Talker(DashScopeTalker):
    _FILES = {'E0': 'talker_easy_E0.md', 'E1': 'talker_easy_E1.md', 'bridge': 'talker_bridge.md',
              'progress': 'talker_progress.md', 'final': 'talker_final.md'}

    def __init__(self, ws: Any, *, model: str, client: V2Client, state: ReasoningState, params: TalkerParams,
                 prompts_dir: Path = PROMPTS_V3) -> None:
        super().__init__(ws, prompts_dir=R.SNAPSHOT / 'prompts', easy_model=model, thinking_model=model,
                         max_prior_turns=0, enable_thinking=True, llm_client=client)
        self._v2 = {k: (prompts_dir / f).read_text(encoding='utf-8') for k, f in self._FILES.items()}
        self._state = state
        self._p = params

    def _should_enable_thinking(self, mode: Any) -> bool:
        return False  # every talker call is non-thinking, as in the recovered system

    def _spoken_so_far(self, turn_id: TurnId) -> str:
        # Segment texts are chunk texts joined without separators; restore a space after punctuation.
        lines = [re.sub(r'([.!?,;:])(?=[A-Za-z])', r'\1 ', s.text.strip())
                 for s in self._ws.current_turn_talker(turn_id) if s.text.strip()]
        return '\n'.join(lines) if lines else '(nothing yet)'

    def _build_v2_prompt(self, signal: ControlSignal, mode: TalkerMode) -> str:
        user_text = self._ws.current_user_text(signal.turn_id) or ''
        st = self._state
        if mode == TalkerMode.EASY:
            return self._v2[self._p.easy_variant].format(user_text=user_text, prior_dialog='')
        if mode == TalkerMode.HARD_FIRST:
            return self._v2['bridge'].format(user_text=user_text)
        if mode == TalkerMode.HARD_MID:
            a = getattr(signal, 'reasoning_from', 0)
            b = getattr(signal, 'reasoning_to', len(st.text))
            ctx = st.text[max(0, a - self._p.progress_overlap_chars):a].strip()
            return self._v2['progress'].format(user_text=user_text, reasoning_context=ctx or '(none)',
                                               reasoning_new=st.text[a:b].strip() or '(none)',
                                               spoken_so_far=self._spoken_so_far(signal.turn_id))
        if mode == TalkerMode.HARD_FINAL:
            status = ('The reasoning about this question was cut off by its time limit before it finished; '
                      'choose the single most likely answer from what it established.' if st.forced
                      else 'The reasoning about this question has finished.')
            spoken = '\n'.join(st.kept_spoken) if st.kept_spoken else self._spoken_so_far(signal.turn_id)
            return self._v2['final'].format(status_line=status, user_text=user_text,
                                            reasoning_tail=st.text[-self._p.final_tail_chars:].strip() or '(none)',
                                            spoken_so_far=spoken or '(nothing yet)')
        raise ValueError(mode)

    async def dispatch(self, signal: ControlSignal, mode: TalkerMode) -> AsyncIterator[str]:
        prompt = self._build_v2_prompt(signal, mode)
        cap = {TalkerMode.EASY: None, TalkerMode.HARD_FIRST: self._p.bridge_max_tokens,
               TalkerMode.HARD_MID: self._p.progress_max_tokens, TalkerMode.HARD_FINAL: self._p.final_max_tokens}[mode]
        self._ws.record_event('v2_talker_prompt', {'turn_id': signal.turn_id, 'mode': mode.value,
                                                   'prompt_chars': len(prompt), 'max_tokens': cap,
                                                   'reasoning_from': getattr(signal, 'reasoning_from', None),
                                                   'reasoning_to': getattr(signal, 'reasoning_to', None)})
        bridge = mode == TalkerMode.HARD_FIRST
        buffer = ''
        emitted_chars = 0
        first_chunk_yielded = False
        first_token_recorded = False
        agen = self._client.stream_chat(model=self._model_for_mode(mode), messages=[{'role': 'user', 'content': prompt}],
                                        enable_thinking=False, thinking_budget=None, max_tokens=cap)
        done = False
        try:
            async for text in agen:
                if not text:
                    continue
                if not first_token_recorded:
                    first_token_recorded = True
                    self._ws.record_event('talker_first_token', {'turn_id': signal.turn_id,
                                                                 'snapshot_seq': signal.snapshot_seq,
                                                                 'mode': mode.value, 'ts': self._ws.now(signal.turn_id)})
                buffer += text
                sentences, buffer = self._split_complete_sentences(
                    buffer, accept_comma=self._faster_first_response and not first_chunk_yielded)
                for sentence in sentences:
                    if mode == TalkerMode.HARD_MID and _SKIP_RE.match(sentence):
                        continue  # the progress talker declined: nothing new to say
                    if mode == TalkerMode.HARD_MID and unhedged_answer_claim(sentence):
                        self._ws.record_event('v2_progress_sentence_suppressed',
                                              {'turn_id': signal.turn_id, 'text': sentence, 'reason': 'unhedged_answer'})
                        continue
                    if bridge:
                        sentence, stop = self._bridge_cap(sentence, emitted_chars)
                        if sentence:
                            emitted_chars += len(sentence) + 1
                            yield sentence
                            first_chunk_yielded = True
                        if stop:
                            done = True
                            break
                    else:
                        yield sentence
                        first_chunk_yielded = True
                if done:
                    break
                if bridge and emitted_chars + len(buffer) > self._p.bridge_max_chars:
                    sentence, _ = self._bridge_cap(buffer, emitted_chars)
                    if sentence:
                        yield sentence
                    buffer = ''
                    done = True
                    break
        finally:
            await agen.aclose()
        if not done:
            tail = buffer.strip()
            if mode == TalkerMode.HARD_MID and (_SKIP_RE.match(tail) or unhedged_answer_claim(tail)):
                if tail and not _SKIP_RE.match(tail):
                    self._ws.record_event('v2_progress_sentence_suppressed',
                                          {'turn_id': signal.turn_id, 'text': tail, 'reason': 'unhedged_answer'})
                tail = ''
            if tail:
                if bridge:
                    tail, _ = self._bridge_cap(tail, emitted_chars)
                if tail:
                    yield tail
        self._last_completed[signal.turn_id] = max(self._last_completed.get(signal.turn_id, -1), signal.snapshot_seq)

    def _bridge_cap(self, sentence: str, emitted: int) -> Tuple[str, bool]:
        """Bridge hard cap (change 1): only the first sentence is spoken, at most bridge_max_chars.
        Returns (text to speak, stop). A comma-terminated first chunk does not stop the bridge."""
        stop = sentence.rstrip()[-1:] in '.!?。！？;；\n' if sentence.strip() else False
        room = self._p.bridge_max_chars - emitted
        if len(sentence) > room:
            cut = sentence[:max(0, room)]
            cut = cut[:cut.rfind(' ')] if ' ' in cut else cut
            sentence = cut.rstrip(' ,;:') + ('.' if cut else '')
            stop = True
        return sentence.strip(), stop


# ---------------------------------------------------------------------------
# Controller (changes 2, 6, 8)
# ---------------------------------------------------------------------------


@dataclass
class ControllerParams:
    k1_tokens: int = 200          # first progress: reasoning tokens needed (fast path, no low-water wait)
    k2_tokens: int = 250          # later progress: new reasoning tokens since the previous progress dispatch
    backlog_seconds: float = 5.0  # later progress: audio backlog must be below this (as in EXP-20260930-003)


class V2Controller(Controller):
    def __init__(self, ws: Any, router: Any, thinker: Any, talker: Any, *, state: ReasoningState,
                 params: ControllerParams) -> None:
        super().__init__(ws, router, thinker, talker, mid_trigger_tts_backlog_seconds=params.backlog_seconds,
                         mid_trigger_min_backlog=1, mid_trigger_cooldown=0.0,
                         enable_hard_first_bridge=True, enable_hard_mid=True)
        self._rs = state
        self._cp = params
        self._dispatch_counter = 0
        self._mid_count = 0
        self._last_mid_char = 0
        self._last_mid_ntok = 0
        self._final_requested = False
        self._seq_mode: Dict[int, str] = {}
        self._seq_task: Dict[int, asyncio.Task] = {}
        self._preempt_cb: Optional[Callable[[TurnId, Set[int]], List[str]]] = None
        self.preemption: Optional[dict] = None

    def set_preempt(self, cb: Callable[[TurnId, Set[int]], List[str]]) -> None:
        self._preempt_cb = cb

    def seq_mode(self) -> Dict[int, str]:
        return dict(self._seq_mode)

    def _state_now(self) -> Optional[SessionState]:
        return self._ws._ws.state  # direct read; avoids a deep-copied snapshot every 50 ms

    # --- progress gating (change 2) ---
    def _should_dispatch_mid(self) -> Optional[str]:
        turn = self._active_turn
        if turn is None:
            return 'no_active_turn'
        if self._final_requested or self._rs.ended:
            return 'finalizing'
        if self._state_now() != SessionState.HARD_MID:
            return 'state_not_hard_mid'
        if self._has_in_flight_talker(turn):
            return 'talker_in_flight'
        if self._mid_count == 0:
            return None if self._rs.ntok >= self._cp.k1_tokens else 'below_k1'
        if self._tts_backlog_seconds >= self._cp.backlog_seconds:
            return 'backlog_above_threshold'
        if self._rs.ntok - self._last_mid_ntok < self._cp.k2_tokens:
            return 'below_k2'
        return None

    def _maybe_dispatch_mid(self, cause: Any) -> None:
        if self._should_dispatch_mid() is not None:
            return
        turn = self._active_turn
        a, b = self._last_mid_char, len(self._rs.text)
        self._last_mid_char, self._last_mid_ntok = b, self._rs.ntok
        self._mid_count += 1
        self._send_talker(turn, cause=cause, snapshot_seq=0, reasoning=(a, b))

    # --- dispatch with V2Signal and per-seq bookkeeping ---
    def _send_talker(self, turn_id: TurnId, cause: Any, snapshot_seq: int,
                     reasoning: Tuple[int, int] = (0, 0)) -> None:
        if turn_id in self._cancelled_turns:
            return
        state = self._state_now()
        if state is None:
            return
        mode = {SessionState.EASY: TalkerMode.EASY, SessionState.HARD_FIRST: TalkerMode.HARD_FIRST,
                SessionState.HARD_MID: TalkerMode.HARD_MID, SessionState.HARD_FINAL: TalkerMode.HARD_FINAL}[state]
        seg_seq = self._reserve_talker_seq(turn_id)
        self._dispatch_counter += 1
        signal = V2Signal(turn_id=turn_id, triggered_at=self._ws.now(turn_id), cause=cause,
                          snapshot_seq=self._dispatch_counter, reasoning_from=reasoning[0], reasoning_to=reasoning[1])
        self._seq_mode[seg_seq] = mode.value
        self._ws.record_event('v2_dispatch', {'turn_id': turn_id, 'seq': seg_seq, 'mode': mode.value, 'cause': cause,
                                              'ts': signal.triggered_at, 'reasoning_tokens': self._rs.ntok,
                                              'reasoning_from': reasoning[0], 'reasoning_to': reasoning[1],
                                              'backlog_seconds': self._tts_backlog_seconds})
        task = asyncio.create_task(self._supervise_talker(signal, mode, seg_seq, self._turn_generation.get(turn_id, 0)))
        self._talker_supervisors.setdefault(turn_id, []).append(task)
        self._seq_task[seg_seq] = task
        if mode == TalkerMode.HARD_FIRST and turn_id not in self._first_hard_dispatched:
            self._first_hard_dispatched.add(turn_id)
            self._set_state(SessionState.HARD_MID)

    # --- final answer preempts (change 6) ---
    async def _handle_end_of_thinking(self, eot: EndOfThinking) -> None:
        if eot.turn_id in self._cancelled_turns or eot.turn_id != self._active_turn:
            return
        self._final_requested = True

        def record_eot(w: Any) -> None:
            w.thinker_history.append(eot)
            w.state = SessionState.HARD_FINAL

        self._ws.mutate(record_eot)
        turn = eot.turn_id
        cancelled = []
        for seq, task in list(self._seq_task.items()):
            if self._seq_mode.get(seq) == 'hard_mid' and not task.done():
                task.cancel()
                cancelled.append(seq)
        mid_seqs = {s for s, m in self._seq_mode.items() if m == 'hard_mid'}
        kept: List[str] = []
        if self._preempt_cb is not None:
            kept = self._preempt_cb(turn, mid_seqs)
        self._rs.kept_spoken = kept
        bridge_tasks = [t for s, t in self._seq_task.items() if self._seq_mode.get(s) == 'hard_first' and not t.done()]
        self.preemption = {'ts': self._ws.now(turn), 'cancelled_in_flight_progress': cancelled,
                           'progress_seqs': sorted(mid_seqs), 'kept_spoken_n': len(kept),
                           'waited_for_bridge': bool(bridge_tasks)}
        self._ws.record_event('v2_preempt', {'turn_id': turn, **self.preemption})
        if bridge_tasks:
            asyncio.create_task(self._final_after(bridge_tasks, turn))
        else:
            self._send_talker(turn, cause='end_of_thinking', snapshot_seq=0)

    async def _final_after(self, tasks: List[asyncio.Task], turn: TurnId) -> None:
        await asyncio.gather(*tasks, return_exceptions=True)
        if turn == self._active_turn and turn not in self._cancelled_turns:
            self._send_talker(turn, cause='end_of_thinking', snapshot_seq=0)


# ---------------------------------------------------------------------------
# SpeakerBridge with a droppable virtual playback queue (change 6)
# ---------------------------------------------------------------------------


class V2Bridge(SpeakerBridge):
    """SpeakerBridge.run() with three additions; the playback arithmetic is unchanged.

    * Every synthesized chunk gets an id and is recorded with its virtual playback window.
    * ``preempt(turn, seqs)``: chunks of ``seqs`` whose playback has not started are dropped (never
      played, never deferred); later-arriving chunks of those dispatches are dropped on arrival.
      The playback cursor moves back to the end of the last kept chunk, so the next chunk starts
      when the sentence currently playing ends. Returns the texts still audible (played or playing).
    * The backlog fed to the controller is the virtual queue's own remaining duration.
    For conditions that never call ``preempt`` the behaviour equals the recovered bridge.
    """

    def __init__(self, llm: Any, tts: Any, **kw: Any) -> None:
        super().__init__(llm, tts, **kw)
        self.log: Any = None
        self.last_window: Optional[PlaybackWindow] = None
        self.last_synth: Optional[tuple] = None
        self.last_chunk_id: Optional[int] = None
        self.last_dropped: Optional[str] = None
        self._scheduled: List[dict] = []
        self._preempted: Dict[TurnId, Set[int]] = {}
        self._chunk_counter = 0
        self._anchor: Optional[float] = None
        self.is_terminal_seq: Callable[[int], bool] = lambda seq: False
        self.terminal_scheduled = False

    def backlog_now(self) -> float:
        # Once the terminal dispatch (easy, hard_final, or a baseline's single call) is fully synthesized
        # and scheduled, nothing else can be dispatched; the rest of its playback is virtual, so the turn
        # may end without waiting for it in wall-clock time (EXP-20260930-003 amendment 3 behaviour for
        # every condition). Recorded virtual timings are unchanged; only wall time shrinks.
        if self.terminal_scheduled:
            return 0.0
        cursor = self._play_until_monotonic
        return 0.0 if cursor is None else max(0.0, cursor - time.monotonic())

    async def _poll_backlog(self) -> None:
        while self._running:
            try:
                self._llm.notify_tts_backlog(self.backlog_now())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('notify_tts_backlog failed')
            await asyncio.sleep(self._backlog_poll_interval)

    def preempt(self, turn_id: TurnId, seqs: Set[int]) -> List[str]:
        now = time.monotonic()
        self._preempted.setdefault(turn_id, set()).update(seqs)
        kept: List[str] = []
        cursor = None
        for entry in self._scheduled:
            if entry['turn_id'] != turn_id or entry['dropped']:
                continue
            if entry['seq'] in seqs and entry['start_mono'] > now:
                entry['dropped'] = 'preempted_queued'
                if self.log is not None:
                    self.log.emit('chunk_dropped', chunk_id=entry['chunk_id'], seq=entry['seq'], text=entry['text'],
                                  reason='preempted_queued', scheduled_start=entry['start_mono'] - self._anchor,
                                  scheduled_end=entry['end_mono'] - self._anchor, at=now - self._anchor)
                continue
            kept.append(entry['text'])
            cursor = entry['end_mono'] if cursor is None else max(cursor, entry['end_mono'])
        self._play_until_monotonic = cursor
        if self.log is not None:
            self.log.emit('preempt', seqs=sorted(seqs), at=now - self._anchor,
                          cursor=(cursor - self._anchor) if cursor is not None else None, kept_n=len(kept))
        return kept

    async def run(self, chunks: AsyncIterator[TalkerChunk], *, turn_anchor_monotonic: float) -> AsyncIterator[TalkerChunk]:
        self._anchor = turn_anchor_monotonic
        seen_keys: Set[Tuple[TurnId, int]] = set()
        pipeline: asyncio.Queue = asyncio.Queue(maxsize=self._max_in_flight_synth + 1)
        sentinel: Tuple[Optional[TalkerChunk], Optional[asyncio.Task]] = (None, None)
        producer_error: List[BaseException] = []

        def preempted(chunk: TalkerChunk) -> bool:
            return chunk.talker_seg_seq in self._preempted.get(chunk.turn_id, ())

        async def producer() -> None:
            cancelled = False
            try:
                async for chunk in chunks:
                    if chunk.text and not preempted(chunk):
                        task = asyncio.create_task(self._tts.async_synthesize_text(chunk.text, turn_anchor_monotonic))
                    else:
                        task = None
                    await pipeline.put((chunk, task))
            except asyncio.CancelledError:
                cancelled = True
                raise
            except Exception as exc:
                producer_error.append(exc)
            finally:
                if cancelled:
                    try:
                        pipeline.put_nowait(sentinel)
                    except asyncio.QueueFull:
                        pass
                else:
                    await pipeline.put(sentinel)

        producer_task = asyncio.create_task(producer())
        in_flight: List[asyncio.Task] = []
        try:
            while True:
                item = await pipeline.get()
                if item is sentinel:
                    break
                chunk, synth_task = item
                self.last_dropped = None
                self.last_chunk_id = None
                self.last_window = None
                self.last_synth = None
                if chunk.text and (synth_task is None or preempted(chunk)):
                    # Dropped before playback could be scheduled: never played.
                    if synth_task is not None:
                        await asyncio.gather(synth_task, return_exceptions=True)
                    self._chunk_counter += 1
                    self.last_chunk_id = self._chunk_counter
                    self.last_dropped = 'preempted_unscheduled'
                    yield chunk
                    continue
                if synth_task is not None:
                    in_flight.append(synth_task)
                key = (chunk.turn_id, chunk.talker_seg_seq)
                dispatch = self._dispatches.get(key)
                if dispatch is None:
                    dispatch = _DispatchPlayback(talker_seg_seq=chunk.talker_seg_seq, audio_started_at=None,
                                                 audio_finished_at=None, first_chunk_seen=False)
                    self._dispatches[key] = dispatch
                    seen_keys.add(key)
                if synth_task is not None:
                    result = await synth_task
                    if synth_task in in_flight:
                        in_flight.remove(synth_task)
                    if result.first_byte_at is not None and not dispatch.first_chunk_seen:
                        try:
                            self._llm.notify_tts_first_byte(chunk.turn_id, chunk.talker_seg_seq, result.first_byte_at)
                        except Exception:
                            logger.exception('notify_tts_first_byte failed')
                    synth_finished_mono = turn_anchor_monotonic + result.finished_at
                    window = self._compute_playback_window(synth_finished_at_monotonic=synth_finished_mono,
                                                           synth_duration_s=result.duration_s,
                                                           turn_anchor_monotonic=turn_anchor_monotonic)
                    self._chunk_counter += 1
                    self.last_chunk_id = self._chunk_counter
                    self.last_window = window
                    self.last_synth = (result.finished_at, result.duration_s)
                    self._scheduled.append({'chunk_id': self._chunk_counter, 'turn_id': chunk.turn_id,
                                            'seq': chunk.talker_seg_seq, 'text': chunk.text,
                                            'start_mono': turn_anchor_monotonic + window.started_at,
                                            'end_mono': turn_anchor_monotonic + window.finished_at, 'dropped': None})
                else:
                    play_until = self._play_until_monotonic
                    if play_until is None:
                        play_until = turn_anchor_monotonic
                    tail = play_until - turn_anchor_monotonic
                    window = PlaybackWindow(started_at=tail, finished_at=tail)
                if not dispatch.first_chunk_seen:
                    dispatch.first_chunk_seen = True
                    dispatch.audio_started_at = window.started_at
                    try:
                        self._llm.notify_tts_first_audio(chunk.turn_id, chunk.talker_seg_seq, window.started_at)
                    except Exception:
                        logger.exception('notify_tts_first_audio failed')
                if chunk.is_last_in_segment:
                    dispatch.audio_finished_at = window.finished_at
                    try:
                        self._llm.notify_tts_finished_at(chunk.turn_id, chunk.talker_seg_seq, window.finished_at)
                    except Exception:
                        logger.exception('notify_tts_finished_at failed')
                    if self.is_terminal_seq(chunk.talker_seg_seq):
                        self.terminal_scheduled = True
                        if self.log is not None:
                            self.log.emit('terminal_scheduled', seq=chunk.talker_seg_seq,
                                          virtual_playback_end=window.finished_at)
                yield chunk
            if producer_error:
                raise producer_error[0]
        finally:
            for task in in_flight:
                if not task.done():
                    task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
            in_flight.clear()
            if not producer_task.done():
                producer_task.cancel()
            await asyncio.gather(producer_task, return_exceptions=True)
            for key in seen_keys:
                self._dispatches.pop(key, None)

    def scheduled(self) -> List[dict]:
        return [dict(e) for e in self._scheduled]
