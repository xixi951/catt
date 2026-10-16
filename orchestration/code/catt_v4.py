"""EXP-20261009-001: CATT v4 as experiment-local subclasses of EXP-20261007-001's CATT v3.

EXP-20261007-001's package is copied read-only to ``pkg/v3`` and never modified. BRIEF section 3.1 lists
the changes; every change has a switch in ``V4Options`` so the development rules can fall back to the v3
behaviour of a single change.

    V4Options      switches of changes 1, 3, 4, 5 (change 2 is a deployment setting, change 7 a router)
    V4Talker       fixed splitter and bridge guard (5); clause chunks for progress speech (1);
                   open-ended final and easy prompts (4); records the final prompt inputs (replay)
    V4BaselineTalker  the matched baselines' talker with the fixed splitter (5)
    V4Controller   routing in parallel with the thinker and the bridge; cancel both on an easy route (3)
    V4Bridge       V2Bridge whose preempt() merges kept clause chunks back into sentences
    NormTTS        spoken-form normalizer in front of Piper, every condition (5)
    NvRouter       NVIDIA prompt-task-and-complexity classifier on CPU as the router (7)
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'v3' / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import catt_v3 as V3  # noqa: E402  (puts v2, EXP-20260930-003's snapshot and run_v2 on sys.path)
import speech_norm as N  # noqa: E402

V2, R = V3.V2, V3.R
from metahuman.types import EndOfThinking, RouteDecision, SessionState, TalkerMode, UserUtterance  # noqa: E402

PROMPTS_V4 = PKG / 'prompts'


@dataclass
class V4Options:
    clause_cut: bool = True          # change 1: progress speech in clause chunks (F1); False = v3 (F0)
    parallel_route: bool = True      # change 3 (route and nvroute configurations only)
    open_final: bool = True          # change 4: open-ended final prompt (O1); False = v3 final prompt (O0)
    easy_open: bool = True           # change 4: open-ended easy prompt (E1o); False = v3 E1
    fix_splitter: bool = True        # change 5
    fix_bridge_guard: bool = True    # change 5
    speech_norm: bool = True         # change 5 (applies to every condition)


# ---------------------------------------------------------------------------
# Bridge guard (change 5): v3 matched the topic phrase "identifying the correct answer choice". The fixed
# guard only fires on an answer *statement*.
# ---------------------------------------------------------------------------

_BRIDGE_ANSWER_RE_V4 = re.compile(
    r"\banswers?\s+(?:is|are|was|would\s+be|should\s+be|must\s+be|seems?\s+to\s+be|might\s+be|may\s+be|could\s+be|looks?\s+like)\b"
    r"|\banswer\s*:|\\boxed"
    r"|\boption\s+\(?[A-J]\)?\s+(?:is|seems|looks|would\s+be|must\s+be)\b"
    r"|\b(?:correct|right|best)\s+(?:answer|option|choice)\s+(?:is|would\s+be|seems|looks|appears|must\s+be|should\s+be)\b"
    r"|\b\(?[A-J]\)?\s+is\s+(?:the\s+)?(?:correct|right|best)\b", re.IGNORECASE)


def bridge_states_answer_v4(sentence: str) -> bool:
    return bool(_BRIDGE_ANSWER_RE_V4.search(sentence))


# ---------------------------------------------------------------------------
# Talker
# ---------------------------------------------------------------------------


class V4Talker(V3.V3Talker):
    _V4_FILES = {'final_open': 'talker_final_open.md', 'easy_open': 'talker_easy_E1open.md'}

    def __init__(self, ws: Any, *, model: str, client: Any, state: Any, params: Any, thinker_prompt_tokens: int,
                 opts: V4Options, kind: str) -> None:
        super().__init__(ws, model=model, client=client, state=state, params=params,
                         thinker_prompt_tokens=thinker_prompt_tokens)
        self._opts = opts
        self._kind = kind
        self._v4 = {k: (PROMPTS_V4 / f).read_text(encoding='utf-8') for k, f in self._V4_FILES.items()}
        self.final_inputs: Optional[dict] = None
        self.prompt_variant: Dict[str, str] = {}
        self.bridge_guard_fired = 0
        self.route_gate: Optional[asyncio.Event] = None   # change 3: bridge text is held until the route is known
        self.bridge_gate_wait_s: Optional[float] = None

    # change 5: splitter
    def _split_complete_sentences(self, text: str, *, accept_comma: bool = False):
        if self._opts.fix_splitter:
            return N.split_sentences(text, accept_comma=accept_comma)
        return super()._split_complete_sentences(text, accept_comma=accept_comma)

    # change 4: prompts by question kind; record final inputs for the development replay
    def _build_v2_prompt(self, signal: Any, mode: TalkerMode) -> str:
        st = self._state
        user_text = self._ws.current_user_text(signal.turn_id) or ''
        if mode == TalkerMode.EASY:
            if self._kind == 'open' and self._opts.easy_open:
                self.prompt_variant['easy'] = 'E1open'
                return self._v4['easy_open'].format(user_text=user_text, prior_dialog='')
            self.prompt_variant['easy'] = 'E1'
            return super()._build_v2_prompt(signal, mode)
        if mode == TalkerMode.HARD_FINAL:
            forced_written = st.forced and st.forced_answer.strip()
            spoken = '\n'.join(st.kept_spoken) if st.kept_spoken else self._spoken_so_far(signal.turn_id)
            status = ('The reasoning about this question was cut off by its time limit before it finished; '
                      'choose the single most likely answer from what it established.' if st.forced
                      else 'The reasoning about this question has finished.')
            self.final_inputs = {'user_text': user_text, 'status_line': status, 'forced': bool(st.forced),
                                 'forced_answer': st.forced_answer if forced_written else '',
                                 'reasoning_tail': st.text[-self._p.final_tail_chars:].strip() or '(none)',
                                 'spoken_so_far': spoken or '(nothing yet)', 'kind': self._kind}
            if self._kind == 'open' and self._opts.open_final and not forced_written:
                self.prompt_variant['final'] = 'O1'
                return self._v4['final_open'].format(status_line=status, user_text=user_text,
                                                     reasoning_tail=self.final_inputs['reasoning_tail'],
                                                     spoken_so_far=self.final_inputs['spoken_so_far'])
            self.prompt_variant['final'] = 'forced' if forced_written else 'O0'
        return super()._build_v2_prompt(signal, mode)

    # change 5: bridge guard (v3 cap logic with the fixed answer pattern)
    def _bridge_cap(self, sentence: str, emitted: int):
        if not self._opts.fix_bridge_guard:
            return super()._bridge_cap(sentence, emitted)
        text = sentence.strip()
        if not text:
            return '', False
        if bridge_states_answer_v4(text):
            self.bridge_guard_fired += 1
            self._ws.record_event('v3_bridge_sentence_suppressed', {'turn_id': self._bridge_turn, 'text': text,
                                                                    'reason': 'answer_statement', 'guard': 'v4'})
            return '', True
        complete = text[-1:] in '.!?。！？;；\n'
        words = V3._WORD.findall(text)
        room = self._bridge_words_cap - self._bridge_words
        if len(words) > room:
            if room <= 0:
                return '', True
            count, cut = 0, len(text)
            for m in V3._WORD.finditer(text):
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

    # change 1: progress speech in clause chunks; change 3: a speculative bridge is held until the route is known
    async def dispatch(self, signal: Any, mode: TalkerMode) -> AsyncIterator[str]:
        gate = self.route_gate if mode == TalkerMode.HARD_FIRST else None
        async for text in super().dispatch(signal, mode):
            if gate is not None and not gate.is_set():
                t0 = time.monotonic()
                await gate.wait()
                self.bridge_gate_wait_s = time.monotonic() - t0
            if mode == TalkerMode.HARD_MID and self._opts.clause_cut:
                for clause in N.split_clauses(text):
                    yield clause
            else:
                yield text


class V4BaselineTalker(R.MatchedBaselineTalker):
    """The prompt-matched baselines (thinking / non-thinking) with the fixed splitter (change 5)."""

    def __init__(self, *a: Any, opts: V4Options, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self._opts = opts

    def _split_complete_sentences(self, text: str, *, accept_comma: bool = False):
        if self._opts.fix_splitter:
            return N.split_sentences(text, accept_comma=accept_comma)
        return super()._split_complete_sentences(text, accept_comma=accept_comma)


# ---------------------------------------------------------------------------
# Controller (change 3)
# ---------------------------------------------------------------------------


class V4Controller(V3.V3Controller):
    """V3Controller; with ``parallel=True`` the thinker and the bridge start together with the router call.
    The bridge's text is held back until the route is known, so no bridge audio plays before an easy answer.
    A hard route releases the bridge and confirms the thinker. An easy route cancels the thinker request (the
    stream is closed, vLLM aborts it) and the bridge call, and dispatches the easy answer. Progress updates are
    held until the route is known; an end of reasoning that arrives first is processed after a hard route."""

    def __init__(self, *a: Any, parallel: bool = False, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self._parallel = parallel
        self._route_pending = False
        self._pending_eot: Optional[EndOfThinking] = None
        self._spec_easy = False
        self.speculative: Optional[dict] = None

    async def _handle_user_utterance(self, u: UserUtterance) -> None:
        await super()._handle_user_utterance(u)
        if not self._parallel or u.turn_id in self._cancelled_turns or self._active_turn != u.turn_id:
            return
        self._route_pending = True
        self._route_gate = asyncio.Event()
        if hasattr(self._talker, 'route_gate'):
            self._talker.route_gate = self._route_gate

        def to_first(w: Any) -> None:
            w.state = SessionState.HARD_FIRST

        self._ws.mutate(to_first)
        self._send_talker(u.turn_id, cause='speculative_hard_first', snapshot_seq=0)
        prompt = self._ws.current_user_text(u.turn_id) or ''
        self._thinker_task = asyncio.create_task(self._run_thinker(u.turn_id, prompt))
        self.speculative = {'started_ts': self._ws.now(u.turn_id)}
        self._ws.record_event('v4_speculative_start', {'turn_id': u.turn_id, 'ts': self._ws.now(u.turn_id)})

    async def _handle_route_decision(self, r: RouteDecision) -> None:
        if not self._parallel:
            await super()._handle_route_decision(r)
            return
        if r.turn_id in self._cancelled_turns or r.turn_id != self._active_turn:
            return
        self._route_pending = False
        turn = r.turn_id
        ts = self._ws.now(turn)
        if r.route == 'hard':
            self._ws.mutate(lambda w: w.route_history.append(r))
            self._route_gate.set()
            self.speculative.update(route='hard', route_ts=ts, confidence=r.confidence)
            self._ws.record_event('v4_route_confirmed', {'turn_id': turn, 'route': 'hard', 'ts': ts})
            if self._pending_eot is not None:
                eot, self._pending_eot = self._pending_eot, None
                await self._handle_end_of_thinking(eot)
            return
        # easy: cancel the speculative thinker and bridge
        self._spec_easy = True
        self._pending_eot = None
        self._thinker.cancel(turn)
        thinker_running = self._thinker_task is not None and not self._thinker_task.done()
        if thinker_running:
            self._thinker_task.cancel()
        bridge_seqs = {s for s, m in self._seq_mode.items() if m == 'hard_first'}
        cancelled = []
        for s in bridge_seqs:
            task = self._seq_task.get(s)
            if task is not None and not task.done():
                task.cancel()
                cancelled.append(s)
        kept = self._preempt_cb(turn, bridge_seqs) if self._preempt_cb is not None else []
        self._route_gate.set()   # after the cancellation, so the cancelled bridge never yields

        def to_easy(w: Any) -> None:
            w.route_history.append(r)
            w.state = SessionState.EASY

        self._ws.mutate(to_easy)
        self.speculative.update(route='easy', route_ts=ts, confidence=r.confidence, thinker_cancelled=thinker_running,
                                bridge_cancelled_in_flight=cancelled, bridge_kept_n=len(kept),
                                reasoning_tokens_at_cancel=self._rs.ntok)
        self._ws.record_event('v4_speculative_cancel', {'turn_id': turn, **self.speculative})
        self._send_talker(turn, cause='route_easy', snapshot_seq=0)

    def _should_dispatch_mid(self) -> Optional[str]:
        if self._route_pending:
            return 'route_pending'
        if self._spec_easy:
            return 'easy_route'
        return super()._should_dispatch_mid()

    async def _handle_end_of_thinking(self, eot: EndOfThinking) -> None:
        if self._spec_easy:
            return
        if self._route_pending:
            self._pending_eot = eot
            return
        await super()._handle_end_of_thinking(eot)


# ---------------------------------------------------------------------------
# Playback (change 1 bookkeeping) and TTS input (change 5)
# ---------------------------------------------------------------------------


class V4Bridge(V2.V2Bridge):
    """V2Bridge whose preempt() returns the audible texts with consecutive chunks of one dispatch joined,
    so the final talker's "already said" lists whole (possibly cut) sentences, not clause fragments."""

    def preempt(self, turn_id: Any, seqs: Set[int]) -> List[str]:
        before = {e['chunk_id'] for e in self._scheduled if e['dropped']}
        super().preempt(turn_id, seqs)
        merged: List[List[Any]] = []
        for e in self._scheduled:
            if e['turn_id'] != turn_id or e['dropped']:
                continue
            if merged and merged[-1][0] == e['seq']:
                merged[-1][1] += ' ' + e['text']
            else:
                merged.append([e['seq'], e['text']])
        self.last_preempt_dropped = [e['chunk_id'] for e in self._scheduled if e['dropped'] and e['chunk_id'] not in before]
        return [t for _, t in merged]


class NormTTS:
    """Inner TTS wrapper: synthesizes N.spoken(text) instead of text (all conditions, change 5)."""

    def __init__(self, inner: Any, enabled: bool = True) -> None:
        self.inner = inner
        self.enabled = enabled
        self.counts = {'changed': 0, 'long_formula': 0, 'math_spans': 0, 'errors': 0}
        self._lock = threading.Lock()

    def synthesize_text(self, text: str, anchor: float) -> Any:
        if self.enabled and text.strip():
            info: dict = {}
            text2 = N.spoken(text, info)
            with self._lock:
                self.counts['changed'] += int(bool(info.get('changed')))
                self.counts['long_formula'] += info.get('long_formula', 0)
                self.counts['math_spans'] += info.get('math_spans', 0)
                self.counts['errors'] += int('error' in info)
            text = text2
        return self.inner.synthesize_text(text, anchor)

    def backlog_seconds(self) -> float:
        return self.inner.backlog_seconds()

    def reset(self) -> None:
        self.inner.reset()
        with self._lock:
            self.counts = {'changed': 0, 'long_formula': 0, 'math_spans': 0, 'errors': 0}


# ---------------------------------------------------------------------------
# NVIDIA router (change 7)
# ---------------------------------------------------------------------------


class NvScorer:
    """nvidia/prompt-task-and-complexity-classifier @fea11215, model-card formula, exactly as EXP-20261002-001's
    remote_score_routers.py (padding to 512, truncation), on CPU. Thread-safe (one forward at a time)."""

    def __init__(self, ckpt: Path, deberta_cfg: Path, threads: int = 4) -> None:
        import nv_model
        import torch
        torch.set_num_threads(threads)
        self.torch = torch
        self.model, self.tok = nv_model.build_nvidia(Path(ckpt), Path(deberta_cfg))
        self._lock = threading.Lock()

    def score(self, text: str) -> float:
        with self._lock:
            enc = self.tok([text], return_tensors='pt', add_special_tokens=True, max_length=512,
                           padding='max_length', truncation=True)
            with self.torch.no_grad():
                r = self.model(enc)
            return float(r['prompt_complexity_score'][0])


class NvRouter:
    """Routes hard iff the NVIDIA complexity score >= tau_nv (threshold from calibration data only)."""

    def __init__(self, scorer: NvScorer, tau: float) -> None:
        self.scorer, self.tau = scorer, float(tau)

    async def classify(self, utterance: UserUtterance) -> RouteDecision:
        t0 = time.monotonic()
        s = await asyncio.to_thread(self.scorer.score, utterance.text)
        route = 'hard' if s >= self.tau else 'easy'
        return RouteDecision(turn_id=utterance.turn_id, route=route, confidence=s,
                             rationale=f'nvidia_complexity={s:.5f} tau_nv={self.tau:.6f} cpu_s={time.monotonic() - t0:.3f}',
                             decided_at=utterance.received_at)
