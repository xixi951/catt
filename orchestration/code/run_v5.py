"""EXP-20261009-001 runner: CATT v4 (route, nvroute, allhard) vs the prompt-matched thinking and
non-thinking baselines on the two-engine deployment (BRIEF sections 3-4). Derived from EXP-20261007-001's
run_v4.py (read-only copy in pkg/v3/code); changes are marked "v4".

Every condition runs through the same LLMBundle -> Controller -> V4Bridge -> NormTTS -> Piper -> virtual
playback path. Baselines: EXP-20260930-003's matched prompt with the fixed splitter, on the thinker engine,
with the shared V3Client (budget continuation + loop guard).

Usage (newsrv, from the package root):
  python code/run_v5.py --config config_v5.json --set MAIN --out runs/<run>/MAIN
  python code/run_v5.py --config config_v5.json --set DEVD --conditions catt_v4_allhard,thinking --out runs/<run>/DEVD
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.dont_write_bytecode = True
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import catt_v4 as V4  # noqa: E402
import speech_norm as N  # noqa: E402
V3, V2, R = V4.V3, V4.V2, V4.R
import httpx  # noqa: E402
import baseline_direct_llm as B  # noqa: E402
from metahuman.controller import Controller  # noqa: E402
from metahuman.llm import LLMBundle  # noqa: E402
from metahuman.types import AudioMeta  # noqa: E402

SETS = {'MAIN': 'v3/fixtures/main.jsonl', 'DEVD': 'fixtures/dev_D.jsonl', 'DEVSMOKE': 'fixtures/dev_smoke.jsonl'}
FORMAL_SETS = ('MAIN',)
FORMAL_CONDITIONS = ('catt_v4_route', 'catt_v4_nvroute', 'catt_v4_allhard', 'thinking', 'non_thinking')
DEV_CONDITIONS = ('catt_v4_allhard_F0',)       # development only: v3 cut-in (finish the playing sentence)
CATT = ('catt_v4_route', 'catt_v4_nvroute', 'catt_v4_allhard', 'catt_v4_allhard_F0')
ROUTED = ('catt_v4_route', 'catt_v4_nvroute')
ENGINES = ('thinker', 'talker')


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_hashes() -> Dict[str, str]:
    """v4: every file of the v4 code, prompts and judges plus every file of the read-only v3 package."""
    files = sorted(list((PKG / 'code').glob('*.py')) + list((PKG / 'code').glob('*.sh'))
                   + list((PKG / 'prompts').glob('*.md')) + list((PKG / 'judges').glob('*.md')))
    files += sorted(p for p in (PKG / 'v3').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.relative_to(PKG)): sha_file(p) for p in files if p.exists()}


def build_schedule(cfg: dict, questions: List[Any], set_name: str, conditions: List[str]) -> List[dict]:
    rng = random.Random(f"{cfg['schedule_seed']}-{set_name}")
    order = [q.id for q in questions]
    rng.shuffle(order)
    schedule = []
    for qid in order:
        conds = list(conditions)
        rng.shuffle(conds)
        schedule += [{'set': set_name, 'question_id': qid, 'condition': c, 'repeat': 0} for c in conds]
    return schedule


class Shared:
    def __init__(self, cfg: dict, tokenizer: Any, http: httpx.AsyncClient, tts: Any, rpc: Any, nvrouter: Any,
                 template: str, questions: Dict[str, Any]) -> None:
        self.cfg, self.tokenizer, self.http, self.tts, self.rpc, self.nvrouter = cfg, tokenizer, http, tts, rpc, nvrouter
        self.template, self.questions = template, questions
        self.ecfg = {e: V3.engine_cfg(cfg, e) for e in ENGINES}


async def reset_prefix_caches(shared: Shared) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {}
    for e in ENGINES:
        if not shared.cfg.get('reset_prefix_cache'):
            out[e] = None
            continue
        resp = await shared.http.post(shared.ecfg[e]['server_root'] + '/reset_prefix_cache', timeout=30.0)
        out[e] = resp.status_code
    return out


async def ensure_engines(shared: Shared, max_wait: float = 600.0) -> None:
    """v4: before each attempt both engines must answer /v1/models; if they do not within max_wait seconds the
    runner exits with code 3 without recording the attempt (the pipeline restarts the engine and resumes)."""
    t0 = time.monotonic()
    while True:
        ok = True
        for e in ENGINES:
            try:
                r = await shared.http.get(shared.ecfg[e]['base_url'] + '/models', timeout=10)
                ok = ok and r.status_code == 200
            except Exception:
                ok = False
        if ok:
            return
        if time.monotonic() - t0 > max_wait:
            print(json.dumps({'engines_unreachable_s': round(time.monotonic() - t0)}), flush=True)
            raise SystemExit(3)
        await asyncio.sleep(15)


async def wait_idle(shared: Shared, timeout: float = 120.0) -> Dict[str, Any]:
    return {e: await R.wait_server_idle(shared.http, shared.ecfg[e], timeout=timeout) for e in ENGINES}


def options_for(cfg: dict, condition: str) -> V4.V4Options:
    opts = V4.V4Options(**cfg['v4_options'])
    if condition == 'catt_v4_allhard_F0':
        opts = dataclasses.replace(opts, clause_cut=False)
    return opts


async def run_attempt(shared: Shared, entry: dict, out: Path, number: int, label: str) -> dict:
    cfg = shared.cfg
    tcfg, kcfg = shared.ecfg['thinker'], shared.ecfg['talker']
    condition = entry['condition']
    question = shared.questions[entry['question_id']]
    kind = N.question_kind(question.question)                       # v4 change 4
    opts = options_for(cfg, condition)
    cell = out / f'attempt-{number:04d}'
    cell.mkdir()
    (cell / 'audio').mkdir()
    log = R.AttemptLog(cell, entry)
    norm = V4.NormTTS(shared.tts, enabled=opts.speech_norm)          # v4 change 5, every condition
    tts = R.RecordedTTS(norm, log, cell)
    model = cfg['model']
    log.emit('attempt_start', label=label, question_kind=kind, options=dataclasses.asdict(opts), **entry)
    status = 'COMPLETE'
    turn_id = f'{condition}/{question.id}'
    ws_log: Optional[Path] = None
    chunks_log: List[dict] = []
    m_before = m_after = None
    wall_start = wall_end = None
    bundle = bridge = None
    rstate = V3.V3ReasoningState()
    controller_ref: List[Any] = []
    talker_ref: List[Any] = []
    clients: Dict[str, Any] = {}
    try:
        if condition in CATT:
            if condition == 'catt_v4_route':
                router = shared.rpc
            elif condition == 'catt_v4_nvroute':
                router = shared.nvrouter
            else:
                router = V3.AlwaysHardRouter()
            parallel = opts.parallel_route and condition in ROUTED
            tparams = V3.V3TalkerParams(**cfg['talker_params'])
            cparams = V3.V3ControllerParams(**cfg['controller_params'])
            prompt_tokens = shared.tokenizer.count(shared.tokenizer.render(
                [{'role': 'user', 'content': shared.template.format(user_text=question.question)}]))
            seed_cond = 'catt_v4_allhard' if condition == 'catt_v4_allhard_F0' else condition

            def thinker_factory(ws: Any) -> Any:
                client = V3.V3Client(tcfg, log, 'thinker', shared.tokenizer, shared.http, ('thinking', question.id),
                                     seed_role='baseline', on_budget='continue')
                clients['thinker'] = client
                return V3.V3Thinker(ws, template=shared.template, model=model, budget=cfg['reasoning_budget'],
                                    client=client, state=rstate)

            def talker_factory(ws: Any) -> Any:
                client = V3.V3Client(kcfg, log, 'talker', shared.tokenizer, shared.http, (seed_cond, question.id))
                t = V4.V4Talker(ws, model=model, client=client, state=rstate, params=tparams,
                                thinker_prompt_tokens=prompt_tokens, opts=opts, kind=kind)
                talker_ref.append(t)
                return t

            def controller_factory(ws: Any, router_: Any, thinker: Any, talker: Any) -> Controller:
                c = V4.V4Controller(ws, router_, thinker, talker, state=rstate, params=cparams, parallel=parallel)
                controller_ref.append(c)
                return c
        elif condition in ('non_thinking', 'thinking'):
            router = R.SingleCallRouter()
            thinking = condition == 'thinking'

            def thinker_factory(ws: Any) -> Any:
                return R.UnusedThinker(ws, prompts_dir=R.SNAPSHOT / 'prompts', thinking_model=model,
                                       llm_client=R.BudgetedClient(tcfg, log, 'unused', shared.tokenizer, shared.http,
                                                                   (condition, question.id)))

            def talker_factory(ws: Any) -> Any:
                client = V3.V3Client(tcfg, log, 'baseline', shared.tokenizer, shared.http, (condition, question.id))
                clients['baseline'] = client
                return V4.V4BaselineTalker(ws, template=shared.template, thinking=thinking,
                                           budget=cfg['reasoning_budget'], model=model, client=client, opts=opts)

            def controller_factory(ws: Any, router_: Any, thinker: Any, talker: Any) -> Controller:
                c = Controller(ws, router_, thinker, talker,
                               mid_trigger_tts_backlog_seconds=cfg['controller_params']['backlog_seconds'],
                               mid_trigger_min_backlog=1, mid_trigger_cooldown=0.0,
                               enable_hard_first_bridge=True, enable_hard_mid=True)
                controller_ref.append(c)
                return c
        else:
            raise ValueError(condition)

        bundle = LLMBundle(router, thinker_factory, talker_factory, controller_factory, log_dir=cell)
        bundle.begin_dialog(re.sub(r'[^A-Za-z0-9_.-]', '_', f'{condition}-{question.id}'))
        await tts.async_synthesize_text(' ', time.monotonic())  # TTS warm-up, as in EXP-20260930-003
        tts.reset()
        log.emit('prefix_cache_reset', status=await reset_prefix_caches(shared))
        m_before = await wait_idle(shared)
        bridge = V4.V4Bridge(bundle, tts)
        bridge.log = log
        if condition in CATT:
            controller_ref[0].set_preempt(bridge.preempt)
        await bridge.start()
        anchor = time.monotonic()
        wall_start = time.time()
        log.anchor = anchor
        log.emit('turn_anchor', turn_id=turn_id, anchor_wall=wall_start)
        chunks = bundle.handle_user_text(turn_id=turn_id, text=question.question,
                                         audio_meta=AudioMeta(duration_s=0.0, sample_rate=16000, started_at=0.0, finished_at=0.0),
                                         turn_anchor_monotonic=anchor, turn_anchor_walltime=wall_start)

        def mode_of(seq: int) -> Optional[str]:
            c = controller_ref[0] if controller_ref else None
            if c is not None and hasattr(c, 'seq_mode'):
                return c.seq_mode().get(seq)
            return 'easy'  # baselines: one easy (single-call) dispatch

        bridge.is_terminal_seq = lambda seq: mode_of(seq) in ('easy', 'hard_final')

        async def consume() -> None:
            async for chunk in bridge.run(chunks, turn_anchor_monotonic=anchor):
                row = {'seq': chunk.talker_seg_seq, 'mode': mode_of(chunk.talker_seg_seq), 'text': chunk.text,
                       'is_last': chunk.is_last_in_segment, 'produced_at': chunk.produced_at,
                       'yielded_at': time.monotonic() - anchor}
                if chunk.text:
                    row['chunk_id'] = bridge.last_chunk_id
                    if bridge.last_dropped:
                        row['dropped'] = bridge.last_dropped
                    else:
                        row.update(synth_ready_at=bridge.last_synth[0], duration_s=bridge.last_synth[1],
                                   playback_start=bridge.last_window.started_at, playback_end=bridge.last_window.finished_at)
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
        scheduled = bridge.scheduled() if bridge is not None else []
        if bridge is not None:
            await bridge.stop()
        if bundle is not None:
            ws_log = bundle.workspace_log_path()
            await bundle.end_dialog()
        try:
            m_after = await wait_idle(shared, timeout=180.0)
        except Exception as exc:
            log.emit('server_not_idle_after_attempt', error=str(exc))
        norm_counts = dict(norm.counts)
        tts.reset()
    dropped_ids = {e['chunk_id'] for e in scheduled if e.get('dropped')}
    for row in chunks_log:
        if row.get('chunk_id') in dropped_ids and not row.get('dropped'):
            row['dropped'] = 'preempted_queued'
    played = sorted([c for c in chunks_log if c.get('text') and not c.get('dropped')], key=lambda c: c['playback_start'])
    segment_text, segments = ('', [])
    if ws_log is not None and ws_log.exists():
        segment_text, segments = R.spoken_text_from_ws(ws_log, turn_id)
    spoken = ' '.join(c['text'] for c in played)
    if status == 'COMPLETE':
        if not any(s[1] in ('easy', 'hard_final') for s in segments):
            log.errors.append('terminal (easy/hard_final) talker segment absent')
        if not played:
            log.errors.append('no spoken chunk')
        if log.errors:
            status = 'FAILED'
    calls = log.calls
    ctrl = controller_ref[0] if controller_ref else None
    talker = talker_ref[0] if talker_ref else None
    think_client = clients.get('thinker') or (clients.get('baseline') if condition == 'thinking' else None)
    reasoning_text = rstate.text if condition in CATT else (think_client.last_reasoning if think_client else '')
    spec = getattr(ctrl, 'speculative', None)
    routed_easy = bool(spec and spec.get('route') == 'easy')
    summary = {
        **entry, 'attempt': number, 'label': label, 'status': status, 'errors': log.errors, 'turn_id': turn_id,
        'kind': kind, 'options': dataclasses.asdict(opts),
        'ws_log': ws_log.name if ws_log else None, 'wall_start': wall_start, 'wall_end': wall_end,
        'spoken_text': spoken, 'segment_text': segment_text,
        'segments': [{'seq': s[0], 'mode': s[1], 'text': s[2]} for s in segments],
        'played_chunks': [dict({k: c.get(k) for k in ('chunk_id', 'seq', 'mode', 'text', 'playback_start', 'playback_end',
                                                        'synth_ready_at', 'duration_s')},
                               spoken=(N.spoken(c['text']) if opts.speech_norm else c['text'])) for c in played],
        'dropped_chunks': [{k: c.get(k) for k in ('chunk_id', 'seq', 'mode', 'text', 'dropped')}
                           for c in chunks_log if c.get('dropped')],
        'seq_mode': (ctrl.seq_mode() if ctrl is not None and hasattr(ctrl, 'seq_mode') else None),
        'preemption': getattr(ctrl, 'preemption', None),
        'speculative': spec,
        'trigger_counts': getattr(ctrl, 'trigger_counts', None),
        'bridge_plan': getattr(talker, 'bridge_plan', None),
        'prompt_variant': getattr(talker, 'prompt_variant', None),
        'final_inputs': getattr(talker, 'final_inputs', None),
        'bridge_guard_fired_v4': getattr(talker, 'bridge_guard_fired', None),
        'bridge_gate_wait_s': getattr(talker, 'bridge_gate_wait_s', None),
        'norm_counts': norm_counts,
        'reasoning': ({'tokens': rstate.ntok, 'chars': len(rstate.text), 'stop_reason': rstate.stop_reason,
                       'forced': rstate.forced, 'force_reason': rstate.force_reason,
                       'forced_answer_chars': len(rstate.forced_answer), 'cancelled_by_easy_route': routed_easy}
                      if condition in CATT else None),
        'reasoning_sha256': (R.sha_bytes(reasoning_text.encode()) if (reasoning_text and not routed_easy) else None),
        'reasoning_chars': len(reasoning_text),
        'continuation_sha256': (R.sha_bytes(think_client.last_continuation.encode())
                                if think_client is not None and think_client.last_forced and not routed_easy else None),
        'llm_calls': len(calls),
        'llm_calls_by_role': {r: sum(1 for c in calls.values() if c['role'] == r) for r in sorted({c['role'] for c in calls.values()})},
        'llm_calls_by_engine': {e: sum(1 for c in calls.values() if c.get('engine') == e) for e in ENGINES},
        'tts_calls': max(0, tts.submitted - 1), 'peak_concurrent_llm_calls': log.peak_calls,
        'budget_forced_calls': sum(1 for c in calls.values() if c.get('budget_forced')), 'calls': calls,
        'server_metrics_before': m_before, 'server_metrics_after': m_after,
        'first_playback_start': played[0]['playback_start'] if played else None,
        'last_playback_end': max((c['playback_end'] for c in played), default=None),
        'n_chunks': len(played),
    }
    log.emit('attempt_end', status=status, errors=log.errors)
    log.close()
    (cell / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
    return summary


async def warmup(shared: Shared, out: Path, label: str) -> None:
    """Fixed non-fixture prompts on both engines before a set; labelled warm-up, never outcomes."""
    cell = out / f'warmup-{label}-{int(time.time())}'
    cell.mkdir()
    log = R.AttemptLog(cell, {'warmup': label})
    for e in ENGINES:
        for i, prompt in enumerate(R.WARMUP_PROMPTS):
            for thinking in (False, True):
                client = V3.V3Client(shared.ecfg[e], log, f'warmup{i}', shared.tokenizer, shared.http, ('warmup', label, e))
                async for _ in client.stream_chat(model=shared.cfg['model'], messages=[{'role': 'user', 'content': prompt}],
                                                  enable_thinking=thinking,
                                                  thinking_budget=shared.cfg['reasoning_budget'] if thinking else None,
                                                  max_tokens=300 if thinking else None):
                    pass
    log.close()


async def main_async(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    conditions = args.conditions.split(',')
    for c in conditions:
        if c not in FORMAL_CONDITIONS + DEV_CONDITIONS:
            raise ValueError(f'unknown condition {c}')
    fixture = PKG / SETS[args.set]
    expected = cfg['fixture_sha256'][args.set]
    if sha_file(fixture) != expected:
        raise ValueError(f'fixture hash mismatch for {args.set}')
    questions = B.load_fixture(fixture)
    if args.ids:
        keep = {l.strip() for l in open(args.ids) if l.strip()}
        questions = [q for q in questions if q.id in keep]
    schedule = build_schedule(cfg, questions, args.set, conditions)
    formal = args.set in FORMAL_SETS
    label = f'SET_{args.set}' if formal else f'DEV_{args.set}'
    fixture_info = {'path': SETS[args.set], 'sha256': expected, 'ids_filter': args.ids}
    if formal:
        if tuple(sorted(conditions)) != tuple(sorted(FORMAL_CONDITIONS)) or args.ids or args.limit:
            raise ValueError('the formal set runs exactly the five formal conditions on every item')
        freeze = json.loads((PKG.parent / 'freeze' / 'freeze_record.json').read_text())
        current = package_hashes()
        diff = {k: (freeze['package_sha256'].get(k), v) for k, v in current.items() if freeze['package_sha256'].get(k) != v}
        missing = [k for k in freeze['package_sha256'] if k not in current]
        if diff or missing or freeze['config_sha256'] != sha_file(Path(args.config)):
            raise ValueError(f'formal run refused: package differs from freeze record: {diff} {missing}')
        if freeze['schedule_sha256'] != R.sha_bytes(R.canonical(schedule)):
            raise ValueError('formal run refused: schedule differs from freeze record')
    if args.limit:
        schedule = schedule[:args.limit]
    qmap = {q.id: q for q in questions}
    template = (PKG / 'v3' / 'v2' / 'exp003' / 'prompts_v2' / 'baseline_matched.md').read_text(encoding='utf-8')
    from metahuman.adapters_impl.sherpa_onnx_tts import SherpaOnnxTTS
    from metahuman.adapters_impl.router_rpc_client import RouterRPCClient
    tts = SherpaOnnxTTS(model_dir=Path(cfg['tts_model_dir']))
    rpc = RouterRPCClient(uds_path=cfg['router_uds'], hard_threshold=cfg['router_threshold']) if 'catt_v4_route' in conditions else None
    nvrouter = None
    if 'catt_v4_nvroute' in conditions:
        nv = cfg['nvroute']
        nvrouter = V4.NvRouter(V4.NvScorer(Path(nv['ckpt']), Path(nv['deberta_cfg']), threads=nv.get('threads', 4)), nv['tau'])
    tokenizer = R.Tokenizer(cfg['weights_path'])
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=0)
    async with httpx.AsyncClient(limits=limits) as http:
        shared = Shared(cfg, tokenizer, http, tts, rpc, nvrouter, template, qmap)
        freeze_path = out / 'freeze.json'
        attempts_path = out / 'attempts.jsonl'
        done = {row['attempt']: row for row in R.read_jsonl(attempts_path)} if attempts_path.exists() else {}
        runner_sha = sha_file(Path(__file__))
        if not freeze_path.exists():
            servers = {}
            for e in ENGINES:
                ec = shared.ecfg[e]
                servers[e] = {'models': (await http.get(ec['base_url'] + '/models', timeout=10)).json(),
                              'version': (await http.get(ec['server_root'] + '/version', timeout=10)).json()}
            freeze = {'evidence_kind': label, 'frozen_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
                      'config_sha256': sha_file(Path(args.config)), 'fixture': fixture_info, 'conditions': conditions,
                      'schedule': schedule, 'schedule_sha256': R.sha_bytes(R.canonical(schedule)),
                      'runner_sha256': runner_sha, 'package_sha256': package_hashes(), 'servers': servers,
                      'playback': 'virtual FIFO from synthesized durations; preempted progress chunks dropped; no device playback'}
            freeze_path.write_text(json.dumps(freeze, indent=1) + '\n')
        else:
            frozen = json.loads(freeze_path.read_text())
            if frozen['schedule_sha256'] != R.sha_bytes(R.canonical(schedule)):
                raise ValueError('resume refused: schedule differs from freeze.json')
            if frozen['runner_sha256'] != runner_sha or frozen['package_sha256'] != package_hashes():
                raise ValueError('resume refused: code changed since freeze.json')
        await warmup(shared, out, label)
        for number, entry in enumerate(schedule):
            if number in done:
                continue
            stale = out / f'attempt-{number:04d}'
            if stale.exists():
                k = 1
                while (out / f'attempt-{number:04d}.interrupted-{k}').exists():
                    k += 1
                stale.rename(out / f'attempt-{number:04d}.interrupted-{k}')
                entry = dict(entry, rerun_after_interruption=k)
            await ensure_engines(shared)
            summary = await run_attempt(shared, entry, out, number, label)
            row = {key: summary.get(key) for key in (
                'set', 'question_id', 'condition', 'repeat', 'attempt', 'status', 'errors', 'llm_calls', 'tts_calls',
                'peak_concurrent_llm_calls', 'budget_forced_calls', 'wall_start', 'wall_end', 'rerun_after_interruption')}
            with attempts_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            secs = (row['wall_end'] or 0) - (row['wall_start'] or row['wall_end'] or 0)
            print(json.dumps({'attempt': number, 'condition': row['condition'], 'q': row['question_id'],
                              'status': row['status'], 'calls': row['llm_calls'], 'secs': round(secs, 1)}), flush=True)
    if rpc is not None:
        close = getattr(rpc, 'close', None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--set', choices=sorted(SETS), required=True)
    parser.add_argument('--conditions', default=','.join(FORMAL_CONDITIONS))
    parser.add_argument('--ids', default=None, help='development only: run only these item ids')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
