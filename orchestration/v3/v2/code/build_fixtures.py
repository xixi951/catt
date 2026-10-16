"""EXP-20261005-001: build the confirmation and development fixtures (BRIEF sections 5.2, 5.3).

Model-free data processing on newsrv (GPQA Arrow caches live there). Inputs are read-only:
EXP-20260930-003's recovered fixtures and fetch script (copied into this package under exp003/),
and the Wanfq/gpqa Arrow files in the HF cache.

Outputs (OUT_DIR):
  confirm_gpqa108.jsonl       GPQA Diamond rows not in the test set, test-set formatting
  dev_pool.jsonl              every development-eligible item (AIME 2024/2025, GPQA main-not-diamond,
                              calibration MMSU and AlpacaEval minus test overlaps)
  dev_A.jsonl, dev_B.jsonl    disjoint seeded slices for tuning and the final development check
  overlap_audit.json          exact and normalized overlap counts (all must be zero)

Usage: python build_fixtures.py PKG_ROOT GPQA_CACHE_DIR OUT_DIR
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PKG = Path(sys.argv[1])
GPQA = Path(sys.argv[2])
OUT = Path(sys.argv[3])
SEED = 20261006
sys.path.insert(0, str(PKG / 'exp003' / 'snapshot' / 'scripts'))
import fetch_paper_main_questions as F  # noqa: E402
from datasets import Dataset  # noqa: E402

DATA = PKG / 'exp003' / 'snapshot' / 'scripts' / 'data'
TEST_SHA = '3a1cdac1162a9b236b62222b453564955305be1f7f02cbba8427defd506f5276'


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows, path):
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(text: str) -> str:
    """Lowercase, drop answer-format suffixes and option lines' letters, collapse whitespace."""
    t = text.lower()
    t = re.sub(r'please reason step by step.*$', '', t, flags=re.S)
    t = re.sub(r'remember to put your answer inside \\boxed\{\}\.?', '', t)
    t = re.sub(r'what is the answer to the above multiple choice question\?.*$', '', t, flags=re.S)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def stem_norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower()).strip()


def load_gpqa(name):
    files = list((GPQA / name).glob('0.0.0/*/gpqa-train.arrow'))
    assert len(files) == 1, files
    return Dataset.from_file(str(files[0]))


def gpqa_rows(ds, records, source, id_prefix, rng):
    rows = []
    for idx in records:
        row = ds[idx]
        stem = row['Question'].strip()
        correct = str(row['Correct Answer']).strip()
        incorrects = [str(row[f'Incorrect Answer {i}']).strip() for i in (1, 2, 3)]
        formatted, letter = F._gpqa_format_mcq(stem, correct, incorrects, rng)
        domain = row['High-level domain']
        rows.append({'id': f"{id_prefix}/{domain.lower()}/{row['Record ID']}", 'source': source,
                     'subject': domain, 'difficulty_hint': 'graduate', 'question': formatted,
                     'answer': letter, 'raw_problem': stem})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test_path = DATA / 'router_sweep_questions_paper_main.jsonl'
    assert hashlib.sha256(test_path.read_bytes()).hexdigest() == TEST_SHA, 'test fixture hash mismatch'
    test = read_jsonl(test_path)
    train = read_jsonl(DATA / 'router_sweep_questions_paper_train.jsonl')
    test_gpqa_ids = {r['id'].rsplit('/', 1)[1] for r in test if r['source'] == 'gpqa_diamond'}
    assert len(test_gpqa_ids) == 90

    diamond = load_gpqa('gpqa_diamond')
    main_split = load_gpqa('gpqa_main')
    diamond_ids = {r['Record ID'] for r in diamond}
    diamond_stems = {stem_norm(r['Question']) for r in diamond}

    # Confirmation set: Diamond rows not in the test set, Record-ID order, fresh seeded option shuffle.
    conf_idx = sorted((i for i, r in enumerate(diamond) if r['Record ID'] not in test_gpqa_ids),
                      key=lambda i: diamond[i]['Record ID'])
    confirm = gpqa_rows(diamond, conf_idx, 'gpqa_diamond', 'gpqa_diamond', random.Random(SEED))
    assert len(confirm) == 108, len(confirm)

    # Development GPQA: main split rows that are not Diamond rows (by Record ID and by stem text).
    dev_g_idx = sorted((i for i, r in enumerate(main_split)
                        if r['Record ID'] not in diamond_ids and stem_norm(r['Question']) not in diamond_stems),
                       key=lambda i: main_split[i]['Record ID'])
    dev_gpqa = gpqa_rows(main_split, dev_g_idx, 'gpqa_main', 'gpqa_main', random.Random(SEED + 1))

    test_norm = {norm(r['question']) for r in test}
    confirm_norm = {norm(r['question']) for r in confirm}
    excluded = []
    dev_other = []
    for r in train:
        if r['source'] == 'gpqa_diamond':
            excluded.append((r['id'], 'calibration GPQA is Diamond (could overlap confirmation)'))
            continue
        if norm(r['question']) in test_norm or norm(r['question']) in confirm_norm:
            excluded.append((r['id'], 'overlaps test/confirmation text'))
            continue
        row = dict(r)
        row['id'] = 'dev/' + r['id']
        dev_other.append(row)
    pool = dev_other + dev_gpqa
    # Within-pool duplicates (e.g. reworded AIME 2024 duplicates): keep the first of each normalized text.
    seen, dedup = set(), []
    for r in pool:
        key = norm(r['question'])
        if key in seen:
            excluded.append((r['id'], 'duplicate within development pool'))
            continue
        seen.add(key)
        dedup.append(r)
    pool = dedup
    write_jsonl(pool, OUT / 'dev_pool.jsonl')

    rng = random.Random(SEED + 2)
    by_src = {}
    for r in pool:
        by_src.setdefault(r['source'], []).append(r)
    for rows in by_src.values():
        rng.shuffle(rows)
    take = {'A': {'aime_2025': 2, 'gpqa_main': 8, 'voicebench_mmsu': 6, 'voicebench_alpaca_eval': 4},
            'B': {'aime_2025': 3, 'gpqa_main': 10, 'voicebench_mmsu': 8, 'voicebench_alpaca_eval': 7}}
    slices = {'A': [], 'B': []}
    for src, rows in by_src.items():
        offset = 0
        for name in ('A', 'B'):
            n = take[name].get(src, 0)
            slices[name] += rows[offset:offset + n]
            offset += n
    hashes = {'confirm_gpqa108.jsonl': write_jsonl(confirm, OUT / 'confirm_gpqa108.jsonl')}
    for name, rows in slices.items():
        rows.sort(key=lambda r: r['id'])
        hashes[f'dev_{name}.jsonl'] = write_jsonl(rows, OUT / f'dev_{name}.jsonl')
    hashes['dev_pool.jsonl'] = hashlib.sha256((OUT / 'dev_pool.jsonl').read_bytes()).hexdigest()

    all_norm = {name: {norm(r['question']) for r in rows} for name, rows in
                (('test', test), ('confirm', confirm), ('dev_pool', pool))}
    ids = {name: {r['id'] for r in rows} for name, rows in (('test', test), ('confirm', confirm), ('dev_pool', pool))}
    audit = {
        'seed': SEED,
        'counts': {'test': len(test), 'confirm': len(confirm), 'dev_pool': len(pool),
                   'dev_pool_by_source': Counter(r['source'] for r in pool),
                   'dev_A': Counter(r['source'] for r in slices['A']),
                   'dev_B': Counter(r['source'] for r in slices['B'])},
        'overlap_normalized_text': {
            'test_confirm': len(all_norm['test'] & all_norm['confirm']),
            'test_dev': len(all_norm['test'] & all_norm['dev_pool']),
            'confirm_dev': len(all_norm['confirm'] & all_norm['dev_pool'])},
        'overlap_ids': {'test_confirm': len(ids['test'] & ids['confirm']),
                        'test_dev': len(ids['test'] & ids['dev_pool']),
                        'confirm_dev': len(ids['confirm'] & ids['dev_pool'])},
        'dev_gpqa_record_ids_in_diamond': sum(r['id'].rsplit('/', 1)[1] in diamond_ids for r in dev_gpqa),
        'excluded': excluded,
        'sha256': hashes,
        'gpqa_arrow_sha256': {n: hashlib.sha256(next((GPQA / n).glob('0.0.0/*/gpqa-train.arrow')).read_bytes()).hexdigest()
                              for n in ('gpqa_diamond', 'gpqa_main')},
    }
    (OUT / 'overlap_audit.json').write_text(json.dumps(audit, indent=1, default=dict) + '\n')
    bad = [k for k, v in {**audit['overlap_normalized_text'], **audit['overlap_ids']}.items() if v]
    print(json.dumps({'counts': audit['counts'], 'overlaps_nonzero': bad,
                      'excluded': len(excluded)}, indent=1, default=dict))
    assert not bad and audit['dev_gpqa_record_ids_in_diamond'] == 0, 'overlap found'


if __name__ == '__main__':
    main()
