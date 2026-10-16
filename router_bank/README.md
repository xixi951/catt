# Router reference bank (released without the GPQA vectors)

The CATT router (`orchestration/v3/v2/exp003/snapshot/src/metahuman/adapters_impl/sw_ranking_qwen3_router.py`)
scores a request by its embedding similarity to 761 labelled reference requests (`battles.csv`: id, which mode
won, weight) and a weighted Bradley-Terry fit. The reference vectors are released **without the 171 GPQA
(non-Diamond) rows**, because the GPQA authors ask that their questions not be published and the vectors encode
the question text. Everything else is here.

| File | Contents |
|---|---|
| `embeddings_without_gpqa.npy` | 590 x 1024 float32, L2-normalised vectors of the non-GPQA reference requests (AIME 2023 to 2025, MMLU-Pro, Alpaca, GSM8K), in bank order. |
| `rows.csv` | All 761 bank rows: row index, `prompt_id` (same order as `battles.csv`), and whether its vector is released. |
| `fetch_router_train_questions.py` | Builds the reference requests from their public sources (seed 17). |
| `build_router_artifacts.py` | Embeds the requests and writes `embeddings.npy`, `battles.csv`, `manifest.json`. |
| `assemble_bank.py` | Inserts rebuilt GPQA vectors into the released ones to form a complete `embeddings.npy`. |

The two scripts come from the project repository (sha256 `40ec01c8…c064` and `1289cba3…b434`, see `SHA256SUMS`);
the original bank is `orchestration/v3/v2/exp003/snapshot/data/router/` (`battles.csv`, `manifest.json`; the
original `embeddings.npy` had sha256 `43af71cc…d7bf`).

## What this means for reproducing routing decisions

- The scores and routes used in the study are released per item: `data/items.csv` (`router_score`,
  `route_router`) and, per turn, `data/v4/turns.csv` (`route_confidence`, `route`). The released results of the main
  study use these recorded scores and routes, so level 1 and level 2 reproduction do not need the bank.
- The released bank alone does not reproduce the router's scores: dropping 171 of 761 reference rows changes every
  similarity-weighted fit. Scores are dense near the threshold (most items lie within ±0.005 of τ*), so many routes
  would change. Use the complete bank for a rerun.

## Rebuilding the GPQA vectors (for readers with GPQA access)

1. Accept the GPQA terms and run `fetch_router_train_questions.py --seed 17` with the Hugging Face `datasets`
   library. It draws 200 GPQA main and extended rows outside Diamond from `Wanfq/gpqa` and formats each as
   `"{question}\n\nA) ...\nB) ...\nC) ...\nD) ...\n\nPlease reason step by step, then conclude your response with
   \"The answer is X.\" where X is one of A, B, C, or D."` with a seeded option order. The bank uses the 171 of them
   listed in `rows.csv` (`vector_released` = 0). The study's fixture had sha256
   `2cab778513870f281bffbfbcb38e65aee2b50f1f47b8c5bf085d606a57b63487` (`manifest.json`, `train_fixture_sha256`);
   a different dataset revision or sampling order will give different rows.
2. Embed each request as `"Instruct: {instruction}\nQuery: {question}"`, with `{instruction}` from `manifest.json`
   (`route_query_v1`), using `Qwen/Qwen3-Embedding-0.6B` (revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, the
   revision in the study server's model cache) through `sentence-transformers` with `normalize_embeddings=True` and batch size 32,
   exactly as in `build_router_artifacts.py`.
3. `python assemble_bank.py rows.csv embeddings_without_gpqa.npy gpqa_rebuilt.npy OUT/embeddings.npy` and place the
   file in the router's data directory. Vectors computed on different hardware or library versions can differ in the
   last bits; compare the resulting scores with `data/items.csv` to check.
