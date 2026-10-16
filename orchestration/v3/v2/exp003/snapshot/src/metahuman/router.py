"""Router: easy / hard classifier (§6.2).

The default concrete adapter lives in ``src/metahuman/adapters_impl/``.
Selection history (full rationale in research-plan §6.2):

* RouteLLM mf / sw_ranking — abandoned: depend on OpenAI embeddings;
  HKUST IP block returns 403.
* NVIDIA prompt-task-and-complexity-classifier — abandoned after §8.5
  R12 real run: training objective is task-type / 6-dim complexity
  (not "should we use the strong model"), and MATH-Hard Level 5
  prompts scored 0.20-0.35 (below default threshold) because the
  prompt text looks short/structured even when the problem is deep.
  Adapter kept for back-compat in
  ``adapters_impl.nvidia_complexity_router.NvidiaComplexityRouter``.
* RouteLLM ``bert`` (R15, current) — direct training on
  ``strong-model wins`` preference pairs (~80K Chatbot Arena), no
  OpenAI dependency at inference. See
  ``adapters_impl.routellm_bert_router.RouteLLMBertRouter``.

This file keeps only the abstract interface so the controller / tests
can target ``RouterInterface`` independently of which concrete model
backs it.
"""
from __future__ import annotations

import abc

from .types import RouteDecision, UserUtterance


class RouterInterface(abc.ABC):
    @abc.abstractmethod
    async def classify(self, utterance: UserUtterance) -> RouteDecision:
        ...
