"""Minimal OpenAI-compatible chat client for the judges (standard library only).

Derived from the study's judge client: same request body, retries and back-off. The only change for the release is
where credentials come from: environment variables instead of a private credentials file.

  CATT_JUDGE_API_KEY    API key of an OpenAI-compatible endpoint that serves the judge models
  CATT_JUDGE_BASE_URL   base URL of that endpoint (with or without the trailing /v1)

The key is never printed, logged or written.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class ChatClient:
    def __init__(self, timeout: float = 600.0) -> None:
        try:
            self._key = os.environ['CATT_JUDGE_API_KEY']
            base = os.environ['CATT_JUDGE_BASE_URL']
        except KeyError as exc:
            raise SystemExit(f'set {exc.args[0]} (see evaluation/judging/README.md)')
        self._bases = [base.rstrip('/')]
        self.timeout = timeout

    def __repr__(self) -> str:  # never expose the key
        return f'ChatClient(bases={len(self._bases)})'

    def chat(self, model: str, messages: List[dict], *, max_tokens: Optional[int] = None,
             temperature: Optional[float] = None, extra: Optional[dict] = None, retries: int = 6) -> Dict[str, Any]:
        body: Dict[str, Any] = {'model': model, 'messages': messages}
        if max_tokens is not None:
            body['max_tokens'] = max_tokens
        if temperature is not None:
            body['temperature'] = temperature
        body.update(extra or {})
        data = json.dumps(body).encode()
        last = None
        for attempt in range(retries):
            base = self._bases[0]
            url = base + ('/chat/completions' if base.endswith('/v1') else '/v1/chat/completions')
            req = urllib.request.Request(url, data=data, method='POST', headers={
                'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._key})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read().decode())
                out['_elapsed_s'] = time.time() - t0
                out['_attempt'] = attempt
                return out
            except urllib.error.HTTPError as exc:
                msg = exc.read().decode(errors='replace')[:400]
                last = f'HTTP {exc.code}: {msg}'
                if exc.code in (400, 401, 403, 404):
                    break
            except Exception as exc:  # timeouts, connection resets
                last = f'{type(exc).__name__}: {str(exc)[:200]}'
            time.sleep(min(60, 3 * 2 ** attempt))
        raise RuntimeError(f'chat failed for {model}: {last}')
