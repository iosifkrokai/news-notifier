import asyncio
import json
import random

import httpx

from app.config import get_settings

# arq only retries a job if it explicitly raises arq.worker.Retry — any other
# exception (e.g. a plain HTTPStatusError) just fails the job outright, so
# transient/rate-limit errors from OpenRouter must be retried here, not left
# to the queue.
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3


class OpenRouterClient:
    """Thin async wrapper over OpenRouter's OpenAI-compatible HTTP API."""

    async def generate_json(self, model: str, system: str, prompt: str, schema: dict, name: str) -> dict:
        settings = get_settings()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
        async with httpx.AsyncClient(base_url=settings.openrouter_base_url, timeout=120) as client:
            resp = await self._post_with_retry(client, "/chat/completions", payload, headers)
        return json.loads(resp.json()["choices"][0]["message"]["content"])

    async def _post_with_retry(
        self, client: httpx.AsyncClient, path: str, payload: dict, headers: dict
    ) -> httpx.Response:
        for attempt in range(_MAX_RETRIES + 1):
            resp = await client.post(path, json=payload, headers=headers)
            if resp.status_code not in _RETRYABLE_STATUSES or attempt == _MAX_RETRIES:
                resp.raise_for_status()
                return resp
            await asyncio.sleep(_retry_delay(resp, attempt))


def _retry_delay(resp: httpx.Response, attempt: int) -> float:
    retry_after = resp.headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return (2**attempt) + random.random()
