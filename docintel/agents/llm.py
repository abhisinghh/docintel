"""Minimal LLM client used by the agent and schema extractor.

Supports any OpenAI-compatible endpoint (vLLM / Ollama / OpenAI / OpenRouter)
and Anthropic. Only two operations are needed: ``complete`` (text → text) and
``complete_json`` (text → parsed JSON). Images can be attached so the agent
can look at a region when verifying a value.
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Optional

import httpx
from PIL import Image

from ..config import LLMConfig


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.available = cfg.provider != "none" and (cfg.provider == "transformers" or bool(cfg.api_key))
        self._local = None
        if cfg.provider == "transformers":
            from ..ocr.backends import TransformersVLM
            self._local = TransformersVLM(cfg.model, cfg.device)
        elif cfg.provider == "anthropic":
            self._client = httpx.Client(base_url="https://api.anthropic.com", timeout=cfg.timeout,
                                        headers={"x-api-key": cfg.api_key or "", "anthropic-version": "2023-06-01"})
        else:
            self._client = httpx.Client(base_url=cfg.base_url or "", timeout=cfg.timeout,
                                        headers={"Authorization": f"Bearer {cfg.api_key}"})

    # ------------------------------------------------------------------
    def complete(self, prompt: str, system: str = "", images: Optional[list[Image.Image]] = None,
                 max_tokens: Optional[int] = None) -> str:
        images = images or []
        if self._local is not None:
            from PIL import Image as _Image
            img = images[0] if images else _Image.new("RGB", (8, 8), "white")
            return self._local.chat(img, (system + "\n\n" if system else "") + prompt, max_tokens or self.cfg.max_tokens)
        if self.cfg.provider == "anthropic":
            content: list[dict[str, Any]] = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(im)}}
                for im in images
            ] + [{"type": "text", "text": prompt}]
            payload = {"model": self.cfg.model, "max_tokens": max_tokens or self.cfg.max_tokens,
                       "temperature": self.cfg.temperature, "messages": [{"role": "user", "content": content}]}
            if system:
                payload["system"] = system
            r = self._client.post("/v1/messages", json=payload)
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"])
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(im)}"}} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        payload = {"model": self.cfg.model, "messages": messages, "temperature": self.cfg.temperature,
                   "max_tokens": max_tokens or self.cfg.max_tokens}
        r = self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def complete_json(self, prompt: str, system: str = "", images: Optional[list[Image.Image]] = None) -> Any:
        raw = self.complete(prompt + "\n\nReturn ONLY valid JSON.", system=system, images=images)
        return parse_json(raw)


def parse_json(raw: str) -> Any:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
        if not m:
            raise
        return json.loads(m.group(1))


def probe(llm: LLM, timeout: float = 10.0) -> bool:
    """Return True if the endpoint answers; used to fall back to offline heuristics (short timeout)."""
    client = getattr(llm, "_client", None)
    old = getattr(client, "timeout", None)
    try:
        if client is not None and old is not None:
            client.timeout = httpx.Timeout(timeout)
        return bool(llm.complete("Reply with OK.", max_tokens=5))
    except Exception:
        return False
    finally:
        if client is not None and old is not None:
            client.timeout = old
