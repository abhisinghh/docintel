"""Runtime configuration (environment variables / YAML)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Any OpenAI-compatible chat endpoint (vLLM, Ollama, OpenAI, OpenRouter) or Anthropic."""

    provider: str = Field(default_factory=lambda: os.getenv("DOCINTEL_LLM_PROVIDER", "openai"))  # openai | anthropic | transformers | none
    model: str = Field(default_factory=lambda: os.getenv("DOCINTEL_LLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    device: str = Field(default_factory=lambda: os.getenv("DOCINTEL_LLM_DEVICE", "auto"))   # transformers only
    base_url: Optional[str] = Field(default_factory=lambda: os.getenv("DOCINTEL_LLM_BASE_URL", "http://localhost:8000/v1"))
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("DOCINTEL_LLM_API_KEY", "EMPTY"))
    max_tokens: int = 8192
    temperature: float = 0.0
    timeout: float = 180.0


class VLMConfig(BaseModel):
    """Document-parsing VLM used for OCR + layout + grounding (page → chunks).

    provider = where the model runs:
      - "openai"       any OpenAI-compatible HTTP endpoint: on-prem vLLM / SGLang / TGI / Ollama / LM Studio,
                       or hosted APIs (OpenAI, OpenRouter, Together, Fireworks, Gemini's OpenAI endpoint, ...)
      - "anthropic"    Anthropic Messages API (Claude as the page parser / verifier)
      - "transformers" load the weights in-process with HF transformers (no server; GPU or CPU)
      - "landingai"    LandingAI Agentic Document Extraction API (hosted parser; returns chunks + groundings)
    family selects the prompt template / output parser:
      - "generic"      any VLM that can return JSON (Qwen3-VL, InternVL, GPT, Gemini, Claude ...)
      - "dots_ocr"     dots.ocr (rednote-hilab/dots.ocr) layout+text JSON prompt
      - "paddleocr_vl" PaddleOCR-VL served via vLLM
      - "deepseek_ocr" DeepSeek-OCR grounding prompt
    """

    enabled: bool = Field(default_factory=lambda: os.getenv("DOCINTEL_VLM_ENABLED", "0") == "1")
    provider: str = Field(default_factory=lambda: os.getenv("DOCINTEL_VLM_PROVIDER", "openai"))
    device: str = Field(default_factory=lambda: os.getenv("DOCINTEL_VLM_DEVICE", "auto"))   # transformers only
    family: str = Field(default_factory=lambda: os.getenv("DOCINTEL_VLM_FAMILY", "generic"))
    model: str = Field(default_factory=lambda: os.getenv("DOCINTEL_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    base_url: str = Field(default_factory=lambda: os.getenv("DOCINTEL_VLM_BASE_URL", "http://localhost:8000/v1"))
    api_key: str = Field(default_factory=lambda: os.getenv("DOCINTEL_VLM_API_KEY", "EMPTY"))
    max_tokens: int = 16384
    timeout: float = 300.0


class OCRConfig(BaseModel):
    engine: str = Field(default_factory=lambda: os.getenv("DOCINTEL_OCR_ENGINE", "tesseract"))  # tesseract|paddle|surya|none
    languages: list[str] = Field(default_factory=lambda: os.getenv("DOCINTEL_OCR_LANGS", "eng+hin").split("+"))
    dpi: int = 200
    force_ocr: bool = False          # OCR even when a native text layer exists


class LayoutConfig(BaseModel):
    engine: str = Field(default_factory=lambda: os.getenv("DOCINTEL_LAYOUT_ENGINE", "heuristic"))  # heuristic|doclayout_yolo|pp_doclayout|vlm
    model_path: Optional[str] = None
    conf_threshold: float = 0.25


class Settings(BaseModel):
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("DOCINTEL_DATA_DIR", "./data")))
    render_dpi: int = 150
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    agent_max_steps: int = 12
    verify_extractions: bool = True
    strict_grounding: bool = False   # null out values that could not be verified against the page
    extraction_mode: str = Field(default_factory=lambda: os.getenv("DOCINTEL_EXTRACTION_MODE", "parse"))  # parse | schema | auto
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "Settings":
        path = path or os.getenv("DOCINTEL_CONFIG")
        if path and Path(path).exists():
            with open(path) as f:
                return cls.model_validate(yaml.safe_load(f) or {})
        return cls()

    def ensure_dirs(self) -> None:
        for sub in ("uploads", "pages", "crops", "parsed", "exports"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
        _settings.ensure_dirs()
    return _settings
