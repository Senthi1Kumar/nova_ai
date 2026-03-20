"""
Local LLM configuration — Pydantic-based model registry.

Add entries to LOCAL_LLM_REGISTRY to support additional models.
Change LOCAL_LLM_SETTINGS.active to switch the default local fallback.

LFM2.5-1.2B-Instruct is the default:
  - 1.2B dense, 32K context, native tool calling
  - HF: LiquidAI/LFM2.5-1.2B-Instruct
  - Requires: transformers>=5.0.0
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class SamplingParams(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_k: Optional[int] = 50
    top_p: Optional[float] = None        # None = disabled
    repetition_penalty: float = 1.05
    do_sample: bool = True


class LocalLLMConfig(BaseModel):
    model_id: str
    display_name: str
    supports_tools: bool = False
    context_length: int = 4096
    vram_mb: int = 1000                  # approximate GPU memory when loaded (4-bit)
    # Quantisation — BitsAndBytes 4-bit keeps VRAM low on consumer GPUs
    load_in_4bit: bool = True
    compute_dtype: str = "bfloat16"      # bfloat16 recommended for LFM2.5; float16 for older GPUs
    sampling: SamplingParams = Field(default_factory=SamplingParams)


# ── Model registry ────────────────────────────────────────────────────────────

LOCAL_LLM_REGISTRY: dict[str, LocalLLMConfig] = {

    # Liquid AI LFM2.5-1.2B — recommended default
    # Chat, instruction-following, native tool calling, 32K context
    "lfm2.5-1.2b": LocalLLMConfig(
        model_id="LiquidAI/LFM2.5-1.2B-Instruct",
        display_name="LFM2.5-1.2B (Liquid AI)",
        supports_tools=True,
        context_length=32768,
        vram_mb=1000,              # ~1 GB in 4-bit
        load_in_4bit=True,
        compute_dtype="bfloat16",
        sampling=SamplingParams(
            max_new_tokens=256,
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.05,
            do_sample=True,
        ),
    ),

    # Liquid AI LFM2-700M — fastest, lowest VRAM
    "lfm2-700m": LocalLLMConfig(
        model_id="LiquidAI/LFM2-700M",
        display_name="LFM2-700M (Liquid AI)",
        supports_tools=False,
        context_length=4096,
        vram_mb=500,               # ~0.5 GB in 4-bit
        # load_in_4bit=True,
        compute_dtype="bfloat16",
        sampling=SamplingParams(
            max_new_tokens=128,
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.05,
            do_sample=True,
        ),
    ),

    # Qwen3.5-0.8B — legacy fallback, kept for compatibility
    "qwen3.5-0.8b": LocalLLMConfig(
        model_id="Qwen/Qwen3.5-0.8B",
        display_name="Qwen3.5-0.8B",
        supports_tools=False,
        context_length=4096,
        vram_mb=600,               # ~0.6 GB in 4-bit
        load_in_4bit=True,
        compute_dtype="float16",
        sampling=SamplingParams(
            max_new_tokens=128,
            temperature=0.7,
            top_k=None,
            top_p=0.8,
            repetition_penalty=1.0,
            do_sample=True,
        ),
    ),
}


# ── VRAM estimates for TTS engines (used by gateway for budget warnings) ──────
TTS_VRAM_ESTIMATES: dict[str, int] = {
    "faster-qwen3": 2750,     # FasterQwen3TTS 0.6B + CUDA graphs
    "pocket-tts": 150,        # small model, can also run on CPU
}

# Baseline VRAM consumed by STT + KWS + CUDA runtime overhead
BASELINE_VRAM_MB: int = 850   # Moonshine ~500 + KWS ~50 + CUDA overhead ~300


class LocalLLMSettings(BaseModel):
    active: str = "lfm2-700m"

    @property
    def config(self) -> LocalLLMConfig:
        if self.active not in LOCAL_LLM_REGISTRY:
            raise ValueError(
                f"Unknown local LLM '{self.active}'. "
                f"Available: {list(LOCAL_LLM_REGISTRY)}"
            )
        return LOCAL_LLM_REGISTRY[self.active]


# Single shared settings instance — import this wherever needed
LOCAL_LLM_SETTINGS = LocalLLMSettings()
