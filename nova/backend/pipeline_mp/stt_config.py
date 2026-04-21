"""
STT configuration — variant registry across backends.

Supports three worker paths:
  - moonshine_voice    (model_arch_name)  — streaming Moonshine
  - HF Transformers    (model_id)         — Moonshine fine-tunes
  - Kyutai moshi       (kyutai_hf_repo)   — mimi + LMGen, optional semantic VAD

Add entries to STT_VARIANT_REGISTRY to support additional variants.
Change STT_SETTINGS.active to switch the default (or set NOVA_STT_VARIANT env).
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel


class STTVariantConfig(BaseModel):
    backend: Literal["moonshine", "kyutai"] = "moonshine"
    model_arch_name: str = ""   # ModelArch enum name for moonshine_voice (e.g. "SMALL_STREAMING")
    model_id: str = ""          # HuggingFace model ID for Transformers-based worker
    display_name: str
    vram_mb: int                # approximate GPU memory when loaded
    rtf_target: float           # typical real-time factor (lower = faster)
    language: str = "en"
    vad_threshold: float = 0.2
    is_streaming: bool = True   # True for MoonshineStreaming*, False for base Moonshine
    processor_id: str = ""      # Override processor source (defaults to model_id if empty)
    # Kyutai-specific:
    kyutai_hf_repo: str = ""                 # e.g. "kyutai/stt-1b-en_fr"
    kyutai_use_semantic_vad: bool = False    # True when the repo ships VAD heads (prs[2])


STT_VARIANT_REGISTRY: dict[str, STTVariantConfig] = {
    "tiny": STTVariantConfig(
        model_arch_name="TINY_STREAMING",
        model_id="usefulsensors/moonshine-streaming-tiny",
        display_name="Moonshine Streaming Tiny (fastest)",
        vram_mb=200,
        rtf_target=0.05,
    ),
    "small": STTVariantConfig(
        model_arch_name="SMALL_STREAMING",
        model_id="usefulsensors/moonshine-streaming-small",
        display_name="Moonshine Streaming Small (recommended)",
        vram_mb=500,
        rtf_target=0.08,
    ),
    "medium": STTVariantConfig(
        model_arch_name="MEDIUM_STREAMING",
        model_id="usefulsensors/moonshine-streaming-medium",
        display_name="Moonshine Streaming Medium (best accuracy)",
        vram_mb=1000,
        rtf_target=0.15,
    ),
    "in_en": STTVariantConfig(
        model_id="pavandheeraj05/moonshine-nova-indian-english",
        processor_id="UsefulSensors/moonshine-base",
        display_name="Fine-tuned moonshine base",
        vram_mb=200,
        rtf_target=0.1,
        is_streaming=False,
    ),
    "kyutai_stt_1b_en_fr": STTVariantConfig(
        backend="kyutai",
        kyutai_hf_repo="kyutai/stt-1b-en_fr-candle",  # candle variant ships VAD heads usable from PyTorch LMGen
        display_name="Kyutai STT 1B en/fr (semantic VAD)",
        vram_mb=2500,
        rtf_target=0.12,
        kyutai_use_semantic_vad=True,
    ),
    "kyutai_stt_2_6b_en": STTVariantConfig(
        backend="kyutai",
        kyutai_hf_repo="kyutai/stt-2.6b-en",
        display_name="Kyutai STT 2.6B en (no built-in VAD)",
        vram_mb=5500,
        rtf_target=0.20,
        kyutai_use_semantic_vad=False,
    ),
}


class STTSettings(BaseModel):
    active: str = "small"

    @property
    def config(self) -> STTVariantConfig:
        if self.active not in STT_VARIANT_REGISTRY:
            raise ValueError(
                f"Unknown STT variant '{self.active}'. "
                f"Available: {list(STT_VARIANT_REGISTRY)}"
            )
        return STT_VARIANT_REGISTRY[self.active]


_env_variant = os.getenv("NOVA_STT_VARIANT")
STT_SETTINGS = STTSettings(
    active=_env_variant if _env_variant in STT_VARIANT_REGISTRY else "small"
)
