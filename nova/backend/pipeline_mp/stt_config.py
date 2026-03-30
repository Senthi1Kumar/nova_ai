"""
STT configuration — Moonshine model variant registry.

Supports both moonshine_voice (model_arch_name) and HF Transformers (model_id).
Add entries to STT_VARIANT_REGISTRY to support additional variants.
Change STT_SETTINGS.active to switch the default.
"""

from __future__ import annotations
from pydantic import BaseModel


class STTVariantConfig(BaseModel):
    model_arch_name: str = ""   # ModelArch enum name for moonshine_voice (e.g. "SMALL_STREAMING")
    model_id: str = ""          # HuggingFace model ID for Transformers-based worker
    display_name: str
    vram_mb: int                # approximate GPU memory when loaded
    rtf_target: float           # typical real-time factor (lower = faster)
    language: str = "en"
    vad_threshold: float = 0.2
    is_streaming: bool = True   # True for MoonshineStreaming*, False for base Moonshine
    processor_id: str = ""      # Override processor source (defaults to model_id if empty)


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


# Single shared settings instance — import this wherever needed
STT_SETTINGS = STTSettings()
