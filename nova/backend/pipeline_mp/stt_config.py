"""
STT configuration — Moonshine model variant registry.

Add entries to STT_VARIANT_REGISTRY to support additional variants.
Change STT_SETTINGS.active to switch the default.

Default: small-streaming (~400 MB VRAM, good accuracy/speed balance)
  - tiny-streaming: fastest, lowest VRAM (~200 MB), slightly lower accuracy
  - small-streaming: recommended default (~400 MB)
  - medium-streaming: highest accuracy (~800 MB), more VRAM
"""

from __future__ import annotations
from pydantic import BaseModel


class STTVariantConfig(BaseModel):
    model_arch_name: str    # ModelArch enum name (e.g. "SMALL_STREAMING")
    display_name: str
    vram_mb: int            # approximate GPU memory when loaded
    rtf_target: float       # typical real-time factor (lower = faster)
    language: str = "en"
    vad_threshold: float = 0.3  # VAD sensitivity [0..1]. Lower = more sensitive
                                # (catches softer speech / accented pauses).
                                # Default 0.3 works better for Indian English than 0.5.


STT_VARIANT_REGISTRY: dict[str, STTVariantConfig] = {
    "tiny": STTVariantConfig(
        model_arch_name="TINY_STREAMING",
        display_name="Moonshine Tiny (fastest)",
        vram_mb=200,
        rtf_target=0.05,
    ),
    "small": STTVariantConfig(
        model_arch_name="SMALL_STREAMING",
        display_name="Moonshine Small (recommended)",
        vram_mb=400,
        rtf_target=0.08,
    ),
    "medium": STTVariantConfig(
        model_arch_name="MEDIUM_STREAMING",
        display_name="Moonshine Medium (highest accuracy)",
        vram_mb=800,
        rtf_target=0.15,
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
