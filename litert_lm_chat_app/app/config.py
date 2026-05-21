from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # LiteRT-LM
    litert_model_path: str = ''
    litert_backend: str = 'GPU'
    litert_audio_backend: str = 'CPU'
    litert_vision_backend: str = 'GPU'
    litert_cache_dir: str | None = None
    litert_system_prompt: str = 'You are a helpful local AI assistant running with LiteRT-LM.'
    litert_enable_speculative: bool = True
    audio_prompt_hint: str = 'Respond conversationally to what the user just said.'

    # Pocket TTS
    tts_enabled: bool = True
    pocket_tts_voice: str = 'alba'
    pocket_tts_language: str = 'english'
    tts_output_dir: str = 'runtime/tts'

    # Clause splitter (token-stream → TTS chunks)
    clause_min_chars: int = 8
    clause_comma_min_chars: int = 32

    # Web app
    app_host: str = '0.0.0.0'
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
