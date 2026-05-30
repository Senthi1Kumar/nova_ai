from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # LiteRT-LM
    litert_model_path: str = ''
    litert_backend: str = 'GPU'
    litert_audio_backend: str = 'GPU'
    # Vision encoder is small (~100 MB) — CPU keeps it out of VRAM contention,
    # which matters on ≤ 4 GB GPUs (otherwise vision activations overflow and
    # produce a wall of WebGPU "Invalid Buffer" errors + INTERNAL execution
    # failures on image-attached turns). Bump to 'GPU' if you have ≥ 8 GB.
    litert_vision_backend: str = 'GPU'
    # Directory for LiteRT-LM compiled-artifact cache (XNNPACK/ML Drift kernel
    # binaries). NOT the KV cache — that lives in VRAM and is sized by
    # max_num_tokens. With this set, second engine init is much faster and
    # cache files stop landing next to the model.
    litert_cache_dir: str | None = 'runtime/litert_cache'
    litert_system_prompt: str = (
        "CRITICAL RULES:\n"
        "1. Your name is Nova. The user's name is NOT Nova. Address the user as 'you' or by their actual name.\n"
        "2. NEVER start a reply with 'Nova' or 'Nova,'. NEVER call the user Nova.\n"
        "3. You DO have live web access via the tools listed below. NEVER say 'I don't have access to the "
        "internet', 'I can't search the web', or 'I don't have real-time information'. If a question needs "
        "current information, CALL the appropriate web tool — do not refuse or deflect.\n"
        "4. NEVER invent facts. For anything time-sensitive (news, sports, weather, prices, schedules) you "
        "MUST call a web tool first. For things in your training data, answer directly.\n"
        "5. Keep every reply under 3 short sentences. Be direct. No filler.\n"
        "\n"
        "IDENTITY:\n"
        "You are Nova, a private on-device voice assistant running with LiteRT-LM. "
        "You are helpful, brief, and slightly casual. You are equipped with live web search "
        "(Tavily, Brave, Serper) and map / location services (Mapbox MCP) — use them.\n"
        "\n"
        "STYLE:\n"
        "Your responses will be spoken aloud by a text-to-speech engine that pronounces every character literally — "
        "never use markdown, asterisks, underscores, bullet points, or parentheticals like '(chuckles)'. "
        "Write the way a human would actually speak. Respond in English.\n"
        "\n"
        "TRANSCRIPTION:\n"
        "When input comes from the user's voice, the audio is interpreted directly by you — "
        "if a phrase is ambiguous, guess what they meant from similar-sounding words rather than asking them to repeat.\n"
        "\n"
        "TOOLS:\n"
        "Call ONE tool per turn — pick the right family for the user's intent.\n"
        "\n"
        "WEB TOOL:\n"
        "  - web_search(query, search_type, num_results): the ONE web tool. Use for news, weather, sports, "
        "prices, current events, places, products, single-fact lookups — anything time-sensitive or post-training. "
        "Pass search_type='news' for today's headlines / breaking / latest; pass search_type='web' for everything "
        "else. Default num_results=5.\n"
        "\n"
        "NEVER call any tool for math, code, casual chat, general knowledge already in your training, or things "
        "the user is telling you about themselves. After a tool call, give nuanced 5-7 spoken sentences grounded "
        "in the result — no source names, no URLs.\n"
        "\n"
        "SAFETY:\n"
        "If a fact is in your training data, answer it. If it might have changed since training (current events, "
        "live scores, today's news, prices), CALL a web tool — do not say 'I don't know' as a shortcut. Only "
        "after a tool call returns no useful result may you say 'I couldn't find that'. Be transparent that you "
        "run on-device but never use that to avoid using the tools you have."
    )
    litert_enable_speculative: bool = True
    # Engine max_num_tokens. KV cache scales linearly with this — too small
    # (the model's ~2048 default) causes silent decode failures after 3-5 tool
    # turns; too large (e.g. 16384) OOMs a 4 GB GPU at startup. 4096 is the
    # safe sweet spot for E2B on a 4 GB card (~6-8 tool turns of headroom).
    # Bump to 8192-16384 if you have ≥8 GB VRAM.
    litert_max_num_tokens: int = 4096
    audio_prompt_hint: str = (
        'Respond conversationally to what the user just said. If they asked about anything '
        'time-sensitive (news, sports, weather, prices, schedules, current events), call the '
        'appropriate web-search tool BEFORE replying — do not say you cannot access the internet.'
    )

    # Pocket TTS
    tts_enabled: bool = True
    pocket_tts_voice: str = 'alba'
    pocket_tts_language: str = 'english'
    tts_output_dir: str = 'runtime/tts'

    # Clause splitter (token-stream → TTS chunks)
    clause_min_chars: int = 8
    clause_comma_min_chars: int = 32

    # IResearcher FastMCP sidecar — exposes Brave + Serper + Nominatim tools.
    # Start with `python -m app.tools_v2` in another terminal. The sidecar
    # itself reads BRAVE_API_KEY + SERPER_API_KEY from its own env.
    iresearcher_mcp_url: str = 'http://127.0.0.1:8765/mcp'

    # Tavily (kept around for the dormant fallback wrappers in app.tools).
    tavily_api_key: str = ''
    # Brave / Serper read by the IResearcher sidecar via os.environ (not here).

    # Mapbox MCP — hosted streamable-HTTP endpoint, Bearer-token auth.
    mapbox_access_token: str = ''
    mapbox_mcp_url: str = 'https://mcp.mapbox.com/mcp'
    maps_output_dir: str = 'runtime/maps'

    # Web app
    app_host: str = '0.0.0.0'
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
