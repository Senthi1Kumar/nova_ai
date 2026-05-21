#!/usr/bin/env bash
set -euo pipefail

WHISPER_MODEL="${WHISPER_MODEL:-base}"
POCKET_TTS_VOICE="${POCKET_TTS_VOICE:-alba}"
POCKET_TTS_LANGUAGE="${POCKET_TTS_LANGUAGE:-english}"

python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt
python scripts/download_voice_models.py \
  --install \
  --whisper-model "$WHISPER_MODEL" \
  --pocket-voice "$POCKET_TTS_VOICE" \
  --pocket-language "$POCKET_TTS_LANGUAGE" \
  --write-env
