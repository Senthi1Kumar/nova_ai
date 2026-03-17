# Nova Voice Reference Audio

Place `nova_ref.wav` here — a clear, noise-free recording of the voice you want
Nova to use for TTS output.

## Requirements
- Format: WAV, mono, any sample rate (resampled internally)
- Duration: 5–10 seconds of natural speech
- Environment: quiet room, close-mic capture preferred

## Quick recording (Linux)
```bash
arecord -d 8 -f cd -r 24000 -c 1 nova_ref.wav
```

## What happens without nova_ref.wav
- On first startup the TTS worker generates a 3-second silence placeholder.
- The model will produce speech in an unvoiced/default timbre until you replace it.
- You can hot-swap the file — just restart the TTS worker process.
