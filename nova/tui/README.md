# nova/tui — Textual TUI for nova_loop

A terminal mirror of `nova/backend/static/` (the browser UI). Connects to the
running nova_loop FastAPI server over the same `/ws` WebSocket. Browser UI and
TUI can run side-by-side — each WebSocket connection is its own `Session` in
[nova_loop.py:1604](../backend/nova_loop.py#L1604), so they don't share
history. That's intentional.

## Install

```bash
pip install textual websockets httpx sounddevice numpy
```

(`sounddevice` needs PortAudio: `apt install libportaudio2` on Debian/Ubuntu.)

## Run

Start nova_loop first (whatever your usual launch command is), then in another
terminal:

```bash
python -m nova.tui                       # connects to 127.0.0.1:8001
python -m nova.tui --host 192.168.1.10   # remote
python -m nova.tui --tls                 # use https/wss
```

## Keybindings

| Key     | Action                                  |
|---------|-----------------------------------------|
| `m`     | Toggle mic (start/stop session)         |
| `space` | Interrupt TTS playback (barge-in)       |
| `r`     | Reset session history (clears server)   |
| `q`     | Quit                                    |

You can also click the `🎙 MIC` button at the footer, the Persona tabs, or the
voice-enrollment buttons.

## What's mirrored from the web UI

| `static/app.js` behavior                    | TUI equivalent              |
|---------------------------------------------|-----------------------------|
| WebSocket to `/ws`, PCM up / events down    | `client.NovaClient`         |
| 16 kHz mono i16 mic capture (AudioWorklet)  | `audio.MicCapture` (sounddevice) |
| 24 kHz playback queue + spacebar interrupt  | `audio.PCMPlayer`           |
| Status dot (idle/listen/think/speak)        | `#status-dot` reactive      |
| Chat turns (user/assistant)                 | `widgets.turn.Turn`         |
| Mic level meter                             | `ProgressBar` (`#meter`)    |
| Last-token latency                          | `#lat` Static               |
| Persona tabs (USER/MEMORY/CHAT)             | `Tabs` widget               |
| pVAD badge                                  | `#pvad-badge` Static        |
| 5-phrase voice enrollment                   | `widgets.enrollment.EnrollPanel` |

## What's intentionally different

- No `MediaRecorder`. Enrollment records 3.5 s of raw PCM via `sounddevice`,
  wraps as WAV in memory, posts to `/enroll/voice-sample`.
- No CSS ring-pulse around the mic. State is conveyed by dot color + button
  color + status text. Acceptable downgrade for a terminal.
- The TUI's `r` reset sends `{"type":"reset"}` to the server (the browser UI
  doesn't expose this — it's a small bonus).
