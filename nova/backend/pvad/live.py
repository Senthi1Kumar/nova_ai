"""
live.py — Two-Stage pVAD live terminal demo

Reads from microphone, runs FireRedGate (Stage 1) + SpeakerGate (Stage 2)
and prints real-time gate state to the terminal.

Usage
-----
    cd nova/backend
    python pvad/live.py --driver_id driver1

    # WAV file playback instead of mic:
    python pvad/live.py --wav path/to/audio.wav

    # Skip Stage 2 (just watch FireRedVAD):
    python pvad/live.py --no_speaker

    # Skip Stage 1 (just ECAPA with RMS gate):
    python pvad/live.py --no_firered

Output columns:
    [t=  0.01s]  Stage1: VOICE  prob=0.82  |  Stage2: score=0.61  frac=0.72  gate=OPEN
    [t=  0.25s]  Stage1: SILEN  prob=0.04  |  Stage2: --- (skipped, frac=0.04)
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_HERE, ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

logging.basicConfig(
    level=logging.WARNING,          # keep libraries quiet during live demo
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pvad.live")

# ── ANSI colours ──────────────────────────────────────────────────────────────
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _gate_colour(gate: bool) -> str:
    return f"{_GREEN}{_BOLD}OPEN {_RESET}" if gate else f"{_RED}{_BOLD}CLOSED{_RESET}"


def _voice_colour(is_voice: bool, prob: float) -> str:
    label = "VOICE" if is_voice else "SILEN"
    colour = _GREEN if is_voice else _YELLOW
    return f"{colour}{label}{_RESET} p={prob:.2f}"


# ── Argument parsing ──────────────────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(description="Two-stage pVAD live demo")

    # Input
    ap.add_argument("--wav",        default=None,
                    help="WAV file instead of microphone (16kHz mono int16)")
    ap.add_argument("--device",     default=None,
                    help="sounddevice input device index/name (mic only)")
    ap.add_argument("--chunk_ms",   type=int, default=20,
                    help="Mic chunk size in ms (default 20ms)")

    # Stage 1
    ap.add_argument("--no_firered", action="store_true",
                    help="Disable Stage-1 FireRedGate; use RMS threshold instead")
    ap.add_argument("--fr_model_dir", default=None,
                    help="Path to FireRedVAD model weights")
    ap.add_argument("--fr_threshold", type=float, default=0.4,
                    help="FireRedVAD speech_threshold (default 0.4)")
    ap.add_argument("--rms_threshold", type=float, default=0.035,
                    help="RMS fallback threshold for Stage-1 (default 0.035)")

    # Stage 2
    ap.add_argument("--no_speaker", action="store_true",
                    help="Disable Stage-2 SpeakerGate (only show FireRedVAD output)")
    ap.add_argument("--driver_id",  default="driver1")
    ap.add_argument("--l3_dir",     default=None)
    ap.add_argument("--sp_threshold", type=float, default=0.35,
                    help="ECAPA cosine similarity threshold (default 0.35)")
    ap.add_argument("--voice_frac",   type=float, default=0.30,
                    help="Min voiced fraction per 250ms stride to run ECAPA (default 0.30)")
    ap.add_argument("--hysteresis",   type=int,   default=2,
                    help="Consecutive same-direction strides before gate changes (default 2)")

    # Display
    ap.add_argument("--verbose", action="store_true",
                    help="Print every 10ms FireRedVAD frame (not just stride summaries)")
    return ap.parse_args()


# ── Stage 1: FireRedGate / RMS fallback ──────────────────────────────────────
def _load_stage1(args):
    if args.no_firered:
        return None

    try:
        from pvad.firered_gate import FireRedGate  # type: ignore
        gate = FireRedGate.load(
            model_dir=args.fr_model_dir,
            speech_threshold=args.fr_threshold,
        )
        print(f"{_CYAN}Stage-1:{_RESET} FireRedGate loaded  "
              f"(threshold={args.fr_threshold})")
        return gate
    except Exception as e:
        print(f"{_YELLOW}Stage-1:{_RESET} FireRedGate unavailable ({e})")
        print(f"         Falling back to RMS threshold={args.rms_threshold}")
        return None


# ── Stage 2: SpeakerGate ─────────────────────────────────────────────────────
def _load_stage2(args):
    if args.no_speaker:
        print(f"{_YELLOW}Stage-2:{_RESET} disabled (--no_speaker)")
        return None

    try:
        from pvad.speaker_gate import SpeakerGate  # type: ignore
        gate = SpeakerGate.load(
            l3_dir=args.l3_dir,
            driver_id=args.driver_id,
            threshold=args.sp_threshold,
            hysteresis=args.hysteresis,
            voice_frac_min=args.voice_frac,
        )
        ecapa_ok = gate._ecapa is not None
        emb_ok   = gate._driver_emb is not None
        status = "ready" if (ecapa_ok and emb_ok) else "fail-open (no model/voiceprint)"
        print(f"{_CYAN}Stage-2:{_RESET} SpeakerGate {status}  "
              f"driver='{args.driver_id}' threshold={args.sp_threshold}")
        return gate
    except Exception as e:
        print(f"{_YELLOW}Stage-2:{_RESET} SpeakerGate unavailable ({e})")
        return None


# ── RMS fallback voice_flags helper ──────────────────────────────────────────
_STEP_SAMPLES = 160

def _make_rms_flag_generator(rms_threshold: float):
    """Returns a stateful function that generates voice_flags from f32 chunks."""
    buf = np.zeros(_STEP_SAMPLES, dtype=np.float32)
    pos = [0]

    def generate(audio_f32: np.ndarray):
        flags = []
        p = 0
        while p < len(audio_f32):
            space = _STEP_SAMPLES - pos[0]
            take  = min(space, len(audio_f32) - p)
            buf[pos[0]:pos[0]+take] = audio_f32[p:p+take]
            pos[0] += take
            p      += take
            if pos[0] == _STEP_SAMPLES:
                rms = float(np.sqrt(np.mean(buf**2)))
                flags.append(rms >= rms_threshold)
                pos[0] = 0
        return flags

    return generate


# ── Core processing loop ──────────────────────────────────────────────────────
def _process_chunk(
    audio_int16: np.ndarray,
    audio_f32:   np.ndarray,
    firered_gate,
    speaker_gate,
    rms_flags_fn,
    t_start:     float,
    args,
):
    # Stage 1
    if firered_gate is not None:
        frames      = firered_gate.feed(audio_int16)
        voice_flags = [f.is_voice for f in frames]

        if args.verbose:
            for f in frames:
                t_ms = (time.monotonic() - t_start) * 1000
                print(f"[t={t_ms/1000:7.3f}s]  "
                      f"S1 {_voice_colour(f.is_voice, f.raw_prob)}  "
                      f"latency={f.latency_ms:.1f}ms")
    else:
        voice_flags = rms_flags_fn(audio_f32)

        if args.verbose and voice_flags:
            t_ms = (time.monotonic() - t_start) * 1000
            rms  = float(np.sqrt(np.mean(audio_f32**2)))
            is_v = voice_flags[-1]
            print(f"[t={t_ms/1000:7.3f}s]  "
                  f"S1(rms) {_voice_colour(is_v, rms)}")

    # Stage 2
    if speaker_gate is None:
        return

    result = speaker_gate.feed(audio_f32, voice_flags)
    if result is None:
        return  # stride not complete yet

    t_s = time.monotonic() - t_start
    if result.score is not None:
        score_str = (f"score={_GREEN}{result.score:.3f}{_RESET}"
                     if result.gate_vote else
                     f"score={_RED}{result.score:.3f}{_RESET}")
        lat_str   = f"  ecapa={result.latency_ms:.0f}ms"
    else:
        score_str = "---"
        lat_str   = ""

    print(
        f"[t={t_s:7.3f}s]  "
        f"S2: {score_str}  "
        f"voiced={result.voice_frac:.0%}  "
        f"gate={_gate_colour(result.gate_vote)}"
        f"{lat_str}"
    )


# ── Microphone mode ───────────────────────────────────────────────────────────
def _run_mic(args, firered_gate, speaker_gate, rms_flags_fn):
    import sounddevice as sd

    sample_rate   = 16_000
    chunk_samples = int(sample_rate * args.chunk_ms / 1000)
    t_start       = time.monotonic()

    print(f"\n{_BOLD}Listening from microphone{_RESET} "
          f"(chunk={args.chunk_ms}ms, Ctrl+C to stop)\n")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=chunk_samples,
        device=args.device,
    ) as stream:
        try:
            while True:
                data, _ = stream.read(chunk_samples)
                audio_int16 = data[:, 0]
                audio_f32   = audio_int16.astype(np.float32) / 32768.0
                _process_chunk(audio_int16, audio_f32,
                               firered_gate, speaker_gate, rms_flags_fn,
                               t_start, args)
        except KeyboardInterrupt:
            print(f"\n{_YELLOW}Stopped.{_RESET}")


# ── WAV file mode ─────────────────────────────────────────────────────────────
def _run_wav(args, firered_gate, speaker_gate, rms_flags_fn):
    import soundfile as sf

    audio, sr = sf.read(args.wav, dtype="int16")
    assert sr == 16_000, f"Need 16kHz WAV, got {sr}Hz"
    if audio.ndim > 1:
        audio = audio[:, 0]

    sample_rate   = 16_000
    chunk_samples = int(sample_rate * args.chunk_ms / 1000)
    t_start       = time.monotonic()
    dur_s         = len(audio) / sample_rate

    print(f"\n{_BOLD}Processing WAV file:{_RESET} {args.wav}  ({dur_s:.2f}s)\n")

    for start in range(0, len(audio), chunk_samples):
        chunk       = audio[start:start + chunk_samples]
        audio_int16 = chunk
        audio_f32   = chunk.astype(np.float32) / 32768.0
        _process_chunk(audio_int16, audio_f32,
                       firered_gate, speaker_gate, rms_flags_fn,
                       t_start, args)

    print(f"\n{_BOLD}Done.{_RESET}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    args = _parse_args()

    print(f"\n{_BOLD}=== Nova pVAD two-stage demo ==={_RESET}")
    print(f"Stage-1: {'FireRedGate (DFSMN)' if not args.no_firered else 'RMS fallback'}")
    print(f"Stage-2: {'SpeakerGate (ECAPA)' if not args.no_speaker else 'disabled'}\n")

    firered_gate = _load_stage1(args)
    speaker_gate = _load_stage2(args)
    rms_flags_fn = _make_rms_flag_generator(args.rms_threshold)

    print()

    if args.wav:
        _run_wav(args, firered_gate, speaker_gate, rms_flags_fn)
    else:
        _run_mic(args, firered_gate, speaker_gate, rms_flags_fn)


if __name__ == "__main__":
    main()
