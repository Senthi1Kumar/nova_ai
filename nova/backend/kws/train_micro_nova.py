#!/usr/bin/env python3
"""
Training data preparation for the "Nova" micro-wake-word model.

This script prepares all training data, then invokes the micro-wake-word
training pipeline to produce a streaming quantised TFLite model.

Prerequisites (training only — NOT needed at inference time):
    pip install tensorflow>=2.16 audiomentations mmap_ninja pymicro-features
    pip install piper-phonemize-cross==1.2.1
    uv pip install -e piper-sample-generator/    # local repo (already installed)
    pip install -e micro-wake-word/              # local repo

For Qwen3-TTS voice cloning (alternative to Piper):
    pip install -e faster-qwen3-tts/             # local repo
    pip install soundfile

Usage:
    cd nova/backend/kws

    # With Piper TTS (default):
    python train_micro_nova.py                # Step 1: prepare data
    python train_micro_nova.py --train        # Step 2: train model
    python train_micro_nova.py --export       # Step 3: copy model to kws/models/

    # With Qwen3-TTS voice cloning:
    python train_micro_nova.py --tts-engine qwen3 --voices-dir /path/to/reference_voices/
    python train_micro_nova.py --train
    python train_micro_nova.py --export

    # Record 20-30 people saying random sentences (~10s each), put WAVs in voices-dir.
    # Qwen3-TTS clones each voice to say the wake word variants → diverse training set.

Recording reference voices with arecord (ALSA):
    mkdir -p reference_voices
    # Record 10s clips, 16kHz mono WAV — have each person say any random sentence:
    arecord -f S16_LE -r 16000 -c 1 -d 10 reference_voices/voice_01.wav
    arecord -f S16_LE -r 16000 -c 1 -d 10 reference_voices/voice_02.wav
    # ... repeat for 20-30 different speakers
    # Then pass the directory:
    python train_micro_nova.py --tts-engine qwen3 --voices-dir ./reference_voices/

The full pipeline (data prep → train → export) can also be run in one shot:
    python train_micro_nova.py --all
    python train_micro_nova.py --all --tts-engine qwen3 --voices-dir ./reference_voices/
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PIPER_DIR = SCRIPT_DIR / "piper-sample-generator"
MWW_DIR = SCRIPT_DIR / "micro-wake-word"
DATA_DIR = SCRIPT_DIR / "micro_training_data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = SCRIPT_DIR / "models"
TRAINED_DIR = DATA_DIR / "trained_models" / "nova"
CONFIG_FILE = SCRIPT_DIR / "train_micro_nova.yaml"

# Piper generator model
PIPER_MODEL_URL = "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"
PIPER_MODEL_PATH = PIPER_DIR / "models" / "en_US-libritts_r-medium.pt"

# Wake word variants — phonetic spellings often produce better TTS samples
POSITIVE_WORDS = [
    "nova", "Nova", "NOVA",
    "noh vuh", "Hey Nova", "hi Nova",
    "noah", "hey Noah", "Hello Nova"
]

# Hard negatives — confusable words the model must learn to reject
HARD_NEGATIVES = [
    "Nava", "Never", "Over", "Mover", "Rover",
    "Sofa", "Lova", "Nola", "Boba", "Dova", "Cova",
    "No way", "No uh", "Motor", "Nota",
    "Hello", "Okay", "Hey there",
    "Alexa", "Hey Siri", "Hey Google", "Hey Bixby",
]

SAMPLES_PER_POSITIVE = 250   # per word variant (Piper) — Qwen3 uses QWEN3_SAMPLES_PER_VOICE * num_voices
SAMPLES_PER_NEGATIVE = 100   # per hard negative word (Piper) — same logic for Qwen3

QWEN3_SAMPLES_PER_VOICE = 100  # each reference voice generates this many samples per word

# MicroKWS expects 1500ms windows — clips must be short.
# Anything beyond the wake word is noise/gibberish that hurts training.
MAX_CLIP_DURATION_S = 1.5  # hard trim all generated samples to this length


# ── Helpers ──────────────────────────────────────────────────────────────────

def trim_wav_dir(directory: Path, max_duration_s: float = MAX_CLIP_DURATION_S):
    """Trim all WAV files in a directory to max_duration_s.

    Reads each file, truncates to max_duration_s * sample_rate samples,
    overwrites in-place.  Skips files already within the limit.
    """
    import soundfile as sf

    wavs = list(directory.glob("*.wav"))
    trimmed = 0
    for wav_path in wavs:
        info = sf.info(str(wav_path))
        if info.duration <= max_duration_s:
            continue
        audio, sr = sf.read(str(wav_path))
        max_samples = int(max_duration_s * sr)
        audio = audio[:max_samples]
        sf.write(str(wav_path), audio, sr)
        trimmed += 1

    if trimmed:
        print(f"    Trimmed {trimmed}/{len(wavs)} files to {max_duration_s}s in {directory.name}/")


def ensure_piper_model():
    """Download the Piper .pt generator model if not present."""
    if PIPER_MODEL_PATH.exists():
        return
    if not PIPER_DIR.exists():
        print(f"ERROR: piper-sample-generator not found at {PIPER_DIR}")
        print("Clone it:  git clone https://github.com/rhasspy/piper-sample-generator.git")
        sys.exit(1)

    PIPER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("  Downloading Piper generator model...")
    print(f"  {PIPER_MODEL_URL}")
    # Use wget — GitHub release URLs redirect and urllib can produce truncated files
    subprocess.run(
        ["wget", "-O", str(PIPER_MODEL_PATH), PIPER_MODEL_URL],
        check=True,
    )
    print(f"  Saved to {PIPER_MODEL_PATH}")


def run_piper(word: str, output_dir: str, max_samples: int, batch_size: int = 50):
    """Invoke piper-sample-generator as a Python module.

    CLI: python -m piper_sample_generator TEXT --model MODEL --max-samples N --output-dir DIR

    The piper_train package lives in the piper-sample-generator repo but is NOT
    installed as a pip package — it must be on PYTHONPATH so the import works.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "piper_sample_generator",
        word,
        "--model", str(PIPER_MODEL_PATH),
        "--max-samples", str(max_samples),
        "--batch-size", str(batch_size),
        "--output-dir", output_dir,
    ]
    # piper_train is a sibling package in the repo, not pip-installed.
    # CUDA_VISIBLE_DEVICES=-1 forces CPU — avoids nvrtc version mismatch
    # (system CUDA 13 vs PyTorch built for CUDA 12).
    import os
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PIPER_DIR) + (f":{existing}" if existing else "")
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.run(cmd, check=True, env=env)


def run_piper_prefixed(word: str, output_dir: str, max_samples: int, prefix: str):
    """Generate samples into a temp dir, then move with prefix to avoid overwrites.

    Piper names output 0.wav, 1.wav, ... — running multiple words into the same
    directory would overwrite. This renames to {prefix}_0.wav, {prefix}_1.wav, etc.
    """
    tmp_dir = output_dir + f"_tmp_{prefix}"
    run_piper(word, tmp_dir, max_samples)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(tmp_dir).glob("*.wav")):
        dest = out / f"{prefix}_{f.name}"
        shutil.move(str(f), str(dest))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    trim_wav_dir(out)


# ── Qwen3-TTS voice cloning ────────────────────────────────────────────────

# Default Qwen3-TTS model — 0.6B is fastest, fits on most GPUs
QWEN3_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

def _load_qwen3_model(model_id: str):
    """Load FasterQwen3TTS once and cache it."""
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"  Loading Qwen3-TTS ({model_id}) on {device}...")
    model = FasterQwen3TTS.from_pretrained(
        model_id,
        device=device,
        dtype=dtype,
        attn_implementation="sdpa",
        max_seq_len=2048,
    )
    print("  Qwen3-TTS ready.")
    return model


def _discover_voices(voices_dir: Path) -> list[Path]:
    """Find all WAV/FLAC/MP3 reference voice files in a directory."""
    exts = {".wav", ".flac", ".mp3", ".ogg"}
    voices = sorted(
        f for f in voices_dir.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    )
    if not voices:
        print(f"ERROR: No audio files found in {voices_dir}")
        print("  Record 20-30 people saying random sentences (~10s each)")
        print(f"  and place the WAV files in: {voices_dir}")
        sys.exit(1)
    return voices


def run_qwen3_cloned(
    model,
    word: str,
    output_dir: str,
    prefix: str,
    voices: list[Path],
    samples_per_voice: int = QWEN3_SAMPLES_PER_VOICE,
):
    """Generate samples for one word by cloning each reference voice.

    Each voice produces ``samples_per_voice`` samples (default 100) with
    temperature sampling for natural variation.  Total samples per word =
    num_voices * samples_per_voice.

    Output is 24 kHz WAV, hard-trimmed to MAX_CLIP_DURATION_S (1.5s).
    The training pipeline's Clips loader resamples to 16 kHz as needed.
    """
    import numpy as np
    import soundfile as sf

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Qwen3-TTS 12Hz codec: 12 tokens ≈ 1s audio.
    # 24 tokens ≈ 2s — enough for any wake word, avoids gibberish tail.
    max_tokens = int(MAX_CLIP_DURATION_S * 12) + 12  # small headroom

    idx = 0
    for voice_path in voices:
        voice_name = voice_path.stem
        for j in range(samples_per_voice):
            try:
                audio_list, sr = model.generate_voice_clone(
                    text=word,
                    language="English",
                    ref_audio=str(voice_path),
                    ref_text="",  # xvec_only mode ignores ref_text
                    xvec_only=True,
                    do_sample=True,
                    temperature=0.9,
                    top_k=50,
                    repetition_penalty=1.05,
                    max_new_tokens=max_tokens,
                )
                audio = audio_list[0]
                if isinstance(audio, np.ndarray) and len(audio) > 0:
                    # Hard trim to MAX_CLIP_DURATION_S
                    max_samples = int(MAX_CLIP_DURATION_S * sr)
                    audio = audio[:max_samples]
                    dest = out / f"{prefix}_{idx}.wav"
                    sf.write(str(dest), audio, sr)
                    idx += 1
            except Exception as e:
                print(f"    WARN: Failed {prefix} voice={voice_name} attempt={j}: {e}")
                continue

    total = len(voices) * samples_per_voice
    print(f"    [{prefix}] Generated {idx}/{total} samples via voice cloning ({MAX_CLIP_DURATION_S}s max)")


def download_file(url: str, dest: Path):
    """Download a file using wget for reliable redirect handling."""
    print(f"  Downloading {dest.name} ...")
    subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)


# ── Data Generation ──────────────────────────────────────────────────────────

def generate_positive_samples(tts_engine: str = "piper", qwen3_model=None, voices: list | None = None):
    """Generate positive wake-word audio using Piper TTS or Qwen3-TTS voice cloning.

    All variants are placed into ONE flat directory with prefixed filenames
    so that Clips(input_directory=...) can load them all at once.
    """
    if tts_engine == "piper":
        ensure_piper_model()

    pos_dir = DATA_DIR / "positive_samples"

    for word in POSITIVE_WORDS:
        prefix = word.replace(" ", "_").lower()
        existing = list(pos_dir.glob(f"{prefix}_*.wav")) if pos_dir.exists() else []

        if tts_engine == "qwen3":
            target = QWEN3_SAMPLES_PER_VOICE * len(voices)
            if len(existing) >= target:
                print(f"  [{prefix}] already has {len(existing)} samples, skipping")
                continue
            print(f"  Generating {target} samples for '{word}' ({len(voices)} voices x {QWEN3_SAMPLES_PER_VOICE})...")
            run_qwen3_cloned(qwen3_model, word, str(pos_dir), prefix, voices)
        else:
            if len(existing) >= SAMPLES_PER_POSITIVE:
                print(f"  [{prefix}] already has {len(existing)} samples, skipping")
                continue
            print(f"  Generating {SAMPLES_PER_POSITIVE} samples for '{word}' (piper)...")
            run_piper_prefixed(word, str(pos_dir), SAMPLES_PER_POSITIVE, prefix)


def generate_hard_negatives(tts_engine: str = "piper", qwen3_model=None, voices: list | None = None):
    """Generate hard-negative audio using Piper TTS or Qwen3-TTS voice cloning.

    All variants into ONE flat directory, same pattern as positives.
    """
    if tts_engine == "piper":
        ensure_piper_model()

    neg_dir = DATA_DIR / "hard_negative_samples"

    for word in HARD_NEGATIVES:
        prefix = word.replace(" ", "_").lower()
        existing = list(neg_dir.glob(f"{prefix}_*.wav")) if neg_dir.exists() else []

        if tts_engine == "qwen3":
            target = QWEN3_SAMPLES_PER_VOICE * len(voices)
            if len(existing) >= target:
                print(f"  [{prefix}] already has {len(existing)} samples, skipping")
                continue
            print(f"  Generating {target} hard negatives for '{word}' ({len(voices)} voices x {QWEN3_SAMPLES_PER_VOICE})...")
            run_qwen3_cloned(qwen3_model, word, str(neg_dir), prefix, voices)
        else:
            if len(existing) >= SAMPLES_PER_NEGATIVE:
                print(f"  [{prefix}] already has {len(existing)} samples, skipping")
                continue
            print(f"  Generating {SAMPLES_PER_NEGATIVE} hard negatives for '{word}' (piper)...")
            run_piper_prefixed(word, str(neg_dir), SAMPLES_PER_NEGATIVE, prefix)


def download_negative_datasets():
    """Download pre-generated negative spectrogram features from HuggingFace.

    These are zip files containing pre-built RaggedMmap folders with the expected
    directory structure: speech/training/speech_mmap/, etc.
    """
    neg_dir = DATA_DIR / "negative_datasets"
    neg_dir.mkdir(parents=True, exist_ok=True)

    hf_root = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
    archives = [
        "speech.zip",
        "no_speech.zip",
        "dinner_party.zip",
        "dinner_party_eval.zip",
    ]

    for fname in archives:
        short = fname.replace(".zip", "")
        target = neg_dir / short
        if target.exists() and any(target.iterdir()):
            print(f"  [{short}] already downloaded, skipping")
            continue

        zip_path = neg_dir / fname
        download_file(hf_root + fname, zip_path)
        print(f"  Extracting {fname}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(neg_dir)
        zip_path.unlink()  # remove zip after extraction


def generate_spectrograms():
    """Convert audio clips → augmented spectrograms → RaggedMmap.

    Follows the exact workflow from the micro-wake-word training notebook:
      Clips(input_directory) → Augmentation → SpectrogramGeneration
      → RaggedMmap.from_generator(spectrogram_generator(split, repeat))
    """
    try:
        from microwakeword.audio.clips import Clips
        from microwakeword.audio.spectrograms import SpectrogramGeneration
        from microwakeword.audio.augmentation import Augmentation
        from mmap_ninja.ragged import RaggedMmap
    except ImportError:
        print("Missing dependencies. Install:")
        print("  pip install -e micro-wake-word/")
        print("  pip install mmap_ninja")
        sys.exit(1)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    def _generate_mmaps(clip_dir: Path, features_name: str, label: str):
        """Generate train/validation/testing RaggedMmaps for a clip directory."""
        out_base = FEATURES_DIR / features_name

        if (out_base / "training" / "wakeword_mmap").exists():
            print(f"  {label} spectrograms already exist, skipping")
            return

        print(f"  Generating {label} spectrograms from {clip_dir.name}/...")

        clips = Clips(
            input_directory=str(clip_dir),
            file_pattern="*.wav",
            max_clip_duration_s=None,
            remove_silence=False,
            random_split_seed=10,
            split_count=0.1,
        )

        augmenter = Augmentation(augmentation_duration_s=3.2)

        for split, split_name, repetition, slide in [
            ("training",   "train",      2, 10),
            ("validation", "validation", 1, 10),
            ("testing",    "test",       1,  1),
        ]:
            out_dir = out_base / split
            out_dir.mkdir(parents=True, exist_ok=True)

            spec_gen = SpectrogramGeneration(
                clips=clips,
                augmenter=augmenter,
                slide_frames=slide,
                step_ms=10,
            )

            RaggedMmap.from_generator(
                out_dir=str(out_dir / "wakeword_mmap"),
                sample_generator=spec_gen.spectrogram_generator(
                    split=split_name, repeat=repetition
                ),
                batch_size=100,
                verbose=True,
            )

    # Positive samples
    pos_dir = DATA_DIR / "positive_samples"
    if pos_dir.exists() and any(pos_dir.glob("*.wav")):
        _generate_mmaps(pos_dir, "positive", "positive")

    # Hard negatives
    neg_dir = DATA_DIR / "hard_negative_samples"
    if neg_dir.exists() and any(neg_dir.glob("*.wav")):
        _generate_mmaps(neg_dir, "hard_negatives", "hard-negative")


def run_training():
    """Invoke micro-wake-word training pipeline.

    Runs from the micro-wake-word/ directory so that YAML paths resolve correctly.
    """
    if not CONFIG_FILE.exists():
        print(f"Config not found: {CONFIG_FILE}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "microwakeword.model_train_eval",
        f"--training_config={CONFIG_FILE}",
        "--train", "1",
        "--restore_checkpoint", "1",
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        "mixednet",
        "--pointwise_filters", "64,64,64,64",
        "--repeat_in_block", "1,1,1,1",
        "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
        "--residual_connection", "0,0,0,0",
        "--first_conv_filters", "32",
        "--first_conv_kernel_size", "5",
        "--stride", "3",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(MWW_DIR), check=True)


def export_model():
    """Export trained TFLite model with versioned snapshot.

    Creates:
        kws/models/versions/<timestamp>/micro_nova.tflite   (immutable archive)
        kws/models/micro_nova.tflite                        (active copy)
        kws/models/active_version.txt                       (points to timestamp)

    MicroKWS.load_version() can load any previous version by timestamp.
    """
    from datetime import datetime

    src = TRAINED_DIR / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
    if not src.exists():
        print(f"Trained model not found at {src}")
        print("Run training first: python train_micro_nova.py --train")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Versioned snapshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = MODELS_DIR / "versions" / timestamp
    version_dir.mkdir(parents=True, exist_ok=True)
    versioned_dst = version_dir / "micro_nova.tflite"
    shutil.copy2(src, versioned_dst)

    # Also copy best_weights for reference
    best_weights = TRAINED_DIR / "best_weights.weights.h5"
    if best_weights.exists():
        shutil.copy2(best_weights, version_dir / "best_weights.weights.h5")

    # Active copy
    active_dst = MODELS_DIR / "micro_nova.tflite"
    shutil.copy2(src, active_dst)

    # Update active version pointer
    version_file = MODELS_DIR / "active_version.txt"
    version_file.write_text(timestamp)

    size_kb = active_dst.stat().st_size / 1024
    print(f"Exported: {active_dst}  ({size_kb:.1f} KB)")
    print(f"Version:  {timestamp} → {versioned_dst}")
    print(f"Active:   {version_file} → {timestamp}")


def main():
    parser = argparse.ArgumentParser(description="Train a 'Nova' micro-wake-word model")
    parser.add_argument("--prepare", action="store_true", help="Generate/download training data")
    parser.add_argument("--train", action="store_true", help="Run the training pipeline")
    parser.add_argument("--export", action="store_true", help="Copy trained model to kws/models/")
    parser.add_argument("--all", action="store_true", help="Run full pipeline (prepare + train + export)")
    parser.add_argument(
        "--tts-engine", choices=["piper", "qwen3"], default="piper",
        help="TTS engine for sample generation (default: piper)",
    )
    parser.add_argument(
        "--voices-dir", type=Path, default=None,
        help="Directory of reference voice recordings for Qwen3-TTS cloning. "
             "Record 20-30 people saying random sentences (~10s each).",
    )
    parser.add_argument(
        "--qwen3-model", default=QWEN3_DEFAULT_MODEL,
        help=f"Qwen3-TTS model ID or local path (default: {QWEN3_DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    if not any([args.prepare, args.train, args.export, args.all]):
        args.prepare = True

    # Validate Qwen3-TTS args
    if args.tts_engine == "qwen3" and (args.prepare or args.all):
        if args.voices_dir is None:
            parser.error("--voices-dir is required when using --tts-engine qwen3")
        if not args.voices_dir.is_dir():
            parser.error(f"--voices-dir does not exist: {args.voices_dir}")

    # Load Qwen3 model once if needed
    qwen3_model = None
    voices = None
    if args.tts_engine == "qwen3" and (args.prepare or args.all):
        voices = _discover_voices(args.voices_dir)
        print(f"  Found {len(voices)} reference voices in {args.voices_dir}")
        qwen3_model = _load_qwen3_model(args.qwen3_model)

    if args.prepare or args.all:
        engine_label = "Qwen3-TTS voice cloning" if args.tts_engine == "qwen3" else "Piper TTS"
        print("=" * 60)
        print(f"Step 1: Generating positive samples via {engine_label}")
        print("=" * 60)
        generate_positive_samples(args.tts_engine, qwen3_model, voices)

        print(f"\nStep 2: Generating hard negatives via {engine_label}")
        generate_hard_negatives(args.tts_engine, qwen3_model, voices)

        print("\nStep 3: Downloading negative datasets from HuggingFace")
        download_negative_datasets()

        print("\nStep 4: Generating spectrograms")
        generate_spectrograms()

    if args.train or args.all:
        print("\n" + "=" * 60)
        print("Step 5: Training MixedNet model")
        print("=" * 60)
        run_training()

    if args.export or args.all:
        print("\n" + "=" * 60)
        print("Step 6: Exporting model")
        print("=" * 60)
        export_model()

    print("\nDone!")


if __name__ == "__main__":
    main()
