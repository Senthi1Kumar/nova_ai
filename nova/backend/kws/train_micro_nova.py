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

Usage:
    cd nova/backend/kws
    python train_micro_nova.py                # Step 1: prepare data
    python train_micro_nova.py --train        # Step 2: train model
    python train_micro_nova.py --export       # Step 3: copy model to kws/models/

The full pipeline (data prep → train → export) can also be run in one shot:
    python train_micro_nova.py --all
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
    "noh vuh", "noh_vuh", "no_va",
    "hey nova", "hey Nova", "Hey Nova",
    "Hi Nova", "hi nova", "hi Nova",
    "noah", "hey noah", "hey Noah"
]

# Hard negatives — confusable words the model must learn to reject
HARD_NEGATIVES = [
    "Nava", "Never", "Over", "Mover", "Rover",
    "Sofa", "Lova", "Nola", "Boba", "Dova", "Cova",
    "No way", "No uh", "Motor", "Nota",
    "Hello", "Okay", "Hey there",
    "Alexa", "Hey Siri", "Hey Google", "Hey Bixby",
]

SAMPLES_PER_POSITIVE = 250   # per word variant
SAMPLES_PER_NEGATIVE = 100   # per hard negative word


# ── Helpers ──────────────────────────────────────────────────────────────────

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

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(tmp_dir).glob("*.wav")):
        dest = Path(output_dir) / f"{prefix}_{f.name}"
        shutil.move(str(f), str(dest))
    shutil.rmtree(tmp_dir, ignore_errors=True)


def download_file(url: str, dest: Path):
    """Download a file using wget for reliable redirect handling."""
    print(f"  Downloading {dest.name} ...")
    subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)


# ── Data Generation ──────────────────────────────────────────────────────────

def generate_positive_samples():
    """Generate positive wake-word audio using Piper TTS.

    All variants are placed into ONE flat directory with prefixed filenames
    so that Clips(input_directory=...) can load them all at once.
    """
    ensure_piper_model()
    pos_dir = DATA_DIR / "positive_samples"

    for word in POSITIVE_WORDS:
        prefix = word.replace(" ", "_").lower()
        # Check if this variant already generated
        existing = list(pos_dir.glob(f"{prefix}_*.wav")) if pos_dir.exists() else []
        if len(existing) >= SAMPLES_PER_POSITIVE:
            print(f"  [{prefix}] already has {len(existing)} samples, skipping")
            continue

        print(f"  Generating {SAMPLES_PER_POSITIVE} samples for '{word}'...")
        run_piper_prefixed(word, str(pos_dir), SAMPLES_PER_POSITIVE, prefix)


def generate_hard_negatives():
    """Generate hard-negative audio using Piper TTS.

    All variants into ONE flat directory, same pattern as positives.
    """
    ensure_piper_model()
    neg_dir = DATA_DIR / "hard_negative_samples"

    for word in HARD_NEGATIVES:
        prefix = word.replace(" ", "_").lower()
        existing = list(neg_dir.glob(f"{prefix}_*.wav")) if neg_dir.exists() else []
        if len(existing) >= SAMPLES_PER_NEGATIVE:
            print(f"  [{prefix}] already has {len(existing)} samples, skipping")
            continue

        print(f"  Generating {SAMPLES_PER_NEGATIVE} hard negatives for '{word}'...")
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
    """Copy the trained TFLite model to kws/models/micro_nova.tflite."""
    src = TRAINED_DIR / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
    if not src.exists():
        print(f"Trained model not found at {src}")
        print("Run training first: python train_micro_nova.py --train")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_DIR / "micro_nova.tflite"
    shutil.copy2(src, dst)
    size_kb = dst.stat().st_size / 1024
    print(f"Exported: {dst}  ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Train a 'Nova' micro-wake-word model")
    parser.add_argument("--prepare", action="store_true", help="Generate/download training data")
    parser.add_argument("--train", action="store_true", help="Run the training pipeline")
    parser.add_argument("--export", action="store_true", help="Copy trained model to kws/models/")
    parser.add_argument("--all", action="store_true", help="Run full pipeline (prepare + train + export)")
    args = parser.parse_args()

    if not any([args.prepare, args.train, args.export, args.all]):
        args.prepare = True

    if args.prepare or args.all:
        print("=" * 60)
        print("Step 1: Generating positive samples via Piper TTS")
        print("=" * 60)
        generate_positive_samples()

        print("\nStep 2: Generating hard negatives via Piper TTS")
        generate_hard_negatives()

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
