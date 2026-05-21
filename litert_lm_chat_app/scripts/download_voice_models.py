#!/usr/bin/env python3
"""Install/check voice dependencies and pre-download local voice models.

Fixes included:
- macOS Python certificate bootstrap helper
- certifi CA bundle option
- optional checksum-verified insecure fallback download for Whisper when a
  corporate/self-signed proxy breaks Python TLS verification

Usage:
  python scripts/download_voice_models.py --install --whisper-model base --pocket-tts --write-env
  python scripts/download_voice_models.py --fix-macos-certs --use-certifi --whisper-model base
  python scripts/download_voice_models.py --whisper-model base --insecure-whisper-download
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_WHISPER_ROOT = Path.home() / ".cache" / "whisper"


def run(cmd: list[str], check: bool = True) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def pip_install(packages: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", *packages])


def fix_macos_certificates() -> None:
    """Run the Python.org macOS Install Certificates.command if present."""
    if sys.platform != "darwin":
        print("ℹ️  --fix-macos-certs is only needed on macOS Python.org installs.")
        return

    candidates = [
        Path(sys.exec_prefix).parent / "Install Certificates.command",
        Path("/Applications") / f"Python {sys.version_info.major}.{sys.version_info.minor}" / "Install Certificates.command",
    ]
    for script in candidates:
        if script.exists():
            print(f"✅ Found macOS certificate installer: {script}")
            run(["/bin/bash", str(script)], check=False)
            return
    print("⚠️  Could not find Install Certificates.command.")
    print("   Try: python -m pip install --upgrade certifi")


def configure_certifi() -> None:
    try:
        import certifi  # type: ignore
    except Exception:
        print("Installing certifi...")
        pip_install(["certifi"])
        import certifi  # type: ignore

    cafile = certifi.where()
    os.environ["SSL_CERT_FILE"] = cafile
    os.environ["REQUESTS_CA_BUNDLE"] = cafile
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=cafile)  # type: ignore[attr-defined]
    print(f"✅ Using certifi CA bundle: {cafile}")


def check_ffmpeg() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(f"✅ ffmpeg found: {ffmpeg}")
        return
    print("⚠️  ffmpeg not found. Whisper needs ffmpeg for browser audio formats like webm.")
    if sys.platform == "darwin":
        print("   Install on macOS: brew install ffmpeg")
    elif sys.platform.startswith("linux"):
        print("   Install on Ubuntu/Debian: sudo apt update && sudo apt install -y ffmpeg")
    else:
        print("   Install ffmpeg and ensure it is on PATH.")


def _copy_stream_with_progress(source, output_path: Path) -> None:
    total = source.headers.get("Content-Length")
    total_i = int(total) if total and total.isdigit() else 0
    done = 0
    with output_path.open("wb") as output:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            done += len(chunk)
            if total_i:
                pct = done * 100 / total_i
                print(f"\r   Downloaded {done / 1024 / 1024:.1f} MB / {total_i / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r   Downloaded {done / 1024 / 1024:.1f} MB", end="", flush=True)
    print()


def _verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    actual = h.hexdigest()
    if actual != expected:
        path.unlink(missing_ok=True)
        raise SystemExit(f"❌ SHA256 mismatch for {path.name}\nExpected: {expected}\nActual:   {actual}")
    print(f"✅ SHA256 verified: {path.name}")


def insecure_download_whisper_model(whisper_module, model_name: str, download_root: Path) -> Path:
    """Download Whisper model with TLS verification disabled, then verify SHA256.

    This is intended only for environments where a corporate/self-signed TLS
    inspection proxy breaks Python certificate verification. Whisper model URLs
    include the expected SHA256 in the path, so we verify content integrity after
    download.
    """
    if model_name not in whisper_module._MODELS:  # noqa: SLF001
        choices = ", ".join(sorted(whisper_module._MODELS.keys()))
        raise SystemExit(f"❌ Unknown Whisper model '{model_name}'. Choices: {choices}")

    url = whisper_module._MODELS[model_name]  # noqa: SLF001
    expected_sha256 = url.split("/")[-2]
    download_root.mkdir(parents=True, exist_ok=True)
    target = download_root / Path(url).name

    if target.exists():
        try:
            _verify_sha256(target, expected_sha256)
            return target
        except SystemExit:
            print("Existing file failed checksum; re-downloading.")

    print("⚠️  Using --insecure-whisper-download: TLS certificate verification is disabled only for this download.")
    print("   The downloaded model will still be SHA256 verified before use.")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx) as source:  # noqa: S310 - intentional opt-in fallback
        _copy_stream_with_progress(source, target)
    _verify_sha256(target, expected_sha256)
    return target


def download_whisper(model_name: str, language: str | None, download_root: str | None, insecure: bool) -> None:
    print(f"\nDownloading/loading Whisper model: {model_name}")
    try:
        import whisper  # type: ignore
    except Exception as exc:
        raise SystemExit(f"❌ Cannot import whisper: {exc}\nRun: pip install openai-whisper") from exc

    root_path = Path(download_root).expanduser() if download_root else DEFAULT_WHISPER_ROOT
    try:
        model = whisper.load_model(model_name, download_root=str(root_path))
    except urllib.error.URLError as exc:
        msg = str(exc)
        if "CERTIFICATE_VERIFY_FAILED" in msg and insecure:
            insecure_download_whisper_model(whisper, model_name, root_path)
            model = whisper.load_model(model_name, download_root=str(root_path))
        else:
            raise SystemExit(
                "❌ Whisper download failed because Python could not verify the HTTPS certificate.\n\n"
                "Recommended macOS fix:\n"
                "  python scripts/download_voice_models.py --fix-macos-certs --use-certifi --whisper-model " + model_name + "\n\n"
                "If you are behind a corporate/self-signed proxy and still fail, use checksum-verified fallback:\n"
                "  python scripts/download_voice_models.py --whisper-model " + model_name + " --insecure-whisper-download --write-env\n\n"
                f"Original error: {exc}"
            ) from exc
    except ssl.SSLError as exc:
        raise SystemExit(
            "❌ SSL error while downloading Whisper. Try:\n"
            f"  python scripts/download_voice_models.py --fix-macos-certs --use-certifi --whisper-model {model_name}\n"
            f"Original error: {exc}"
        ) from exc

    print(f"✅ Whisper model ready: {model_name}")
    print(f"   Whisper cache: {root_path}")
    del model
    if language:
        print(f"   App language hint will be: {language}")


def download_pocket_tts(voice: str, language: str) -> None:
    print(f"\nDownloading/loading Pocket TTS model: language={language}, voice={voice}")
    try:
        from pocket_tts import TTSModel  # type: ignore
    except Exception as exc:
        raise SystemExit(f"❌ Cannot import pocket_tts: {exc}\nRun: pip install pocket-tts scipy") from exc

    try:
        try:
            model = TTSModel.load_model(language=language)
        except TypeError:
            model = TTSModel.load_model()
        state = model.get_state_for_audio_prompt(voice)
        print("✅ Pocket TTS model and voice state ready")
        del state, model
    except Exception as exc:
        raise SystemExit(
            f"❌ Pocket TTS failed to load voice/model: {exc}\n"
            "Try the built-in voice 'alba', or check the voice name/path."
        ) from exc


def update_env(args: argparse.Namespace) -> None:
    env_path = Path(args.env_file)
    values = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()
    values["WHISPER_MODEL"] = args.whisper_model
    values["WHISPER_LANGUAGE"] = args.whisper_language or ""
    values["WHISPER_DOWNLOAD_ROOT"] = str(Path(args.whisper_download_root).expanduser())
    values["TTS_ENABLED"] = "true" if args.pocket_tts and not args.skip_pocket_tts else values.get("TTS_ENABLED", "true")
    values["POCKET_TTS_VOICE"] = args.pocket_voice
    values["POCKET_TTS_LANGUAGE"] = args.pocket_language

    lines = []
    if env_path.exists():
        seen = set()
        for line in env_path.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                k = line.split("=", 1)[0].strip()
                if k in values:
                    lines.append(f"{k}={values[k]}")
                    seen.add(k)
                else:
                    lines.append(line)
            else:
                lines.append(line)
        for k, v in values.items():
            if k not in seen:
                lines.append(f"{k}={v}")
    else:
        lines = [f"{k}={v}" for k, v in values.items()]
    env_path.write_text("\n".join(lines).rstrip() + "\n")
    print(f"✅ Updated {env_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/check Whisper and Pocket TTS models for the LiteRT-LM voice app")
    parser.add_argument("--install", action="store_true", help="Install/upgrade voice dependencies with pip first")
    parser.add_argument("--fix-macos-certs", action="store_true", help="Run macOS Python Install Certificates.command when available")
    parser.add_argument("--use-certifi", action="store_true", help="Force Python HTTPS downloads to use certifi CA bundle")
    parser.add_argument("--insecure-whisper-download", action="store_true", help="Fallback for corporate/self-signed TLS proxies. Disables TLS verification for Whisper download only, then verifies SHA256.")
    parser.add_argument("--whisper-model", default=os.getenv("WHISPER_MODEL", "base"), help="Whisper model: tiny/base/small/medium/large")
    parser.add_argument("--whisper-language", default=os.getenv("WHISPER_LANGUAGE", ""), help="Optional language code, e.g. en")
    parser.add_argument("--whisper-download-root", default=os.getenv("WHISPER_DOWNLOAD_ROOT", str(DEFAULT_WHISPER_ROOT)), help="Directory for Whisper model cache")
    parser.add_argument("--skip-whisper", action="store_true", help="Do not download/check Whisper")
    parser.add_argument("--pocket-tts", action="store_true", default=True, help="Download/check Pocket TTS")
    parser.add_argument("--skip-pocket-tts", action="store_true", help="Do not download/check Pocket TTS")
    parser.add_argument("--pocket-voice", default=os.getenv("POCKET_TTS_VOICE", "alba"), help="Pocket TTS built-in voice or prompt path")
    parser.add_argument("--pocket-language", default=os.getenv("POCKET_TTS_LANGUAGE", "english"), help="Pocket TTS language")
    parser.add_argument("--env-file", default=".env", help="Path to .env file to update")
    parser.add_argument("--write-env", action="store_true", help="Write selected voice settings into .env")
    args = parser.parse_args()

    if args.fix_macos_certs:
        fix_macos_certificates()
    if args.install:
        pip_install(["openai-whisper", "pocket-tts", "scipy", "soundfile", "certifi"])
    if args.use_certifi:
        configure_certifi()

    check_ffmpeg()
    if not args.skip_whisper:
        download_whisper(args.whisper_model, args.whisper_language or None, args.whisper_download_root, args.insecure_whisper_download)
    if args.pocket_tts and not args.skip_pocket_tts:
        download_pocket_tts(args.pocket_voice, args.pocket_language)
    if args.write_env:
        update_env(args)

    print("\n✅ Voice setup completed.")
    print("Start the app with: python run.py")


if __name__ == "__main__":
    main()
