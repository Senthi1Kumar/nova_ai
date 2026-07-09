#!/bin/bash

# Patches the micro-wake-word submodule for Nova compatibility:
#   inference.py — lazy-import generate_features_for_clip to avoid pulling in
#                  webrtcvad (training dep) at module load time
#   setup.py     — move heavy training deps to extras_require[train] so
#                  the inference-only install stays lean

set -e

SUBMODULE="nova/backend/kws/micro-wake-word"
INFERENCE="$SUBMODULE/microwakeword/inference.py"
SETUP="$SUBMODULE/setup.py"

if [ ! -f "$INFERENCE" ] || [ ! -f "$SETUP" ]; then
    echo "Error: submodule files not found. Did you run:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

echo "Patching $INFERENCE ..."

python3 - <<'PYEOF'
import sys

path = "nova/backend/kws/micro-wake-word/microwakeword/inference.py"
with open(path) as f:
    src = f.read()

if "_get_generate_features" in src:
    print("inference.py already patched — skipping.")
    sys.exit(0)

# 1. Remove eager top-level import
src = src.replace(
    "from microwakeword.audio.audio_utils import generate_features_for_clip\n",
    "",
)

# 2. Insert lazy wrapper after ai_edge_litert import (correct position)
lazy_fn = (
    "\n\ndef _get_generate_features():\n"
    '    """Lazy import to avoid pulling in webrtcvad (training dep) at module load."""\n'
    "    from microwakeword.audio.audio_utils import generate_features_for_clip\n"
    "    return generate_features_for_clip\n"
)
src = src.replace(
    "from ai_edge_litert.interpreter import Interpreter\n",
    "from ai_edge_litert.interpreter import Interpreter\n" + lazy_fn,
)

# 3. Replace direct call with lazy call
src = src.replace(
    "spectrogram = generate_features_for_clip(data, step_ms=step_ms)",
    "spectrogram = _get_generate_features()(data, step_ms=step_ms)",
)

with open(path, "w") as f:
    f.write(src)

print("inference.py patched.")
PYEOF

echo "Patching $SETUP ..."

# Rewrite setup.py install_requires / extras_require in-place using python
python3 - <<'PYEOF'
import re, sys

setup_path = "nova/backend/kws/micro-wake-word/setup.py"
with open(setup_path) as f:
    src = f.read()

# Only patch if not already patched (check for extras_require marker)
if "extras_require" in src:
    print("setup.py already patched — skipping.")
    sys.exit(0)

new_src = re.sub(
    r'install_requires=\[.*?\],',
    '''install_requires=[
        "numpy",
        "pymicro-features",
        "ai-edge-litert",
    ],
    extras_require={
        "train": [
            "audiomentations",
            "audio_metadata",
            "datasets",
            "mmap_ninja",
            "pyyaml",
            "tensorflow>=2.16",
            "webrtcvad-wheels",
        ],
    },''',
    src,
    flags=re.DOTALL,
)

with open(setup_path, "w") as f:
    f.write(new_src)

print("setup.py patched.")
PYEOF

# Create missing __init__.py files (upstream omits them, breaks Python imports)
for pkg in "$SUBMODULE/microwakeword/audio" "$SUBMODULE/microwakeword/layers"; do
    if [ ! -f "$pkg/__init__.py" ]; then
        touch "$pkg/__init__.py"
        echo "Created $pkg/__init__.py"
    else
        echo "$pkg/__init__.py already exists — skipping."
    fi
done

echo "Patch applied successfully!"
echo "You can now run 'uv sync'."
