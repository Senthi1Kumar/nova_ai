#!/bin/bash

echo "Patching pocket-tts submodule for CUDA compatibility..."
FILE="pocket-tts/pyproject.toml"

if [ -f "$FILE" ]; then
    # 1. Comment out the [tool.uv.sources] block, stopping exactly at its closing bracket
    sed -i '/^\[tool\.uv\.sources\]/,/^\]/ s/^/# /' "$FILE"
    
    # 2. Comment out the [[tool.uv.index]] block, stopping exactly at "explicit = true"
    sed -i '/^\[\[tool\.uv\.index\]\]/,/^explicit = true/ s/^/# /' "$FILE"
    
    # Clean up any accidental double-comments if the script is run twice
    sed -i 's/^# # /# /g' "$FILE"
    
    echo "Patch applied successfully! You can now run 'uv sync'."
else
    echo "Error: $FILE not found. Did you run 'git submodule update --init --recursive'?"
    exit 1
fi