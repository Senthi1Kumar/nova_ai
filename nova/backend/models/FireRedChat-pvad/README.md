---
license: apache-2.0
language:
- zh
- en
base_model:
- speechbrain/spkrec-ecapa-voxceleb
tags:
- agent
- voice-activity-detection
---

<div align="center">
<h1>FireRedChat-pVAD</h1>
</div>
<div align="center">
  <a href="https://fireredteam.github.io/demos/firered_chat/">Demo</a> •
  <a href="https://arxiv.org/pdf/2509.06502">Paper</a> •
  <a href="https://huggingface.co/FireRedTeam">Huggingface</a>
</div>

## Descriptions
FireRedChat's personalized Voice Activity Detection (pVAD) model, an open-weight model for detecting voice activity with speaker embedding updates.. [LiveKit plugin available here](https://github.com/fireredchat-submodules/livekit-plugins-fireredchat-pvad)

- Supports speaker embedding updates for improved voice activity detection.
- The plugin requires a compatible LiveKit Agents fork or modification to include `update_speaker` call for the first user utterance.

## Roadmap
- [x] 2025/09
  - [x] Release the pVAD model weights and LiveKit plugin.

## Usage
For inference, please use the [LiveKit plugin](https://github.com/fireredchat-submodules/livekit-plugins-fireredchat-pvad). Install and configure as follows:

```python
from livekit.plugins import fireredchat_pvad as pvad

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = pvad.VAD.load(activation_threshold=0.5)

# After the first utterance (or when primary speaker switches based on RMS), call VADStream's update_speaker() to update speaker embedding.
```

## License
The model weights and plugin code are licensed under the Apache-2.0 license.

### Acknowledgment
- Speaker embedding model: speechbrain/spkrec-ecapa-voxceleb

---