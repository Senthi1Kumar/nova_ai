from unittest.mock import MagicMock

import numpy as np
import torch

from app.config import Settings
from app.voice_service import PocketTTSService


def _make_service_with_mock_model(sample_rate: int = 24000):
    settings = Settings(tts_enabled=True, tts_output_dir="runtime/tts")
    svc = PocketTTSService(settings)
    fake_model = MagicMock()
    fake_model.sample_rate = sample_rate
    svc.model = fake_model
    svc.voice_state = object()
    return svc, fake_model


def test_synthesize_stream_yields_int16_le_pcm_chunks():
    svc, model = _make_service_with_mock_model()
    model.generate_audio_stream.return_value = iter([
        torch.tensor([0.0, 0.5, -0.5, 1.0], dtype=torch.float32),
        torch.tensor([-1.0, 0.25], dtype=torch.float32),
    ])

    chunks = list(svc.synthesize_stream("hello world"))

    assert len(chunks) == 2
    pcm0, sr0 = chunks[0]
    assert sr0 == 24000
    arr0 = np.frombuffer(pcm0, dtype="<i2")
    assert arr0.tolist() == [0, 16383, -16383, 32767]

    pcm1, _ = chunks[1]
    arr1 = np.frombuffer(pcm1, dtype="<i2")
    assert arr1.tolist() == [-32767, 8191]


def test_synthesize_stream_empty_text_yields_nothing():
    svc, model = _make_service_with_mock_model()
    chunks = list(svc.synthesize_stream("   "))
    assert chunks == []
    model.generate_audio_stream.assert_not_called()


def test_synthesize_stream_clips_out_of_range():
    svc, model = _make_service_with_mock_model()
    model.generate_audio_stream.return_value = iter([
        torch.tensor([2.0, -3.0], dtype=torch.float32),
    ])
    pcm, _ = next(svc.synthesize_stream("clip me"))
    arr = np.frombuffer(pcm, dtype="<i2")
    assert arr.tolist() == [32767, -32767]
