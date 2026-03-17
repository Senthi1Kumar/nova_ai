import multiprocessing as mp
import multiprocessing.synchronize
import time
import numpy as np
import logging
import os
import sys
from queue import Empty
from pathlib import Path
import setproctitle

logger = logging.getLogger("KWSWorker")

def run_kws_worker(
    kws_in_queue: mp.Queue,  # type: ignore[type-arg]
    ws_out_queue: mp.Queue,  # type: ignore[type-arg]
    stop_event: multiprocessing.synchronize.Event,
    bypass_kws: bool,
):
    setproctitle.setproctitle("nova-kws-worker")

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    if bypass_kws:
        logger.info("KWS Bypassed. It will not trigger automatically.")
        while not stop_event.is_set():
            time.sleep(1)
        return

    from kws.kws_engine_v2 import StreamingKWSv2
    from kws.kws_engine_v2 import GoogleEmbeddingModel

    kws_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "kws", "google_speech_embedding.onnx"
    )
    kws_engine = StreamingKWSv2(model_path=None)  # type: ignore[arg-type]
    kws_engine.model = GoogleEmbeddingModel(kws_model_path)
    
    if not kws_engine.load_model():
        logger.warning("KWS MLP/Refs not found. Please enroll first.")
    else:
        logger.info("KWS Model loaded successfully.")

    state = "IDLE"

    while not stop_event.is_set():
        try:
            msg = kws_in_queue.get(timeout=0.05)
        except Empty:
            continue

        if isinstance(msg, dict):
            if msg.get("type") == "enroll":
                refs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kws", "refs")
                ref_paths = sorted([str(p) for p in Path(refs_dir).glob("nova_*.wav")])
                noise_paths = sorted([str(p) for p in Path(refs_dir).glob("noise_*.wav")])
                if ref_paths:
                    logger.info("KWS Worker running enrollment...")
                    version_tag: str = msg.get("version_tag") or ""
                    kws_engine.enroll(ref_paths, noise_paths, version_tag=version_tag or None)  # type: ignore[arg-type]
            elif msg.get("type") == "activate_version":
                success = kws_engine.load_version(msg["version"])
                ws_out_queue.put({"type": "kws_version_activated", "version": msg["version"], "success": success})
            elif msg.get("type") == "set_state":
                state = msg.get("state", "IDLE")
                # We can keep KWS unmuted even in GENERATING state to allow barge-in via Wake Word
                if state == "LISTENING":
                    kws_engine.mute()
                else:
                    kws_engine.unmute()
        elif isinstance(msg, bytes):
            if state in ["IDLE", "GENERATING"]:
                audio_int16 = np.frombuffer(msg, dtype=np.int16)
                audio_np = audio_int16.astype(np.float32) / 32768.0
                detected, latency = kws_engine.process_chunk(audio_np)
                if detected:
                    logger.info(f"Wake word detected! Latency: {latency:.2f}ms")
                    ws_out_queue.put({
                        "type": "kws_detected",
                        "data": {"latency": latency}
                    })
                    kws_engine.mute()
