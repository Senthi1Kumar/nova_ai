import multiprocessing as mp
import multiprocessing.synchronize
import time
import numpy as np
import logging
import torch
from queue import Empty

logger = logging.getLogger("STTWorker")

def run_stt_worker(
    stt_in_queue: mp.Queue,  # type: ignore[type-arg]
    llm_in_queue: mp.Queue,  # type: ignore[type-arg]
    ws_out_queue: mp.Queue,  # type: ignore[type-arg]
    stop_event: multiprocessing.synchronize.Event,
):
    """
    Multiprocessing worker for HF Moonshine STT.
    """
    import setproctitle
    setproctitle.setproctitle("nova-stt-worker")

    from transformers import MoonshineStreamingForConditionalGeneration, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info(f"Initializing HF Moonshine STT (Streaming Small) on {device}...")
    model_id = "UsefulSensors/moonshine-streaming-small"

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = MoonshineStreamingForConditionalGeneration.from_pretrained(model_id).to(device).to(dtype)

        # Silero VAD runs on CPU — avoids nvrtc JIT compilation issues on systems
        # where CUDA runtime is present but libnvrtc-builtins is not installed.
        vad_model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
        vad_model = vad_model.cpu()

        logger.info("HF Moonshine STT & VAD initialized and waiting for commands.")
    except Exception as e:
        logger.error(f"Failed to load HF Moonshine STT or VAD: {e}")
        return

    session_active = False
    is_ptt = False
    audio_buffer: list[bytes] = []
    silence_chunks = 0
    max_silence_chunks = 4

    while not stop_event.is_set():
        try:
            msg = stt_in_queue.get(timeout=0.05)
        except Empty:
            continue

        if isinstance(msg, dict):
            if msg.get("type") == "start":
                new_ptt = msg.get("ptt", False)
                if not session_active:
                    session_active = True
                    is_ptt = new_ptt
                    audio_buffer = []
                    silence_chunks = 0
                    logger.info(f"STT Session Started (PTT: {is_ptt})")
                elif new_ptt != is_ptt:
                    # Session already active — update PTT mode without resetting buffer
                    is_ptt = new_ptt
                    silence_chunks = 0
                    logger.info(f"STT PTT mode changed to {is_ptt}")
            elif msg.get("type") == "stop":
                if session_active:
                    session_active = False
                    logger.info("STT Session Stopped. Running inference...")

                    if len(audio_buffer) > 0:
                        full_audio = b"".join(audio_buffer)
                        audio_int16 = np.frombuffer(full_audio, dtype=np.int16)
                        audio_np = audio_int16.astype(np.float32) / 32768.0

                        t0 = time.time()
                        try:
                            inputs = processor(
                                audio_np,
                                return_tensors="pt",
                                sampling_rate=16000,
                            )
                            # Move inputs to device and cast to dtype if it's float
                            inputs = {k: (v.to(device).to(dtype) if v.dtype in [torch.float32, torch.float16] else v.to(device)) for k, v in inputs.items()}

                            token_limit_factor = 6.5 / 16000
                            seq_lens = inputs["attention_mask"].sum(dim=-1)
                            max_length = int((seq_lens * token_limit_factor).max().item())
                            max_length = max(max_length, 20)

                            with torch.no_grad():
                                generated_ids = model.generate(**inputs, max_length=max_length)

                            transcript = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
                            ttfb = time.time() - t0
                            audio_dur = len(audio_np) / 16000.0
                            rtf = ttfb / audio_dur if audio_dur > 0 else 0

                            logger.info(f"STT Final: '{transcript}' | Latency: {ttfb:.2f}s | RTF: {rtf:.2f}")
                            ws_out_queue.put({
                                "type": "transcript",
                                "data": transcript,
                                "latency": {"stt_ttfb": ttfb, "stt_rtf": rtf}
                            })

                            if transcript:
                                llm_in_queue.put({
                                    "type": "text",
                                    "text": transcript,
                                    "audio_data": audio_np.tolist(),
                                })
                                ws_out_queue.put({"type": "generation_start"})
                            else:
                                ws_out_queue.put({"type": "recording_stopped"})
                        except Exception as e:
                            logger.error(f"STT Inference error: {e}")
                            ws_out_queue.put({"type": "recording_stopped"})
                    else:
                        ws_out_queue.put({"type": "recording_stopped"})
            elif msg.get("type") == "interrupted":
                if session_active:
                    session_active = False
                    audio_buffer = []
                    logger.info("STT Session Interrupted")
        elif isinstance(msg, bytes):
            if session_active:
                audio_buffer.append(msg)

                if not is_ptt:
                    # Run VAD on CPU to check silence
                    audio_int16 = np.frombuffer(msg, dtype=np.int16)
                    audio_np = audio_int16.astype(np.float32) / 32768.0
                    vad_chunk = audio_np[-512:] if len(audio_np) >= 512 else np.pad(audio_np, (0, 512 - len(audio_np)))
                    with torch.no_grad():
                        prob = vad_model(torch.from_numpy(vad_chunk).unsqueeze(0), 16000).item()

                    if prob < 0.4:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0

                    if silence_chunks >= max_silence_chunks:
                        logger.info("VAD Auto-stop triggered by silence.")
                        stt_in_queue.put({"type": "stop"})

    logger.info("STT Worker exiting.")
