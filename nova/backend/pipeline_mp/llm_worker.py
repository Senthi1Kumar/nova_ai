"""
pipeline_mp/llm_worker.py
-------------------------
The real LLM worker. This file previously contained a patch-generator script
(BEFORE_CHANGE_*/AFTER_CHANGE_* string pairs meant to be applied to a *different*
target file) and never defined `run_llm_worker` — but `pipeline_mp/__init__.py`
imports that name unconditionally, so the whole `pipeline_mp` package failed to
import and the app couldn't start. See `_retired/README.md` for the full story.

Contract (matches the real callers):
  Input  (llm_in_queue)  : {"type": "text", "text": str, "audio_data"?: list[float],
                            "bio_score"?: float, "bio_pass"?: bool}
                            — pushed by the STT workers' DriveAuthGate.intercept()
  Output (tts_in_queue)  : {"type": "text_to_speak", "text": str}
                            {"type": "eof"}
                            — matches tts_worker.py's real input protocol
  Output (ws_out_queue)  : {"type": "assistant_start"}
                            {"type": "llm_token", "data": str, "latency"?: {...}}
                            {"type": "llm_route", "data": {"backend": str, "model": str}}
                            — matches the real frontend contract in
                            nova/frontend/index.html (cases 'assistant_start',
                            'llm_token', 'llm_route')

Payment gating (second layer): STT workers already run DriveAuthGate.intercept()
on the raw transcript before a message ever reaches this queue (first layer).
This worker re-checks payment intents with DriveAuthGate.require_auth() before
treating them as tool calls — covering the case where the dialogue manager
decomposes a multi-intent utterance into a payment sub-intent that the STT-level
regex on the ORIGINAL utterance didn't catch. Both layers use the same
DriveAuthGate class but call it independently, matching the original two-layer
design intent (see the deprecated-but-preserved comments this replaces).
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import multiprocessing as mp
import multiprocessing.synchronize
from queue import Empty
from typing import Any

logger = logging.getLogger("LLMWorker")

# Payment intent — triggers the second-layer DriveAuth gate before tool use.
_PAYMENT_INTENT_RE = re.compile(
    r"\b(order|buy|purchase|pay|send money|transfer|checkout"
    r"|coffee|burger|pizza|food|latte|cappuccino"
    r"|add to cart|top.?up|recharge)\b",
    re.IGNORECASE,
)

# Utterances that never need tool use / generation beyond a quick ack.
_CONVERSATIONAL_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|bye|goodbye|cool|nice)\b",
    re.IGNORECASE,
)

_SENTENCE_BOUNDARY = re.compile(r"[.!?\n]")


def _needs_tools(prompt: str) -> bool:
    """Return False if the prompt is clearly conversational and should never trigger a tool call."""
    stripped = prompt.strip()
    if len(stripped.split()) <= 3 and "?" not in stripped:
        return False
    if _CONVERSATIONAL_RE.match(stripped):
        return False
    return True


def _is_payment_intent(prompt: str) -> bool:
    """True when the utterance looks like a payment / purchase command."""
    return bool(_PAYMENT_INTENT_RE.search(prompt))


def run_llm_worker(
    llm_in_queue: mp.Queue,  # type: ignore[type-arg]
    tts_in_queue: mp.Queue,  # type: ignore[type-arg]
    ws_out_queue: mp.Queue,  # type: ignore[type-arg]
    stop_event: multiprocessing.synchronize.Event,
    tts_interrupt_event: multiprocessing.synchronize.Event | None = None,
):
    import setproctitle
    setproctitle.setproctitle("nova-llm-worker")

    # Same convention every real worker in this package uses: make sibling
    # modules (llm_config, driveauth, biometric_gate, dynamic_orchestrator)
    # importable bare, without needing pipeline_mp installed as a package.
    pipeline_mp_dir = os.path.dirname(os.path.abspath(__file__))
    if pipeline_mp_dir not in sys.path:
        sys.path.insert(0, pipeline_mp_dir)

    from llm_config import LOCAL_LLM_SETTINGS

    # ── Second-layer DriveAuth payment gate ────────────────────────────────
    driver_id = os.getenv("NOVA_DRIVER_ID", "driver1")
    gate_enabled = os.getenv("NOVA_BIO_GATE_ENABLED", "1") == "1"
    payment_gate = None
    if gate_enabled:
        try:
            from driveauth.gate import DriveAuthGate
            payment_gate = DriveAuthGate.load(driver_id=driver_id)
            logger.info(f"LLM worker: DriveAuthGate loaded (driver={driver_id})")
        except Exception as exc:
            logger.error(
                f"LLM worker: DriveAuthGate load failed ({exc}) — payment "
                "tool-use will NOT be re-gated at this layer. The STT-level "
                "gate (first layer) may still cover most cases."
            )
            payment_gate = None
    use_local_tools = os.getenv("NOVA_LLM_LOCAL_TOOLS", "1") == "1"

    def _payment_gate_check(prompt: str) -> bool:
        """Returns True if generation/tool-use should proceed."""
        if payment_gate is None:
            return True
        try:
            auth = payment_gate.require_auth(tier="payment")
        except Exception as exc:
            logger.error(f"LLM worker: payment gate error (failing open): {exc}")
            return True
        if auth.legacy_decision != "pass":
            if auth.legacy_decision == "step_up":
                msg = (
                    "I need to verify your identity before processing this "
                    "payment. I've sent a one-time code to your registered "
                    "mobile number — please read it out."
                )
            else:
                msg = "I couldn't verify your identity. Payment cancelled."
            ws_out_queue.put({"type": "tts_speak", "text": msg})
            ws_out_queue.put({"type": "recording_stopped"})
            logger.warning(
                f"LLM worker: payment blocked by DriveAuth gate — "
                f"decision={auth.decision.value} trust={auth.trust_score:.3f} "
                f"risk={auth.risk_score:.3f} rule={auth.policy_rule}"
            )
            return False
        return True

    # ── Local model (transformers, per llm_config registry) ────────────────
    model = None
    tokenizer = None
    active_cfg = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        active_cfg = LOCAL_LLM_SETTINGS.config
        logger.info(f"LLM worker: loading {active_cfg.model_id} ...")

        tokenizer = AutoTokenizer.from_pretrained(active_cfg.model_id)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": getattr(torch, active_cfg.compute_dtype, torch.bfloat16),
        }
        if active_cfg.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except Exception as bnb_exc:
                logger.warning(
                    f"LLM worker: 4-bit quantisation unavailable ({bnb_exc}) — "
                    "loading at full precision instead."
                )
        model = AutoModelForCausalLM.from_pretrained(
            active_cfg.model_id, device_map="auto", **load_kwargs,
        )
        logger.info(f"LLM worker: {active_cfg.model_id} ready.")
    except Exception as exc:
        logger.error(
            f"LLM worker: model load failed ({exc}) — running in degraded "
            "mode (acknowledges but cannot generate)."
        )
        model = None

    def _speak(text: str) -> None:
        """Push a finished utterance to TTS and close the generation."""
        if text.strip():
            tts_in_queue.put({"type": "text_to_speak", "text": text.strip()})
        tts_in_queue.put({"type": "eof"})

    def generate_response(prompt: str, bio_pass: bool = False) -> None:
        """
        Generate a streamed reply for ``prompt``. Mirrors the original
        two-layer gate design: the STT worker already ran DriveAuthGate on the
        raw utterance (first layer); this re-checks payment intents
        specifically at the tool-use boundary (second layer), since a
        multi-intent utterance may decompose into a payment sub-intent that
        wasn't obviously a payment in the original transcript.
        """
        if use_local_tools and _needs_tools(prompt) and _is_payment_intent(prompt):
            if not _payment_gate_check(prompt):
                return  # gate already notified the user; abort this turn

        ws_out_queue.put({"type": "assistant_start"})
        ws_out_queue.put({
            "type": "llm_route",
            "data": {
                "backend": "local",
                "model": active_cfg.model_id if active_cfg else "unavailable",
            },
        })

        if model is None or tokenizer is None:
            reply = "I'm having trouble reaching my language model right now."
            ws_out_queue.put({"type": "llm_token", "data": reply})
            _speak(reply)
            return

        t0 = time.time()
        try:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            ).to(model.device)

            from transformers import TextIteratorStreamer
            import threading as _threading

            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True,
            )
            sp = active_cfg.sampling
            gen_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                max_new_tokens=sp.max_new_tokens,
                temperature=sp.temperature,
                top_k=sp.top_k,
                top_p=sp.top_p,
                repetition_penalty=sp.repetition_penalty,
                do_sample=sp.do_sample,
            )
            gen_thread = _threading.Thread(
                target=model.generate, kwargs=gen_kwargs, daemon=True,
            )
            gen_thread.start()

            first_token = True
            sentence_buf = ""
            token_count = 0
            for token_text in streamer:
                if stop_event.is_set() or (tts_interrupt_event and tts_interrupt_event.is_set()):
                    logger.info("LLM worker: generation interrupted mid-stream.")
                    break
                token_count += 1
                latency: dict[str, float] = {}
                if first_token:
                    latency["llm_ttft"] = time.time() - t0
                    first_token = False
                ws_out_queue.put({"type": "llm_token", "data": token_text, "latency": latency})
                sentence_buf += token_text
                # Flush on sentence boundaries so TTS can start speaking before
                # the full response finishes generating.
                if _SENTENCE_BOUNDARY.search(token_text):
                    sentence = sentence_buf.strip()
                    if sentence:
                        tts_in_queue.put({"type": "text_to_speak", "text": sentence})
                    sentence_buf = ""

            if sentence_buf.strip():
                tts_in_queue.put({"type": "text_to_speak", "text": sentence_buf.strip()})
            tts_in_queue.put({"type": "eof"})

            dt = max(1e-3, time.time() - t0)
            throughput = token_count / dt
            ws_out_queue.put({"type": "llm_token", "data": "", "latency": {"llm_throughput": throughput}})
            logger.info(f"LLM worker: {token_count} tokens in {dt:.2f}s ({throughput:.1f} t/s)")
        except Exception as exc:
            logger.error(f"LLM worker: generation failed: {exc}")
            _speak("Sorry, something went wrong while I was thinking about that.")

    logger.info("LLM worker ready — waiting for messages.")
    while not stop_event.is_set():
        try:
            msg = llm_in_queue.get(timeout=0.1)
        except Empty:
            continue

        if not isinstance(msg, dict) or msg.get("type") != "text":
            continue

        prompt = (msg.get("text") or "").strip()
        if not prompt:
            continue

        try:
            generate_response(prompt, bio_pass=bool(msg.get("bio_pass", False)))
        except Exception as exc:
            logger.error(f"LLM worker: unhandled error processing message: {exc}")
            ws_out_queue.put({"type": "recording_stopped"})

    logger.info("LLM worker shutting down.")
