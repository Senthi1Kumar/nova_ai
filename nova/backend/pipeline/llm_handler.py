import logging
import time
import re
import os
import threading
from pipeline.base_handler import BaseHandler
from transformers import AutoTokenizer, TextIteratorStreamer

logger = logging.getLogger(__name__)

class LLMHandler(BaseHandler):
    """
    Handles Layer 7 Dialogue routing and Qwen/OpenRouter Text Generation.
    Reads from: `queue_in` (text transcript dicts from STT)
    Writes to: `queue_out` (streamed text sentences for TTS)
    """

    def setup(self, ws_queue, dm, llm_model, llm_tokenizer, openai_client=None):
        self.ws_queue = ws_queue
        self.dm = dm
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.openai_client = openai_client
        self.current_llm_name = "qwen/qwen3.5-9b"

    def process(self, stt_output):
        # Allow passing custom instructions/interrupts through the queue
        if isinstance(stt_output, dict) and stt_output.get("type") == "interrupted":
            yield {"type": "interrupted"}
            return

        if not isinstance(stt_output, dict) or "text" not in stt_output:
            return

        transcript = stt_output["text"]
        audio_np = stt_output.get("audio_np")

        is_verifying = self.dm and self.dm.state.fsm_state == "VERIFY"

        if not transcript and not is_verifying:
            self.ws_queue.put({"type": "recording_stopped"})
            return

        if self.dm:
            # If we are verifying, we MUST pass the audio buffer even if the user just mumbled
            safe_transcript = transcript if transcript else "voice verification audio"
            
            # Pass to L7
            dm_response = self.dm.process(safe_transcript, audio_buffer=audio_np)
            logger.info(f"DM Response: {dm_response}")

            # Send DM state update to UI
            self.ws_queue.put({
                "type": "dm_state", 
                "data": {
                    "fsm_state": dm_response.get("fsm_state"),
                    "intent": dm_response.get("intent"),
                    "routing": dm_response.get("routing"),
                    "entities": dm_response.get("entities"),
                    "otp": dm_response.get("otp"),
                    "driver_name": dm_response.get("driver_name"),
                }
            })

            if dm_response.get("intent") == "emergency":
                yield {"type": "interrupted", "trigger": "emergency"}
                time.sleep(0.1)
                yield {"type": "text_to_speak", "text": dm_response["nova_says"]}
                return

            if dm_response.get("intent") == "stop":
                yield {"type": "interrupted", "trigger": "stop"}
                return

            if dm_response.get("queued", False):
                yield {"type": "text_to_speak", "text": dm_response["nova_says"]}
                return

            if dm_response.get("intent") != "general_question":
                # Deterministic FSM Response (No LLM generation needed)
                yield {"type": "text_to_speak", "text": dm_response["nova_says"]}
                
                # Check if we should re-enable listening immediately (slot filling, etc)
                active_states = ["VERIFY", "VERIFY_PIN", "VERIFY_FACE", "SLOT_FILL", "CONFIRM_PENDING", "OTP_PENDING"]
                if self.dm.state.fsm_state in active_states:
                    yield {"type": "trigger_listen"}
                return

        # If it IS a general question, or we have no DM, use the LLM
        yield from self._generate_response(transcript)

    def _generate_response(self, prompt: str):
        if self.dm:
            self.dm.speaking_started()
            
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Nova, an ultra-fast AI voice assistant built into an EV dashboard. "
                    "You are directly connected to the car's CAN bus. "
                    "If asked about vehicle status (battery, gas, tire pressure, speed), you MUST invent a realistic reading (e.g., 'Battery is at 82%'). "
                    "NEVER say you are an AI, a language model, or that you lack access. "
                    "Keep answers incredibly concise and direct."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        
        self.ws_queue.put({"type": "assistant_start"})
        start_time = time.time()
        ttft_llm = None
        is_thinking = False
        sentence_buffer = ""
        tokens_count = 0

        # We will yield chunks of text (sentences) to the TTS
        if self.openai_client:
            # We are inside a standard thread, but the OpenAI client from main.py is an AsyncOpenAI client.
            # To run it here synchronously, we need a small event loop runner for the generator, or we use the synchronous client.
            # Actually, since we passed it in, we can run it synchronously using asyncio.run if it's the only async thing in this thread.
            import asyncio
            
            async def _run_openrouter():
                nonlocal tokens_count, ttft_llm, is_thinking, sentence_buffer
                stream = await self.openai_client.chat.completions.create(
                    model=self.current_llm_name,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True}
                )
                
                results = []
                async for chunk in stream:
                    if self.stop_event.is_set():
                        break
                        
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue
                        
                    content = delta.content
                    if content:
                        tokens_count += 1
                        if ttft_llm is None:
                            ttft_llm = time.time() - start_time
                        
                        latency_data = {"llm_ttft": ttft_llm}
                        elapsed = time.time() - (start_time + ttft_llm)
                        if elapsed > 0:
                            latency_data["llm_throughput"] = tokens_count / elapsed

                        self.ws_queue.put({
                            "type": "llm_token",
                            "data": content,
                            "latency": latency_data,
                        })

                        if "<think>" in content:
                            is_thinking = True
                        if not is_thinking:
                            sentence_buffer += content
                            if any(p in content for p in [".", "!", "?", "\n", ","]):
                                clean = self.clean_for_tts(sentence_buffer)
                                if clean:
                                    results.append({"type": "text_to_speak", "text": clean})
                                sentence_buffer = ""
                        if "</think>" in content:
                            is_thinking = False
                            sentence_buffer = ""
                
                if sentence_buffer and not self.stop_event.is_set():
                    clean = self.clean_for_tts(sentence_buffer)
                    if clean:
                        results.append({"type": "text_to_speak", "text": clean})
                return results
                
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            results = loop.run_until_complete(_run_openrouter())
            for r in results:
                yield r

        elif self.llm_model and self.llm_tokenizer:
            text_prompt = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.llm_tokenizer(
                text_prompt, return_tensors="pt", add_special_tokens=False
            ).to("cuda")

            streamer = TextIteratorStreamer(
                self.llm_tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.0,
            )
            
            # Start generation in background thread
            gen_thread = threading.Thread(target=self.llm_model.generate, kwargs=generation_kwargs)
            gen_thread.start()

            for new_text in streamer:
                # Stop checking interrupt_flag directly; the run loop handles it via stop_event or queue
                if self.stop_event.is_set():
                    break
                    
                tokens_count += 1
                if ttft_llm is None:
                    ttft_llm = time.time() - start_time
                
                latency_data = {"llm_ttft": ttft_llm}
                elapsed = time.time() - (start_time + ttft_llm)
                if elapsed > 0:
                    latency_data["llm_throughput"] = tokens_count / elapsed

                self.ws_queue.put({
                    "type": "llm_token",
                    "data": new_text,
                    "latency": latency_data,
                })

                if "<think>" in new_text:
                    is_thinking = True
                if not is_thinking:
                    sentence_buffer += new_text
                    if any(p in new_text for p in [".", "!", "?", "\n", ","]):
                        clean = self.clean_for_tts(sentence_buffer)
                        if clean:
                            yield {"type": "text_to_speak", "text": clean}
                        sentence_buffer = ""
                if "</think>" in new_text:
                    is_thinking = False
                    sentence_buffer = ""

            if sentence_buffer and not self.stop_event.is_set():
                clean = self.clean_for_tts(sentence_buffer)
                if clean:
                    yield {"type": "text_to_speak", "text": clean}
                    
        # Check buffer from DM if they asked a question while speaking earlier
        if self.dm:
            next_response = self.dm.speaking_done()
            if next_response:
                logger.info(f"LLM: DM Buffered Response: {next_response}")
                if next_response.get("intent") == "general_question":
                    original_text = next_response.get("original_text", "")
                    if original_text:
                        yield from self._generate_response(original_text)
                    else:
                        yield {"type": "text_to_speak", "text": next_response["nova_says"]}
                else:
                    yield {"type": "text_to_speak", "text": next_response["nova_says"]}

    def clean_for_tts(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.replace("<think>", "").replace("</think>", "")
        text = re.sub(r'\bAC\b', 'A C', text, flags=re.IGNORECASE)
        text = re.sub(r'\bEV\b', 'E V', text)
        text = re.sub(r'\bUI\b', 'U I', text)
        text = re.sub(r'\bAPI\b', 'A P I', text)
        text = re.sub(r'([a-zA-Z]),([a-zA-Z])', r'\1, \2', text)
        return text.strip()
