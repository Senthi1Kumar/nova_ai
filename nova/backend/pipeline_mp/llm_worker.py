import multiprocessing as mp
import time
import logging
import os
import sys
import re
import json
from queue import Empty

# Monkeypatch torchaudio for speechbrain compatibility
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]  # type: ignore[attr-defined]

logger = logging.getLogger("LLMWorker")


# ── Tool definitions for OpenRouter function calling ─────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this for: "
                "recent news, financial updates, stock prices, weather, "
                "current events, sports scores, or any question requiring "
                "up-to-date information beyond your training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# Models that support tool/function calling via OpenRouter
TOOL_CAPABLE_MODELS = {
    "qwen/qwen3.5-9b",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-2.5-flash",
}


def _execute_web_search(query: str, max_results: int = 5) -> str:
    """Execute a DuckDuckGo search. Uses news() for news queries, text() otherwise."""
    try:
        from ddgs import DDGS
        results = []
        query_lower = query.lower()
        is_news = any(kw in query_lower for kw in [
            "news", "headline", "latest", "breaking", "today",
            "current events", "what happened",
        ])

        with DDGS() as ddgs:
            if is_news:
                for r in ddgs.news(query, max_results=min(max_results, 10)):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "source": r.get("source", ""),
                        "date": r.get("date", ""),
                    })
            else:
                for r in ddgs.text(query, max_results=min(max_results, 10)):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    })
        if not results:
            return json.dumps({"error": "No results found", "query": query})
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return json.dumps({"error": str(e), "query": query})


TOOL_MAPPING = {
    "web_search": _execute_web_search,
}


def run_llm_worker(llm_in_queue: "mp.Queue[dict]", tts_in_queue: "mp.Queue[dict]", ws_out_queue: "mp.Queue[dict]", stop_event: "mp.Event", tts_interrupt_event: "mp.Event | None" = None) -> None:
    import setproctitle
    setproctitle.setproctitle("nova-llm-worker")

    # Setup paths
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(backend_dir)
    sys.path.append(os.path.join(backend_dir, "nova-l7", "L-7"))

    # Init OpenRouter / LLM
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    openai_client = None
    llm_model = None
    llm_tokenizer = None
    current_llm_name = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

    if api_key:
        openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        logger.info("OpenRouter initialized in LLM worker.")
    else:
        logger.warning("No OpenRouter API key found. Initializing local Qwen fallback...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        primary_model = "Qwen/Qwen3.5-0.8B"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(primary_model)
            llm_model = AutoModelForCausalLM.from_pretrained(
                primary_model, quantization_config=bnb_config, device_map="auto"
            )
            logger.info("Local fallback model loaded.")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")

    # Init Dialogue Manager
    try:
        from dialogue_manager import DialogueManager
        dm = DialogueManager()
        # Skip the auto-greeting on first utterance — let the user's actual
        # question go through intent classification instead of being replaced
        # by a canned "Hello, driver!" response.
        dm.state.greeted = True
        logger.info("Dialogue Manager initialized (auto-greeting disabled).")
    except Exception as e:
        logger.error(f"Failed to init Dialogue Manager: {e}")
        dm = None

    def clean_for_tts(text: str) -> str:
        """Strip markdown, URLs, and non-speakable chars so TTS gets clean prose."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.replace("<think>", "").replace("</think>", "")
        # Remove markdown links: [text](url) → text
        text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
        # Remove bare URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove markdown bold/italic markers
        text = re.sub(r'\*{1,3}', '', text)
        text = re.sub(r'_{1,3}', ' ', text)
        # Remove markdown headers
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        # Remove bullet points and numbered list markers
        text = re.sub(r'^\s*[-*•]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+[.)]\s*', '', text, flags=re.MULTILINE)
        # Remove inline code backticks
        text = re.sub(r'`+', '', text)
        # Remove stray markdown/special chars that produce noise
        text = re.sub(r'[/\\|<>{}[\]~^]', ' ', text)
        # Collapse multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text)
        # Abbreviation expansions
        text = re.sub(r'\bAC\b', 'A C', text, flags=re.IGNORECASE)
        text = re.sub(r'\bEV\b', 'E V', text)
        text = re.sub(r'\bUI\b', 'U I', text)
        text = re.sub(r'\bAPI\b', 'A P I', text)
        text = re.sub(r'([a-zA-Z]),([a-zA-Z])', r'\1, \2', text)
        text = text.strip()
        # Skip fragments that are too short or have no real words (e.g. '/', ')')
        if len(text) < 3 or not re.search(r'[a-zA-Z]{2,}', text):
            return ""
        return text

    def _stream_and_speak(stream, start_time):
        """Stream LLM response tokens to UI and TTS.

        Returns (tokens_count, ttft, tool_calls_list).
        tool_calls_list is non-empty if the model requested tool calls instead of text.
        """
        ttft_llm = None
        is_thinking = False
        sentence_buffer = ""
        tokens_count = 0
        # Accumulate tool calls from streaming deltas
        tool_calls_acc = {}  # index -> {id, name, arguments_parts}

        for chunk in stream:
            if stop_event.is_set():
                break
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = choice.delta
            if not delta:
                continue

            # Handle tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": tc_delta.id or "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments
                continue

            content = delta.content
            if content:
                tokens_count += 1
                if ttft_llm is None:
                    ttft_llm = time.time() - start_time

                elapsed = time.time() - (start_time + ttft_llm)
                latency_data = {"llm_ttft": ttft_llm}
                if elapsed > 0:
                    latency_data["llm_throughput"] = tokens_count / elapsed

                ws_out_queue.put({"type": "llm_token", "data": content, "latency": latency_data})

                if "<think>" in content:
                    is_thinking = True
                if not is_thinking:
                    sentence_buffer += content
                    if any(p in content for p in [".", "!", "?", "\n", ","]):
                        clean = clean_for_tts(sentence_buffer)
                        if clean:
                            tts_in_queue.put({"type": "text_to_speak", "text": clean})
                        sentence_buffer = ""
                if "</think>" in content:
                    is_thinking = False
                    sentence_buffer = ""

        if sentence_buffer and not stop_event.is_set():
            clean = clean_for_tts(sentence_buffer)
            if clean:
                tts_in_queue.put({"type": "text_to_speak", "text": clean})

        # Convert accumulated tool calls to list
        tool_calls_list = []
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            if tc["name"]:
                tool_calls_list.append(tc)

        return tokens_count, ttft_llm, tool_calls_list

    def generate_response(prompt: str):
        if dm:
            dm.speaking_started()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Nova, an ultra-fast AI voice assistant built into an EV dashboard. "
                    "You are directly connected to the car's CAN bus. "
                    "If asked about vehicle status (battery, gas, tire pressure, speed), you MUST invent a realistic reading (e.g., 'Battery is at 82%'). "
                    "NEVER say you are an AI, a language model, or that you lack access. "
                    "Keep answers incredibly concise and direct — 2-3 sentences max. "
                    "IMPORTANT: Your output is read aloud by TTS. NEVER use markdown, bullet points, "
                    "numbered lists, URLs, links, or special formatting. Write plain conversational sentences only. "
                    "You have a web_search tool — use it for any question about current events, "
                    "news, financial markets, stock prices, weather, or anything requiring recent data. "
                    "After getting search results, synthesize a brief spoken summary in plain sentences — no lists or links."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        ws_out_queue.put({"type": "assistant_start"})
        start_time = time.time()

        try:
            if openai_client:
                use_tools = current_llm_name in TOOL_CAPABLE_MODELS

                # Stream-first approach: always stream, detect tool calls from deltas.
                # For non-tool queries (the common case), tokens arrive immediately — no double call.
                for _round in range(3):
                    if stop_event.is_set():
                        break

                    stream_kwargs = {
                        "model": current_llm_name,
                        "messages": messages,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    if use_tools:
                        stream_kwargs["tools"] = TOOLS
                        stream_kwargs["tool_choice"] = "auto"

                    stream = openai_client.chat.completions.create(**stream_kwargs)  # type: ignore[call-overload]
                    _tokens, _ttft, tool_calls = _stream_and_speak(stream, start_time)

                    if not tool_calls:
                        # Normal text response — already streamed to UI + TTS
                        break

                    # Model requested tool calls — execute and loop
                    # Build assistant message with tool_calls for the conversation
                    tc_objects = []
                    for tc in tool_calls:
                        tc_objects.append({
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        })
                    messages.append({"role": "assistant", "tool_calls": tc_objects})  # type: ignore[arg-type]

                    ws_out_queue.put({"type": "llm_token", "data": "[searching...] ",
                                      "latency": {"llm_ttft": time.time() - start_time}})

                    for tc in tool_calls:
                        fn_name = tc["name"]
                        fn_args = json.loads(tc["arguments"])
                        logger.info(f"Tool call: {fn_name}({fn_args})")

                        if fn_name in TOOL_MAPPING:
                            result = TOOL_MAPPING[fn_name](**fn_args)
                        else:
                            result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        })
                    # Loop — next streaming call includes tool results

                else:
                    # Exhausted tool rounds — force a final answer
                    messages.append({"role": "user", "content": "Please provide your final answer now."})
                    stream = openai_client.chat.completions.create(  # type: ignore[call-overload]
                        model=current_llm_name,
                        messages=messages,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    _stream_and_speak(stream, start_time)

            elif llm_model and llm_tokenizer:
                from transformers import TextIteratorStreamer
                import threading

                text_prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = llm_tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

                streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=128, temperature=0.7, top_p=0.8)

                gen_thread = threading.Thread(target=llm_model.generate, kwargs=generation_kwargs)
                gen_thread.start()

                ttft_llm = None
                is_thinking = False
                sentence_buffer = ""
                tokens_count = 0

                for new_text in streamer:
                    if stop_event.is_set(): break
                    tokens_count += 1
                    if ttft_llm is None: ttft_llm = time.time() - start_time

                    elapsed = time.time() - (start_time + ttft_llm)
                    latency_data = {"llm_ttft": ttft_llm}
                    if elapsed > 0: latency_data["llm_throughput"] = tokens_count / elapsed

                    ws_out_queue.put({"type": "llm_token", "data": new_text, "latency": latency_data})

                    if "<think>" in new_text: is_thinking = True
                    if not is_thinking:
                        sentence_buffer += new_text
                        if any(p in new_text for p in [".", "!", "?", "\n", ","]):
                            clean = clean_for_tts(sentence_buffer)
                            if clean: tts_in_queue.put({"type": "text_to_speak", "text": clean})
                            sentence_buffer = ""
                    if "</think>" in new_text:
                        is_thinking = False
                        sentence_buffer = ""

                if sentence_buffer and not stop_event.is_set():
                    clean = clean_for_tts(sentence_buffer)
                    if clean: tts_in_queue.put({"type": "text_to_speak", "text": clean})

        except Exception as e:
            logger.error(f"LLM generation error: {e}")

        # EOF signal to TTS
        logger.info("LLM generation complete → sending eof to TTS queue")
        tts_in_queue.put({"type": "eof"})

        if dm:
            next_response = dm.speaking_done()
            if next_response:
                logger.info(f"DM speaking_done returned queued response: intent={next_response.get('intent')}")
                if next_response.get("intent") == "general_question" and next_response.get("original_text"):
                    generate_response(str(next_response["original_text"]))
                else:
                    tts_in_queue.put({"type": "text_to_speak", "text": next_response["nova_says"]})
                    tts_in_queue.put({"type": "eof"})
            else:
                logger.info("DM speaking_done: no queued response")

    while not stop_event.is_set():
        try:
            msg = llm_in_queue.get(timeout=0.05)
        except Empty:
            continue

        if not isinstance(msg, dict):
            continue

        if msg.get("type") == "interrupted":
            if tts_interrupt_event:
                tts_interrupt_event.set()
            if dm: dm.state.is_speaking = False
            continue
            
        if msg.get("type") == "change_llm":
            current_llm_name = msg.get("data", current_llm_name)
            continue

        if msg.get("type") == "text":
            transcript = msg.get("text")
            audio_data = msg.get("audio_data")
            logger.info(f"Received transcript: '{transcript}'")

            is_verifying = dm and dm.state.fsm_state == "VERIFY"
            if not transcript and not is_verifying:
                ws_out_queue.put({"type": "recording_stopped"})
                continue

            if dm:
                safe_transcript = transcript if transcript else "voice verification audio"

                # Convert audio_data list to numpy array if present
                audio_np = None
                if audio_data:
                    import numpy as np
                    audio_np = np.array(audio_data, dtype=np.float32)
                    logger.info(f"Audio buffer present: {len(audio_np)} samples ({len(audio_np)/16000:.1f}s)")

                # Log DM state before processing (helps trace verification flow)
                if dm.state.fsm_state not in ("IDLE", "READY"):
                    logger.info(f"DM state before process: {dm.state.fsm_state}")

                dm_response = dm.process(safe_transcript, audio_buffer=audio_np)
                logger.info(f"DM response: intent={dm_response.get('intent')}, routing={dm_response.get('routing')}, nova_says='{str(dm_response.get('nova_says',''))[:80]}'")

                # Log verification results
                if dm_response.get("verification_status"):
                    logger.info(f"Voice verification: {dm_response['verification_status']} | driver={dm_response.get('driver_name', 'unknown')}")
                if dm_response.get("session_started"):
                    logger.info(f"Session started for driver: {dm_response.get('driver_name')}")

                ws_out_queue.put({
                    "type": "dm_state",
                    "data": {
                        "fsm_state": dm_response.get("fsm_state"),
                        "intent": dm_response.get("intent"),
                        "routing": dm_response.get("routing"),
                        "entities": dm_response.get("entities"),
                        "otp": dm_response.get("otp"),
                        "driver_name": dm_response.get("driver_name"),
                        "session_warning": dm_response.get("session_warning"),
                        "nova_says": dm_response.get("nova_says"),
                    }
                })

                if dm_response.get("intent") == "emergency":
                    logger.info(f"DM routed as EMERGENCY → TTS: '{dm_response['nova_says'][:60]}'")
                    if tts_interrupt_event:
                        tts_interrupt_event.set()
                    time.sleep(0.1)
                    tts_in_queue.put({"type": "text_to_speak", "text": dm_response["nova_says"]})
                    tts_in_queue.put({"type": "eof"})
                    continue

                if dm_response.get("intent") == "stop":
                    logger.info("DM routed as STOP → interrupting")
                    if tts_interrupt_event:
                        tts_interrupt_event.set()
                    tts_in_queue.put({"type": "eof"})  # Reset interrupted flag in TTS
                    ws_out_queue.put({"type": "interrupted"})
                    continue

                if dm_response.get("queued", False):
                    logger.info(f"DM queued response → TTS: '{dm_response['nova_says'][:60]}'")
                    tts_in_queue.put({"type": "text_to_speak", "text": dm_response["nova_says"]})
                    tts_in_queue.put({"type": "eof"})
                    continue

                if dm_response.get("intent") != "general_question":
                    logger.info(f"DM intent={dm_response.get('intent')} (not general_question) → TTS: '{dm_response['nova_says'][:60]}'")
                    tts_in_queue.put({"type": "text_to_speak", "text": dm_response["nova_says"]})
                    tts_in_queue.put({"type": "eof"})
                    logger.info("Sent text_to_speak + eof to TTS queue")

                    # Multi-turn DM states (slot fill, verification, OTP) don't need
                    # special handling — generation_done will restart auto-listen.
                    # NOTE: Previously sent ptt_started here, but ws_queue_reader
                    # treated it as a real PTT press and interrupted TTS mid-speech.
                    continue

            # General question -> LLM
            logger.info(f"Routing to LLM: '{transcript[:80] if transcript else ''}'")
            generate_response(transcript or "")

    logger.info("LLM Worker exiting.")
