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
    """Execute a Meta search using DDGS. Uses news() for news queries, text() otherwise."""
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


def run_llm_worker(llm_in_queue: "mp.Queue[dict]", tts_in_queue: "mp.Queue[dict]", ws_out_queue: "mp.Queue[dict]", stop_event: "mp.Event", tts_interrupt_event: "mp.Event | None" = None) -> None:
    import setproctitle
    setproctitle.setproctitle("nova-llm-worker")

    # Setup paths
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(backend_dir)
    sys.path.append(os.path.join(backend_dir, "nova-l7", "L-7"))

    from pipeline_mp.llm_config import LOCAL_LLM_REGISTRY

    # ── Init OpenRouter (cloud) ──────────────────────────────────────────────
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    openai_client = None
    if api_key:
        openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        logger.info("OpenRouter initialized in LLM worker.")

    # ── Local LLM state (lazy-loaded on demand) ─────────────────────────────
    llm_model = None
    llm_tokenizer = None
    local_cfg = None
    local_key = None       # e.g. "lfm2.5-1.2b" — None means no local model loaded

    # Default cloud model (used when routing to OpenRouter)
    cloud_llm_name = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    # Overall mode: "cloud", "local", or "auto" (query routing decides per-query)
    llm_mode = "auto" if api_key else "local"

    def _load_local_model(key: str):
        """Load a local model from the registry. Unloads any previous local model."""
        nonlocal llm_model, llm_tokenizer, local_cfg, local_key
        if key == local_key and llm_model is not None:
            return  # already loaded
        _unload_local_model()

        if key not in LOCAL_LLM_REGISTRY:
            logger.error(f"Unknown local model key: '{key}'. Available: {list(LOCAL_LLM_REGISTRY)}")
            return

        import torch
        if not torch.cuda.is_available():
            logger.warning(
                f"No CUDA GPU detected — skipping local LLM '{key}'. "
                "Using cloud (OpenRouter) for all queries."
            )
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        local_cfg = LOCAL_LLM_REGISTRY[key]
        logger.info(f"Loading local LLM: {local_cfg.display_name} ({local_cfg.model_id}) …")
        ws_out_queue.put({"type": "llm_loading", "data": {"model": key, "status": "loading"}})

        compute_dtype = torch.bfloat16 if local_cfg.compute_dtype == "bfloat16" else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=local_cfg.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(local_cfg.model_id)
            llm_model = AutoModelForCausalLM.from_pretrained(
                local_cfg.model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
            local_key = key
            logger.info(f"Local model loaded: {local_cfg.display_name}")
            ws_out_queue.put({"type": "llm_loading", "data": {"model": key, "status": "ready"}})
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            llm_model = llm_tokenizer = local_cfg = None
            local_key = None

    def _unload_local_model():
        """Free GPU memory from current local model.

        Models loaded with device_map="auto" keep accelerate dispatch hooks
        and internal tensor references that survive a plain ``del``.
        We must: unhook → move to meta device → delete → GC → flush CUDA.
        """
        nonlocal llm_model, llm_tokenizer, local_cfg, local_key
        if llm_model is None:
            return
        import gc
        import torch

        old_name = local_key

        # 1. Remove accelerate dispatch hooks (they hold tensor refs)
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(llm_model)
        except Exception:
            pass

        # 2. Move all parameters to meta device (instantly frees CUDA tensors)
        try:
            llm_model.to("meta")
        except Exception:
            # Fallback: try .cpu() then delete
            try:
                llm_model.to("cpu")
            except Exception:
                pass

        # 3. Drop all Python references
        del llm_model
        del llm_tokenizer
        llm_model = None  # type: ignore[assignment]
        llm_tokenizer = None  # type: ignore[assignment]
        local_cfg = None
        local_key = None

        # 4. Force GC — two passes to break weak-ref and closure cycles
        gc.collect()
        gc.collect()

        # 5. Flush CUDA allocator
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 6. Log actual VRAM to verify it's freed
        try:
            vram_mb = torch.cuda.memory_allocated() / 1024 ** 2
            logger.info(f"Local LLM '{old_name}' unloaded. VRAM now: {vram_mb:.0f} MB")
        except Exception:
            logger.info(f"Local LLM '{old_name}' unloaded.")

    # ── Query router — decides local vs cloud per query ──────────────────────
    # Keywords that indicate the query needs cloud LLM (web search, complex reasoning)
    _CLOUD_KEYWORDS = re.compile(
        r"\b(news|headline|latest|breaking|trending|today|current events|"
        r"what happened|stock|market|weather|forecast|score|election|"
        r"compare|analyze|explain in detail|write me|essay|summarize this article|"
        r"translate|code|debug|how does .+ work in detail)\b",
        re.IGNORECASE,
    )

    def _route_query(prompt: str, dm_intent: str | None = None) -> str:
        """Return 'local' or 'cloud' based on query complexity.

        Routing rules (evaluated in order):
          1. If only one backend is available, use that.
          2. If DM intent is vehicle-related or simple factual → local.
          3. If query matches cloud keywords (news, trending, complex) → cloud.
          4. Short queries (≤ 12 words) → local; longer → cloud.
        """
        if llm_mode == "cloud":
            return "cloud"
        if llm_mode == "local":
            return "local"
        # llm_mode == "auto"
        if not api_key:
            return "local"
        if llm_model is None and local_key is None:
            return "cloud"

        # Vehicle/simple intents handled by DM never reach here, but just in case
        if dm_intent in ("vehicle_control", "navigation", "media", "communication"):
            return "local"

        # Cloud keywords → needs web search or complex reasoning
        if _CLOUD_KEYWORDS.search(prompt):
            return "cloud"

        # Short simple queries → local is fast enough
        if len(prompt.split()) <= 12:
            return "local"

        return "cloud"

    # ── Startup: load default local model ────────────────────────────────────
    # _load_local_model skips silently when CUDA is not available (no GPU = cloud only).
    from pipeline_mp.llm_config import LOCAL_LLM_SETTINGS
    _load_local_model(LOCAL_LLM_SETTINGS.active)

    # Track the current LLM name for the UI display
    current_llm_name = cloud_llm_name if api_key else f"local:{LOCAL_LLM_SETTINGS.active}"

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

    def generate_response(prompt: str, dm_intent: str | None = None):
        if dm:
            dm.speaking_started()

        # ── Query routing: decide local vs cloud for this query ──────────
        route = _route_query(prompt, dm_intent)
        use_cloud = (route == "cloud" and openai_client is not None)
        use_local = (not use_cloud and llm_model is not None)

        if use_cloud:
            logger.info(f"Query routed → CLOUD ({cloud_llm_name}): '{prompt[:50]}'")
        elif use_local:
            logger.info(f"Query routed → LOCAL ({local_key}): '{prompt[:50]}'")
        else:
            logger.warning(f"No LLM backend available for query: '{prompt[:50]}'")

        # System prompt — omit web_search mention for local models (no tool support or limited)
        system_base = (
            "Your name is Nova. You are a voice assistant embedded in an electric vehicle dashboard. "
            "The user is the driver — never address the driver as 'Nova'. "
            "Keep answers concise and direct, 2-3 sentences max. "
            "Vehicle controls like lights, AC, and windows are handled by a separate system. "
            "If the driver asks about vehicle status you do not have access to, say so honestly. "
            "Do not make up sensor readings or vehicle data. "
            "Your output is read aloud by TTS. Never use markdown, bullet points, "
            "numbered lists, URLs, links, or special formatting. Write plain conversational sentences only. "
            "Stay on topic. If a question is outside your knowledge, say you are not sure rather than guessing."
        )
        if use_cloud:
            system_base += (
                " You have a web_search tool — use it for any question about current events, "
                "news, financial markets, stock prices, weather, or anything requiring recent data. "
                "After getting search results, synthesize a brief spoken summary in plain sentences — no lists or links."
            )

        messages = [
            {"role": "system", "content": system_base},
            {"role": "user", "content": prompt},
        ]

        ws_out_queue.put({"type": "assistant_start"})
        start_time = time.time()

        try:
            if use_cloud:
                use_tools = cloud_llm_name in TOOL_CAPABLE_MODELS

                # Stream-first approach: always stream, detect tool calls from deltas.
                # For non-tool queries (the common case), tokens arrive immediately — no double call.
                for _round in range(3):
                    if stop_event.is_set():
                        break

                    stream_kwargs = {
                        "model": cloud_llm_name,
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
                        model=cloud_llm_name,
                        messages=messages,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    _stream_and_speak(stream, start_time)

            elif use_local and llm_model and llm_tokenizer:
                import ast
                import threading
                from transformers import TextIteratorStreamer

                _sp = local_cfg.sampling
                _gen_base = dict(
                    max_new_tokens=_sp.max_new_tokens,
                    do_sample=_sp.do_sample,
                    temperature=_sp.temperature,
                    repetition_penalty=_sp.repetition_penalty,
                    **({"top_k": _sp.top_k} if _sp.top_k is not None else {}),
                    **({"top_p": _sp.top_p} if _sp.top_p is not None else {}),
                )
                use_local_tools = local_cfg.supports_tools
                # LFM2.5 tool call delimiter tokens
                TC_START = "<|tool_call_start|>"
                TC_END   = "<|tool_call_end|>"

                for _round in range(3):
                    if stop_event.is_set():
                        break

                    # apply_chat_template — pass tools list for LFM2.5 (transformers>=5.0)
                    tmpl_kwargs: dict = dict(
                        conversation=messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if use_local_tools:
                        try:
                            tmpl_kwargs["tools"] = [t["function"] for t in TOOLS]
                        except Exception:
                            pass

                    text_prompt = llm_tokenizer.apply_chat_template(**tmpl_kwargs)
                    inputs = llm_tokenizer(
                        text_prompt, return_tensors="pt", add_special_tokens=False
                    ).to("cuda")

                    if use_local_tools:
                        # Non-streaming first pass so we can inspect for tool calls
                        import torch
                        with torch.no_grad():
                            output_ids = llm_model.generate(**inputs, **_gen_base)
                        input_len = inputs["input_ids"].shape[1]
                        generated = llm_tokenizer.decode(
                            output_ids[0][input_len:], skip_special_tokens=False
                        )

                        if TC_START in generated and TC_END in generated:
                            # ── Tool call round ──────────────────────────────
                            tc_raw = generated[
                                generated.index(TC_START) + len(TC_START):
                                generated.index(TC_END)
                            ].strip()

                            ws_out_queue.put({
                                "type": "llm_token",
                                "data": "[searching...] ",
                                "latency": {"llm_ttft": time.time() - start_time},
                            })

                            # Parse Pythonic list: [func(arg="val"), ...]
                            tool_results = []
                            try:
                                tree = ast.parse(tc_raw, mode="eval")
                                calls = (
                                    tree.body.elts
                                    if isinstance(tree.body, ast.List)
                                    else [tree.body]
                                )
                                for call in calls:
                                    fn_name = call.func.id  # type: ignore[attr-defined]
                                    fn_args = {
                                        kw.arg: ast.literal_eval(kw.value)
                                        for kw in call.keywords
                                    }
                                    logger.info(f"Local tool call: {fn_name}({fn_args})")
                                    result = (
                                        TOOL_MAPPING[fn_name](**fn_args)
                                        if fn_name in TOOL_MAPPING
                                        else json.dumps({"error": f"Unknown tool: {fn_name}"})
                                    )
                                    tool_results.append(result)
                            except Exception as parse_err:
                                logger.warning(f"Tool call parse error '{tc_raw}': {parse_err}")
                                tool_results.append(json.dumps({"error": "parse failure"}))

                            messages.append({"role": "assistant", "content": generated})
                            messages.append({"role": "tool", "content": "\n".join(tool_results)})
                            continue  # re-generate with tool result in context

                        # No tool call — strip special tokens and stream as text
                        clean_generated = re.sub(r"<\|[^|]+\|>", "", generated).strip()
                        ttft_llm = time.time() - start_time
                        ws_out_queue.put({
                            "type": "llm_token",
                            "data": clean_generated,
                            "latency": {"llm_ttft": ttft_llm},
                        })
                        clean = clean_for_tts(clean_generated)
                        if clean:
                            tts_in_queue.put({"type": "text_to_speak", "text": clean})
                        break

                    else:
                        # No tool support — pure streaming
                        streamer = TextIteratorStreamer(
                            llm_tokenizer, skip_prompt=True, skip_special_tokens=True
                        )
                        gen_thread = threading.Thread(
                            target=llm_model.generate,
                            kwargs=dict(**inputs, streamer=streamer, **_gen_base),
                        )
                        gen_thread.start()

                        ttft_llm = None
                        is_thinking = False
                        sentence_buffer = ""
                        tokens_count = 0

                        for new_text in streamer:
                            if stop_event.is_set():
                                break
                            tokens_count += 1
                            if ttft_llm is None:
                                ttft_llm = time.time() - start_time

                            elapsed = time.time() - (start_time + ttft_llm)
                            latency_data = {"llm_ttft": ttft_llm}
                            if elapsed > 0:
                                latency_data["llm_throughput"] = tokens_count / elapsed

                            ws_out_queue.put({"type": "llm_token", "data": new_text, "latency": latency_data})

                            if "<think>" in new_text:
                                is_thinking = True
                            if not is_thinking:
                                sentence_buffer += new_text
                                if any(p in new_text for p in [".", "!", "?", "\n", ","]):
                                    clean = clean_for_tts(sentence_buffer)
                                    if clean:
                                        tts_in_queue.put({"type": "text_to_speak", "text": clean})
                                    sentence_buffer = ""
                            if "</think>" in new_text:
                                is_thinking = False
                                sentence_buffer = ""

                        if sentence_buffer and not stop_event.is_set():
                            clean = clean_for_tts(sentence_buffer)
                            if clean:
                                tts_in_queue.put({"type": "text_to_speak", "text": clean})
                        break

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
            new_name = msg.get("data", current_llm_name)
            if new_name.startswith("local:"):
                # Switch to local model: "local:lfm2.5-1.2b"
                key = new_name[6:]
                _load_local_model(key)
                llm_mode = "local"
                current_llm_name = new_name
            elif new_name == "auto":
                llm_mode = "auto"
                current_llm_name = "auto"
            else:
                # Cloud model — unload local to free VRAM
                _unload_local_model()
                cloud_llm_name = new_name
                llm_mode = "cloud"
                current_llm_name = new_name
            logger.info(f"LLM switched → mode={llm_mode}, name={current_llm_name}")
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

            # General question -> LLM (query router decides local vs cloud)
            dm_intent = dm_response.get("intent") if dm else None
            logger.info(f"Routing to LLM: '{transcript[:80] if transcript else ''}'")
            generate_response(transcript or "", dm_intent=dm_intent)

    logger.info("LLM Worker exiting.")
