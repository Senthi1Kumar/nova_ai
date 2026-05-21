const messagesEl = document.getElementById('messages');
const form = document.getElementById('chatForm');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('sendBtn');
const recordBtn = document.getElementById('recordBtn');
const enableMicBtn = document.getElementById('enableMicBtn');
const voiceState = document.getElementById('voiceState');
const micState = document.getElementById('micState');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const statusDetails = document.getElementById('statusDetails');

let sessionId = localStorage.getItem('litert_session_id') || null;
let busy = false;
let mediaRecorder = null;
let recordedChunks = [];
let recording = false;
let engineLoaded = false;
let sttAvailable = false;
let microphoneAllowed = false;
let microphoneSupported = false;
let permissionWatcher = null;

function addMessage(role, text = '') {
  const row = document.createElement('div');
  row.className = `message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function appendText(el, text) {
  el.textContent += text;
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioCtx;
}

class AudioQueue {
  constructor(ctx) {
    this.ctx = ctx;
    this.nextStart = 0;
  }
  pushPcm16(bytes, sampleRate) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const f32 = new Float32Array(bytes.length / 2);
    for (let i = 0; i < f32.length; i++) f32[i] = view.getInt16(i * 2, true) / 32768;
    const buf = this.ctx.createBuffer(1, f32.length, sampleRate);
    buf.copyToChannel(f32, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    const start = Math.max(this.ctx.currentTime, this.nextStart);
    src.start(start);
    this.nextStart = start + buf.duration;
  }
  reset() { this.nextStart = 0; }
}

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function addToolPill(name, label) {
  const row = document.createElement('div');
  row.className = 'message assistant tool-row';
  const pill = document.createElement('div');
  pill.className = 'bubble tool-bubble';
  pill.textContent = `🔎 ${name}: ${label}`;
  row.appendChild(pill);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return pill;
}

function isLocalHost() {
  return ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
}

function setMicStatus(text, state = 'unknown') {
  if (!micState) return;
  micState.textContent = text;
  micState.dataset.state = state;
}

function updateVoiceButtons() {
  sendBtn.disabled = !engineLoaded || busy;

  // Keep Enable Microphone ALWAYS clickable.
  // Even when navigator.mediaDevices is unavailable, a click should show the user
  // exactly why the browser cannot request permission instead of becoming a dead button.
  enableMicBtn.disabled = false;
  enableMicBtn.textContent = microphoneAllowed ? 'Microphone Enabled' : 'Enable Microphone';

  // Hold-to-talk should only unlock after mic permission + STT + engine are ready.
  recordBtn.disabled = !engineLoaded || busy || !sttAvailable || !microphoneAllowed;
}

function explainMicrophoneError(err) {
  const name = err?.name || '';
  const msg = err?.message || String(err);

  if (!window.isSecureContext) {
    return 'Microphone is blocked because this page is not a secure browser context. Open http://localhost:8000 or http://127.0.0.1:8000, not http://0.0.0.0 or a LAN IP over plain HTTP.';
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return 'Browser microphone API is unavailable. Use Chrome, Safari, Edge, or Brave on http://localhost:8000.';
  }
  if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError') {
    return 'Microphone permission was denied or blocked. Enable it in the browser address-bar permission icon and macOS System Settings → Privacy & Security → Microphone.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No microphone device was found. Connect/select a microphone and retry.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'The microphone is already in use by another app or cannot be opened. Close Zoom/Teams/other audio apps and retry.';
  }
  return msg;
}

async function refreshMicrophonePermissionState() {
  microphoneSupported = Boolean(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  if (!window.isSecureContext) {
    microphoneAllowed = false;
    setMicStatus('Microphone blocked: open http://localhost:8000', 'blocked');
    updateVoiceButtons();
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    microphoneAllowed = false;
    setMicStatus('Microphone API unavailable in this browser/context', 'blocked');
    updateVoiceButtons();
    return;
  }

  try {
    if (navigator.permissions && navigator.permissions.query) {
      const permission = await navigator.permissions.query({ name: 'microphone' });
      microphoneAllowed = permission.state === 'granted';

      if (permission.state === 'granted') {
        setMicStatus('Microphone permission granted', 'granted');
      } else if (permission.state === 'prompt') {
        setMicStatus('Click Enable Microphone to allow access', 'prompt');
      } else if (permission.state === 'denied') {
        setMicStatus('Microphone permission denied in browser', 'denied');
      }

      if (!permissionWatcher) {
        permissionWatcher = permission;
        permission.onchange = () => refreshMicrophonePermissionState();
      }
    } else {
      // Safari may not fully support navigator.permissions for microphone.
      setMicStatus(microphoneAllowed ? 'Microphone permission granted' : 'Click Enable Microphone to allow access', microphoneAllowed ? 'granted' : 'prompt');
    }
  } catch (_) {
    // Permission API is optional. getUserMedia will still trigger the real browser prompt.
    setMicStatus(microphoneAllowed ? 'Microphone permission granted' : 'Click Enable Microphone to allow access', microphoneAllowed ? 'granted' : 'prompt');
  }

  updateVoiceButtons();
}

async function getAudioStreamForPermission() {
  // Modern browsers. This triggers the real browser permission dialog
  // when called from a user click.
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    return await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  }

  // Older Safari fallback.
  const legacyGetUserMedia =
    navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;

  if (legacyGetUserMedia) {
    return await new Promise((resolve, reject) => {
      legacyGetUserMedia.call(navigator, { audio: true }, resolve, reject);
    });
  }

  throw new Error('Browser microphone API is unavailable. Open http://localhost:8000 in Chrome/Safari/Edge and make sure microphone access is allowed.');
}

async function requestMicrophonePermission(showSuccessMessage = true) {
  try {
    // Keep the click alive and visible. Do not silently disable this button.
    setMicStatus('Requesting microphone permission...', 'prompt');
    enableMicBtn.textContent = 'Requesting...';

    if (!window.isSecureContext) {
      throw new Error('Microphone requires a secure browser context. Open http://localhost:8000 or http://127.0.0.1:8000, not http://0.0.0.0 or a LAN IP over plain HTTP.');
    }

    const stream = await getAudioStreamForPermission();

    // Permission is now granted. Stop this test stream; recording opens a fresh stream.
    stream.getTracks().forEach(track => track.stop());
    microphoneSupported = true;
    microphoneAllowed = true;
    setMicStatus('Microphone permission granted', 'granted');
    if (showSuccessMessage) addMessage('assistant', 'Microphone permission granted. Press Hold to Talk and speak.');
  } catch (err) {
    microphoneAllowed = false;
    const message = explainMicrophoneError(err);
    setMicStatus(message, 'denied');
    addMessage('assistant', `[Microphone permission error] ${message}`);
  } finally {
    enableMicBtn.textContent = microphoneAllowed ? 'Microphone Enabled' : 'Enable Microphone';
    updateVoiceButtons();
  }
}

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();

    engineLoaded = Boolean(data.engine_loaded);
    statusDot.classList.toggle('ok', engineLoaded);
    statusDot.classList.toggle('bad', !engineLoaded);
    statusText.textContent = engineLoaded ? 'Engine ready' : 'Engine not ready';
    statusDetails.textContent = engineLoaded
      ? `${data.backend} (LLM) · ${data.audio_backend} (audio) · MTP=${data.speculative_decoding} · ${data.model_path}`
      : (data.error || 'Check server logs');

    const ttsOk = Boolean(data.tts?.available && data.tts?.enabled);
    const ttsLoaded = Boolean(data.tts?.loaded);
    voiceState.textContent = `Voice: Gemma-4 native audio · Pocket TTS ${ttsOk ? (ttsLoaded ? 'loaded' : 'available') : 'missing/disabled'}`;

    sttAvailable = engineLoaded;
    updateVoiceButtons();
  } catch (err) {
    engineLoaded = false;
    sttAvailable = false;
    statusDot.className = 'dot bad';
    statusText.textContent = 'Server offline';
    statusDetails.textContent = String(err);
    voiceState.textContent = 'Voice unavailable: server offline';
    sendBtn.disabled = true;
    recordBtn.disabled = true;
    enableMicBtn.disabled = false;
  }
}

async function readSseResponse(res, onEvent) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split('\n\n');
    buffer = frames.pop() || '';
    for (const frame of frames) {
      if (!frame.trim()) continue;
      const lines = frame.split('\n');
      const event = (lines.find(l => l.startsWith('event:')) || '').replace('event:', '').trim();
      const dataLine = (lines.find(l => l.startsWith('data:')) || '').replace('data:', '').trim();
      const data = dataLine ? JSON.parse(dataLine) : {};
      onEvent(event, data);
    }
  }
}

async function postJsonStream(url, payload, onEvent) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  await readSseResponse(res, onEvent);
}

async function postFormStream(url, formData, onEvent) {
  const res = await fetch(url, {
    method: 'POST',
    body: formData,
  });
  await readSseResponse(res, onEvent);
}

async function sendTextMessage(text) {
  if (!text || busy) return;
  busy = true;
  updateVoiceButtons();
  addMessage('user', text);
  const assistantBubble = addMessage('assistant', '');

  try {
    await postJsonStream('/api/chat/stream', { message: text, session_id: sessionId }, (event, data) => {
      if (event === 'session') {
        sessionId = data.session_id;
        localStorage.setItem('litert_session_id', sessionId);
      } else if (event === 'token') {
        appendText(assistantBubble, data.text || '');
      } else if (event === 'error') {
        appendText(assistantBubble, `\n\n[Error] ${data.error}`);
      }
    });
  } catch (err) {
    assistantBubble.textContent = `[Error] ${err.message}`;
  } finally {
    busy = false;
    await checkHealth();
    promptEl.focus();
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = promptEl.value.trim();
  promptEl.value = '';
  await sendTextMessage(text);
});

promptEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

enableMicBtn.addEventListener('click', async (event) => {
  event.preventDefault();
  event.stopPropagation();
  await requestMicrophonePermission(true);
});

enableMicBtn.addEventListener('keydown', async (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    await requestMicrophonePermission(true);
  }
});

async function startRecording() {
  if (busy || recording) return;

  if (!microphoneAllowed) {
    await requestMicrophonePermission(false);
    if (!microphoneAllowed) return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    recordedChunks = [];

    let options = {};
    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
      options = { mimeType: 'audio/webm;codecs=opus' };
    } else if (MediaRecorder.isTypeSupported('audio/webm')) {
      options = { mimeType: 'audio/webm' };
    }

    mediaRecorder = new MediaRecorder(stream, options);
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) recordedChunks.push(event.data);
    };
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(track => track.stop());
      await uploadRecordingStreaming();
    };
    mediaRecorder.start();
    recording = true;
    recordBtn.classList.add('recording');
    recordBtn.textContent = 'Release to Send';
    voiceState.textContent = 'Recording... speak now';
  } catch (err) {
    const message = explainMicrophoneError(err);
    addMessage('assistant', `[Microphone error] ${message}`);
    microphoneAllowed = false;
    await refreshMicrophonePermissionState();
  }
}

function stopRecording() {
  if (!recording || !mediaRecorder) return;
  recording = false;
  recordBtn.classList.remove('recording');
  recordBtn.textContent = 'Hold to Talk';
  voiceState.textContent = 'Processing voice stream...';
  mediaRecorder.stop();
}

async function uploadRecordingStreaming() {
  if (!recordedChunks.length) return;
  busy = true;
  updateVoiceButtons();

  const blob = new Blob(recordedChunks, { type: mediaRecorder?.mimeType || 'audio/webm' });
  const formData = new FormData();
  formData.append('audio', blob, 'recording.webm');
  if (sessionId) formData.append('session_id', sessionId);

  const queue = new AudioQueue(getAudioCtx());
  if (audioCtx.state === 'suspended') {
    try { await audioCtx.resume(); } catch (_) {}
  }

  addMessage('user', '🎙️ Voice message');
  const assistantBubble = addMessage('assistant', '');

  try {
    await postFormStream('/api/voice/chat/stream', formData, (event, data) => {
      if (event === 'status') {
        voiceState.textContent = data.stage === 'encode' ? 'Listening to audio...' : (data.message || 'Processing...');
      } else if (event === 'session') {
        sessionId = data.session_id;
        localStorage.setItem('litert_session_id', sessionId);
      } else if (event === 'token') {
        appendText(assistantBubble, data.text || '');
      } else if (event === 'audio_chunk') {
        const bytes = b64ToBytes(data.b64);
        queue.pushPcm16(bytes, data.sample_rate || 24000);
      } else if (event === 'tool_call') {
        addToolPill(data.name || 'tool', JSON.stringify(data.args || {}));
      } else if (event === 'tool_result') {
        addToolPill(data.name || 'tool', data.ok ? 'ok' : 'failed');
      } else if (event === 'clause' || event === 'clause_end') {
        // debug-only; canonical text is the token stream
      } else if (event === 'turn_metrics') {
        console.debug('turn_metrics', data);
        voiceState.textContent = `TTFB ${data.total_ttfb_ms ?? '?'}ms · ${data.decode_tok_per_s ?? '?'} tok/s`;
      } else if (event === 'warning') {
        appendText(assistantBubble, `\n\n[TTS warning] ${data.warning}`);
      } else if (event === 'error') {
        appendText(assistantBubble, `\n\n[Voice error] ${data.error}`);
      }
    });
  } catch (err) {
    assistantBubble.textContent = `[Voice error] ${err.message}`;
  } finally {
    busy = false;
    await checkHealth();
  }
}

recordBtn.addEventListener('mousedown', startRecording);
recordBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
window.addEventListener('mouseup', stopRecording);
window.addEventListener('touchend', stopRecording);

(async function init() {
  await refreshMicrophonePermissionState();
  await checkHealth();

  if (!isLocalHost() && !window.isSecureContext) {
    addMessage('assistant', 'Microphone access requires http://localhost:8000, http://127.0.0.1:8000, or HTTPS. Do not use http://0.0.0.0:8000 for microphone testing.');
  }
})();

setInterval(checkHealth, 5000);
