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
const statusChips = document.getElementById('statusChips');
const statusModel = document.getElementById('statusModel');
const metricsPill = document.getElementById('metricsPill');
const metricsSummaryText = document.getElementById('metricsSummaryText');
const mAudio = document.getElementById('m-audio');
const mTtft = document.getElementById('m-ttft');
const mTts = document.getElementById('m-tts');
const mTotal = document.getElementById('m-total');
const mTokens = document.getElementById('m-tokens');
const mTps = document.getElementById('m-tps');
const snapBtn = document.getElementById('snapBtn');
const attachBtn = document.getElementById('attachBtn');
const fileInput = document.getElementById('fileInput');
const attachStrip = document.getElementById('attachStrip');
const attachThumb = document.getElementById('attachThumb');
const attachLabel = document.getElementById('attachLabel');
const attachClear = document.getElementById('attachClear');
const camModal = document.getElementById('camModal');
const camVideo = document.getElementById('camVideo');
const camCanvas = document.getElementById('camCanvas');
const camCapture = document.getElementById('camCapture');
const camCancel = document.getElementById('camCancel');

let stagedImage = null;          // { blob, url, filename }
let camStream = null;

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

function basename(path) {
  if (!path) return '';
  const parts = String(path).split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function renderEngineStatus(data) {
  statusChips.innerHTML = '';
  const chips = [
    { key: 'LLM', val: data.backend || '?' },
    { key: 'audio', val: data.audio_backend || '?' },
    { key: 'vision', val: data.vision_backend || '?' },
    { key: 'MTP', val: data.speculative_decoding ? 'on' : 'off',
      cls: data.speculative_decoding ? 'on' : 'off' },
  ];
  for (const c of chips) {
    const el = document.createElement('span');
    el.className = 'chip' + (c.cls ? ' ' + c.cls : '');
    el.innerHTML = `<span class="chip-key">${c.key}</span> ${c.val}`;
    statusChips.appendChild(el);
  }
  const model = basename(data.model_path);
  statusModel.textContent = model || '';
  statusModel.title = data.model_path || '';
}

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();

    engineLoaded = Boolean(data.engine_loaded);
    statusDot.classList.toggle('ok', engineLoaded);
    statusDot.classList.toggle('bad', !engineLoaded);
    statusText.textContent = engineLoaded ? 'Engine ready' : 'Engine not ready';

    if (engineLoaded) {
      renderEngineStatus(data);
    } else {
      statusChips.innerHTML = '';
      statusModel.textContent = data.error || 'Check server logs';
      statusModel.title = '';
    }

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
    statusChips.innerHTML = '';
    statusModel.textContent = String(err);
    statusModel.title = '';
    voiceState.textContent = 'Voice unavailable: server offline';
    sendBtn.disabled = true;
    recordBtn.disabled = true;
    enableMicBtn.disabled = false;
  }
}

function fmtMs(v) { return v == null ? '—' : `${v} ms`; }
function fmtN(v)  { return v == null ? '—' : String(v); }

function renderMetrics(m) {
  if (!m) return;
  metricsSummaryText.textContent =
    `${m.total_ttfb_ms ?? '—'} ms · ${m.decode_tok_per_s ?? '—'} tok/s`;
  mAudio.textContent  = fmtMs(m.audio_dur_ms);
  mTtft.textContent   = fmtMs(m.llm_ttft_ms);
  mTts.textContent    = fmtMs(m.tts_ttfb_ms);
  mTotal.textContent  = fmtMs(m.total_ttfb_ms);
  mTokens.textContent = fmtN(m.tokens);
  mTps.textContent    = m.decode_tok_per_s == null ? '—' : `${m.decode_tok_per_s} /s`;
  metricsPill.hidden = false;
}

metricsPill.addEventListener('click', () => {
  const open = metricsPill.getAttribute('aria-expanded') === 'true';
  metricsPill.setAttribute('aria-expanded', open ? 'false' : 'true');
});

function setStagedImage(blob, filename) {
  clearStagedImage();
  stagedImage = { blob, url: URL.createObjectURL(blob), filename };
  attachThumb.src = stagedImage.url;
  attachLabel.textContent = filename;
  attachStrip.hidden = false;
}

function clearStagedImage() {
  if (stagedImage) {
    URL.revokeObjectURL(stagedImage.url);
    stagedImage = null;
  }
  attachThumb.removeAttribute('src');
  attachStrip.hidden = true;
}

function addImageMessage(blob) {
  const row = document.createElement('div');
  row.className = 'message user';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const img = document.createElement('img');
  img.src = URL.createObjectURL(blob);
  img.style.maxWidth = '320px';
  img.style.borderRadius = '12px';
  img.style.display = 'block';
  bubble.appendChild(img);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

attachClear.addEventListener('click', clearStagedImage);

attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  const f = fileInput.files && fileInput.files[0];
  if (!f) return;
  setStagedImage(f, f.name);
  fileInput.value = '';
});

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    addMessage('assistant', '[Camera error] getUserMedia not available in this browser context.');
    return;
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    camVideo.srcObject = camStream;
    camModal.hidden = false;
  } catch (err) {
    addMessage('assistant', `[Camera error] ${err?.message || err}`);
  }
}

function closeCamera() {
  if (camStream) {
    camStream.getTracks().forEach(t => t.stop());
    camStream = null;
  }
  camVideo.srcObject = null;
  camModal.hidden = true;
}

snapBtn.addEventListener('click', openCamera);
camCancel.addEventListener('click', closeCamera);

function extOfMime(type) {
  const map = { 'image/jpeg': 'jpg', 'image/jpg': 'jpg', 'image/png': 'png',
                'image/webp': 'webp', 'image/gif': 'gif', 'image/bmp': 'bmp' };
  return map[type] || (type?.split('/')[1] || 'png');
}

// Paste image from clipboard (screenshot, copy-from-webpage, etc.)
document.addEventListener('paste', (e) => {
  if (busy) return;
  // Skip if paste target is the textarea AND clipboard has text — let normal text paste happen.
  const items = e.clipboardData?.items || [];
  for (const item of items) {
    if (item.type && item.type.startsWith('image/')) {
      const blob = item.getAsFile();
      if (blob) {
        const name = `paste-${Date.now()}.${extOfMime(blob.type)}`;
        setStagedImage(blob, name);
        e.preventDefault();
        return;
      }
    }
  }
});

// Drag & drop — accept files or image URLs dragged from other tabs.
const dropTarget = document.querySelector('.chat-card');
function isDragWithImage(e) {
  const types = e.dataTransfer?.types || [];
  return types.includes('Files') || types.includes('text/uri-list') || types.includes('text/plain');
}
['dragenter', 'dragover'].forEach((ev) => {
  dropTarget.addEventListener(ev, (e) => {
    if (!isDragWithImage(e)) return;
    e.preventDefault();
    dropTarget.classList.add('drop-hover');
  });
});
['dragleave', 'dragend', 'drop'].forEach((ev) => {
  dropTarget.addEventListener(ev, () => dropTarget.classList.remove('drop-hover'));
});
dropTarget.addEventListener('drop', async (e) => {
  if (busy) return;
  e.preventDefault();
  // 1) Files dropped from desktop / file manager
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) {
    setStagedImage(file, file.name);
    return;
  }
  // 2) Image URL dropped from another browser tab
  const url = e.dataTransfer?.getData('text/uri-list')
           || e.dataTransfer?.getData('text/plain');
  if (!url) return;
  try {
    const res = await fetch(url, { mode: 'cors' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    if (!blob.type.startsWith('image/')) throw new Error(`not an image (got ${blob.type || 'unknown'})`);
    const ext = extOfMime(blob.type);
    const name = url.split('/').pop()?.split('?')[0] || `drop-${Date.now()}.${ext}`;
    setStagedImage(blob, name);
  } catch (err) {
    addMessage('assistant',
      `[Drop hint] couldn't fetch the image directly (${err.message}). ` +
      `Many sites block cross-origin fetches — drag the image to your desktop first, then drop it here.`);
  }
});

camCapture.addEventListener('click', () => {
  if (!camStream) return closeCamera();
  const w = camVideo.videoWidth || 1280;
  const h = camVideo.videoHeight || 720;
  // Cap longest edge at 1024 px to keep prefill tokens reasonable.
  const scale = Math.min(1, 1024 / Math.max(w, h));
  const cw = Math.round(w * scale);
  const ch = Math.round(h * scale);
  camCanvas.width = cw;
  camCanvas.height = ch;
  const ctx = camCanvas.getContext('2d');
  ctx.drawImage(camVideo, 0, 0, cw, ch);
  camCanvas.toBlob((blob) => {
    if (blob) setStagedImage(blob, `snap-${Date.now()}.jpg`);
    closeCamera();
  }, 'image/jpeg', 0.85);
});

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
  if ((!text && !stagedImage) || busy) return;
  busy = true;
  updateVoiceButtons();

  const fd = new FormData();
  fd.append('message', text || '');
  if (sessionId) fd.append('session_id', sessionId);
  if (stagedImage) {
    fd.append('image', stagedImage.blob, stagedImage.filename);
    addImageMessage(stagedImage.blob);
  }
  if (text) addMessage('user', text);
  const assistantBubble = addMessage('assistant', '');

  // Consume the staged image now — one-shot binding.
  clearStagedImage();

  try {
    await postFormStream('/api/chat/stream', fd, (event, data) => {
      if (event === 'session') {
        sessionId = data.session_id;
        localStorage.setItem('litert_session_id', sessionId);
      } else if (event === 'token') {
        appendText(assistantBubble, data.text || '');
      } else if (event === 'tool_call') {
        addToolPill(data.name || 'tool', JSON.stringify(data.args || {}));
      } else if (event === 'tool_result') {
        addToolPill(data.name || 'tool', data.ok ? 'ok' : 'failed');
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
  if (!text && !stagedImage) return;
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
  if (stagedImage) {
    formData.append('image', stagedImage.blob, stagedImage.filename);
    addImageMessage(stagedImage.blob);
    clearStagedImage();
  }

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
        renderMetrics(data);
        voiceState.textContent = 'Voice turn complete';
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
