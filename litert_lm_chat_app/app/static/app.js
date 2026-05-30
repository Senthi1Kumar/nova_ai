// ──────────────────────────────────────────────────────────────────────────
// Nova HUD — orb state machine + caption stream + typewriter + settings
// Voice-only input. Text composer removed.
// ──────────────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

// Header / status
const statusDot       = $('statusDot');
const brandSub        = $('brandSub');
const sessionTag      = $('sessionTag');
const hudClock        = $('hudClock');
const settingsBtn     = $('settingsBtn');

// Stage
const orb             = $('orb');
const stage           = document.querySelector('.stage');
const stateLabel      = $('stateLabel');
const captionsUser    = $('captionsUser');
const captionsAsst    = $('captionsAssistant');
const toolPill        = $('toolPill');
const typewriter      = $('typewriter');
const tweText         = $('tweText');

// Left rail
const recordBtn       = $('recordBtn');
const micState        = $('micState');
const snapBtn         = $('snapBtn');
const attachBtn       = $('attachBtn');
const fileInput       = $('fileInput');
const enableMicBtn    = $('enableMicBtn');
const newChatBtn      = $('newChatBtn');

// Attachment chip
const attachChip      = $('attachChip');
const attachThumb     = $('attachThumb');
const attachLabel     = $('attachLabel');
const attachClear     = $('attachClear');

// Right rail
const statusChips     = $('statusChips');
const statusModel     = $('statusModel');
const metricsPanel    = $('metricsPanel');
const mAudio          = $('m-audio');
const mTtft           = $('m-ttft');
const mTts            = $('m-tts');
const mTotal          = $('m-total');
const mTokens         = $('m-tokens');
const mTps            = $('m-tps');

// Settings modal
const settingsModal   = $('settingsModal');
const settingsForm    = $('settingsForm');
const setBackend      = $('setBackend');
const setAudio        = $('setAudio');
const setVision       = $('setVision');
const setMaxTokens    = $('setMaxTokens');
const setMTP          = $('setMTP');
const setStatus       = $('setStatus');
const setApply        = $('setApply');

// Camera modal
const camModal        = $('camModal');
const camVideo        = $('camVideo');
const camCanvas       = $('camCanvas');
const camCapture      = $('camCapture');
const camCancel       = $('camCancel');

// ── State ─────────────────────────────────────────────────────────────────

let sessionId         = localStorage.getItem('litert_session_id') || null;
let busy              = false;
let mediaRecorder     = null;
let recordedChunks    = [];
let recording         = false;
let recordStartMs     = 0;
let engineLoaded      = false;
let microphoneAllowed = false;
let permissionWatcher = null;
let stagedImage       = null;
let camStream         = null;

const MIN_REC_MS  = 350;
const MIN_BLOB_B  = 800;
const TYPEWRITER_AFTER_TURNS = 2;   // hide typewriter once user has had >N turns
let turnCount     = 0;

// ── Orb state machine ─────────────────────────────────────────────────────

const STATES = {
  idle:      'TAP MIC TO TALK',
  listening: 'LISTENING…',
  thinking:  'THINKING…',
  speaking:  'SPEAKING…',
};

function setOrbState(state) {
  if (!STATES[state]) return;
  orb.dataset.state = state;
  stateLabel.textContent = STATES[state];
  // Typewriter rules:
  //  - hide entirely once the user has completed N turns (it's a fresh-start hint)
  //  - pause for any non-idle state
  if (turnCount > TYPEWRITER_AFTER_TURNS) {
    typewriter.hidden = true;
  } else {
    typewriter.hidden = false;
    typewriter.dataset.paused = state === 'idle' ? 'false' : 'true';
  }
}

// ── Captions ──────────────────────────────────────────────────────────────

function setUserCaption(text) {
  captionsUser.textContent = text || '';
  captionsUser.classList.remove('streaming');
}
function clearAssistantCaption() {
  captionsAsst.textContent = '';
  captionsAsst.classList.remove('streaming');
}
function startAssistantCaption() {
  clearAssistantCaption();
  captionsAsst.classList.add('streaming');
}
function appendAssistantToken(text) {
  if (!text) return;
  captionsAsst.textContent += text;
}
function finishAssistantCaption() {
  captionsAsst.classList.remove('streaming');
}

// ── Typewriter cycler (idle-only) ─────────────────────────────────────────

const PROMPTS = [
  "What's the latest tech news?",
  "Weather in Bangalore today",
  "Search for Sony WH-1000XM5 price",
  "Find coffee shops near MG Road",
  "Tell me a short joke",
  "Explain MCP in one line",
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function typewriterLoop() {
  let i = 0;
  while (true) {
    // Pause while not idle.
    while (orb.dataset.state !== 'idle') {
      tweText.textContent = '';
      await sleep(400);
    }
    const text = PROMPTS[i % PROMPTS.length];
    // Type in
    for (let n = 0; n <= text.length; n++) {
      if (orb.dataset.state !== 'idle') break;
      tweText.textContent = text.slice(0, n);
      await sleep(38);
    }
    // Hold
    await sleep(2400);
    // Erase
    for (let n = tweText.textContent.length; n >= 0; n--) {
      if (orb.dataset.state !== 'idle') break;
      tweText.textContent = text.slice(0, n);
      await sleep(18);
    }
    i++;
    await sleep(300);
  }
}

// ── Health / status chips ─────────────────────────────────────────────────

function basename(p) {
  if (!p) return '';
  const parts = String(p).split(/[\\/]/);
  return parts[parts.length - 1] || p;
}

function renderStatusChips(data) {
  statusChips.innerHTML = '';
  const chips = [
    { k: 'LLM',    v: data.backend || '?' },
    { k: 'audio',  v: data.audio_backend || '?' },
    { k: 'vision', v: data.vision_backend || '?' },
    { k: 'MTP',    v: data.speculative_decoding ? 'on' : 'off',
      cls: data.speculative_decoding ? 'on' : 'off' },
    { k: 'ctx',    v: String(data.max_num_tokens || '?') },
  ];
  for (const c of chips) {
    const el = document.createElement('span');
    el.className = 'chip' + (c.cls ? ' ' + c.cls : '');
    el.innerHTML = `<span class="k">${c.k}</span><span>${c.v}</span>`;
    statusChips.appendChild(el);
  }
  statusModel.textContent = basename(data.model_path) || '';
  statusModel.title = data.model_path || '';
}

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    engineLoaded = Boolean(data.engine_loaded);
    statusDot.classList.toggle('ok',  engineLoaded);
    statusDot.classList.toggle('bad', !engineLoaded);
    brandSub.textContent = engineLoaded ? 'on-device · ready'
                           : (data.error ? 'engine error' : 'loading…');
    renderStatusChips(data);
    seedSettingsForm(data);
    updateButtons();
  } catch (err) {
    engineLoaded = false;
    statusDot.classList.remove('ok');
    statusDot.classList.add('bad');
    brandSub.textContent = 'server offline';
    statusChips.innerHTML = '';
    statusModel.textContent = String(err);
  }
}

function updateButtons() {
  recordBtn.disabled = !engineLoaded || busy || !microphoneAllowed;
}

// ── Microphone permission ─────────────────────────────────────────────────

function setMicState(text) { micState.textContent = text; }

async function refreshMicrophonePermissionState() {
  if (!window.isSecureContext) { microphoneAllowed = false; setMicState('Open via localhost'); updateButtons(); return; }
  if (!navigator.mediaDevices?.getUserMedia) { microphoneAllowed = false; setMicState('Browser API unavailable'); updateButtons(); return; }
  try {
    if (navigator.permissions?.query) {
      const p = await navigator.permissions.query({ name: 'microphone' });
      microphoneAllowed = p.state === 'granted';
      setMicState(microphoneAllowed ? 'Permission granted'
        : (p.state === 'denied' ? 'Permission denied' : 'Tap Enable Mic'));
      if (!permissionWatcher) {
        permissionWatcher = p;
        p.onchange = () => refreshMicrophonePermissionState();
      }
    }
  } catch (_) { /* Safari */ }
  updateButtons();
}

async function requestMicrophonePermission() {
  try {
    setMicState('Requesting…');
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    stream.getTracks().forEach(t => t.stop());
    microphoneAllowed = true;
    setMicState('Permission granted');
  } catch (err) {
    microphoneAllowed = false;
    setMicState((err?.message || 'Permission denied').slice(0, 40));
  } finally { updateButtons(); }
}

enableMicBtn.addEventListener('click', requestMicrophonePermission);

// ── Audio playback ────────────────────────────────────────────────────────

let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

class AudioQueue {
  constructor(ctx) { this.ctx = ctx; this.nextStart = 0; }
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
}

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// ── Staged image ──────────────────────────────────────────────────────────

function setStagedImage(blob, filename) {
  clearStagedImage();
  stagedImage = { blob, url: URL.createObjectURL(blob), filename };
  attachThumb.src = stagedImage.url;
  attachLabel.textContent = filename;
  attachChip.hidden = false;
}
function clearStagedImage() {
  if (stagedImage) { URL.revokeObjectURL(stagedImage.url); stagedImage = null; }
  attachThumb.removeAttribute('src');
  attachChip.hidden = true;
}
attachClear.addEventListener('click', clearStagedImage);
attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  const f = fileInput.files?.[0];
  if (f) setStagedImage(f, f.name);
  fileInput.value = '';
});

function extOfMime(mime) {
  const m = { 'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp',
              'image/gif': 'gif', 'image/bmp': 'bmp' };
  return m[mime] || (mime?.split('/')?.[1] || 'png');
}

document.addEventListener('paste', (e) => {
  if (busy) return;
  for (const item of (e.clipboardData?.items || [])) {
    if (item.type?.startsWith('image/')) {
      const blob = item.getAsFile();
      if (blob) {
        setStagedImage(blob, `paste-${Date.now()}.${extOfMime(blob.type)}`);
        e.preventDefault();
        return;
      }
    }
  }
});

['dragenter', 'dragover'].forEach((ev) =>
  stage.addEventListener(ev, (e) => {
    const t = e.dataTransfer?.types || [];
    if (t.includes('Files') || t.includes('text/uri-list')) {
      e.preventDefault();
      stage.classList.add('drop-hover');
    }
  }));
['dragleave', 'dragend', 'drop'].forEach((ev) =>
  stage.addEventListener(ev, () => stage.classList.remove('drop-hover')));
stage.addEventListener('drop', async (e) => {
  if (busy) return;
  e.preventDefault();
  const file = e.dataTransfer?.files?.[0];
  if (file?.type?.startsWith('image/')) { setStagedImage(file, file.name); return; }
  const url = e.dataTransfer?.getData('text/uri-list') || e.dataTransfer?.getData('text/plain');
  if (!url) return;
  try {
    const res = await fetch(url, { mode: 'cors' });
    const blob = await res.blob();
    if (!blob.type?.startsWith('image/')) throw new Error('not an image');
    setStagedImage(blob, url.split('/').pop()?.split('?')[0] || `drop-${Date.now()}.${extOfMime(blob.type)}`);
  } catch (_) { /* CORS — silent */ }
});

// ── Camera ────────────────────────────────────────────────────────────────

snapBtn.addEventListener('click', async () => {
  if (!navigator.mediaDevices?.getUserMedia) return;
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    camVideo.srcObject = camStream;
    camModal.hidden = false;
  } catch (err) {
    captionsUser.textContent = `Camera error: ${err.message || err}`;
  }
});
function closeCam() {
  if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; }
  camVideo.srcObject = null;
  camModal.hidden = true;
}
camCancel.addEventListener('click', closeCam);
camCapture.addEventListener('click', () => {
  if (!camStream) return closeCam();
  const w = camVideo.videoWidth || 1280;
  const h = camVideo.videoHeight || 720;
  const scale = Math.min(1, 1024 / Math.max(w, h));
  const cw = Math.round(w * scale), ch = Math.round(h * scale);
  camCanvas.width = cw; camCanvas.height = ch;
  camCanvas.getContext('2d').drawImage(camVideo, 0, 0, cw, ch);
  camCanvas.toBlob((blob) => {
    if (blob) setStagedImage(blob, `snap-${Date.now()}.jpg`);
    closeCam();
  }, 'image/jpeg', 0.85);
});

// ── SSE plumbing ──────────────────────────────────────────────────────────

async function readSseResponse(res, onEvent) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const frames = buf.split('\n\n');
    buf = frames.pop() || '';
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

async function postFormStream(url, formData, onEvent) {
  const res = await fetch(url, { method: 'POST', body: formData });
  await readSseResponse(res, onEvent);
}

// ── Metrics ───────────────────────────────────────────────────────────────

function fmtMs(v) { return v == null ? '—' : `${v}ms`; }
function fmtN(v)  { return v == null ? '—' : String(v); }
function renderMetrics(m) {
  if (!m) return;
  mAudio.textContent  = fmtMs(m.audio_dur_ms);
  mTtft.textContent   = fmtMs(m.llm_ttft_ms);
  mTts.textContent    = fmtMs(m.tts_ttfb_ms);
  mTotal.textContent  = fmtMs(m.total_ttfb_ms);
  mTokens.textContent = fmtN(m.tokens);
  mTps.textContent    = m.decode_tok_per_s == null ? '—' : `${m.decode_tok_per_s}/s`;
  metricsPanel.hidden = false;
}

// ── Tool pill ─────────────────────────────────────────────────────────────

function showToolPill(name, label) {
  toolPill.textContent = `${label} · ${name}`;
  toolPill.hidden = false;
}
function hideToolPill() { toolPill.hidden = true; }

// ── Voice upload ──────────────────────────────────────────────────────────

async function uploadRecordingStreaming() {
  if (!recordedChunks.length) return;

  const blob = new Blob(recordedChunks, { type: mediaRecorder?.mimeType || 'audio/webm' });
  if (blob.size < MIN_BLOB_B) {
    setUserCaption('Hold longer to record — release after you finish speaking.');
    setOrbState('idle');
    return;
  }

  busy = true; updateButtons();

  const fd = new FormData();
  fd.append('audio', blob, 'recording.webm');
  if (sessionId) fd.append('session_id', sessionId);
  if (stagedImage) { fd.append('image', stagedImage.blob, stagedImage.filename); clearStagedImage(); }

  const queue = new AudioQueue(getAudioCtx());
  if (audioCtx.state === 'suspended') { try { await audioCtx.resume(); } catch (_) {} }

  setUserCaption('🎙 (voice message)');
  startAssistantCaption();
  setOrbState('thinking');

  let gotFirstToken = false;
  try {
    await postFormStream('/api/voice/chat/stream', fd, (event, data) => {
      if (event === 'status') {
        if (data.stage === 'encode' || data.stage === 'transcode') setOrbState('thinking');
      } else if (event === 'session') {
        sessionId = data.session_id;
        localStorage.setItem('litert_session_id', sessionId);
        if (sessionTag) sessionTag.textContent = sessionId.slice(0, 8).toUpperCase();
      } else if (event === 'token') {
        if (!gotFirstToken) { gotFirstToken = true; setOrbState('speaking'); }
        appendAssistantToken(data.text || '');
      } else if (event === 'audio_chunk') {
        const bytes = b64ToBytes(data.b64);
        queue.pushPcm16(bytes, data.sample_rate || 24000);
      } else if (event === 'tool_call') {
        showToolPill(data.name || 'tool', '🔎');
      } else if (event === 'tool_result') {
        showToolPill(data.name || 'tool', data.ok ? '✓' : '✗');
        setTimeout(hideToolPill, 1500);
      } else if (event === 'turn_metrics') {
        renderMetrics(data);
      } else if (event === 'warning') {
        appendAssistantToken(`\n[tts warning] ${data.warning}`);
      } else if (event === 'error') {
        appendAssistantToken(`\n[error] ${data.error}`);
      }
    });
  } catch (err) {
    appendAssistantToken(`\n[error] ${err.message}`);
  } finally {
    finishAssistantCaption();
    turnCount += 1;
    setOrbState('idle');           // re-evaluates typewriter visibility against turnCount
    hideToolPill();
    busy = false;
    await checkHealth();
  }
}

// ── Hold-to-talk ──────────────────────────────────────────────────────────

async function startRecording() {
  if (busy || recording) return;
  if (!microphoneAllowed) {
    await requestMicrophonePermission();
    if (!microphoneAllowed) return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    recordedChunks = [];
    let options = {};
    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) options = { mimeType: 'audio/webm;codecs=opus' };
    else if (MediaRecorder.isTypeSupported('audio/webm'))         options = { mimeType: 'audio/webm' };

    mediaRecorder = new MediaRecorder(stream, options);
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      await uploadRecordingStreaming();
    };
    mediaRecorder.start(250);
    recording = true;
    recordStartMs = performance.now();
    recordBtn.classList.add('recording');
    setOrbState('listening');
    setUserCaption('Listening…');
    clearAssistantCaption();
  } catch (err) {
    captionsUser.textContent = `[mic error] ${err?.message || err}`;
  }
}

function stopRecording() {
  if (!recording || !mediaRecorder) return;
  const elapsed = performance.now() - recordStartMs;
  if (elapsed < MIN_REC_MS) {
    setTimeout(stopRecording, MIN_REC_MS - elapsed);
    return;
  }
  recording = false;
  recordBtn.classList.remove('recording');
  mediaRecorder.stop();
}

recordBtn.addEventListener('mousedown', startRecording);
recordBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
window.addEventListener('mouseup', stopRecording);
window.addEventListener('touchend', stopRecording);

// ── New chat ──────────────────────────────────────────────────────────────

newChatBtn.addEventListener('click', async () => {
  if (busy) return;
  const prev = sessionId;
  sessionId = null;
  localStorage.removeItem('litert_session_id');
  if (prev) {
    try { await fetch(`/api/session/${encodeURIComponent(prev)}`, { method: 'DELETE' }); } catch (_) {}
  }
  if (audioCtx) { try { audioCtx.close(); } catch (_) {} audioCtx = null; }
  setUserCaption('');
  clearAssistantCaption();
  hideToolPill();
  metricsPanel.hidden = true;
  if (sessionTag) sessionTag.textContent = '—';
  clearStagedImage();
  turnCount = 0;                  // re-enable typewriter for the fresh session
  setOrbState('idle');
});

// ── Settings modal ────────────────────────────────────────────────────────

function seedSettingsForm(health) {
  if (!health) return;
  if (setBackend.value === '' || !settingsModal.open) {
    setBackend.value    = (health.backend || 'GPU').toUpperCase();
    setAudio.value      = (health.audio_backend || 'CPU').toUpperCase();
    setVision.value     = (health.vision_backend || 'CPU').toUpperCase();
    setMaxTokens.value  = String(health.max_num_tokens || 4096);
    setMTP.checked      = Boolean(health.speculative_decoding);
  }
}

function openSettings() {
  settingsModal.hidden = false;
  settingsModal.setAttribute('aria-hidden', 'false');
  setStatus.textContent = '';
  setStatus.className = 'modal-status';
}
function closeSettings() {
  settingsModal.hidden = true;
  settingsModal.setAttribute('aria-hidden', 'true');
}

settingsBtn.addEventListener('click', openSettings);
settingsModal.addEventListener('click', (e) => {
  if (e.target.dataset?.close !== undefined || e.target.matches('.modal-backdrop')) {
    closeSettings();
  }
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !settingsModal.hidden) closeSettings();
});

settingsForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (busy) return;
  setApply.disabled = true;
  setStatus.className = 'modal-status';
  setStatus.textContent = 'RESTARTING ENGINE…';

  const payload = {
    backend:            setBackend.value,
    audio_backend:      setAudio.value,
    vision_backend:     setVision.value,
    max_num_tokens:     parseInt(setMaxTokens.value, 10),
    enable_speculative: setMTP.checked,
  };

  try {
    const res = await fetch('/api/engine/reconfigure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    setStatus.className = 'modal-status ok';
    setStatus.textContent = 'ENGINE RESTARTED';
    // Drop the current session — KV cache + history are gone now.
    sessionId = null;
    localStorage.removeItem('litert_session_id');
    if (sessionTag) sessionTag.textContent = '—';
    clearAssistantCaption();
    setUserCaption('');
    turnCount = 0;                  // typewriter re-appears for the fresh engine
    setOrbState('idle');
    await checkHealth();
    setTimeout(closeSettings, 700);
  } catch (err) {
    setStatus.className = 'modal-status error';
    setStatus.textContent = `FAILED · ${err.message || err}`;
  } finally {
    setApply.disabled = false;
  }
});

// ── HUD clock ─────────────────────────────────────────────────────────────

function tickClock() {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  hudClock.textContent = `${hh}:${mm}`;
}
tickClock();
setInterval(tickClock, 30_000);

// ── Init ──────────────────────────────────────────────────────────────────

(async function init() {
  if (sessionId) sessionTag.textContent = sessionId.slice(0, 8).toUpperCase();
  setOrbState('idle');
  await refreshMicrophonePermissionState();
  await checkHealth();
  typewriterLoop().catch(() => {});  // fire-and-forget infinite loop
})();

setInterval(checkHealth, 5000);
