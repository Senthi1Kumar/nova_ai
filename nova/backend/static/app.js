(() => {
  const $ = id => document.getElementById(id);
  const setStatus = (k, t) => {
    document.body.dataset.state = k === 'listen' ? 'listening' : k === 'think' ? 'thinking' : k === 'speak' ? 'speaking' : 'idle';
    $('dot').className = 'dot ' + k; $('status').textContent = t;
  };
  setStatus('idle', 'idle');
  const turns = $('turns');
  const newTurn = (who, text) => {
    const el = document.createElement('div');
    el.className = 'turn ' + who;
    el.innerHTML = `<div class="who">${who}</div><div class="t"></div>`;
    el.querySelector('.t').textContent = text;
    turns.appendChild(el); el.scrollIntoView({ behavior: 'smooth', block: 'end' });
    $('nTurns').textContent = turns.children.length;
    return el.querySelector('.t');
  };
  const showErr = m => { $('err').textContent = m; setTimeout(() => $('err').textContent = '', 4000); };

  let ws, micCtx, playCtx, playT = 0, on = false, srcNode, workletNode, t0 = 0;
  let assistantSpan = null;
  let pendingAudioSr = 0;
  function playPCM(i16, sr) {
    if (!playCtx) return;
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
    const buf = playCtx.createBuffer(1, f32.length, sr);
    buf.copyToChannel(f32, 0);
    const s = playCtx.createBufferSource(); s.buffer = buf; s.connect(playCtx.destination);
    const t = Math.max(playT, playCtx.currentTime); s.start(t); playT = t + buf.duration;
  }

  async function start() {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
    } catch (e) { showErr('mic permission denied'); return; }

    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => setStatus('listen', 'listening');
    ws.onclose = () => setStatus('idle', 'disconnected');
    ws.onerror = () => showErr('connection error');
    ws.onmessage = onMessage;
    // Track expected binary audio frame from preceding header
    pendingAudioSr = 0;

    micCtx = new AudioContext({ sampleRate: 16000 });
    await micCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([`
      class P extends AudioWorkletProcessor {
        process(inputs) {
          const ch = inputs[0][0]; if (!ch) return true;
          let peak = 0;
          const i16 = new Int16Array(ch.length);
          for (let i = 0; i < ch.length; i++) {
            const v = Math.max(-1, Math.min(1, ch[i]));
            i16[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
            const a = v < 0 ? -v : v; if (a > peak) peak = a;
          }
          this.port.postMessage({ buf: i16.buffer, peak }, [i16.buffer]);
          return true;
        }
      }
      registerProcessor('p', P);`], { type: 'application/javascript' })));
    srcNode = micCtx.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(micCtx, 'p');
    workletNode.port.onmessage = e => {
      if (ws && ws.readyState === 1) ws.send(e.data.buf);
      const pct = Math.min(100, Math.round(e.data.peak * 200));
      $('lvlBar').style.width = pct + '%';
      $('lvl').textContent = `${pct}%`;
    };
    srcNode.connect(workletNode);

    playCtx = new AudioContext({ sampleRate: 24000 });
    playT = playCtx.currentTime;
    on = true; $('micBtn').classList.add('on');
  }

  function stop() {
    on = false; $('micBtn').classList.remove('on');
    if (workletNode) workletNode.disconnect();
    if (srcNode) srcNode.disconnect();
    if (micCtx) micCtx.close().catch(()=>{});
    if (playCtx) playCtx.close().catch(()=>{});
    if (ws && ws.readyState === 1) ws.close();
    setStatus('idle', 'idle');
  }

  function onMessage(e) {
    // Binary frame arrives right after an audio_header text frame.
    if (e.data instanceof ArrayBuffer) {
      const i16 = new Int16Array(e.data);
      playPCM(i16, pendingAudioSr || 24000);
      pendingAudioSr = 0;
      return;
    }
    let msg; try { msg = JSON.parse(e.data) } catch { return; }
    switch (msg.type) {
      case 'speech_started':
        setStatus('listen', 'listening'); break;
      case 'transcript':
        newTurn('user', msg.data || '');
        t0 = performance.now();
        setStatus('think', 'thinking…');
        break;
      case 'assistant_start':
        assistantSpan = newTurn('assistant', '');
        break;
      case 'llm_token':
        if (assistantSpan) assistantSpan.textContent += (assistantSpan.textContent ? ' ' : '') + (msg.data || '');
        if (t0) { $('lat').textContent = `${Math.round(performance.now() - t0)} ms`; t0 = 0; }
        setStatus('speak', 'speaking…');
        break;
      case 'generation_done':
        setStatus('listen', 'listening');
        assistantSpan = null;
        // Refresh sidebar after the turn — memory may have grown.
        loadDoc(currentDoc);
        break;
      case 'audio_header': pendingAudioSr = msg.sr || 24000; break;
      case 'audio_out': {
        // Legacy base64 path (NOVA_TTS_BINARY=0).
        const bin = atob(msg.pcm_b64);
        const arr = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
        playPCM(new Int16Array(arr.buffer), msg.sr);
        break;
      }
      case 'error': showErr(msg.data || 'error'); break;
    }
  }

  // Sidebar tabs
  let currentDoc = 'user';
  async function loadDoc(name) {
    try {
      const r = await fetch('/persona/' + name); const j = await r.json();
      $('doc').textContent = j.content || '(empty)';
    } catch { $('doc').textContent = '(failed to load)' }
  }
  document.querySelectorAll('.tab').forEach(t => {
    t.onclick = () => {
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      currentDoc = t.dataset.doc; loadDoc(currentDoc);
    };
  });
  loadDoc('user');

  $('micBtn').onclick = () => on ? stop() : start();
  document.addEventListener('keydown', e => {
    if (e.code === 'Space' && on && playCtx) {
      // local interrupt: drop scheduled playback so user feels barge-in immediately
      try { playCtx.close(); } catch {}
      playCtx = new AudioContext({ sampleRate: 24000 });
      playT = playCtx.currentTime;
      setStatus('listen', 'listening');
      e.preventDefault();
    }
  });
})();
