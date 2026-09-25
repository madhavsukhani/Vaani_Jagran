/**
 * script.js – VAANI JAGRAN Dashboard
 * by: Team EdgeVox (Smart India Hackathon)
 */

'use strict';

// ─────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────
const SERVER_HOST = window.location.hostname || 'localhost';
const SERVER_PORT = 8000;
const WS_URL      = `ws://${SERVER_HOST}:${SERVER_PORT}/ws/frontend`;
const HTTP_BASE   = `http://${SERVER_HOST}:${SERVER_PORT}`;
const RECONNECT_DELAY_MS = 2500;

// ─────────────────────────────────────────────
//  DOM references
// ─────────────────────────────────────────────
const $serverPill       = document.getElementById('server-pill');
const $serverLabel      = document.getElementById('server-label');
const $esp32Pill        = document.getElementById('esp32-pill');
const $esp32Label       = document.getElementById('esp32-label');

const $cardStatus       = document.getElementById('card-status');
const $systemState      = document.getElementById('system-state');
const $systemStateDesc  = document.getElementById('system-state-desc');
const $stateIcon        = document.getElementById('state-icon');

const $esp32Cpu         = document.getElementById('esp32-cpu');
const $esp32CpuCores    = document.getElementById('esp32-cpu-cores');
const $esp32Ram         = document.getElementById('esp32-ram');
const $esp32RamDetail   = document.getElementById('esp32-ram-detail');

const $latencyValue     = document.getElementById('latency-value');
const $bufferValue      = document.getElementById('buffer-value');
const $segmentsCount    = document.getElementById('segments-count');

const $asrBadge         = document.getElementById('asr-badge');
const $transcriptBox    = document.getElementById('transcript-box');
const $transcriptPh     = document.getElementById('transcript-placeholder');
const $transcriptText   = document.getElementById('transcript-text');

const $recordingsList   = document.getElementById('recordings-list');
const $recordingsEmpty  = document.getElementById('recordings-empty');
const $recordingsBadge  = document.getElementById('recordings-count-badge');
const $refreshRecBtn    = document.getElementById('refresh-recordings-btn');

const $logBox           = document.getElementById('log-box');
const $clearLogBtn      = document.getElementById('clear-log-btn');

const $waveformCanvas   = document.getElementById('waveform-canvas');
const waveCtx           = $waveformCanvas.getContext('2d');

// ─────────────────────────────────────────────
//  State
// ─────────────────────────────────────────────
let ws             = null;
let reconnectTimer = null;
let waveAnimId     = null;
let isStreaming    = false;
let displayedFiles = new Set();
const waveformLevels = [];
const MAX_WAVEFORM_LEVELS = 180;
let lastAudioLevelAt = 0;
let waveformStopTimer = null;
const WAVEFORM_IDLE_TIMEOUT_MS = 450;

// ─────────────────────────────────────────────
//  WebSocket connection
// ─────────────────────────────────────────────
function connect() {
  log('info', `Connecting to server at ${WS_URL}…`);
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    log('success', 'Connected to VAANI JAGRAN backend.');
    setServerPill(true);
    fetchRecordings();
  };

  ws.onmessage = (evt) => {
    try {
      const payload = JSON.parse(evt.data);
      handleMessage(payload);
    } catch (e) {
      console.warn('Malformed JSON received:', evt.data);
    }
  };

  ws.onerror = () => {
    setServerPill(false);
  };

  ws.onclose = () => {
    log('warn', 'Connection to server lost. Reconnecting in 2.5s…');
    setServerPill(false);
    updateEsp32Pill(false);
    scheduleReconnect();
  };
}

function scheduleReconnect() {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
}

// ─────────────────────────────────────────────
//  Message router
// ─────────────────────────────────────────────
function handleMessage(payload) {
  const { type } = payload;

  if (payload.esp32_connected !== undefined) {
    updateEsp32Pill(payload.esp32_connected);
  }

  if (payload.system_state !== undefined) {
    updateSystemState(payload.system_state);
  }

  if (payload.telemetry) {
    updateTelemetry(payload.telemetry);
  }

  if (payload.latency_ms !== undefined) {
    updateLatency(payload.latency_ms);
  }

  if (payload.buffer_duration_s !== undefined) {
    $bufferValue.textContent = `${payload.buffer_duration_s.toFixed(2)} s`;
  }

  if (payload.audio_level !== undefined) {
    noteAudioActivity(payload.audio_level);
  }

  if (payload.recordings_count !== undefined) {
    $segmentsCount.textContent = payload.recordings_count;
    $recordingsBadge.textContent = `${payload.recordings_count} clips`;
  }

  if (payload.transcript !== undefined) {
    updateTranscript(payload.transcript);
  }

  // Initial load or full sync with list of recordings
  if (payload.recordings && Array.isArray(payload.recordings)) {
    payload.recordings.forEach(rec => renderRecordingItem(rec, false));
  }

  // Type-specific actions
  if (type === 'new_recording' && payload.recording) {
    renderRecordingItem(payload.recording, true);
    log('success', `Saved new recording: ${payload.recording.filename} (${payload.recording.duration_seconds}s)`);
  }

  if (type === 'event') {
    const { event, message } = payload;
    const level = event.includes('error') ? 'error' : (event.includes('detected') ? 'success' : 'info');
    log(level, `[SYSTEM] ${message}`);
  }
}

// ─────────────────────────────────────────────
//  UI Updaters
// ─────────────────────────────────────────────
function setServerPill(connected) {
  $serverPill.className = `pill ${connected ? 'connected' : 'disconnected'}`;
  $serverLabel.textContent = connected ? 'Server Online' : 'Server Offline';
}

function updateEsp32Pill(connected) {
  $esp32Pill.className = `pill ${connected ? 'connected' : 'disconnected'}`;
  $esp32Label.textContent = connected ? 'ESP32 Online' : 'ESP32 Offline';
  if (!connected) {
    resetWaveformImmediately();
    updateSystemState('Listening');
  }
}

function updateSystemState(state) {
  $cardStatus.dataset.state = state;
  $systemState.textContent = state;

  if (state === 'Listening') {
    $stateIcon.textContent = '👂';
    $systemStateDesc.textContent = 'Waiting for wake-word';
    $asrBadge.textContent = 'Listening…';
    $asrBadge.className = 'panel__badge badge--listening';
    scheduleWaveformStop();
  } else if (state === 'Detected') {
    $stateIcon.textContent = '🎯';
    $systemStateDesc.textContent = 'Wake-word spotted! Triggering…';
    $asrBadge.textContent = 'Detected!';
    $asrBadge.className = 'panel__badge badge--detected';
  } else if (state === 'Streaming') {
    $stateIcon.textContent = '🎙️';
    $systemStateDesc.textContent = 'Live audio streaming to ASR';
    $asrBadge.textContent = 'Live Streaming';
    $asrBadge.className = 'panel__badge badge--streaming';
    if (!isStreaming) {
      isStreaming = true;
      startWaveform();
    }
  }
}

function updateTelemetry(telem) {
  // ESP32 CPU (Dual core)
  const avgCpu = Math.round(telem.cpu_avg || 0);
  const c0 = Math.round(telem.cpu_c0 || 0);
  const c1 = Math.round(telem.cpu_c1 || 0);
  $esp32Cpu.textContent = `${avgCpu}%`;
  $esp32CpuCores.textContent = `Core 0 (WiFi): ${c0}% · Core 1 (ML): ${c1}%`;

  // ESP32 RAM (Internal heap)
  const ramPct = Math.round(telem.ram_pct || 0);
  const usedKb = Math.round(telem.ram_used_kb || 0);
  const totalKb = Math.round(telem.ram_total_kb || 320);
  $esp32Ram.textContent = `${ramPct}%`;
  $esp32RamDetail.textContent = `${usedKb} KB / ${totalKb} KB`;

  if (telem.state && telem.state !== $systemState.textContent) {
    updateSystemState(telem.state);
  }
}

function updateLatency(ms) {
  if (ms > 0) {
    $latencyValue.textContent = `${Math.round(ms)} ms`;
  } else {
    $latencyValue.textContent = '—';
  }
}

function updateTranscript(text) {
  if (!text || text.trim() === '') return;
  $transcriptPh.classList.add('hidden');
  $transcriptText.textContent = text;
  $transcriptBox.scrollTop = $transcriptBox.scrollHeight;
}

// ─────────────────────────────────────────────
//  Recordings: In-Browser Player & Downloads
// ─────────────────────────────────────────────
async function fetchRecordings() {
  try {
    const res = await fetch(`${HTTP_BASE}/api/recordings`);
    if (!res.ok) return;
    const data = await res.json();
    if (!Array.isArray(data.recordings)) return;

    const currentFiles = new Set(data.recordings.map(rec => rec.filename));
    for (const filename of displayedFiles) {
      if (!currentFiles.has(filename)) {
        document.getElementById(`rec-${filename.replace(/[^a-zA-Z0-9]/g, '_')}`)?.remove();
        displayedFiles.delete(filename);
      }
    }

    $recordingsBadge.textContent = `${data.count} clips`;
    $segmentsCount.textContent = data.count;
    $recordingsEmpty.classList.toggle('hidden', data.recordings.length > 0);
    data.recordings.forEach(rec => renderRecordingItem(rec, false));
  } catch (err) {
    console.debug('Failed to fetch recordings list:', err);
  }
}

// WebSocket events update the list immediately; this catches recordings
// completed while the dashboard was reconnecting.
setInterval(fetchRecordings, 5000);

function renderRecordingItem(rec, prepend = true) {
  if (!rec || !rec.filename) return;
  if (displayedFiles.has(rec.filename)) return;
  displayedFiles.add(rec.filename);

  $recordingsEmpty.classList.add('hidden');

  const card = document.createElement('div');
  card.className = 'recording-card';
  card.id = `rec-${rec.filename.replace(/[^a-zA-Z0-9]/g, '_')}`;

  const audioSrc = `${HTTP_BASE}/api/recordings/${encodeURIComponent(rec.filename)}`;
  const durationText = rec.duration_seconds ? `${rec.duration_seconds}s` : 'Audio';
  const transcriptText = rec.transcript && rec.transcript !== '(Saved recording)'
    ? rec.transcript
    : 'Spoken audio segment';

  card.innerHTML = `
    <div class="recording-card__header">
      <div class="recording-card__meta">
        <span class="recording-card__title">🎵 ${escapeHtml(rec.filename)}</span>
        <span class="recording-card__time">${escapeHtml(rec.timestamp || '')}</span>
      </div>
      <span class="recording-card__tag">${durationText}</span>
    </div>

    <!-- Live In-Browser Audio Player -->
    <div class="recording-card__player">
      <span class="recording-card__play-label">PLAYBACK</span>
      <audio controls preload="metadata" src="${audioSrc}" class="native-audio-player"></audio>
    </div>

    <div class="recording-card__transcript">
      <span class="recording-card__quote">“</span>${escapeHtml(transcriptText)}<span class="recording-card__quote">”</span>
    </div>

    <div class="recording-card__footer">
      <a href="${audioSrc}" download="${escapeHtml(rec.filename)}" class="btn btn--download" title="Download WAV directly">
        <span>⬇️</span> Download .WAV
      </a>
      <span class="recording-card__filesize">${formatBytes(rec.size_bytes)}</span>
    </div>
  `;

  if (prepend) {
    $recordingsList.prepend(card);
  } else {
    $recordingsList.appendChild(card);
  }
}

function formatBytes(bytes) {
  if (!bytes) return '';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ─────────────────────────────────────────────
//  Live PCM amplitude waveform
// ─────────────────────────────────────────────
function resizeCanvas() {
  const rect = $waveformCanvas.parentElement.getBoundingClientRect();
  $waveformCanvas.width = rect.width;
  $waveformCanvas.height = 64;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawWave() {
  const w = $waveformCanvas.width;
  const h = $waveformCanvas.height;
  waveCtx.clearRect(0, 0, w, h);

  if (!isStreaming && waveformLevels.length === 0) {
    // Flat standby line
    waveCtx.strokeStyle = 'rgba(74, 222, 128, 0.25)';
    waveCtx.lineWidth = 1.5;
    waveCtx.beginPath();
    waveCtx.moveTo(0, h / 2);
    waveCtx.lineTo(w, h / 2);
    waveCtx.stroke();
    return;
  }

  waveCtx.lineWidth = 2;
  const gradient = waveCtx.createLinearGradient(0, 0, w, 0);
  gradient.addColorStop(0, '#38bdf8');
  gradient.addColorStop(0.5, '#4ade80');
  gradient.addColorStop(1, '#a855f7');
  waveCtx.strokeStyle = gradient;

  const levels = waveformLevels.length ? waveformLevels : [0];
  const sliceWidth = w / Math.max(levels.length - 1, 1);
  waveCtx.beginPath();
  levels.forEach((level, index) => {
    const x = index * sliceWidth;
    const envelope = Math.max(0.04, Math.min(1, level));
    const y = (h / 2) - envelope * (h * 0.42);
    if (index === 0) waveCtx.moveTo(x, y);
    else waveCtx.lineTo(x, y);
  });
  levels.slice().reverse().forEach((level, reverseIndex) => {
    const index = levels.length - 1 - reverseIndex;
    const x = index * sliceWidth;
    const envelope = Math.max(0.04, Math.min(1, level));
    const y = (h / 2) + envelope * (h * 0.42);
    waveCtx.lineTo(x, y);
  });
  waveCtx.closePath();
  waveCtx.globalAlpha = isStreaming ? 0.9 : 0.55;
  waveCtx.fillStyle = gradient;
  waveCtx.fill();
  waveCtx.globalAlpha = 1;

  waveCtx.beginPath();
  levels.forEach((level, index) => {
    const x = index * sliceWidth;
    const y = (h / 2) - Math.max(0.04, Math.min(1, level)) * (h * 0.42);
    if (index === 0) waveCtx.moveTo(x, y);
    else waveCtx.lineTo(x, y);
  });
  waveCtx.stroke();

  if (isStreaming) {
    waveAnimId = requestAnimationFrame(drawWave);
  }
}

function addWaveformLevel(level) {
  waveformLevels.push(Math.max(0, Math.min(1, Number(level) || 0)));
  if (waveformLevels.length > MAX_WAVEFORM_LEVELS) waveformLevels.shift();
  if (!waveAnimId) drawWave();
}

function noteAudioActivity(level) {
  lastAudioLevelAt = performance.now();
  clearTimeout(waveformStopTimer);
  if (!isStreaming) {
    isStreaming = true;
    startWaveform();
  }
  addWaveformLevel(level);
  waveformStopTimer = setTimeout(scheduleWaveformStop, WAVEFORM_IDLE_TIMEOUT_MS);
}

function scheduleWaveformStop() {
  clearTimeout(waveformStopTimer);
  const silenceFor = performance.now() - lastAudioLevelAt;
  if (lastAudioLevelAt && silenceFor < WAVEFORM_IDLE_TIMEOUT_MS) {
    waveformStopTimer = setTimeout(scheduleWaveformStop,
      WAVEFORM_IDLE_TIMEOUT_MS - silenceFor);
    return;
  }
  stopWaveform();
  isStreaming = false;
  waveformLevels.length = 0;
  drawWave();
}

function resetWaveformImmediately() {
  clearTimeout(waveformStopTimer);
  lastAudioLevelAt = 0;
  stopWaveform();
  isStreaming = false;
  waveformLevels.length = 0;
  drawWave();
}

function startWaveform() {
  if (waveAnimId) cancelAnimationFrame(waveAnimId);
  waveAnimId = null;
  waveformLevels.length = 0;
  drawWave();
}

function stopWaveform() {
  if (waveAnimId) {
    cancelAnimationFrame(waveAnimId);
    waveAnimId = null;
  }
  drawWave();
}

// ─────────────────────────────────────────────
//  Event Logger
// ─────────────────────────────────────────────
function log(level, message) {
  const time = new Date().toLocaleTimeString();
  const row = document.createElement('div');
  row.className = `log-row log-row--${level}`;
  row.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-msg">${escapeHtml(message)}</span>`;
  $logBox.appendChild(row);
  $logBox.scrollTop = $logBox.scrollHeight;
}

$clearLogBtn.addEventListener('click', () => {
  $logBox.innerHTML = '';
});

$refreshRecBtn.addEventListener('click', () => {
  displayedFiles.clear();
  $recordingsList.innerHTML = '';
  $recordingsEmpty.classList.remove('hidden');
  fetchRecordings();
  log('info', 'Refreshed recordings list.');
});

// Initialize on page load
connect();
drawWave();
