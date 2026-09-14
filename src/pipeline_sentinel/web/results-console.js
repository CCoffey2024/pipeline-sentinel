(() => {
  const style = document.createElement('style');
  style.textContent = `
    .result-actions { display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; margin:10px 0 16px; }
    .storage-pill { color:var(--muted); border:1px solid var(--line); background:#0d1319; border-radius:999px; padding:7px 10px; font-size:11px; }
    .button.danger { border-color:rgba(255,107,107,.48); background:rgba(255,107,107,.10); color:#ffc8c8; }
    .button.danger:hover { border-color:var(--critical); }
    .result-tabs { display:flex; gap:6px; flex-wrap:wrap; margin:8px 0 12px; }
    .result-tab { border:1px solid var(--line); background:#0d1319; color:var(--muted); border-radius:7px; padding:8px 10px; cursor:pointer; font-size:11px; }
    .result-tab:hover { border-color:var(--accent-2); color:var(--text); }
    .result-tab.active { color:#dcfff8; border-color:var(--accent); background:#18332f; }
    .result-panel { border:1px solid var(--line); background:rgba(13,19,25,.55); border-radius:10px; padding:14px; min-height:88px; }
    .result-grid { display:grid; grid-template-columns:minmax(280px,1.3fr) minmax(230px,.7fr); gap:16px; }
    .result-subtitle { color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.13em; margin:0 0 10px; }
    .bar-list { display:grid; gap:8px; }
    .bar-row { display:grid; grid-template-columns:minmax(75px,120px) 1fr auto; align-items:center; gap:9px; font-size:11px; }
    .bar-label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .bar-track { height:11px; border-radius:999px; background:#101820; overflow:hidden; border:1px solid rgba(38,50,62,.8); }
    .bar-fill { height:100%; background:linear-gradient(90deg,var(--accent),var(--accent-2)); min-width:2px; }
    .bar-value { color:var(--muted); min-width:42px; text-align:right; font-variant-numeric:tabular-nums; }
    .spark-wrap { border:1px solid var(--line); border-radius:8px; background:#0b1015; padding:10px; }
    .spark-wrap svg { width:100%; height:90px; display:block; overflow:visible; }
    .spark-line { fill:none; stroke:var(--accent-2); stroke-width:1.6; vector-effect:non-scaling-stroke; }
    .density-stats { display:grid; grid-template-columns:repeat(3,1fr); gap:7px; margin-top:9px; }
    .density-stat { border:1px solid var(--line); border-radius:7px; padding:8px; text-align:center; }
    .density-stat b { display:block; font-size:15px; }
    .density-stat span { color:var(--muted); font-size:9px; text-transform:uppercase; letter-spacing:.1em; }
    .result-table { max-height:360px; overflow:auto; border:1px solid var(--line); border-radius:8px; }
    .result-table table { font-size:11px; }
    .result-table tbody tr:hover { background:rgba(123,182,255,.06); }
    .result-empty { color:var(--muted); padding:18px 8px; text-align:center; font-size:12px; }
    .result-note { color:var(--muted); font-size:10px; margin-top:8px; line-height:1.45; }
    .download-grid { display:flex; flex-wrap:wrap; gap:8px; }
    .download-grid .artifact { display:inline-flex; flex-direction:column; gap:2px; min-width:130px; }
    .download-grid .artifact small { color:var(--muted); font-size:9px; }
    .frame-preview { margin-top:10px; border:1px solid var(--line); border-radius:9px; background:#0d1319; overflow:hidden; }
    .frame-preview summary { cursor:pointer; padding:10px 12px; color:var(--text); font-size:11px; user-select:none; }
    .frame-preview-body { padding:0 10px 10px; }
    .frame-preview img { display:block; width:100%; max-height:480px; object-fit:contain; background:#000; border:1px solid var(--line); border-radius:8px; }
    .frame-player-toolbar { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:10px; }
    .frame-player-button { border:1px solid var(--line); background:#121b24; color:var(--text); border-radius:7px; padding:7px 10px; cursor:pointer; font-size:11px; min-height:32px; }
    .frame-player-button:hover { border-color:var(--accent-2); }
    .frame-player-button.primary { border-color:#3a5f5b; background:#18332f; color:#dcfff8; min-width:68px; }
    .frame-player-button.primary:hover { border-color:var(--accent); }
    .frame-player-field { display:inline-flex; align-items:center; gap:6px; color:var(--muted); font-size:10px; margin:0; }
    .frame-player-field select { width:auto; min-width:72px; padding:6px 8px; border-radius:7px; font-size:11px; }
    .frame-player-field input[type="checkbox"] { margin:0; }
    .frame-preview-time { margin-left:auto; color:var(--text); font-size:11px; font-variant-numeric:tabular-nums; white-space:nowrap; }
    .frame-preview-controls { display:flex; align-items:center; gap:10px; margin-top:9px; }
    .frame-preview-controls input[type="range"] { flex:1; }
    .frame-preview-label { min-width:78px; text-align:right; color:var(--muted); font-size:10px; font-variant-numeric:tabular-nums; }
    @media (max-width: 850px) { .result-grid { grid-template-columns:1fr; } .frame-preview-time { width:100%; margin-left:0; } }
  `;
  document.head.appendChild(style);

  state.resultTabs = state.resultTabs || {};
  state.framePreviewStop = state.framePreviewStop || null;

  const formatBytes = (bytes) => {
    const value = Number(bytes || 0);
    if (value < 1024) return `${value.toLocaleString()} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let n = value / 1024;
    let i = 0;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
    return `${n >= 100 ? n.toFixed(0) : n >= 10 ? n.toFixed(1) : n.toFixed(2)} ${units[i]}`;
  };

  const numberOrDash = (value, digits=2) => {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : '—';
  };

  const formatTimestamp = (seconds) => {
    const totalMs = Math.max(0, Math.round((Number(seconds) || 0) * 1000));
    const hours = Math.floor(totalMs / 3600000);
    const minutes = Math.floor((totalMs % 3600000) / 60000);
    const wholeSeconds = Math.floor((totalMs % 60000) / 1000);
    const millis = totalMs % 1000;
    const hh = hours ? `${hours}:` : '';
    const mm = hours ? String(minutes).padStart(2, '0') : String(minutes);
    return `${hh}${mm}:${String(wholeSeconds).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
  };

  function classBars(classes, kind='run') {
    if (!classes || !classes.length) return '<div class="result-empty">No class-level results were emitted.</div>';
    const maxValue = Math.max(...classes.map(row => Number(row.detections || 0)), 1);
    const noun = kind === 'fusion' ? 'events' : 'detections';
    return `<div class="bar-list">${classes.map(row => {
      const value = Number(row.detections || 0);
      const width = Math.max(1, (value / maxValue) * 100);
      const conf = row.mean_confidence == null ? 'n/a' : Number(row.mean_confidence).toFixed(3);
      const title = kind === 'fusion'
        ? `${row.label}: ${value.toLocaleString()} fused events`
        : `${row.label}: ${value.toLocaleString()} detections, ${Number(row.tracks || 0).toLocaleString()} tracks, mean confidence ${conf}`;
      return `<div class="bar-row" title="${escapeHtml(title)}"><div class="bar-label">${escapeHtml(row.label)}</div><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><div class="bar-value">${value.toLocaleString()}</div></div>`;
    }).join('')}</div><div class="result-note">Hover over a bar for ${noun}, track, and confidence detail.</div>`;
  }

  function sparkline(series) {
    if (!series || series.length < 2) return '<div class="result-empty">Not enough frame data to graph detection density.</div>';
    const maximum = Math.max(...series.map(point => Number(point.count || 0)), 1);
    const points = series.map((point, index) => {
      const x = series.length === 1 ? 0 : (index / (series.length - 1)) * 100;
      const y = 30 - (Number(point.count || 0) / maximum) * 28;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(' ');
    const first = series[0];
    const last = series[series.length - 1];
    return `<div class="spark-wrap" title="Detection count by frame; sampled for long runs"><svg viewBox="0 0 100 32" preserveAspectRatio="none" aria-label="Detections per frame"><polyline class="spark-line" points="${points}"></polyline></svg><div class="result-note">Frame ${escapeHtml(first.frame)} → ${escapeHtml(last.frame)}. Long runs are downsampled for display only; evidence files remain complete.</div></div>`;
  }

  function overviewHtml(results) {
    const storage = results?.storage || {};
    const stats = results?.detection_frame_stats || {};
    const labelTitle = results?.kind === 'fusion' ? 'Fused events by type' : 'Detections by class';
    const density = results?.kind === 'run'
      ? `<div><div class="result-subtitle">Detections across frames</div>${sparkline(results.detections_by_frame || [])}<div class="density-stats"><div class="density-stat"><b>${numberOrDash(stats.min,0)}</b><span>min/frame</span></div><div class="density-stat"><b>${numberOrDash(stats.mean,1)}</b><span>mean/frame</span></div><div class="density-stat"><b>${numberOrDash(stats.max,0)}</b><span>max/frame</span></div></div></div>`
      : `<div><div class="result-subtitle">Run storage</div><div class="spark-wrap"><b>${formatBytes(storage.total_bytes)}</b><div class="result-note">Generated evidence plus any Pipeline Sentinel-managed uploaded input.</div></div></div>`;
    return `<div class="result-grid"><div><div class="result-subtitle">${labelTitle}</div>${classBars(results?.classes || [], results?.kind)}</div>${density}</div>`;
  }

  function tableWrap(headers, rowsHtml, emptyMessage) {
    if (!rowsHtml) return `<div class="result-empty">${escapeHtml(emptyMessage)}</div>`;
    return `<div class="result-table"><table><thead><tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join('')}</tr></thead><tbody>${rowsHtml}</tbody></table></div>`;
  }

  async function detectionsHtml(jobId) {
    const rows = await jsonFetch(`/api/jobs/${encodeURIComponent(jobId)}/detections?limit=150`);
    const body = rows.slice().reverse().map(row => `<tr title="bbox ${escapeHtml(row.x1)},${escapeHtml(row.y1)} → ${escapeHtml(row.x2)},${escapeHtml(row.y2)}"><td>${fmt(row.frame_number)}</td><td>${numberOrDash(row.timestamp_s,2)}s</td><td>${escapeHtml(row.label)}</td><td>${numberOrDash(row.confidence,3)}</td><td>${escapeHtml(row.detector || '—')}</td></tr>`).join('');
    return tableWrap(['Frame','Time','Class','Confidence','Detector'], body, 'No detections were emitted by this run.') + '<div class="result-note">Showing up to the most recent 150 detection rows. Hover a row to see its bounding box.</div>';
  }

  async function tracksHtml(jobId) {
    const rows = await jsonFetch(`/api/jobs/${encodeURIComponent(jobId)}/tracks?limit=150`);
    const body = rows.slice().reverse().map(row => `<tr title="bbox ${escapeHtml(row.x1)},${escapeHtml(row.y1)} → ${escapeHtml(row.x2)},${escapeHtml(row.y2)}"><td>${escapeHtml(row.track_id)}</td><td>${fmt(row.frame_number)}</td><td>${escapeHtml(row.label)}</td><td>${numberOrDash(row.confidence,3)}</td><td>${fmt(row.hits)}</td><td>${numberOrDash(row.duration_s,2)}s</td></tr>`).join('');
    return tableWrap(['Track','Frame','Class','Confidence','Hits','Duration'], body, 'No track observations were emitted by this run.') + '<div class="result-note">Showing up to the most recent 150 track observations.</div>';
  }

  async function anomaliesHtml(jobId) {
    const rows = await jsonFetch(`/api/jobs/${encodeURIComponent(jobId)}/anomalies?limit=150`);
    const body = rows.slice().reverse().map(row => `<tr><td>${fmt(row.frame_number)}</td><td>${escapeHtml(row.track_id)}</td><td>${escapeHtml(row.label)}</td><td>${numberOrDash(row.score,3)}</td><td>${numberOrDash(row.threshold,3)}</td><td>${row.is_anomaly ? 'YES' : 'no'}</td></tr>`).join('');
    return tableWrap(['Frame','Track','Class','Score','Threshold','Flagged'], body, 'No anomaly observations were emitted by this run.') + '<div class="result-note">Anomaly scoring remains separate from human-facing alerts.</div>';
  }

  async function eventsAlertsHtml(jobId) {
    const [events, alerts] = await Promise.all([
      jsonFetch(`/api/jobs/${encodeURIComponent(jobId)}/events?limit=150`),
      jsonFetch(`/api/jobs/${encodeURIComponent(jobId)}/alerts?limit=150`),
    ]);
    const eventBody = events.slice().reverse().map(row => `<tr><td>${escapeHtml(row.severity || '—')}</td><td>${numberOrDash(row.timestamp_s,2)}s</td><td>${escapeHtml(row.event_type || '—')}</td><td>${escapeHtml(row.label || '—')}</td><td>${escapeHtml(row.message || '—')}</td></tr>`).join('');
    const alertBody = alerts.slice().reverse().map(row => `<tr class="alert-row ${escapeHtml(row.severity)}"><td>${escapeHtml(row.severity || '—')}</td><td>${numberOrDash(row.timestamp_s,2)}s</td><td>${escapeHtml(row.alert_type || '—')}</td><td>${escapeHtml(row.label || '—')}</td><td>${escapeHtml(row.message || '—')}</td></tr>`).join('');
    return `<div class="result-subtitle">Events</div>${tableWrap(['Severity','Time','Type','Label','Message'], eventBody, 'No semantic events were emitted.')}<div class="result-subtitle" style="margin-top:16px">Alerts</div>${tableWrap(['Severity','Time','Type','Label','Message'], alertBody, 'No human-facing alerts were emitted.')}`;
  }

  function downloadsHtml(artifacts) {
    const entries = Object.entries(artifacts || {});
    if (!entries.length) return '<div class="result-empty">No downloadable artifacts are available.</div>';
    return `<div class="download-grid">${entries.map(([name,item]) => `<a class="artifact" href="${escapeHtml(item.url)}" target="_blank"><span>${escapeHtml(name)}</span><small>${escapeHtml(item.filename || '')}</small></a>`).join('')}</div><div class="result-note">Downloads are retained for auditability, but normal inspection can be done directly in this console.</div>`;
  }

  async function loadTab(job, tab, results, artifacts) {
    const panel = document.getElementById('result-panel');
    if (!panel) return;
    panel.innerHTML = '<div class="result-empty">Loading…</div>';
    try {
      if (tab === 'overview') panel.innerHTML = overviewHtml(results);
      else if (tab === 'detections') panel.innerHTML = await detectionsHtml(job.job_id);
      else if (tab === 'tracks') panel.innerHTML = await tracksHtml(job.job_id);
      else if (tab === 'anomalies') panel.innerHTML = await anomaliesHtml(job.job_id);
      else if (tab === 'events') panel.innerHTML = await eventsAlertsHtml(job.job_id);
      else if (tab === 'downloads') panel.innerHTML = downloadsHtml(artifacts);
    } catch (err) {
      panel.innerHTML = `<div class="error" style="display:block">${escapeHtml(err.message)}</div>`;
    }
  }

  renderDetail = async function(jobId, scroll=true) {
    if (state.framePreviewStop) {
      state.framePreviewStop();
      state.framePreviewStop = null;
    }

    const job = state.jobs.find(j => j.job_id === jobId) || await jsonFetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    if (!job) return;
    const s = job.summary || {};
    const metrics = job.kind === 'run'
      ? metric('Frames', s.frames) + metric('Detections', s.detections) + metric('Tracks', s.tracks) + metric('Anomalies', s.anomalies) + metric('Events', s.events) + metric('Alerts', s.alerts)
      : metric('Sensors', s.sensors) + metric('Input events', s.input_events) + metric('Fused events', s.events) + metric('Alerts', s.alerts);

    let results = null;
    let artifacts = {};
    if (job.status === 'completed') {
      const settled = await Promise.allSettled([
        jsonFetch(`/api/jobs/${encodeURIComponent(jobId)}/results`),
        jsonFetch(`/api/jobs/${encodeURIComponent(jobId)}/artifacts`),
      ]);
      if (settled[0].status === 'fulfilled') results = settled[0].value;
      if (settled[1].status === 'fulfilled') artifacts = settled[1].value.artifacts || {};
    }

    let html = `<div class="summary">${metrics}</div>
      <div class="detail-meta"><span><b>Status:</b> ${escapeHtml(job.status)}</span><span><b>Created:</b> ${escapeHtml(shortTime(job.created_at_utc))}</span>${job.started_at_utc ? `<span><b>Started:</b> ${escapeHtml(shortTime(job.started_at_utc))}</span>` : ''}${job.completed_at_utc ? `<span><b>Finished:</b> ${escapeHtml(shortTime(job.completed_at_utc))}</span>` : ''}</div>`;

    if (job.error) html += `<div class="error" style="display:block"><b>${escapeHtml(job.error.type)}</b>: ${escapeHtml(job.error.message)}</div>`;
    if (job.status === 'queued' || job.status === 'running') html += '<div class="hint">This job is active. The console will refresh automatically.</div>';

    if (job.status === 'completed' || job.status === 'failed') {
      const storage = results?.storage;
      const sourceNote = storage?.input_is_managed
        ? 'Deleting removes this run, its evidence, and the uploaded source copy.'
        : 'Deleting removes this run and its evidence. Read-in-place source imagery is never deleted.';
      html += `<div class="result-actions"><div>${storage ? `<span class="storage-pill" title="Generated output + managed input + job metadata">Run storage: ${formatBytes(storage.total_bytes)}</span>` : ''}<div class="result-note">${escapeHtml(sourceNote)}</div></div><button id="delete-job-button" class="button danger" type="button">Delete run & files</button></div>`;
    }

    let activeTab = null;
    let previewBase = null;
    let previewFps = 30;
    if (job.status === 'completed') {
      if (artifacts.annotated_video) {
        const frameCount = Math.max(1, Number(s.frames || 1));
        previewBase = `/api/jobs/${encodeURIComponent(job.job_id)}/preview-frame`;
        if (artifacts.run_manifest) {
          try {
            const manifest = await jsonFetch(artifacts.run_manifest.url);
            const manifestFps = Number(manifest.render_fps);
            if (Number.isFinite(manifestFps) && manifestFps > 0) previewFps = manifestFps;
          } catch (_) {}
        }
        const durationS = frameCount / previewFps;
        html += `<div class="section-title">Annotated video</div><video id="annotated-video" controls preload="metadata" poster="${escapeHtml(previewBase)}?frame=0" src="${escapeHtml(artifacts.annotated_video.url)}"></video><div id="annotated-video-hint" class="hint">Browser playback depends on the installed codec. If playback is unavailable, use the codec-safe annotated frame browser below or download the MP4.</div><details id="frame-preview-details" class="frame-preview"><summary>Browse annotated frames (codec-safe)</summary><div class="frame-preview-body"><img id="annotated-frame-preview" src="${escapeHtml(previewBase)}?frame=0" alt="Annotated frame preview" /><div class="frame-player-toolbar"><button id="frame-prev-button" class="frame-player-button" type="button">Previous</button><button id="frame-play-button" class="frame-player-button primary" type="button">Play</button><button id="frame-next-button" class="frame-player-button" type="button">Next</button><label class="frame-player-field" for="frame-speed-select">Speed <select id="frame-speed-select"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><label class="frame-player-field"><input id="frame-loop-toggle" type="checkbox" /> Loop</label><span id="frame-preview-time" class="frame-preview-time">${formatTimestamp(0)} / ${formatTimestamp(durationS)}</span></div><div class="frame-preview-controls"><input id="frame-preview-slider" type="range" min="0" max="${frameCount - 1}" value="0" step="1" aria-label="Annotated frame" /><span id="frame-preview-label" class="frame-preview-label">1 / ${frameCount}</span></div><div class="result-note">Play uses Pipeline Sentinel's JPEG frame decoder, so it works even when Chrome cannot play the MP4 codec. Playback follows the run FPS when local decoding can keep up.</div></div></details>`;
      }

      const tabs = job.kind === 'run'
        ? [['overview','Overview'],['detections','Detections'],['tracks','Tracks'],['anomalies','Anomalies'],['events','Events / Alerts'],['downloads','Downloads']]
        : [['overview','Overview'],['events','Events / Alerts'],['downloads','Downloads']];
      const rememberedTab = state.resultTabs[job.job_id] || 'overview';
      activeTab = tabs.some(([id]) => id === rememberedTab) ? rememberedTab : 'overview';
      state.resultTabs[job.job_id] = activeTab;
      const initialPanel = activeTab === 'overview'
        ? (results ? overviewHtml(results) : '<div class="result-empty">Result summary unavailable; raw downloads remain available.</div>')
        : '<div class="result-empty">Loading…</div>';
      html += `<div class="section-title">Interactive results</div><div class="result-tabs">${tabs.map(([id,label]) => `<button class="result-tab ${id === activeTab ? 'active' : ''}" type="button" data-result-tab="${id}">${label}</button>`).join('')}</div><div id="result-panel" class="result-panel">${initialPanel}</div>`;
    }

    $('detail').innerHTML = html;

    const deleteButton = document.getElementById('delete-job-button');
    if (deleteButton) {
      deleteButton.addEventListener('click', async () => {
        const sourceMessage = results?.storage?.input_is_managed
          ? 'The Pipeline Sentinel-managed uploaded source copy will also be removed.'
          : 'Your original read-in-place imagery will NOT be touched.';
        const confirmed = window.confirm(`Delete ${job.display_name} and its generated results?\n\n${sourceMessage}\n\nThis cannot be undone.`);
        if (!confirmed) return;
        if (state.framePreviewStop) {
          state.framePreviewStop();
          state.framePreviewStop = null;
        }
        deleteButton.disabled = true;
        deleteButton.textContent = 'Deleting…';
        try {
          const deleted = await jsonFetch(`/api/jobs/${encodeURIComponent(job.job_id)}`, {method:'DELETE'});
          delete state.resultTabs[job.job_id];
          state.selected = null;
          await refreshJobs({keepDetail:false});
          $('detail').innerHTML = `<div class="detail-empty">Deleted run and freed ${escapeHtml(formatBytes(deleted.freed_bytes))}. Select another job or start a new analysis.</div>`;
        } catch (err) {
          deleteButton.disabled = false;
          deleteButton.textContent = 'Delete run & files';
          window.alert(`Delete failed: ${err.message}`);
        }
      });
    }

    document.querySelectorAll('.result-tab').forEach(button => button.addEventListener('click', async () => {
      state.resultTabs[job.job_id] = button.dataset.resultTab;
      document.querySelectorAll('.result-tab').forEach(item => item.classList.toggle('active', item === button));
      await loadTab(job, button.dataset.resultTab, results, artifacts);
    }));

    if (activeTab && activeTab !== 'overview') await loadTab(job, activeTab, results, artifacts);

    const slider = document.getElementById('frame-preview-slider');
    const previewImage = document.getElementById('annotated-frame-preview');
    const previewLabel = document.getElementById('frame-preview-label');
    const previewTime = document.getElementById('frame-preview-time');
    const playButton = document.getElementById('frame-play-button');
    const previousButton = document.getElementById('frame-prev-button');
    const nextButton = document.getElementById('frame-next-button');
    const speedSelect = document.getElementById('frame-speed-select');
    const loopToggle = document.getElementById('frame-loop-toggle');
    if (slider && previewImage && previewLabel && previewTime && playButton && previousButton && nextButton && speedSelect && loopToggle && previewBase) {
      const total = Number(slider.max || 0) + 1;
      const fps = Number.isFinite(Number(previewFps)) && Number(previewFps) > 0 ? Number(previewFps) : 30;
      let playing = false;
      let disposed = false;
      let playbackTimer = null;
      let scrubTimer = null;
      let requestToken = 0;

      const speed = () => Math.max(0.05, Number(speedSelect.value || 1));
      const frameDelay = () => 1000 / (fps * speed());
      const currentIndex = () => Math.max(0, Math.min(total - 1, Number(slider.value || 0)));
      const clearPlaybackTimer = () => {
        if (playbackTimer) clearTimeout(playbackTimer);
        playbackTimer = null;
      };
      const updateMeta = () => {
        const index = currentIndex();
        previewLabel.textContent = `${index + 1} / ${total}`;
        previewTime.textContent = `${formatTimestamp(index / fps)} / ${formatTimestamp(total / fps)}`;
      };
      const updatePlayButton = () => {
        playButton.textContent = playing ? 'Pause' : 'Play';
        playButton.setAttribute('aria-pressed', playing ? 'true' : 'false');
      };
      const pause = () => {
        playing = false;
        clearPlaybackTimer();
        updatePlayButton();
      };
      const scheduleAdvance = (elapsedMs=0) => {
        if (!playing || disposed) return;
        clearPlaybackTimer();
        playbackTimer = setTimeout(advanceFrame, Math.max(0, frameDelay() - elapsedMs));
      };
      const showFrame = (index, {schedule=false}={}) => {
        const clamped = Math.max(0, Math.min(total - 1, Number(index) || 0));
        slider.value = String(clamped);
        updateMeta();
        const token = ++requestToken;
        const started = performance.now();
        const loaded = () => {
          if (token !== requestToken || disposed || !playing || !schedule) return;
          scheduleAdvance(performance.now() - started);
        };
        previewImage.addEventListener('load', loaded, {once:true});
        previewImage.src = `${previewBase}?frame=${clamped}`;
      };
      function advanceFrame() {
        if (!playing || disposed) return;
        let next = currentIndex() + 1;
        if (next >= total) {
          if (loopToggle.checked) next = 0;
          else {
            pause();
            return;
          }
        }
        showFrame(next, {schedule:true});
      }
      const play = () => {
        if (playing || disposed) return;
        if (currentIndex() >= total - 1 && !loopToggle.checked) showFrame(0);
        playing = true;
        updatePlayButton();
        scheduleAdvance();
      };
      const step = (delta) => {
        pause();
        let next = currentIndex() + delta;
        if (loopToggle.checked) next = (next + total) % total;
        else next = Math.max(0, Math.min(total - 1, next));
        showFrame(next);
      };

      playButton.addEventListener('click', () => playing ? pause() : play());
      previousButton.addEventListener('click', () => step(-1));
      nextButton.addEventListener('click', () => step(1));
      speedSelect.addEventListener('change', () => {
        if (playing) scheduleAdvance();
      });
      slider.addEventListener('input', () => {
        pause();
        updateMeta();
        if (scrubTimer) clearTimeout(scrubTimer);
        scrubTimer = setTimeout(() => showFrame(currentIndex()), 90);
      });
      slider.addEventListener('change', () => {
        if (scrubTimer) clearTimeout(scrubTimer);
        scrubTimer = null;
        pause();
        showFrame(currentIndex());
      });
      previewImage.addEventListener('error', () => pause());
      updateMeta();
      updatePlayButton();

      state.framePreviewStop = () => {
        disposed = true;
        pause();
        if (scrubTimer) clearTimeout(scrubTimer);
        scrubTimer = null;
        requestToken += 1;
      };
    }

    const video = document.getElementById('annotated-video');
    const framePreview = document.getElementById('frame-preview-details');
    const videoHint = document.getElementById('annotated-video-hint');
    if (video && framePreview) {
      video.addEventListener('error', () => {
        video.style.display = 'none';
        framePreview.open = true;
        if (videoHint) videoHint.textContent = 'Chrome could not decode this MP4. The annotated frame player below remains fully usable, and the original artifact is still available in Downloads.';
      });
    }

    if (scroll) $('detail').scrollIntoView({behavior:'smooth', block:'nearest'});
  };
})();
