(() => {
  const style = document.createElement('style');
  style.textContent = `
    .source-tabs { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:14px; }
    .source-tab { border:1px solid var(--line); background:#0d1319; color:var(--muted); border-radius:7px; padding:9px 8px; cursor:pointer; font-size:11px; }
    .source-tab.active { color:#dcfff8; border-color:var(--accent); background:#18332f; }
    .source-pane { display:none; }
    .source-pane.active { display:block; }
    .source-status { border:1px solid var(--line); border-radius:8px; padding:9px 10px; margin:10px 0 14px; color:var(--muted); font-size:11px; line-height:1.5; display:none; }
    .source-status.ok { display:block; border-color:rgba(104,211,145,.35); color:#b9efca; background:rgba(104,211,145,.06); }
    .source-status.bad { display:block; border-color:rgba(255,107,107,.35); color:#ffc8c8; background:rgba(255,107,107,.06); }
    .source-actions { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .source-help { color:var(--muted); font-size:11px; line-height:1.5; margin-top:8px; }
  `;
  document.head.appendChild(style);

  const leftStack = document.querySelector('.grid > .stack');
  if (!leftStack) return;
  const firstCard = leftStack.querySelector('.card');
  if (!firstCard) return;

  firstCard.innerHTML = `
    <div class="card-head"><div><div class="eyebrow">Sensor ingest</div><h2>Start Analysis Run</h2></div></div>
    <div class="card-body">
      <div class="source-tabs">
        <button id="source-files-tab" type="button" class="source-tab active">Media files</button>
        <button id="source-local-tab" type="button" class="source-tab">Local folder / dataset</button>
      </div>

      <div id="source-files-pane" class="source-pane active">
        <div class="field">
          <label for="source-files">Video or image frames</label>
          <input id="source-files" type="file" multiple accept=".mp4,.mov,.avi,.mkv,.m4v,.jpg,.jpeg,.png,.bmp,.tif,.tiff" />
        </div>
        <div id="source-file-status" class="source-status"></div>
        <div class="source-help">Choose one encoded video, or choose one or more still images from the same sequence. Still images are ordered naturally by filename. For very large frame collections, use the local-folder mode so source imagery is read in place instead of copied into the operator workspace.</div>
      </div>

      <div id="source-local-pane" class="source-pane">
        <div class="field">
          <label for="local-source-type">Source type</label>
          <select id="local-source-type">
            <option value="image_folder">Image folder</option>
            <option value="visdrone">VisDrone2019-VID</option>
            <option value="uavdt">UAVDT</option>
          </select>
        </div>
        <div class="field">
          <label for="local-root">Local source path</label>
          <input id="local-root" type="text" placeholder="D:\\imagery\\my_sequence" />
        </div>
        <div class="source-actions">
          <button id="local-inspect" type="button" class="button secondary">Inspect source</button>
          <select id="local-sequence" disabled><option value="">Inspect source first</option></select>
        </div>
        <div id="local-source-status" class="source-status"></div>
        <div class="field-row">
          <div class="field"><label for="local-step">Frame step</label><input id="local-step" type="text" value="1" /></div>
          <div class="field"><label for="local-max">Max frames</label><input id="local-max" type="text" placeholder="blank = all" /></div>
        </div>
        <div class="source-help">Local folders and supported datasets are read in place. Pipeline Sentinel decodes one frame at a time and writes only run evidence plus annotated output.</div>
      </div>

      <div class="field-row" style="margin-top:14px">
        <div class="field"><label for="source-modality">Modality</label><select id="source-modality"><option value="EO">EO</option><option value="IR">IR</option><option value="OTHER">Other</option></select></div>
        <div class="field"><label for="source-sensor-id">Sensor ID</label><input id="source-sensor-id" type="text" value="EO_CAM_01" /></div>
      </div>
      <div class="field"><label for="source-fps">Working FPS</label><input id="source-fps" type="text" value="30" /></div>
      <button id="source-start" type="button" class="button full">Start analysis</button>
      <div id="source-progress" class="progress"><span></span></div>
      <div id="source-error" class="error"></div>
      <div class="hint">EO/IR describes the sensor, not the file format. The same analysis pipeline accepts supported video containers or ordered still-image frames.</div>
    </div>`;

  const byId = (id) => document.getElementById(id);
  let mode = 'files';
  let inspectedPath = null;

  function setStatus(id, message, ok=true) {
    const box = byId(id);
    box.textContent = message;
    box.className = `source-status ${ok ? 'ok' : 'bad'}`;
  }

  function clearStatus(id) {
    const box = byId(id);
    box.textContent = '';
    box.className = 'source-status';
  }

  function setMode(nextMode) {
    mode = nextMode;
    byId('source-files-tab').classList.toggle('active', mode === 'files');
    byId('source-local-tab').classList.toggle('active', mode === 'local');
    byId('source-files-pane').classList.toggle('active', mode === 'files');
    byId('source-local-pane').classList.toggle('active', mode === 'local');
    byId('source-error').style.display = 'none';
  }

  function resetInspection() {
    inspectedPath = null;
    const select = byId('local-sequence');
    select.innerHTML = '<option value="">Inspect source first</option>';
    select.disabled = true;
    clearStatus('local-source-status');
  }

  function updateLocalPlaceholder() {
    const sourceType = byId('local-source-type').value;
    if (sourceType === 'visdrone') byId('local-root').placeholder = 'D:\\FMV\\VisDrone\\VisDrone2019-VID-val';
    else if (sourceType === 'uavdt') byId('local-root').placeholder = 'D:\\FMV\\UAVDT';
    else byId('local-root').placeholder = 'D:\\imagery\\my_sequence';
  }

  byId('source-files-tab').addEventListener('click', () => setMode('files'));
  byId('source-local-tab').addEventListener('click', () => setMode('local'));

  byId('source-files').addEventListener('change', () => {
    const files = [...byId('source-files').files];
    if (!files.length) {
      clearStatus('source-file-status');
      return;
    }
    const names = files.slice(0, 3).map(file => file.name).join(', ');
    const suffix = files.length > 3 ? ` + ${files.length - 3} more` : '';
    setStatus('source-file-status', `${files.length.toLocaleString()} file(s) selected: ${names}${suffix}`);
  });

  byId('source-modality').addEventListener('change', () => {
    const mod = byId('source-modality').value;
    const current = byId('source-sensor-id').value;
    if (/^(EO|IR|OTHER)_CAM_01$/.test(current)) byId('source-sensor-id').value = `${mod}_CAM_01`;
  });

  byId('local-source-type').addEventListener('change', () => {
    updateLocalPlaceholder();
    resetInspection();
  });
  byId('local-root').addEventListener('input', resetInspection);

  byId('local-inspect').addEventListener('click', async () => {
    const sourceType = byId('local-source-type').value;
    const path = byId('local-root').value.trim();
    const error = byId('source-error');
    error.style.display = 'none';
    if (!path) {
      error.textContent = 'Enter the local image-folder or dataset path.';
      error.style.display = 'block';
      return;
    }
    byId('local-inspect').disabled = true;
    try {
      const info = await jsonFetch('/api/local-sources/inspect', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({source_type:sourceType, path})
      });
      inspectedPath = path;
      const select = byId('local-sequence');
      const sequences = info.sequences || [];
      select.innerHTML = sequences.map(item => {
        const count = item.frame_count != null ? ` · ${Number(item.frame_count).toLocaleString()} frames` : '';
        return `<option value="${escapeHtml(item.sequence_id)}">${escapeHtml(item.sequence_id)}${count}</option>`;
      }).join('');
      select.disabled = sequences.length <= 1;
      const suffix = info.sequence_list_truncated ? ' (list truncated)' : '';
      setStatus('local-source-status', `${Number(info.sequence_count || sequences.length).toLocaleString()} sequence(s) found${suffix}. Source imagery will be read in place.`);
    } catch (err) {
      setStatus('local-source-status', err.message, false);
    } finally {
      byId('local-inspect').disabled = false;
    }
  });

  function numericInputs() {
    const fps = Number(byId('source-fps').value);
    const frameStep = Number(byId('local-step').value);
    const maxText = byId('local-max').value.trim();
    const maxFrames = maxText ? Number(maxText) : null;
    if (!Number.isFinite(fps) || fps <= 0) throw new Error('Working FPS must be a positive number.');
    if (!Number.isInteger(frameStep) || frameStep <= 0) throw new Error('Frame step must be a positive integer.');
    if (maxFrames != null && (!Number.isInteger(maxFrames) || maxFrames <= 0)) throw new Error('Max frames must be a positive integer or blank.');
    return {fps, frameStep, maxFrames};
  }

  function setBusy(busy) {
    byId('source-start').disabled = busy;
    byId('local-inspect').disabled = busy;
  }

  async function startLocal() {
    const path = byId('local-root').value.trim();
    if (!inspectedPath || path !== inspectedPath) throw new Error('Inspect the local source after entering or changing its path.');
    const {fps, frameStep, maxFrames} = numericInputs();
    const payload = {
      source_type: byId('local-source-type').value,
      path,
      sequence_id: byId('local-sequence').value || null,
      sensor_id: byId('source-sensor-id').value.trim(),
      modality: byId('source-modality').value,
      fps,
      frame_step: frameStep,
      max_frames: maxFrames,
    };
    return jsonFetch('/api/jobs/local-sequence', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload),
    });
  }

  function startUploadedMedia() {
    const selected = [...byId('source-files').files];
    if (!selected.length) return Promise.reject(new Error('Choose one video or one or more image files.'));
    const {fps} = numericInputs();
    const form = new FormData();
    selected.forEach(file => form.append('files', file));
    form.append('sensor_id', byId('source-sensor-id').value.trim());
    form.append('modality', byId('source-modality').value);
    form.append('fps', String(fps));

    const progress = byId('source-progress');
    const bar = progress.querySelector('span');
    progress.style.display = 'block';
    bar.style.width = '0%';

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/jobs/media');
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) bar.style.width = `${Math.min(100, (event.loaded / event.total) * 100)}%`;
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          bar.style.width = '100%';
          try { resolve(JSON.parse(xhr.responseText)); }
          catch (_) { reject(new Error('The service returned an invalid response.')); }
        } else {
          let message = `${xhr.status} ${xhr.statusText}`;
          try { message = JSON.parse(xhr.responseText).detail || message; } catch (_) {}
          reject(new Error(message));
        }
      };
      xhr.onerror = () => reject(new Error('Media upload failed.'));
      xhr.send(form);
    }).finally(() => {
      setTimeout(() => { progress.style.display = 'none'; }, 500);
    });
  }

  byId('source-start').addEventListener('click', async () => {
    const error = byId('source-error');
    error.style.display = 'none';
    if (!byId('source-sensor-id').value.trim()) {
      error.textContent = 'Sensor ID must not be blank.';
      error.style.display = 'block';
      return;
    }
    setBusy(true);
    try {
      const job = mode === 'files' ? await startUploadedMedia() : await startLocal();
      if (mode === 'files') byId('source-files').value = '';
      clearStatus('source-file-status');
      await refreshJobs({keepDetail:false});
      await selectJob(job.job_id);
    } catch (err) {
      error.textContent = err.message;
      error.style.display = 'block';
    } finally {
      setBusy(false);
    }
  });

  updateLocalPlaceholder();
})();
