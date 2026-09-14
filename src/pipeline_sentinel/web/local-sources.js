(() => {
  const style = document.createElement('style');
  style.textContent = `
    .source-tabs { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-bottom:14px; }
    .source-tab { border:1px solid var(--line); background:#0d1319; color:var(--muted); border-radius:7px; padding:8px 6px; cursor:pointer; font-size:11px; }
    .source-tab.active { color:#dcfff8; border-color:var(--accent); background:#18332f; }
    .source-status { border:1px solid var(--line); border-radius:8px; padding:9px 10px; margin:10px 0 14px; color:var(--muted); font-size:11px; line-height:1.5; display:none; }
    .source-status.ok { display:block; border-color:rgba(104,211,145,.35); color:#b9efca; background:rgba(104,211,145,.06); }
    .source-status.bad { display:block; border-color:rgba(255,107,107,.35); color:#ffc8c8; background:rgba(255,107,107,.06); }
    .source-actions { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  `;
  document.head.appendChild(style);

  const leftStack = document.querySelector('.grid > .stack');
  if (!leftStack) return;
  const firstCard = leftStack.querySelector('.card');
  if (!firstCard) return;

  const card = document.createElement('section');
  card.className = 'card';
  card.innerHTML = `
    <div class="card-head"><div><div class="eyebrow">Read in place</div><h2>Local Image Sequence</h2></div></div>
    <div class="card-body">
      <div class="source-tabs">
        <button type="button" class="source-tab active" data-source="image_folder">Image folder</button>
        <button type="button" class="source-tab" data-source="visdrone">VisDrone</button>
        <button type="button" class="source-tab" data-source="uavdt">UAVDT</button>
      </div>
      <div class="field">
        <label for="local-root">Local source path</label>
        <input id="local-root" type="text" placeholder="D:\\FMV\\VisDrone\\VisDrone2019-VID-val" />
      </div>
      <div class="source-actions">
        <button id="local-inspect" type="button" class="button secondary">Inspect source</button>
        <select id="local-sequence" disabled><option value="">Inspect source first</option></select>
      </div>
      <div id="local-source-status" class="source-status"></div>
      <div class="field-row">
        <div class="field"><label for="local-modality">Modality</label><select id="local-modality"><option value="EO">EO</option><option value="IR">IR</option><option value="OTHER">Other</option></select></div>
        <div class="field"><label for="local-sensor-id">Sensor ID</label><input id="local-sensor-id" type="text" value="EO_CAM_01" /></div>
      </div>
      <div class="field-row">
        <div class="field"><label for="local-fps">Working FPS</label><input id="local-fps" type="text" value="30" /></div>
        <div class="field"><label for="local-step">Frame step</label><input id="local-step" type="text" value="1" /></div>
      </div>
      <div class="field"><label for="local-max">Max frames (blank = all)</label><input id="local-max" type="text" placeholder="e.g. 300" /></div>
      <button id="local-start" type="button" class="button full" disabled>Start image-sequence analysis</button>
      <div id="local-error" class="error"></div>
      <div class="hint">Source JPEGs stay where they are. Pipeline Sentinel reads one frame at a time and writes only run evidence and the annotated output video. Browser folder uploads are deliberately avoided.</div>
    </div>`;
  firstCard.insertAdjacentElement('afterend', card);

  const byId = (id) => document.getElementById(id);
  let sourceType = 'image_folder';
  let inspectedPath = null;

  function setStatus(message, ok=true) {
    const box = byId('local-source-status');
    box.textContent = message;
    box.className = `source-status ${ok ? 'ok' : 'bad'}`;
  }

  function resetInspection() {
    inspectedPath = null;
    const select = byId('local-sequence');
    select.innerHTML = '<option value="">Inspect source first</option>';
    select.disabled = true;
    byId('local-start').disabled = true;
    byId('local-source-status').className = 'source-status';
  }

  card.querySelectorAll('.source-tab').forEach(button => {
    button.addEventListener('click', () => {
      sourceType = button.dataset.source;
      card.querySelectorAll('.source-tab').forEach(item => item.classList.toggle('active', item === button));
      resetInspection();
      if (sourceType === 'visdrone') byId('local-root').placeholder = 'D:\\FMV\\VisDrone\\VisDrone2019-VID-val';
      else if (sourceType === 'uavdt') byId('local-root').placeholder = 'D:\\FMV\\UAVDT';
      else byId('local-root').placeholder = 'D:\\imagery\\my_sequence';
    });
  });

  byId('local-root').addEventListener('input', resetInspection);
  byId('local-modality').addEventListener('change', () => {
    const mod = byId('local-modality').value;
    const current = byId('local-sensor-id').value;
    if (/^(EO|IR|OTHER)_CAM_01$/.test(current)) byId('local-sensor-id').value = `${mod}_CAM_01`;
  });

  byId('local-inspect').addEventListener('click', async () => {
    const path = byId('local-root').value.trim();
    const error = byId('local-error');
    error.style.display = 'none';
    if (!path) {
      error.textContent = 'Enter the local dataset or image-folder path.';
      error.style.display = 'block';
      return;
    }
    byId('local-inspect').disabled = true;
    byId('local-start').disabled = true;
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
      byId('local-start').disabled = !sequences.length;
      const suffix = info.sequence_list_truncated ? ' (list truncated)' : '';
      setStatus(`${Number(info.sequence_count || sequences.length).toLocaleString()} sequence(s) found${suffix}. Imagery will be read in place; no source files will be copied.`);
    } catch (err) {
      setStatus(err.message, false);
    } finally {
      byId('local-inspect').disabled = false;
    }
  });

  byId('local-start').addEventListener('click', async () => {
    const error = byId('local-error');
    error.style.display = 'none';
    const path = byId('local-root').value.trim();
    if (!inspectedPath || path !== inspectedPath) {
      error.textContent = 'Inspect the source again after changing its path.';
      error.style.display = 'block';
      return;
    }
    const fps = Number(byId('local-fps').value);
    const frameStep = Number(byId('local-step').value);
    const maxText = byId('local-max').value.trim();
    const maxFrames = maxText ? Number(maxText) : null;
    const payload = {
      source_type: sourceType,
      path,
      sequence_id: byId('local-sequence').value || null,
      sensor_id: byId('local-sensor-id').value.trim(),
      modality: byId('local-modality').value,
      fps,
      frame_step: frameStep,
      max_frames: maxFrames,
    };
    byId('local-start').disabled = true;
    try {
      const job = await jsonFetch('/api/jobs/local-sequence', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(payload),
      });
      await refreshJobs({keepDetail:false});
      await selectJob(job.job_id);
    } catch (err) {
      error.textContent = err.message;
      error.style.display = 'block';
    } finally {
      byId('local-start').disabled = false;
    }
  });
})();
