(function () {
  'use strict';

  const TARGET_SAMPLE_RATE = 22050;
  const MAX_RECORD_SEC = 30;
  const MAX_ANALYZE_SEC = 60;
  const LABELS = { chest: '地声', falsetto: '裏声', unvoiced: '— (無声)' };

  const $ = (id) => document.getElementById(id);
  const els = {
    inputCard: $('inputCard'),
    recordBtn: $('recordBtn'),
    recordLabel: $('recordLabel'),
    fileBtn: $('fileBtn'),
    fileInput: $('fileInput'),
    recStatus: $('recStatus'),
    meterFill: $('meterFill'),
    recTime: $('recTime'),
    message: $('message'),
    resultCard: $('resultCard'),
    verdictEn: $('verdictEn'),
    verdictLabel: $('verdictLabel'),
    ratioChest: $('ratioChest'),
    ratioFalsetto: $('ratioFalsetto'),
    pctChest: $('pctChest'),
    pctFalsetto: $('pctFalsetto'),
    verdictNote: $('verdictNote'),
    featF0: $('featF0'),
    featH1H2: $('featH1H2'),
    featH1H2c: $('featH1H2c'),
    featHrf: $('featHrf'),
    featSlope: $('featSlope'),
    specCard: $('specCard'),
    playBtn: $('playBtn'),
    freqButtons: document.querySelectorAll('.seg-btn[data-freq]'),
    pitchToggle: $('pitchToggle'),
  };

  const view = new SpectrogramView({
    wrap: $('specWrap'),
    canvas: $('specCanvas'),
    overlay: $('specOverlay'),
    tooltip: $('tooltip'),
    labels: LABELS,
  });

  let audioCtx = null;
  let recorder = null;
  let meterRaf = 0;
  let playbackBuffer = null;
  let playing = null;
  let busy = false;

  function getAudioContext() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    return audioCtx;
  }

  function showMessage(text, kind) {
    els.message.textContent = text || '';
    els.message.className = `message ${kind || ''}`;
    els.message.hidden = !text;
  }

  function setBusy(value) {
    busy = value;
    els.recordBtn.disabled = value;
    els.fileBtn.classList.toggle('disabled', value);
    els.fileInput.disabled = value;
  }

  const nextPaint = () => new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));

  // ------------------------------------------------------------ analysis

  async function toMonoSamples(buffer) {
    const seconds = Math.min(buffer.duration, MAX_ANALYZE_SEC);
    const length = Math.max(1, Math.ceil(seconds * TARGET_SAMPLE_RATE));
    const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const offline = new Offline(1, length, TARGET_SAMPLE_RATE);
    const source = offline.createBufferSource();
    source.buffer = buffer;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0);
  }

  async function analyzeBuffer(buffer) {
    stopPlayback();
    setBusy(true);
    showMessage('解析中…');
    await nextPaint();
    try {
      const samples = await toMonoSamples(buffer);
      const result = VoiceAnalyzer.analyze(samples, TARGET_SAMPLE_RATE);
      playbackBuffer = buffer;
      showResult(result);
      showMessage(buffer.duration > MAX_ANALYZE_SEC ? `先頭の ${MAX_ANALYZE_SEC} 秒を解析しました。` : '');
    } catch (err) {
      console.error(err);
      showMessage(`解析に失敗しました（${err.message}）`, 'error');
    } finally {
      setBusy(false);
    }
  }

  function leanTag(leaning) {
    if (!leaning) return '';
    return `<span class="lean ${leaning}">#${LABELS[leaning]}寄り</span>`;
  }

  function formatFeature(value, unit, leaning) {
    if (value === null || !Number.isFinite(value)) return '–';
    return `${value.toFixed(1)} ${unit}${leanTag(leaning)}`;
  }

  function showResult(result) {
    const s = result.summary;
    els.resultCard.hidden = false;
    els.specCard.hidden = false;

    if (!s.label) {
      els.verdictEn.textContent = 'NO RESULT';
      els.verdictLabel.textContent = '判定できません';
      els.verdictLabel.className = 'verdict-label unknown';
      els.ratioChest.style.width = '0%';
      els.ratioFalsetto.style.width = '0%';
      els.pctChest.textContent = '–';
      els.pctFalsetto.textContent = '–';
      els.verdictNote.textContent = '声（音程のある音）を十分に検出できませんでした。マイクに近づいて、はっきり発声してください。';
      for (const el of [els.featF0, els.featH1H2, els.featHrf, els.featSlope]) el.textContent = '–';
      els.featH1H2c.hidden = true;
    } else {
      const falsettoPct = Math.round(s.falsettoRatio * 100);
      els.verdictEn.textContent = s.label === 'falsetto' ? 'FALSETTO' : 'CHEST VOICE';
      els.verdictLabel.textContent = LABELS[s.label];
      els.verdictLabel.className = `verdict-label ${s.label}`;
      els.ratioChest.style.width = `${100 - falsettoPct}%`;
      els.ratioFalsetto.style.width = `${falsettoPct}%`;
      els.pctChest.textContent = `${100 - falsettoPct}%`;
      els.pctFalsetto.textContent = `${falsettoPct}%`;
      const majorityPct = s.label === 'falsetto' ? falsettoPct : 100 - falsettoPct;
      els.verdictNote.textContent = `声を検出した ${s.voicedSeconds.toFixed(1)} 秒のうち ${majorityPct}% が${LABELS[s.label]}と判定されました。`;
      els.featF0.textContent = `${s.medianF0.toFixed(0)} Hz（${s.note}）`;
      els.featH1H2.innerHTML = formatFeature(s.features.h1h2, 'dB', s.leaning.h1h2);
      const showCorrected = s.formantCorrected && Number.isFinite(s.features.h1h2c);
      els.featH1H2c.textContent = showCorrected ? `フォルマント補正後 ${s.features.h1h2c.toFixed(1)} dB` : '';
      els.featH1H2c.hidden = !showCorrected;
      els.featHrf.innerHTML = formatFeature(s.features.hrf, 'dB', s.leaning.hrf);
      els.featSlope.innerHTML = formatFeature(s.features.slope, 'dB/oct', s.leaning.slope);
    }

    view.setResult(result);
    els.resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ------------------------------------------------------------ recording

  function micErrorMessage(err) {
    if (err && (err.name === 'NotAllowedError' || err.name === 'SecurityError')) {
      return 'マイクの使用が許可されていません。ブラウザの設定でマイクを許可してください。';
    }
    if (err && err.name === 'NotFoundError') return 'マイクが見つかりません。';
    return `マイクを開始できませんでした（${err && err.message}）`;
  }

  async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showMessage('このブラウザ・開き方ではマイクを使えません。localhost か https で開いてください。', 'error');
      return;
    }
    stopPlayback();
    showMessage('');
    const rec = new MicRecorder(getAudioContext());
    recorder = rec;
    els.recordBtn.disabled = true;
    try {
      await rec.start();
    } catch (err) {
      rec.release();
      recorder = null;
      els.recordBtn.disabled = false;
      showMessage(micErrorMessage(err), 'error');
      return;
    }
    els.recordBtn.disabled = false;
    els.recordBtn.classList.add('recording');
    els.recordLabel.textContent = '停止して判定';
    els.fileBtn.classList.add('disabled');
    els.fileInput.disabled = true;
    els.recStatus.hidden = false;

    const tick = () => {
      if (recorder !== rec) return;
      const db = rec.level();
      els.meterFill.style.width = `${Math.max(0, Math.min(100, ((db + 60) / 60) * 100))}%`;
      els.recTime.textContent = `${rec.elapsed.toFixed(1)} 秒 / 最長 ${MAX_RECORD_SEC} 秒`;
      if (rec.elapsed >= MAX_RECORD_SEC) finishRecording();
      else meterRaf = requestAnimationFrame(tick);
    };
    tick();
  }

  function finishRecording() {
    cancelAnimationFrame(meterRaf);
    const buffer = recorder.stop();
    recorder = null;
    els.recordBtn.classList.remove('recording');
    els.recordLabel.textContent = '録音';
    els.fileBtn.classList.remove('disabled');
    els.fileInput.disabled = false;
    els.recStatus.hidden = true;
    els.meterFill.style.width = '0%';
    if (!buffer || buffer.duration < 0.3) {
      showMessage('録音が短すぎます。もう一度お試しください。', 'error');
      return;
    }
    analyzeBuffer(buffer);
  }

  els.recordBtn.addEventListener('click', () => {
    if (busy) return;
    if (recorder) finishRecording();
    else startRecording();
  });

  // ------------------------------------------------------------ files

  async function loadFile(file) {
    if (busy || recorder) return;
    showMessage(`「${file.name}」を読み込み中…`);
    try {
      const data = await file.arrayBuffer();
      const buffer = await getAudioContext().decodeAudioData(data);
      await analyzeBuffer(buffer);
    } catch (err) {
      console.error(err);
      showMessage(`「${file.name}」は音声ファイルとして読み込めませんでした。`, 'error');
    }
  }

  els.fileInput.addEventListener('change', () => {
    const file = els.fileInput.files[0];
    els.fileInput.value = '';
    if (file) loadFile(file);
  });

  let dragDepth = 0;
  document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragDepth++;
    els.inputCard.classList.add('dragging');
  });
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) els.inputCard.classList.remove('dragging');
  });
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDepth = 0;
    els.inputCard.classList.remove('dragging');
    const file = e.dataTransfer && e.dataTransfer.files[0];
    if (file) loadFile(file);
  });

  // ------------------------------------------------------------ playback

  function startPlayback() {
    if (!playbackBuffer || !view.result) return;
    const ctx = getAudioContext();
    ctx.resume();
    const source = ctx.createBufferSource();
    source.buffer = playbackBuffer;
    source.connect(ctx.destination);
    const duration = view.result.duration;
    const startedAt = ctx.currentTime;
    source.start(0, 0, duration);
    source.onended = () => {
      if (playing && playing.source === source) stopPlayback();
    };
    playing = { source, raf: 0 };
    els.playBtn.textContent = '■ 停止';
    const tick = () => {
      if (!playing || playing.source !== source) return;
      view.setPlayhead(Math.min(duration, ctx.currentTime - startedAt));
      playing.raf = requestAnimationFrame(tick);
    };
    tick();
  }

  function stopPlayback() {
    if (!playing) return;
    const { source, raf } = playing;
    playing = null;
    source.onended = null;
    try {
      source.stop();
    } catch (_) {
      /* already stopped */
    }
    cancelAnimationFrame(raf);
    view.setPlayhead(null);
    els.playBtn.textContent = '▶ 再生';
  }

  els.playBtn.addEventListener('click', () => (playing ? stopPlayback() : startPlayback()));

  // ------------------------------------------------------------ spectrogram controls

  for (const btn of els.freqButtons) {
    btn.addEventListener('click', () => {
      for (const b of els.freqButtons) b.setAttribute('aria-pressed', String(b === btn));
      view.setMaxFreq(Number(btn.dataset.freq));
    });
  }

  els.pitchToggle.addEventListener('click', () => {
    const show = els.pitchToggle.getAttribute('aria-pressed') !== 'true';
    els.pitchToggle.setAttribute('aria-pressed', String(show));
    view.setShowPitch(show);
  });

  // Web フォントの読み込み後に軸ラベルを描き直す
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => view.render());
})();
