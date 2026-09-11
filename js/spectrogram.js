/*
 * SpectrogramView — スペクトログラムの描画とホバー時の周波数・時間表示
 */
(function (root) {
  'use strict';

  const DYNAMIC_RANGE_DB = 75;
  const MARGIN = { left: 58, right: 14, top: 12, bottom: 44 };
  const STRIP = { gap: 8, height: 8 };

  // inferno カラーマップ（制御点を線形補間）
  const INFERNO = [
    [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103], [188, 55, 84],
    [221, 81, 58], [243, 120, 25], [252, 165, 10], [246, 215, 70], [252, 255, 164],
  ];
  const LUT = (() => {
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const pos = (i / 255) * (INFERNO.length - 1);
      const k = Math.min(INFERNO.length - 2, Math.floor(pos));
      const t = pos - k;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = INFERNO[k][c] + (INFERNO[k + 1][c] - INFERNO[k][c]) * t;
    }
    return lut;
  })();

  const TIME_STEPS = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 30];
  const FREQ_STEPS = [100, 200, 250, 500, 1000, 2000, 2500, 5000];

  function niceStep(steps, range, pixels, minPx) {
    for (const s of steps) if ((s / range) * pixels >= minPx) return s;
    return steps[steps.length - 1];
  }

  function formatFreqTick(f) {
    return f >= 1000 ? `${+(f / 1000).toFixed(2)} kHz` : `${f} Hz`;
  }

  class SpectrogramView {
    constructor({ wrap, canvas, overlay, tooltip, labels }) {
      this.wrap = wrap;
      this.canvas = canvas;
      this.overlay = overlay;
      this.tooltip = tooltip;
      this.labels = labels; // { chest, falsetto, unvoiced }
      this.result = null;
      this.maxFreq = 5000;
      this.showPitch = false;
      this.hover = null;
      this.playhead = null;
      this.image = null;

      new ResizeObserver(() => this.render()).observe(wrap);
      const media = root.matchMedia && root.matchMedia('(prefers-color-scheme: dark)');
      if (media && media.addEventListener) media.addEventListener('change', () => this.render());

      overlay.addEventListener('pointermove', (e) => this.onPointer(e));
      overlay.addEventListener('pointerdown', (e) => this.onPointer(e));
      overlay.addEventListener('pointerleave', () => this.clearHover());
      overlay.addEventListener('pointercancel', () => this.clearHover());
    }

    setResult(result) {
      this.result = result;
      this.hover = null;
      this.playhead = null;
      this.buildImage();
      this.render();
    }

    setMaxFreq(freq) {
      this.maxFreq = freq;
      this.buildImage();
      this.render();
    }

    setShowPitch(show) {
      this.showPitch = show;
      this.render();
    }

    setPlayhead(time) {
      this.playhead = time;
      this.drawOverlay();
    }

    // ------------------------------------------------------------ geometry

    get plot() {
      const w = this.wrap.clientWidth;
      const h = this.wrap.clientHeight;
      return {
        x: MARGIN.left,
        y: MARGIN.top,
        w: Math.max(10, w - MARGIN.left - MARGIN.right),
        h: Math.max(10, h - MARGIN.top - MARGIN.bottom),
      };
    }

    timeToX(t) {
      const p = this.plot;
      return p.x + (t / this.result.duration) * p.w;
    }

    freqToY(f) {
      const p = this.plot;
      return p.y + (1 - f / this.maxFreq) * p.h;
    }

    // ------------------------------------------------------------ drawing

    buildImage() {
      const r = this.result;
      if (!r) return;
      const rows = Math.min(r.nBins, Math.floor(this.maxFreq / r.binHz) + 2);
      const cols = r.nFrames;
      const off = document.createElement('canvas');
      off.width = cols;
      off.height = rows;
      const ctx = off.getContext('2d');
      const img = ctx.createImageData(cols, rows);
      const refDb = 10 * Math.log10(r.maxPower);
      const scale = 255 / DYNAMIC_RANGE_DB;
      for (let i = 0; i < cols; i++) {
        const base = i * r.nBins;
        for (let b = 0; b < rows; b++) {
          const db = 10 * Math.log10(r.power[base + b] + 1e-20) - refDb;
          const v = Math.max(0, Math.min(255, Math.round((db + DYNAMIC_RANGE_DB) * scale)));
          const px = ((rows - 1 - b) * cols + i) * 4;
          img.data[px] = LUT[v * 3];
          img.data[px + 1] = LUT[v * 3 + 1];
          img.data[px + 2] = LUT[v * 3 + 2];
          img.data[px + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      this.image = { canvas: off, rows, cols };
    }

    render() {
      const dpr = root.devicePixelRatio || 1;
      const w = this.wrap.clientWidth;
      const h = this.wrap.clientHeight;
      for (const c of [this.canvas, this.overlay]) {
        c.width = Math.round(w * dpr);
        c.height = Math.round(h * dpr);
      }
      const ctx = this.canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (!this.result || !this.image) return;

      const style = getComputedStyle(this.wrap);
      const textColor = style.getPropertyValue('--text-muted').trim() || '#666';
      const gridColor = style.getPropertyValue('--grid').trim() || 'rgba(255,255,255,0.15)';
      const chestColor = style.getPropertyValue('--chest').trim() || '#e8590c';
      const falsettoColor = style.getPropertyValue('--falsetto').trim() || '#4c6ef5';
      const r = this.result;
      const p = this.plot;
      const hopSec = r.hopSize / r.sampleRate;

      // スペクトログラム本体
      ctx.save();
      ctx.beginPath();
      ctx.rect(p.x, p.y, p.w, p.h);
      ctx.clip();
      ctx.fillStyle = '#000004';
      ctx.fillRect(p.x, p.y, p.w, p.h);
      ctx.imageSmoothingEnabled = true;
      const { canvas: img, rows, cols } = this.image;
      const dx = this.timeToX(-0.5 * hopSec);
      const dw = this.timeToX((cols - 0.5) * hopSec) - dx;
      const dy = this.freqToY((rows - 0.5) * r.binHz);
      const dh = this.freqToY(-0.5 * r.binHz) - dy;
      ctx.drawImage(img, 0, 0, cols, rows, dx, dy, dw, dh);

      // グリッド
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 1;
      const fStep = niceStep(FREQ_STEPS, this.maxFreq, p.h, 36);
      for (let f = fStep; f < this.maxFreq; f += fStep) {
        const y = Math.round(this.freqToY(f)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(p.x, y);
        ctx.lineTo(p.x + p.w, y);
        ctx.stroke();
      }

      // ピッチ線
      if (this.showPitch) {
        ctx.strokeStyle = 'rgba(94, 255, 196, 0.95)';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        let drawing = false;
        for (let i = 0; i < r.nFrames; i++) {
          if (!r.voiced[i]) {
            drawing = false;
            continue;
          }
          const x = this.timeToX(i * hopSec);
          const y = this.freqToY(r.f0[i]);
          if (drawing) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
          drawing = true;
        }
        ctx.stroke();
      }
      ctx.restore();

      // 判定ストリップ（時間ごとの地声／裏声）
      const sy = p.y + p.h + STRIP.gap;
      ctx.fillStyle = gridColor;
      ctx.fillRect(p.x, sy, p.w, STRIP.height);
      for (let i = 0; i < r.nFrames; i++) {
        if (!r.labels[i]) continue;
        let j = i;
        while (j + 1 < r.nFrames && r.labels[j + 1] === r.labels[i]) j++;
        ctx.fillStyle = r.labels[i] === 2 ? falsettoColor : chestColor;
        const x0 = this.timeToX((i - 0.5) * hopSec);
        const x1 = this.timeToX((j + 0.5) * hopSec);
        ctx.fillRect(Math.max(p.x, x0), sy, Math.min(p.x + p.w, x1) - Math.max(p.x, x0), STRIP.height);
        i = j;
      }

      // 軸ラベル
      ctx.fillStyle = textColor;
      ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let f = fStep; f <= this.maxFreq + 1e-6; f += fStep) {
        ctx.fillText(formatFreqTick(f), p.x - 8, this.freqToY(f));
      }
      ctx.fillText('判定', p.x - 8, sy + STRIP.height / 2);

      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const tStep = niceStep(TIME_STEPS, r.duration, p.w, 64);
      const ty = sy + STRIP.height + 6;
      for (let t = 0; t <= r.duration + 1e-9; t += tStep) {
        const x = this.timeToX(t);
        ctx.fillRect(Math.round(x), sy + STRIP.height, 1, 3);
        ctx.fillText(`${+t.toFixed(2)} s`, Math.min(x, p.x + p.w - 14), ty);
      }

      this.drawOverlay();
    }

    drawOverlay() {
      const dpr = root.devicePixelRatio || 1;
      const ctx = this.overlay.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, this.overlay.width / dpr, this.overlay.height / dpr);
      if (!this.result) return;
      const p = this.plot;
      const bottom = p.y + p.h + STRIP.gap + STRIP.height;

      if (this.playhead !== null) {
        const x = Math.round(this.timeToX(this.playhead)) + 0.5;
        ctx.strokeStyle = 'rgba(255,255,255,0.95)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, p.y);
        ctx.lineTo(x, bottom);
        ctx.stroke();
      }

      if (this.hover) {
        const x = Math.round(this.timeToX(this.hover.time)) + 0.5;
        ctx.strokeStyle = 'rgba(255,255,255,0.7)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(x, p.y);
        ctx.lineTo(x, bottom);
        if (this.hover.freq !== null) {
          const y = Math.round(this.freqToY(this.hover.freq)) + 0.5;
          ctx.moveTo(p.x, y);
          ctx.lineTo(p.x + p.w, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // ------------------------------------------------------------ hover

    onPointer(e) {
      if (!this.result) return;
      const rect = this.overlay.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const p = this.plot;
      const bottom = p.y + p.h + STRIP.gap + STRIP.height;
      if (px < p.x || px > p.x + p.w || py < p.y || py > bottom) {
        this.clearHover();
        return;
      }
      const r = this.result;
      const time = ((px - p.x) / p.w) * r.duration;
      const inPlot = py <= p.y + p.h;
      const freq = inPlot ? (1 - (py - p.y) / p.h) * this.maxFreq : null;
      this.hover = { time, freq };
      this.drawOverlay();
      this.showTooltip(px, py, time, freq);
    }

    clearHover() {
      this.hover = null;
      this.tooltip.hidden = true;
      this.drawOverlay();
    }

    showTooltip(px, py, time, freq) {
      const r = this.result;
      const frame = Math.max(0, Math.min(r.nFrames - 1, Math.round((time * r.sampleRate) / r.hopSize)));
      const label = r.labels[frame];
      const labelText = label === 2 ? this.labels.falsetto : label === 1 ? this.labels.chest : this.labels.unvoiced;
      const labelClass = label === 2 ? 'falsetto' : label === 1 ? 'chest' : 'unvoiced';

      let html = `<div class="tt-row"><span>時間</span><b>${time.toFixed(3)} s</b></div>`;
      if (freq !== null) {
        const bin = Math.max(0, Math.min(r.nBins - 1, Math.round(freq / r.binHz)));
        const db = 10 * Math.log10(r.power[frame * r.nBins + bin] + 1e-20) - 10 * Math.log10(r.maxPower);
        html +=
          `<div class="tt-row"><span>周波数</span><b>${freq.toFixed(1)} Hz</b></div>` +
          `<div class="tt-row"><span>音名</span><b>${root.VoiceAnalyzer.noteName(freq)}</b></div>` +
          `<div class="tt-row"><span>強度</span><b>${db.toFixed(1)} dB</b></div>`;
      }
      if (r.voiced[frame]) {
        html += `<div class="tt-row"><span>F0</span><b>${r.f0[frame].toFixed(1)} Hz</b></div>`;
      }
      html += `<div class="tt-row"><span>判定</span><b class="tt-label ${labelClass}">${labelText}</b></div>`;
      this.tooltip.innerHTML = html;
      this.tooltip.hidden = false;

      const tw = this.tooltip.offsetWidth;
      const th = this.tooltip.offsetHeight;
      const W = this.wrap.clientWidth;
      const H = this.wrap.clientHeight;
      let left = px + 14;
      let top = py + 14;
      if (left + tw > W - 4) left = px - tw - 14;
      if (top + th > H - 4) top = py - th - 14;
      this.tooltip.style.left = `${Math.max(4, left)}px`;
      this.tooltip.style.top = `${Math.max(4, top)}px`;
    }
  }

  root.SpectrogramView = SpectrogramView;
})(window);
