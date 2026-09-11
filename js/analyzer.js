/*
 * VoiceAnalyzer — 音声のスペクトラム解析と地声／裏声の推定
 *
 * 処理の流れ
 *   1. STFT（ハン窓）でパワースペクトログラムを計算
 *   2. McLeod Pitch Method（NSDF）でフレームごとの基本周波数 F0 を推定
 *   3. 有声フレームで倍音ごとのパワーを測り、次の特徴量を求める
 *        - H1−H2   : 第1倍音と第2倍音のレベル差 [dB]（裏声ほど大きい）
 *        - H1*−H2* : LPC で推定したフォルマントの影響を補正した H1−H2（Iseli & Alwan, 2004）
 *        - HRF     : 第2倍音以上（〜4 kHz）の合計パワーと第1倍音の比 [dB]（地声ほど大きい）
 *        - 傾斜    : 倍音レベルの倍音次数（対数）に対する傾き [dB/oct]（裏声ほど急）
 *   4. ロジスティック回帰スコアで各フレームを判定し、多数決で全体を判定
 *      フォルマント推定が安定する低めの声では H1*−H2* を、倍音がまばらな高い声では傾斜を使い、
 *      330〜420 Hz でスコアを線形に混ぜる。係数は LF 声門音源モデルによる合成音声で学習した。
 *
 * ブラウザでは window.VoiceAnalyzer、Node.js では module.exports として使える。
 */
(function (root) {
  'use strict';

  const DEFAULTS = {
    frameSize: 2048,        // STFT の窓長（22.05 kHz で約 93 ms）
    hopSize: 256,           // フレーム間隔（約 11.6 ms）
    pitchFrameSize: 1024,   // ピッチ推定の窓長
    minF0: 60,
    maxF0: 1400,
    clarityThreshold: 0.75, // NSDF ピーク値がこれ未満なら無声
    relativeSilenceDb: -35, // 最大フレームからこれ以上小さいフレームは無声
    absoluteSilenceDb: -60, // dBFS
    harmonicMaxFreq: 4000,  // 特徴量に使う倍音の上限周波数
    minVoicedRun: 5,        // これより短い有声区間はノイズとして除外（フレーム数）
    scoreSmoothing: 9,      // スコアの移動中央値の幅（フレーム数）
  };

  // 裏声スコア = Σ weight × 特徴量 + bias（正なら裏声）
  const SCORE_MODELS = {
    low: { weights: { h1h2: 0.0968, hrf: -0.0649, h1h2c: 0.2619 }, bias: -3.611 },
    high: { weights: { h1h2: 0.0985, hrf: -0.0473, slope: -0.1663 }, bias: -3.025, fallback: { slope: -16.7 } },
  };
  const BLEND_F0 = [330, 420]; // この範囲で low → high にスコアを線形に混ぜる

  // 表示用：各特徴量単独での地声／裏声の境目（dir > 0 は大きいほど裏声寄り）
  const FEATURE_THRESHOLDS = {
    h1h2: { threshold: 5.4, dir: +1 },
    hrf: { threshold: -0.2, dir: -1 },
    slope: { threshold: -16.7, dir: -1 },
  };

  // ---------------------------------------------------------------- FFT

  function createFFT(n) {
    const levels = Math.round(Math.log2(n));
    if (1 << levels !== n) throw new Error('FFT size must be a power of 2');
    const cosT = new Float64Array(n / 2);
    const sinT = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      cosT[i] = Math.cos((2 * Math.PI * i) / n);
      sinT[i] = Math.sin((2 * Math.PI * i) / n);
    }
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0, v = i; b < levels; b++, v >>= 1) r = (r << 1) | (v & 1);
      rev[i] = r;
    }
    return function transform(re, im) {
      for (let i = 0; i < n; i++) {
        const j = rev[i];
        if (j > i) {
          let t = re[i]; re[i] = re[j]; re[j] = t;
          t = im[i]; im[i] = im[j]; im[j] = t;
        }
      }
      for (let size = 2; size <= n; size *= 2) {
        const half = size / 2;
        const step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = i, k = 0; j < i + half; j++, k += step) {
            const l = j + half;
            const tre = re[l] * cosT[k] + im[l] * sinT[k];
            const tim = -re[l] * sinT[k] + im[l] * cosT[k];
            re[l] = re[j] - tre;
            im[l] = im[j] - tim;
            re[j] += tre;
            im[j] += tim;
          }
        }
      }
    };
  }

  // ---------------------------------------------------------------- utilities

  function removeDC(input, sampleRate) {
    // 1 次の DC ブロッカー（カットオフ約 20 Hz）
    const R = Math.exp((-2 * Math.PI * 20) / sampleRate);
    const out = new Float32Array(input.length);
    let prevX = 0;
    let prevY = 0;
    for (let i = 0; i < input.length; i++) {
      const y = input[i] - prevX + R * prevY;
      prevX = input[i];
      prevY = y;
      out[i] = y;
    }
    return out;
  }

  function median(values) {
    if (values.length === 0) return NaN;
    const s = Array.from(values).sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  function noteName(freq) {
    if (!(freq > 0)) return '';
    const midi = Math.round(69 + 12 * Math.log2(freq / 440));
    return NOTE_NAMES[((midi % 12) + 12) % 12] + (Math.floor(midi / 12) - 1);
  }

  // ---------------------------------------------------------------- pitch (MPM)

  function createPitchDetector(sampleRate, o) {
    const W = o.pitchFrameSize;
    const M = W * 2;
    const fft = createFFT(M);
    const re = new Float64Array(M);
    const im = new Float64Array(M);
    const nsdf = new Float64Array(W);
    const minLag = Math.max(2, Math.floor(sampleRate / o.maxF0));
    const maxLag = Math.min(W - 2, Math.ceil(sampleRate / o.minF0));

    // frame: 長さ W の Float64Array。{ f0, clarity } を返す（検出不可なら f0 = 0）
    return function detect(frame) {
      let energy = 0;
      for (let n = 0; n < W; n++) {
        re[n] = frame[n];
        energy += frame[n] * frame[n];
      }
      re.fill(0, W);
      im.fill(0);
      if (energy <= 0) return { f0: 0, clarity: 0 };

      // Wiener–Khinchin: |X|^2 を FFT すると自己相関 × M が得られる
      fft(re, im);
      for (let k = 0; k < M; k++) {
        re[k] = re[k] * re[k] + im[k] * im[k];
        im[k] = 0;
      }
      fft(re, im);

      let m = 2 * energy;
      nsdf[0] = 1;
      for (let tau = 1; tau <= maxLag + 1; tau++) {
        const a = frame[tau - 1];
        const b = frame[W - tau];
        m -= a * a + b * b;
        nsdf[tau] = m > 1e-12 ? (2 * (re[tau] / M)) / m : 0;
      }

      // 正の区間ごとの最大値（key maxima）を集める
      const peaks = [];
      let tau = 1;
      while (tau <= maxLag && nsdf[tau] > 0) tau++;
      while (tau <= maxLag) {
        while (tau <= maxLag && nsdf[tau] <= 0) tau++;
        let best = -1;
        let bestTau = -1;
        while (tau <= maxLag && nsdf[tau] > 0) {
          if (nsdf[tau] > best) {
            best = nsdf[tau];
            bestTau = tau;
          }
          tau++;
        }
        if (bestTau >= minLag && bestTau < maxLag) peaks.push(bestTau);
      }
      if (peaks.length === 0) return { f0: 0, clarity: 0 };

      let highest = 0;
      for (const p of peaks) highest = Math.max(highest, nsdf[p]);
      const chosen = peaks.find((p) => nsdf[p] >= 0.9 * highest);

      const a = nsdf[chosen - 1];
      const b = nsdf[chosen];
      const c = nsdf[chosen + 1];
      const denom = a - 2 * b + c;
      const delta = denom !== 0 ? (0.5 * (a - c)) / denom : 0;
      const period = chosen + delta;
      return { f0: sampleRate / period, clarity: b - 0.25 * (a - c) * delta };
    };
  }

  // ---------------------------------------------------------------- formants (selective LPC)

  function levinson(R, order) {
    const a = new Float64Array(order + 1);
    const prev = new Float64Array(order + 1);
    a[0] = 1;
    let err = R[0];
    for (let i = 1; i <= order; i++) {
      let acc = R[i];
      for (let j = 1; j < i; j++) acc += a[j] * R[i - j];
      const k = -acc / err;
      prev.set(a);
      for (let j = 1; j < i; j++) a[j] = prev[j] + k * prev[i - j];
      a[i] = k;
      err *= 1 - k * k;
      if (!(err > 0)) break;
    }
    return a;
  }

  // z^p + c[1] z^(p-1) + ... + c[p] の根（Durand–Kerner 法）
  function polyRoots(c) {
    const p = c.length - 1;
    const re = new Float64Array(p);
    const im = new Float64Array(p);
    for (let i = 0; i < p; i++) {
      const ang = (2 * Math.PI * i) / p + 0.4;
      re[i] = 0.9 * Math.cos(ang);
      im[i] = 0.9 * Math.sin(ang);
    }
    for (let iter = 0; iter < 300; iter++) {
      let maxDelta = 0;
      for (let i = 0; i < p; i++) {
        const zr = re[i];
        const zi = im[i];
        let pr = 1;
        let pi = 0;
        for (let k = 1; k <= p; k++) {
          const nr = pr * zr - pi * zi + c[k];
          pi = pr * zi + pi * zr;
          pr = nr;
        }
        let dr = 1;
        let di = 0;
        for (let j = 0; j < p; j++) {
          if (j === i) continue;
          const ar = zr - re[j];
          const ai = zi - im[j];
          const nr = dr * ar - di * ai;
          di = dr * ai + di * ar;
          dr = nr;
        }
        const den = dr * dr + di * di || 1e-30;
        const qr = (pr * dr + pi * di) / den;
        const qi = (pi * dr - pr * di) / den;
        re[i] -= qr;
        im[i] -= qi;
        maxDelta = Math.max(maxDelta, Math.hypot(qr, qi));
      }
      if (maxDelta < 1e-12) break;
    }
    return { re, im };
  }

  // 0〜5.5 kHz のパワースペクトルに LPC を当てはめ、フォルマント [周波数, 帯域幅] を推定する
  function estimateFormants(power, offset, nBins, df) {
    const order = 12;
    const preEmphasis = 0.9;
    const B = Math.min(nBins - 1, Math.floor(5500 / df));
    const fs = 2 * B * df;
    const R = new Float64Array(order + 1);
    for (let b = 0; b <= B; b++) {
      const w = (Math.PI * b) / B;
      const p =
        power[offset + b] *
        (1 + preEmphasis * preEmphasis - 2 * preEmphasis * Math.cos(w)) *
        (b === 0 || b === B ? 0.5 : 1);
      for (let t = 0; t <= order; t++) R[t] += p * Math.cos(w * t);
    }
    R[0] *= 1.0001;
    const { re, im } = polyRoots(levinson(R, order));
    const formants = [];
    for (let i = 0; i < re.length; i++) {
      if (im[i] <= 0) continue;
      const freq = (Math.atan2(im[i], re[i]) * fs) / (2 * Math.PI);
      const bandwidth = (-Math.log(Math.hypot(re[i], im[i])) * fs) / Math.PI;
      if (freq >= 200 && freq <= 5000 && bandwidth > 0 && bandwidth < 700) formants.push([freq, bandwidth]);
    }
    formants.sort((x, y) => x[0] - y[0]);
    return { formants, fs };
  }

  // 下位 4 つのフォルマント共振が周波数 f に与えるゲイン [dB]（DC で 0 dB）
  function tractGainDb(formants, f, fs) {
    let gain = 0;
    for (let i = 0; i < Math.min(4, formants.length); i++) {
      const F = formants[i][0];
      const r = Math.exp((-Math.PI * Math.max(40, Math.min(500, formants[i][1]))) / fs);
      const c = Math.cos((2 * Math.PI * F) / fs);
      const num = (1 - 2 * r * c + r * r) ** 2;
      const den =
        (1 - 2 * r * Math.cos((2 * Math.PI * (F + f)) / fs) + r * r) *
        (1 - 2 * r * Math.cos((2 * Math.PI * (F - f)) / fs) + r * r);
      gain += 10 * Math.log10(num / den);
    }
    return gain;
  }

  // ---------------------------------------------------------------- harmonic features

  function harmonicFeatures(power, offset, nBins, df, f0, maxFreq) {
    const nyquist = df * (nBins - 1);
    const K = Math.floor(Math.min(maxFreq, nyquist - f0 / 2) / f0);
    if (K < 2) return null;

    const levels = new Float64Array(K + 1);
    const binCount = new Float64Array(K + 1);
    const valleys = [];
    const halfWidth = Math.max(1, Math.round((0.15 * f0) / df));
    for (let k = 1; k <= K; k++) {
      let lo = Math.max(1, Math.ceil(((k - 0.5) * f0) / df));
      let hi = Math.min(nBins - 1, Math.floor(((k + 0.5) * f0) / df));
      if (hi < lo) lo = hi = Math.round((k * f0) / df);
      let sum = 0;
      for (let b = lo; b <= hi; b++) sum += power[offset + b];
      levels[k] = Math.max(sum, 1e-20);
      binCount[k] = hi - lo + 1;
      // 倍音と倍音の間（谷）の最小値を雑音床の推定に使う
      const v = Math.round(((k + 0.5) * f0) / df);
      let min = Infinity;
      for (let b = Math.max(1, v - halfWidth); b <= Math.min(nBins - 1, v + halfWidth); b++) {
        min = Math.min(min, power[offset + b]);
      }
      if (Number.isFinite(min)) valleys.push(min);
    }
    const noisePerBin = valleys.length ? median(valleys) : 0;
    const denoised = (k) => Math.max(levels[k] - noisePerBin * binCount[k], noisePerBin, 1e-20);

    let upper = 0;
    for (let k = 2; k <= K; k++) upper += levels[k];

    let slope = null;
    if (K >= 3) {
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (let k = 1; k <= K; k++) {
        const x = Math.log2(k);
        const y = 10 * Math.log10(levels[k]);
        sx += x; sy += y; sxx += x * x; sxy += x * y;
      }
      slope = (K * sxy - sx * sy) / (K * sxx - sx * sx);
    }

    // 声道（フォルマント）の影響を補正した H1*−H2*（Iseli & Alwan, 2004）
    const { formants, fs } = estimateFormants(power, offset, nBins, df);
    const h1h2c =
      10 * Math.log10(denoised(1) / denoised(2)) - (tractGainDb(formants, f0, fs) - tractGainDb(formants, 2 * f0, fs));

    return {
      h1h2: 10 * Math.log10(levels[1] / levels[2]),
      h1h2c,
      hrf: 10 * Math.log10(upper / levels[1]),
      slope,
    };
  }

  function linearScore(model, features) {
    let score = model.bias;
    for (const key in model.weights) {
      let v = features[key];
      if (v === null || v === undefined || !Number.isFinite(v)) v = model.fallback ? model.fallback[key] : 0;
      score += model.weights[key] * v;
    }
    return score;
  }

  function scoreFeatures(features, f0) {
    const t = Math.max(0, Math.min(1, (f0 - BLEND_F0[0]) / (BLEND_F0[1] - BLEND_F0[0])));
    if (t === 0) return linearScore(SCORE_MODELS.low, features);
    if (t === 1) return linearScore(SCORE_MODELS.high, features);
    return (1 - t) * linearScore(SCORE_MODELS.low, features) + t * linearScore(SCORE_MODELS.high, features);
  }

  // 各特徴量が地声寄りか裏声寄りか（UI 表示用）
  function featureLeaning(key, value) {
    const f = FEATURE_THRESHOLDS[key];
    if (!f || value === null || !Number.isFinite(value)) return null;
    return f.dir * (value - f.threshold) > 0 ? 'falsetto' : 'chest';
  }

  // 連続する有声区間 [start, end) の一覧
  function voicedRuns(voiced) {
    const runs = [];
    let start = -1;
    for (let i = 0; i <= voiced.length; i++) {
      const v = i < voiced.length && voiced[i];
      if (v && start < 0) start = i;
      if (!v && start >= 0) {
        runs.push([start, i]);
        start = -1;
      }
    }
    return runs;
  }

  function medianFilterRuns(values, runs, width) {
    const out = Float32Array.from(values);
    const r = width >> 1;
    for (const [s, e] of runs) {
      for (let i = s; i < e; i++) {
        const win = [];
        for (let j = Math.max(s, i - r); j < Math.min(e, i + r + 1); j++) {
          if (Number.isFinite(values[j])) win.push(values[j]);
        }
        if (win.length) out[i] = median(win);
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- main

  function analyze(input, sampleRate, options) {
    const o = Object.assign({}, DEFAULTS, options);
    const x = removeDC(input, sampleRate);
    const N = o.frameSize;
    const hop = o.hopSize;
    const W = o.pitchFrameSize;
    const nFrames = Math.max(1, Math.floor((x.length - 1) / hop) + 1);
    const nBins = N / 2 + 1;
    const df = sampleRate / N;

    const win = new Float64Array(N);
    let winSum = 0;
    for (let n = 0; n < N; n++) {
      win[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N);
      winSum += win[n];
    }
    const norm = (winSum * winSum) / 4; // 振幅 1 の正弦波 → 0 dB

    const fft = createFFT(N);
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    const detect = createPitchDetector(sampleRate, o);
    const pitchFrame = new Float64Array(W);

    const power = new Float32Array(nFrames * nBins);
    const f0 = new Float32Array(nFrames);
    const clarity = new Float32Array(nFrames);
    const rmsDb = new Float32Array(nFrames);
    let maxPower = 1e-20;

    for (let i = 0; i < nFrames; i++) {
      const center = i * hop;

      for (let n = 0, idx = center - N / 2; n < N; n++, idx++) {
        re[n] = idx >= 0 && idx < x.length ? x[idx] * win[n] : 0;
        im[n] = 0;
      }
      fft(re, im);
      const off = i * nBins;
      for (let b = 0; b < nBins; b++) {
        const p = (re[b] * re[b] + im[b] * im[b]) / norm;
        power[off + b] = p;
        if (p > maxPower) maxPower = p;
      }

      let energy = 0;
      for (let n = 0, idx = center - W / 2; n < W; n++, idx++) {
        const v = idx >= 0 && idx < x.length ? x[idx] : 0;
        pitchFrame[n] = v;
        energy += v * v;
      }
      rmsDb[i] = 10 * Math.log10(energy / W + 1e-20);
      const p = detect(pitchFrame);
      f0[i] = p.f0;
      clarity[i] = p.clarity;
    }

    // 有声判定
    let loudest = -Infinity;
    for (let i = 0; i < nFrames; i++) loudest = Math.max(loudest, rmsDb[i]);
    const voiced = new Uint8Array(nFrames);
    for (let i = 0; i < nFrames; i++) {
      voiced[i] =
        f0[i] >= o.minF0 &&
        f0[i] <= o.maxF0 &&
        clarity[i] >= o.clarityThreshold &&
        rmsDb[i] >= loudest + o.relativeSilenceDb &&
        rmsDb[i] >= o.absoluteSilenceDb
          ? 1
          : 0;
    }
    for (const [s, e] of voicedRuns(voiced)) {
      if (e - s < o.minVoicedRun) voiced.fill(0, s, e);
    }
    let runs = voicedRuns(voiced);

    // F0 のスパイク除去
    const f0Smooth = medianFilterRuns(f0, runs, 5);
    for (let i = 0; i < nFrames; i++) if (!voiced[i]) f0Smooth[i] = 0;

    // フレームごとの特徴量とスコア
    const h1h2 = new Float32Array(nFrames).fill(NaN);
    const h1h2c = new Float32Array(nFrames).fill(NaN);
    const hrf = new Float32Array(nFrames).fill(NaN);
    const slope = new Float32Array(nFrames).fill(NaN);
    const rawScores = new Float32Array(nFrames).fill(NaN);
    for (let i = 0; i < nFrames; i++) {
      if (!voiced[i]) continue;
      const feat = harmonicFeatures(power, i * nBins, nBins, df, f0Smooth[i], o.harmonicMaxFreq);
      if (!feat) {
        voiced[i] = 0;
        f0Smooth[i] = 0;
        continue;
      }
      h1h2[i] = feat.h1h2;
      h1h2c[i] = feat.h1h2c;
      hrf[i] = feat.hrf;
      slope[i] = feat.slope === null ? NaN : feat.slope;
      rawScores[i] = scoreFeatures(feat, f0Smooth[i]);
    }
    runs = voicedRuns(voiced);
    const scores = medianFilterRuns(rawScores, runs, o.scoreSmoothing);

    // 0: 無声, 1: 地声, 2: 裏声
    const labels = new Uint8Array(nFrames);
    const voicedIdx = [];
    let falsettoCount = 0;
    for (let i = 0; i < nFrames; i++) {
      if (!voiced[i]) continue;
      voicedIdx.push(i);
      labels[i] = scores[i] > 0 ? 2 : 1;
      if (labels[i] === 2) falsettoCount++;
    }

    const pick = (arr) => median(voicedIdx.map((i) => arr[i]).filter(Number.isFinite));
    const voicedSeconds = (voicedIdx.length * hop) / sampleRate;
    let summary;
    if (voicedSeconds < 0.2) {
      summary = { label: null, voicedSeconds };
    } else {
      const ratio = falsettoCount / voicedIdx.length;
      const meanScore = voicedIdx.reduce((s, i) => s + scores[i], 0) / voicedIdx.length;
      const features = { h1h2: pick(h1h2), h1h2c: pick(h1h2c), hrf: pick(hrf), slope: pick(slope) };
      const medianF0 = pick(f0Smooth);
      summary = {
        label: ratio > 0.5 || (ratio === 0.5 && meanScore > 0) ? 'falsetto' : 'chest',
        falsettoRatio: ratio,
        meanScore,
        voicedSeconds,
        medianF0,
        note: noteName(medianF0),
        formantCorrected: medianF0 < BLEND_F0[1],
        features,
        leaning: {
          h1h2: featureLeaning('h1h2', features.h1h2),
          hrf: featureLeaning('hrf', features.hrf),
          slope: featureLeaning('slope', features.slope),
        },
      };
    }

    return {
      sampleRate,
      duration: x.length / sampleRate,
      frameSize: N,
      hopSize: hop,
      nFrames,
      nBins,
      binHz: df,
      power,
      maxPower,
      f0: f0Smooth,
      voiced,
      scores,
      labels,
      features: { h1h2, h1h2c, hrf, slope },
      summary,
    };
  }

  const api = { analyze, noteName, createFFT, estimateFormants, DEFAULTS, SCORE_MODELS, FEATURE_THRESHOLDS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.VoiceAnalyzer = api;
})(typeof self !== 'undefined' ? self : this);
