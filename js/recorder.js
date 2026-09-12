/*
 * MicRecorder — マイク入力を PCM のまま録音する
 * （エコーキャンセル・ノイズ抑制・自動ゲインはスペクトルを歪めるので無効化）
 *
 * Bluetooth 対策:
 *   - 対応ブラウザ（Chrome / Edge）では MediaStreamTrackProcessor で録音し、AudioContext を開かない。
 *     録音中に出力デバイスを掴まないので、通話モードに巻き込まれる範囲を小さくできる。
 *   - 非対応ブラウザでは AudioContext + AudioWorklet を使い、終了時に必ず閉じる。
 *   - 入力デバイスを選べるようにし、Bluetooth のマイクかどうかを呼び出し側が判断できるようにする。
 */
(function (root) {
  'use strict';

  const WORKLET_SOURCE = `
    class CaptureProcessor extends AudioWorkletProcessor {
      process(inputs) {
        const input = inputs[0];
        if (input && input.length) {
          const out = new Float32Array(input[0].length);
          for (const ch of input) for (let i = 0; i < out.length; i++) out[i] += ch[i] / input.length;
          this.port.postMessage(out, [out.buffer]);
        }
        return true;
      }
    }
    registerProcessor('capture-processor', CaptureProcessor);
  `;

  const BLUETOOTH_LABEL = /hands[\s-]?free|bluetooth|ヘッドセット|ハンズフリー/i;

  // Safari の Audio Session API（対応ブラウザのみ）
  function setAudioSession(type) {
    try {
      if (navigator.audioSession) navigator.audioSession.type = type;
    } catch (_) {
      /* unsupported */
    }
  }

  class MicRecorder {
    constructor() {
      this.ctx = null;
      this.chunks = [];
      this.length = 0;
      this.nodes = [];
      this.stream = null;
      this.track = null;
      this.reader = null;
      this.analyser = null;
      this.levelBuffer = null;
      this.levelDb = -Infinity;
      this.sampleRate = 48000;
      this.label = '';
    }

    static isBluetooth(label) {
      return BLUETOOTH_LABEL.test(label || '');
    }

    // 入力デバイスの一覧（ラベルはマイクの許可後にしか取れない）
    static async listInputs() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.filter((d) => d.kind === 'audioinput');
    }

    get bluetooth() {
      return MicRecorder.isBluetooth(this.label);
    }

    async start(deviceId) {
      setAudioSession('play-and-record');
      const audio = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
      if (deviceId) audio.deviceId = { exact: deviceId };
      this.stream = await navigator.mediaDevices.getUserMedia({ audio });
      this.track = this.stream.getAudioTracks()[0];
      this.label = (this.track && this.track.label) || '';
      const settings = this.track && this.track.getSettings ? this.track.getSettings() : {};
      this.sampleRate = settings.sampleRate || 48000;

      if (root.MediaStreamTrackProcessor) this.startProcessorCapture();
      else await this.startContextCapture();
      this.startedAt = performance.now();
    }

    // AudioContext を開かずに録音する（出力デバイスを掴まない）
    startProcessorCapture() {
      const processor = new root.MediaStreamTrackProcessor({ track: this.track });
      this.reader = processor.readable.getReader();
      const pump = async () => {
        while (this.reader) {
          let result;
          try {
            result = await this.reader.read();
          } catch (_) {
            break;
          }
          if (result.done || !result.value) break;
          const data = result.value;
          const frames = data.numberOfFrames;
          const channels = data.numberOfChannels || 1;
          const chunk = new Float32Array(frames);
          if (channels === 1) {
            data.copyTo(chunk, { planeIndex: 0, format: 'f32-planar' });
          } else {
            const tmp = new Float32Array(frames);
            for (let c = 0; c < channels; c++) {
              data.copyTo(tmp, { planeIndex: c, format: 'f32-planar' });
              for (let i = 0; i < frames; i++) chunk[i] += tmp[i] / channels;
            }
          }
          this.sampleRate = data.sampleRate || this.sampleRate;
          data.close();
          this.push(chunk);
          this.updateLevel(chunk);
        }
      };
      pump();
    }

    // 旧来の方法（Safari / Firefox）。終了時に必ずコンテキストを閉じる
    async startContextCapture() {
      const ctx = new (root.AudioContext || root.webkitAudioContext)();
      this.ctx = ctx;
      await ctx.resume();
      const source = ctx.createMediaStreamSource(this.stream);
      const mute = ctx.createGain();
      mute.gain.value = 0;
      mute.connect(ctx.destination);

      this.analyser = ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.levelBuffer = new Float32Array(this.analyser.fftSize);
      source.connect(this.analyser);

      let capture;
      try {
        if (!ctx.audioWorklet) throw new Error('AudioWorklet unsupported');
        const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'application/javascript' }));
        try {
          await ctx.audioWorklet.addModule(url);
        } finally {
          URL.revokeObjectURL(url);
        }
        capture = new AudioWorkletNode(ctx, 'capture-processor');
        capture.port.onmessage = (e) => this.push(e.data);
      } catch (err) {
        capture = ctx.createScriptProcessor(4096, 2, 1);
        capture.onaudioprocess = (e) => {
          const input = e.inputBuffer;
          const out = new Float32Array(input.length);
          for (let c = 0; c < input.numberOfChannels; c++) {
            const ch = input.getChannelData(c);
            for (let i = 0; i < out.length; i++) out[i] += ch[i] / input.numberOfChannels;
          }
          this.push(out);
        };
      }
      source.connect(capture);
      capture.connect(mute);
      this.nodes = [source, capture, mute, this.analyser];
    }

    push(chunk) {
      this.chunks.push(chunk);
      this.length += chunk.length;
    }

    updateLevel(chunk) {
      let sum = 0;
      for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
      this.levelDb = 10 * Math.log10(sum / chunk.length + 1e-12);
    }

    get elapsed() {
      return (performance.now() - this.startedAt) / 1000;
    }

    // 現在の入力レベル [dBFS]
    level() {
      if (this.analyser) {
        this.analyser.getFloatTimeDomainData(this.levelBuffer);
        let sum = 0;
        for (const v of this.levelBuffer) sum += v * v;
        return 10 * Math.log10(sum / this.levelBuffer.length + 1e-12);
      }
      return this.levelDb;
    }

    // 録音を終了してマイクとコンテキストを解放し、モノラルの AudioBuffer を返す（何も録れていなければ null）
    stop() {
      const sampleRate = this.ctx ? this.ctx.sampleRate : this.sampleRate;
      let buffer = null;
      if (this.length > 0) {
        buffer = this.ctx
          ? this.ctx.createBuffer(1, this.length, sampleRate)
          : new AudioBuffer({ length: this.length, numberOfChannels: 1, sampleRate });
        const data = buffer.getChannelData(0);
        let offset = 0;
        for (const chunk of this.chunks) {
          data.set(chunk, offset);
          offset += chunk.length;
        }
      }
      this.chunks = [];
      this.release();
      return buffer;
    }

    release() {
      if (this.reader) {
        const reader = this.reader;
        this.reader = null;
        reader.cancel().catch(() => {});
      }
      for (const node of this.nodes) node.disconnect();
      this.nodes = [];
      this.analyser = null;
      if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
      this.track = null;
      if (this.ctx && this.ctx.state !== 'closed') this.ctx.close().catch(() => {});
      this.ctx = null;
      setAudioSession('auto');
    }
  }

  root.MicRecorder = MicRecorder;
})(window);
