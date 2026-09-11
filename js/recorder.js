/*
 * MicRecorder — マイク入力を PCM のまま録音する
 * （エコーキャンセル・ノイズ抑制・自動ゲインはスペクトルを歪めるので無効化）
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

  class MicRecorder {
    constructor(audioContext) {
      this.ctx = audioContext;
      this.chunks = [];
      this.length = 0;
      this.nodes = [];
      this.stream = null;
      this.analyser = null;
      this.levelBuffer = null;
    }

    async start() {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      await this.ctx.resume();
      const ctx = this.ctx;
      const source = ctx.createMediaStreamSource(this.stream);
      const mute = ctx.createGain();
      mute.gain.value = 0;
      mute.connect(ctx.destination);

      this.analyser = ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.levelBuffer = new Float32Array(this.analyser.fftSize);
      source.connect(this.analyser);

      const push = (data) => {
        this.chunks.push(data);
        this.length += data.length;
      };

      let capture;
      try {
        if (!ctx.audioWorklet) throw new Error('AudioWorklet unsupported');
        if (!MicRecorder.workletLoaded) {
          const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'application/javascript' }));
          await ctx.audioWorklet.addModule(url);
          URL.revokeObjectURL(url);
          MicRecorder.workletLoaded = true;
        }
        capture = new AudioWorkletNode(ctx, 'capture-processor');
        capture.port.onmessage = (e) => push(e.data);
      } catch (err) {
        capture = ctx.createScriptProcessor(4096, 2, 1);
        capture.onaudioprocess = (e) => {
          const input = e.inputBuffer;
          const out = new Float32Array(input.length);
          for (let c = 0; c < input.numberOfChannels; c++) {
            const ch = input.getChannelData(c);
            for (let i = 0; i < out.length; i++) out[i] += ch[i] / input.numberOfChannels;
          }
          push(out);
        };
      }
      source.connect(capture);
      capture.connect(mute);
      this.nodes = [source, capture, mute, this.analyser];
      this.startedAt = performance.now();
    }

    get elapsed() {
      return (performance.now() - this.startedAt) / 1000;
    }

    // 現在の入力レベル [dBFS]
    level() {
      if (!this.analyser) return -Infinity;
      this.analyser.getFloatTimeDomainData(this.levelBuffer);
      let sum = 0;
      for (const v of this.levelBuffer) sum += v * v;
      return 10 * Math.log10(sum / this.levelBuffer.length + 1e-12);
    }

    // 録音を終了し、モノラルの AudioBuffer を返す（何も録れていなければ null）
    stop() {
      this.release();
      if (this.length === 0) return null;
      const buffer = this.ctx.createBuffer(1, this.length, this.ctx.sampleRate);
      const data = buffer.getChannelData(0);
      let offset = 0;
      for (const chunk of this.chunks) {
        data.set(chunk, offset);
        offset += chunk.length;
      }
      this.chunks = [];
      return buffer;
    }

    release() {
      for (const node of this.nodes) node.disconnect();
      this.nodes = [];
      this.analyser = null;
      if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
  }

  MicRecorder.workletLoaded = false;
  root.MicRecorder = MicRecorder;
})(window);
