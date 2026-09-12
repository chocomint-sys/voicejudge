/*
 * MicRecorder — マイク入力を録音する
 * （エコーキャンセル・ノイズ抑制・自動ゲインはスペクトルを歪めるので無効化）
 *
 * スマホ対策: 録音中に AudioContext（出力デバイス）を開かない。
 *   開くと Android では通話モード（Bluetooth SCO）、iOS では play-and-record セッションになり、
 *   録音を終えてもメディア音声が戻らないことがあるため。
 *   - Chrome / Edge / Android: MediaStreamTrackProcessor で PCM を直接読む（音量メーターも作れる）
 *   - Safari / Firefox: MediaRecorder で録音し、停止後に OfflineAudioContext でデコードする
 */
(function (root) {
  'use strict';

  const BLUETOOTH_LABEL = /hands[\s-]?free|bluetooth|ヘッドセット|ハンズフリー/i;

  // Safari の Audio Session API（対応ブラウザのみ）
  function setAudioSession(type) {
    try {
      if (navigator.audioSession) navigator.audioSession.type = type;
    } catch (_) {
      /* unsupported */
    }
  }

  function pickMimeType() {
    if (typeof MediaRecorder === 'undefined' || !MediaRecorder.isTypeSupported) return '';
    for (const type of ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus']) {
      if (MediaRecorder.isTypeSupported(type)) return type;
    }
    return '';
  }

  class MicRecorder {
    constructor() {
      this.chunks = [];
      this.length = 0;
      this.stream = null;
      this.track = null;
      this.reader = null;
      this.mediaRecorder = null;
      this.blobParts = [];
      this.levelDb = -Infinity;
      this.sampleRate = 48000;
      this.label = '';
      this.hasLevel = false;
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

    static setAudioSession(type) {
      setAudioSession(type);
    }

    get bluetooth() {
      return MicRecorder.isBluetooth(this.label);
    }

    get elapsed() {
      return (performance.now() - this.startedAt) / 1000;
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
      else this.startMediaRecorderCapture();
      this.startedAt = performance.now();
    }

    // PCM を直接読む（AudioContext もエンコードも不要）
    startProcessorCapture() {
      const processor = new root.MediaStreamTrackProcessor({ track: this.track });
      this.reader = processor.readable.getReader();
      this.hasLevel = true;
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
          this.chunks.push(chunk);
          this.length += chunk.length;
          let sum = 0;
          for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
          this.levelDb = 10 * Math.log10(sum / chunk.length + 1e-12);
        }
      };
      pump();
    }

    // 圧縮したまま録り、停止後にデコードする（Safari / Firefox。音量メーターは出せない）
    startMediaRecorderCapture() {
      const mimeType = pickMimeType();
      this.mediaRecorder = mimeType ? new MediaRecorder(this.stream, { mimeType }) : new MediaRecorder(this.stream);
      this.blobParts = [];
      this.hasLevel = false;
      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size) this.blobParts.push(e.data);
      };
      this.mediaRecorder.start();
    }

    // 現在の入力レベル [dBFS]（MediaRecorder では取得できない）
    level() {
      return this.levelDb;
    }

    // 録音を終了してマイクを解放し、モノラルの AudioBuffer を返す（何も録れていなければ null）
    async stop() {
      let buffer = null;
      try {
        buffer = this.mediaRecorder ? await this.stopMediaRecorder() : this.buildBuffer();
      } catch (_) {
        buffer = null;
      } finally {
        this.release();
      }
      return buffer;
    }

    buildBuffer() {
      if (!this.length) return null;
      const buffer = new AudioBuffer({ length: this.length, numberOfChannels: 1, sampleRate: this.sampleRate });
      const data = buffer.getChannelData(0);
      let offset = 0;
      for (const chunk of this.chunks) {
        data.set(chunk, offset);
        offset += chunk.length;
      }
      this.chunks = [];
      return buffer;
    }

    stopMediaRecorder() {
      return new Promise((resolve) => {
        const rec = this.mediaRecorder;
        const finish = async () => {
          try {
            const blob = new Blob(this.blobParts, { type: rec.mimeType || 'audio/webm' });
            this.blobParts = [];
            if (!blob.size) return resolve(null);
            const Offline = root.OfflineAudioContext || root.webkitOfflineAudioContext;
            resolve(await new Offline(1, 1, 44100).decodeAudioData(await blob.arrayBuffer()));
          } catch (_) {
            resolve(null);
          }
        };
        rec.onstop = finish;
        if (rec.state !== 'inactive') rec.stop();
        else finish();
      });
    }

    release() {
      if (this.reader) {
        const reader = this.reader;
        this.reader = null;
        reader.cancel().catch(() => {});
      }
      if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
        try {
          this.mediaRecorder.stop();
        } catch (_) {
          /* already stopped */
        }
      }
      this.mediaRecorder = null;
      if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
      this.track = null;
      // iOS: 録音用セッションのままだと音が受話口側に回るため、再生側に戻す
      setAudioSession('playback');
    }
  }

  root.MicRecorder = MicRecorder;
})(window);
