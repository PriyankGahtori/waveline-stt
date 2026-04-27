class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options?.processorOptions || {};
    this.chunkFrames = opts.chunkFrames || 16000 * 4; // 4s default
    this.silenceThreshold = opts.silenceThreshold ?? 0.002; // RMS below this = silence
    this.buf = new Float32Array(this.chunkFrames);
    this.wrote = 0;

    this.port.onmessage = (e) => {
      const { type, chunkFrames, silenceThreshold } = e.data || {};
      if (type === 'setChunkFrames' && Number.isInteger(chunkFrames) && chunkFrames > 0) {
        this.chunkFrames = chunkFrames;
        this.buf = new Float32Array(this.chunkFrames);
        this.wrote = 0;
      } else if (type === 'setSilenceThreshold' && typeof silenceThreshold === 'number') {
        this.silenceThreshold = silenceThreshold;
      } else if (type === 'flush') {
        if (this.wrote > 0) {
          const slice = this.buf.slice(0, this.wrote);
          this.port.postMessage({ type: 'chunk', samples: slice, rms: _rms(slice) }, [slice.buffer]);
          this.buf = new Float32Array(this.chunkFrames);
          this.wrote = 0;
        }
        this.port.postMessage({ type: 'flushed' });
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    const ch0 = input[0] || new Float32Array(128);
    const ch1 = input[1];
    const frames = ch0.length;

    let i = 0;
    while (i < frames) {
      const space = this.chunkFrames - this.wrote;
      const copy = Math.min(space, frames - i);

      if (ch1) {
        for (let k = 0; k < copy; k++) {
          this.buf[this.wrote + k] = (ch0[i + k] + ch1[i + k]) * 0.5;
        }
      } else {
        this.buf.set(ch0.subarray(i, i + copy), this.wrote);
      }

      this.wrote += copy;
      i += copy;

      if (this.wrote >= this.chunkFrames) {
        const out = this.buf;
        const rms = _rms(out);
        this.port.postMessage({ type: 'chunk', samples: out, rms }, [out.buffer]);
        this.buf = new Float32Array(this.chunkFrames);
        this.wrote = 0;
      }
    }
    return true;
  }
}

function _rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

registerProcessor('pcm-capture', PcmCaptureProcessor);
