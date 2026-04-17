

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.chunkFrames = options?.processorOptions?.chunkFrames || 16000 * 4; // ~4s default
    this.buf = new Float32Array(this.chunkFrames);
    this.wrote = 0;

 
    this.port.onmessage = (e) => {
      const { type, chunkFrames } = e.data || {};
      if (type === 'setChunkFrames' && Number.isInteger(chunkFrames) && chunkFrames > 0) {
        this.chunkFrames = chunkFrames;
    
        this.buf = new Float32Array(this.chunkFrames);
        this.wrote = 0;
      } else if (type === 'flush') {

        if (this.wrote > 0) {
          const slice = this.buf.slice(0, this.wrote);
          this.port.postMessage({ type: 'chunk', samples: slice }, [slice.buffer]);
          this.buf = new Float32Array(this.chunkFrames);
          this.wrote = 0;
        }
        // tell main thread we’re done
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
    if (ch1) {
   
      while (i < frames) {
        const space = this.chunkFrames - this.wrote;
        const copy = Math.min(space, frames - i);
        for (let k = 0; k < copy; k++) {
          this.buf[this.wrote + k] = (ch0[i + k] + ch1[i + k]) * 0.5;
        }
        this.wrote += copy;
        i += copy;
        if (this.wrote >= this.chunkFrames) {
          const out = this.buf; 
          this.port.postMessage({ type: 'chunk', samples: out }, [out.buffer]);
          this.buf = new Float32Array(this.chunkFrames);
          this.wrote = 0;
        }
      }
    } else {
      
      while (i < frames) {
        const space = this.chunkFrames - this.wrote;
        const copy = Math.min(space, frames - i);
        this.buf.set(ch0.subarray(i, i + copy), this.wrote);
        this.wrote += copy;
        i += copy;
        if (this.wrote >= this.chunkFrames) {
          const out = this.buf; // full
          this.port.postMessage({ type: 'chunk', samples: out }, [out.buffer]);
          this.buf = new Float32Array(this.chunkFrames);
          this.wrote = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor);
