/**
 * AudioWorkletProcessor that captures audio at the browser's native sample rate
 * (typically 48kHz) and downsamples to 16kHz using a polyphase FIR low-pass filter.
 *
 * Outputs Int16 PCM chunks via the port for WebSocket transmission.
 *
 * Why not ScriptProcessor?
 *   - ScriptProcessor runs on the main thread → audio gaps during UI work
 *   - AudioWorklet runs on a dedicated real-time audio thread → glitch-free
 */
class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // Ratio: e.g. 48000 / 16000 = 3
    const nativeSR = options.processorOptions?.nativeSampleRate || sampleRate;
    this.ratio = Math.round(nativeSR / 16000);
    // Build a low-pass FIR filter (windowed-sinc) to prevent aliasing before decimation.
    // Cutoff at 7600 Hz (slightly below Nyquist of 8000 Hz for 16kHz target).
    this.filter = this._buildLowPassFilter(nativeSR, 7600, 127);
    // Ring buffer for the FIR filter (holds last N samples for convolution)
    this.filterState = new Float32Array(this.filter.length);
    this.filterPos = 0;
    // Decimation phase counter
    this.phase = 0;
    // Accumulation buffer for 16kHz int16 output (send ~4096 samples at a time ≈ 256ms)
    this.outBuf = new Int16Array(4096);
    this.outPos = 0;
  }

  /**
   * Windowed-sinc low-pass FIR filter design.
   * @param {number} sr     - native sample rate (e.g. 48000)
   * @param {number} cutoff - cutoff frequency in Hz (e.g. 7600)
   * @param {number} order  - filter length (odd, higher = sharper rolloff)
   */
  _buildLowPassFilter(sr, cutoff, order) {
    const fc = cutoff / sr;          // normalized cutoff [0..0.5]
    const mid = (order - 1) / 2;
    const h = new Float32Array(order);
    let sum = 0;
    for (let i = 0; i < order; i++) {
      const n = i - mid;
      // Sinc function
      if (n === 0) {
        h[i] = 2 * Math.PI * fc;
      } else {
        h[i] = Math.sin(2 * Math.PI * fc * n) / n;
      }
      // Blackman window for good stopband attenuation (~74 dB)
      h[i] *= 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (order - 1))
                     + 0.08 * Math.cos((4 * Math.PI * i) / (order - 1));
      sum += h[i];
    }
    // Normalize for unity gain at DC
    for (let i = 0; i < order; i++) h[i] /= sum;
    return h;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // Float32, native sample rate, 128 frames
    const filterLen = this.filter.length;

    for (let i = 0; i < samples.length; i++) {
      // Push sample into filter ring buffer
      this.filterState[this.filterPos] = samples[i];
      this.filterPos = (this.filterPos + 1) % filterLen;

      // Decimate: only compute output every `ratio` samples
      this.phase++;
      if (this.phase >= this.ratio) {
        this.phase = 0;

        // Apply FIR filter (convolution at this output point)
        let acc = 0;
        let idx = this.filterPos; // oldest sample in ring
        for (let k = 0; k < filterLen; k++) {
          acc += this.filter[k] * this.filterState[idx];
          idx = (idx + 1) % filterLen;
        }

        // Convert to int16 and accumulate
        const clamped = Math.max(-1, Math.min(1, acc));
        this.outBuf[this.outPos++] = clamped * 32767;

        // Flush when buffer is full
        if (this.outPos >= this.outBuf.length) {
          this.port.postMessage(this.outBuf.buffer.slice(0), []);
          this.outPos = 0;
        }
      }
    }

    return true;
  }
}

registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);
