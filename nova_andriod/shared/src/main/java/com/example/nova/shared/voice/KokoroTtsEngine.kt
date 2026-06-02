package com.example.nova.shared.voice

import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig

/**
 * On-device Kokoro TTS via sherpa-onnx OfflineTts. [modelDir] holds the Kokoro
 * bundle: model.onnx, voices.bin, tokens.txt, and espeak-ng-data/ (dataDir).
 */
class KokoroTtsEngine(
    modelDir: String,
    private val speakerId: Int = 0,
    private val speed: Float = 1.0f,
) : TtsEngine {

    // Prefer the int8-quantized acoustic model (~3× smaller, faster on CPU) when
    // present; fall back to the fp32 model.onnx. Same voices/tokens/espeak-ng-data.
    private val modelPath: String =
        listOf("model.int8.onnx", "model.onnx")
            .map { java.io.File(modelDir, it) }
            .firstOrNull { it.exists() }
            ?.absolutePath
            ?: "$modelDir/model.onnx"

    init {
        // Decisive check for "is int8 actually loaded?" — if this prints model.onnx,
        // the int8 file never made it to modelDir and we silently fell back to fp32.
        // instance=<id> lets us tell whether multiple live engines are thrashing cores.
        Log.i(TAG, "init instance=${System.identityHashCode(this)} threads=4 model=$modelPath")
    }

    private val tts = OfflineTts(
        assetManager = null,
        config = OfflineTtsConfig(
            model = OfflineTtsModelConfig(
                kokoro = OfflineTtsKokoroModelConfig(
                    model = modelPath,
                    voices = "$modelDir/voices.bin",
                    tokens = "$modelDir/tokens.txt",
                    dataDir = "$modelDir/espeak-ng-data",
                ),
                // A55 (Exynos 1480) has 4× Cortex-A78 big cores; use them so Kokoro's
                // real-time factor drops below 1 and playback can keep up.
                numThreads = 4,
                provider = "cpu",
            ),
        ),
    )

    override val sampleRate: Int get() = tts.sampleRate()

    /**
     * Pay onnxruntime's one-time graph-init + first-inference cost up front (call
     * during model load), so the user's first real turn doesn't eat the cold start.
     * Output is discarded.
     */
    fun warmUp() {
        // Diagnostic isolation probe: synthesize a representative sentence with NO LLM
        // running, so the logged rtf is the *contention-free* floor for this device +
        // model + thread config. Compare against the in-turn `synth:` rtf:
        //   warmup rtf < 1  → engine is fine alone → in-turn slowness is CPU contention
        //   warmup rtf ~5-8 → the engine itself is slow → chase int8/threads, not pipeline
        val text = "Hello, I am Nova, your on-device voice assistant."
        val startNs = System.nanoTime()
        var firstPcmNs = 0L
        var totalSamples = 0L
        tts.generateWithCallback(text = text, sid = speakerId, speed = speed, callback = object : (FloatArray) -> Int {
            override fun invoke(samples: FloatArray): Int {
                if (firstPcmNs == 0L) firstPcmNs = System.nanoTime()
                totalSamples += samples.size
                return 1
            }
        })
        val endNs = System.nanoTime()
        val ttfbMs = if (firstPcmNs == 0L) -1.0 else (firstPcmNs - startNs) / 1_000_000.0
        val audioSec = totalSamples.toDouble() / sampleRate
        val rtf = if (audioSec > 0) ((endNs - startNs) / 1e9) / audioSec else -1.0
        Log.i(TAG, "warmup: ttfb=%.0fms rtf=%.2f audio=%.1fs (contention-free)".format(ttfbMs, rtf, audioSec))
    }

    override fun speak(text: String, onPcm: (ShortArray) -> Unit, shouldContinue: () -> Boolean) {
        if (text.isBlank()) return
        // sherpa-onnx's native generateWithCallbackImpl looks up the callback method by
        // the exact JNI signature invoke([F)Ljava/lang/Integer;. A Kotlin lambda here is
        // compiled to an invokedynamic `$$ExternalSyntheticLambda` (Kotlin 2.x default,
        // -Xlambdas=indy) that D8 desugars to a class carrying ONLY the erased
        // invoke(Object)Object bridge — the specialized method is absent, so the native
        // GetMethodID fails with NoSuchMethodError and CheckJNI aborts the process.
        // An explicit object expression is compiled to a real anonymous class that DOES
        // carry invoke([F)Ljava/lang/Integer;, which is what the JNI side requires.
        // sherpa-onnx convention: callback returns 1 to CONTINUE, 0 to STOP.
        // Timing for verification. We need three raw numbers:
        //   startNs       — when synth was requested
        //   firstPcmNs    — when the first audio buffer came back (→ TTFB)
        //   totalSamples  — total float samples produced (→ audio seconds = samples/sampleRate)
        // and endNs (after generateWithCallback returns) → synth wall-time.
        val startNs = System.nanoTime()
        var firstPcmNs = 0L
        var totalSamples = 0L
        val callback = object : (FloatArray) -> Int {
            override fun invoke(samples: FloatArray): Int {
                if (firstPcmNs == 0L) firstPcmNs = System.nanoTime()
                totalSamples += samples.size
                onPcm(floatToPcm16(samples))
                return if (shouldContinue()) 1 else 0
            }
        }
        tts.generateWithCallback(text = text, sid = speakerId, speed = speed, callback = callback)
        val endNs = System.nanoTime()

        // Verdict metrics. ttfb = the "lag before sound"; rtf < 1.0 means synth
        // outruns playback (no underrun gaps). Read these straight off logcat and
        // A/B int8 vs fp32 on the same prompt.
        val ttfbMs = if (firstPcmNs == 0L) -1.0 else (firstPcmNs - startNs) / 1_000_000.0
        val audioSec = totalSamples.toDouble() / sampleRate
        val rtf = if (audioSec > 0) ((endNs - startNs) / 1e9) / audioSec else -1.0
        Log.i(TAG, "synth: ttfb=%.0fms rtf=%.2f audio=%.1fs".format(ttfbMs, rtf, audioSec))
    }

    override fun close() {
        Log.i(TAG, "close instance=${System.identityHashCode(this)}")
        tts.free()
    }

    private companion object {
        const val TAG = "KokoroTts"
    }
}
