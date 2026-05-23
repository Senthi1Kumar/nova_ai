package com.example.nova.shared.voice

import kotlin.math.sqrt

/** Mic capture sample rate. Gemma-4's audio encoder expects 16 kHz mono. */
const val SAMPLE_RATE_CAPTURE = 16_000

/** Playback sample rate for Kokoro TTS output (24 kHz). */
const val SAMPLE_RATE_TTS = 24_000

/**
 * Root-mean-square level of the first [count] PCM16 [samples], normalized so a
 * full-scale signal (±32768) is ~1.0. Drives the orb's amplitude animation.
 */
fun rms(samples: ShortArray, count: Int = samples.size): Float {
    if (count <= 0) return 0f
    var sumSq = 0.0
    for (i in 0 until count) {
        val s = samples[i].toDouble()
        sumSq += s * s
    }
    return (sqrt(sumSq / count) / 32768.0).toFloat()
}

/**
 * Converts TTS Float samples in [-1, 1] to signed PCM16. Values outside the
 * range are clamped. Used to feed sherpa-onnx Kokoro output to AudioPlayer.
 */
fun floatToPcm16(samples: FloatArray): ShortArray {
    val out = ShortArray(samples.size)
    for (i in samples.indices) {
        val clamped = samples[i].coerceIn(-1f, 1f)
        out[i] = (clamped * 32767f).toInt().toShort()
    }
    return out
}
