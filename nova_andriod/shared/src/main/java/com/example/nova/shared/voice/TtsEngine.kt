package com.example.nova.shared.voice

/**
 * Synthesizes speech on-device. Swappable (Kokoro now; Pocket-TTS/custom later).
 * Not thread-safe; confine calls to one coroutine.
 */
interface TtsEngine {
    /** Output sample rate (Hz). */
    val sampleRate: Int

    /**
     * Synthesize [text], invoking [onPcm] with PCM16 chunks as they are produced.
     * Blocking; call off the main thread. [shouldContinue] lets the caller stop
     * mid-utterance (barge-in) — return false to abort.
     */
    fun speak(text: String, onPcm: (ShortArray) -> Unit, shouldContinue: () -> Boolean = { true })

    fun close()
}
