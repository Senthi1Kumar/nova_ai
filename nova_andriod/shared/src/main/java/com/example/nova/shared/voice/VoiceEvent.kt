package com.example.nova.shared.voice

/** Events emitted by a VoiceSession over its lifetime. Mirrors the webui SSE order. */
sealed interface VoiceEvent {
    /** Mic is hot, capturing audio. */
    data object Listening : VoiceEvent

    /** Mic released; utterance handed to the engine. */
    data object SpeechEnd : VoiceEvent

    /** One streamed reply token of text. */
    data class Token(val text: String) : VoiceEvent

    /** A clause was flushed to TTS. */
    data class Clause(val text: String) : VoiceEvent

    /** A chunk of synthesized speech, signed 16-bit LE PCM. */
    data class AudioChunk(val pcm: ByteArray, val sampleRate: Int) : VoiceEvent {
        override fun equals(other: Any?): Boolean =
            other is AudioChunk && sampleRate == other.sampleRate && pcm.contentEquals(other.pcm)
        override fun hashCode(): Int = 31 * pcm.contentHashCode() + sampleRate
    }

    /** Per-turn latency snapshot. */
    data class TurnMetrics(val metrics: TurnMetricsData) : VoiceEvent

    /** Non-fatal issue (e.g. TTS degraded to text-only). */
    data class Warning(val message: String) : VoiceEvent

    /** Fatal error ending the turn. */
    data class Error(val message: String, val cause: Throwable? = null) : VoiceEvent

    /** Turn complete. */
    data object Done : VoiceEvent
}

/** The four UI animation states for the Gemini-Live orb. */
enum class VoiceState {
    IDLE, LISTENING, THINKING, SPEAKING;

    companion object {
        fun fromEvent(event: VoiceEvent): VoiceState = when (event) {
            is VoiceEvent.Listening -> LISTENING
            is VoiceEvent.SpeechEnd, is VoiceEvent.Token, is VoiceEvent.Clause -> THINKING
            is VoiceEvent.AudioChunk -> SPEAKING
            is VoiceEvent.TurnMetrics, is VoiceEvent.Warning,
            is VoiceEvent.Error, is VoiceEvent.Done -> IDLE
        }
    }
}
