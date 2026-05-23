package com.example.nova.shared.voice

/** Immutable per-turn latency snapshot surfaced in the metrics drawer. */
data class TurnMetricsData(
    val ttftMs: Long?,            // speech-end -> first token
    val ttfbMs: Long?,            // speech-end -> first audio
    val totalMs: Long,            // speech-end -> finish
    val decodeTokensPerSec: Double,
    val tokenCount: Int,
)

/**
 * Collects boundary timestamps for one turn and computes [TurnMetricsData].
 * [clock] returns a monotonic time in ms (default: System.nanoTime/1e6).
 * Not thread-safe; confined to the orchestration coroutine.
 */
class MetricsCollector(private val clock: () -> Long = { System.nanoTime() / 1_000_000 }) {
    private var speechEndAt: Long? = null
    private var firstTokenAt: Long? = null
    private var firstAudioAt: Long? = null
    private var tokenCount = 0

    fun onSpeechEnd() { speechEndAt = clock() }
    fun onFirstToken() { if (firstTokenAt == null) firstTokenAt = clock() }
    fun onFirstAudio() { if (firstAudioAt == null) firstAudioAt = clock() }
    fun onTokenDecoded() { tokenCount++ }

    fun finish(): TurnMetricsData {
        val end = clock()
        val start = speechEndAt ?: end
        val decodeWindowMs = (end - (firstTokenAt ?: end)).coerceAtLeast(0)
        val tps = if (decodeWindowMs > 0) tokenCount * 1000.0 / decodeWindowMs else 0.0
        return TurnMetricsData(
            ttftMs = firstTokenAt?.let { it - start },
            ttfbMs = firstAudioAt?.let { it - start },
            totalMs = end - start,
            decodeTokensPerSec = tps,
            tokenCount = tokenCount,
        )
    }
}
