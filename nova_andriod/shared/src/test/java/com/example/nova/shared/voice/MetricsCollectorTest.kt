package com.example.nova.shared.voice

import org.junit.Assert.assertEquals
import org.junit.Test

class MetricsCollectorTest {
    private var now = 0L
    private fun collector() = MetricsCollector(clock = { now })

    @Test
    fun `computes ttft ttfb and total from boundaries`() {
        val c = collector()
        now = 1000; c.onSpeechEnd()
        now = 1300; c.onFirstToken()
        now = 1500; c.onFirstAudio()
        now = 2000; c.onTokenDecoded(); c.onTokenDecoded()  // 2 tokens
        now = 2500
        val m = c.finish()
        assertEquals(300L, m.ttftMs)
        assertEquals(500L, m.ttfbMs)
        assertEquals(1500L, m.totalMs)
    }

    @Test
    fun `decode tokens per second uses token count over decode window`() {
        val c = collector()
        now = 0; c.onSpeechEnd(); c.onFirstToken()
        // 10 tokens across 0..1000ms -> 10 tok/s
        repeat(10) { c.onTokenDecoded() }
        now = 1000
        val m = c.finish()
        assertEquals(10.0, m.decodeTokensPerSec, 0.01)
    }

    @Test
    fun `missing first audio leaves ttfb null`() {
        val c = collector()
        now = 0; c.onSpeechEnd(); c.onFirstToken()
        now = 100
        val m = c.finish()
        assertEquals(null, m.ttfbMs)
    }
}
