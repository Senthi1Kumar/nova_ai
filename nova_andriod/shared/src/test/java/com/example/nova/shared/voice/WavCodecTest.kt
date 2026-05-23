package com.example.nova.shared.voice

import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteOrder

class WavCodecTest {
    @Test
    fun `header has correct size and RIFF or WAVE tags`() {
        val pcm = shortArrayOf(0, 1, -1, 2)            // 4 samples = 8 data bytes
        val wav = pcmToWav(pcm, sampleRate = 16000)
        assertEquals(44 + 8, wav.size)
        assertEquals("RIFF", String(wav, 0, 4, Charsets.US_ASCII))
        assertEquals("WAVE", String(wav, 8, 4, Charsets.US_ASCII))
        assertEquals("fmt ", String(wav, 12, 4, Charsets.US_ASCII))
        assertEquals("data", String(wav, 36, 4, Charsets.US_ASCII))
    }

    @Test
    fun `fmt chunk encodes mono 16-bit at given sample rate`() {
        val wav = pcmToWav(shortArrayOf(0), sampleRate = 16000)
        val bb = java.nio.ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(1, bb.getShort(20).toInt())
        assertEquals(1, bb.getShort(22).toInt())
        assertEquals(16000, bb.getInt(24))
        assertEquals(16, bb.getShort(34).toInt())
    }

    @Test
    fun `samples are written little-endian after the header`() {
        val wav = pcmToWav(shortArrayOf(0x0102), sampleRate = 8000)
        assertEquals(0x02.toByte(), wav[44])
        assertEquals(0x01.toByte(), wav[45])
    }
}
