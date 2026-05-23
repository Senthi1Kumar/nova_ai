package com.example.nova.shared.voice

import org.junit.Assert.assertEquals
import org.junit.Test

class AudioMathTest {
    @Test
    fun `rms of empty array is zero`() {
        assertEquals(0f, rms(shortArrayOf()), 0f)
    }

    @Test
    fun `rms of silence is zero`() {
        assertEquals(0f, rms(ShortArray(128)), 0f)
    }

    @Test
    fun `rms of half-scale square wave is one half`() {
        assertEquals(0.5f, rms(shortArrayOf(16384, -16384, 16384, -16384)), 1e-3f)
    }

    @Test
    fun `rms honors count parameter`() {
        assertEquals(0.5f, rms(shortArrayOf(16384, 9999, 12345), count = 1), 1e-3f)
    }

    @Test
    fun `floatToPcm16 maps full scale and clamps`() {
        val pcm = floatToPcm16(floatArrayOf(0f, 1f, -1f, 2f, -2f, 0.5f))
        assertEquals(0, pcm[0].toInt())
        assertEquals(32767, pcm[1].toInt())
        assertEquals(-32767, pcm[2].toInt())
        assertEquals(32767, pcm[3].toInt())
        assertEquals(-32767, pcm[4].toInt())
        assertEquals(16383, pcm[5].toInt())
    }

    @Test
    fun `floatToPcm16 empty is empty`() {
        assertEquals(0, floatToPcm16(floatArrayOf()).size)
    }
}
