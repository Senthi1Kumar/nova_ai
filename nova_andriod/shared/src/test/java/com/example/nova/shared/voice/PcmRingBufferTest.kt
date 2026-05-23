package com.example.nova.shared.voice

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class PcmRingBufferTest {
    @Test
    fun `writes under capacity read back in order`() {
        val buf = PcmRingBuffer(capacity = 4)
        buf.write(shortArrayOf(1, 2, 3))
        assertEquals(3, buf.size())
        assertArrayEquals(shortArrayOf(1, 2, 3), buf.toShortArray())
    }

    @Test
    fun `overflow drops oldest samples`() {
        val buf = PcmRingBuffer(capacity = 3)
        buf.write(shortArrayOf(1, 2, 3, 4, 5))
        assertEquals(3, buf.size())
        assertArrayEquals(shortArrayOf(3, 4, 5), buf.toShortArray())
    }

    @Test
    fun `clear resets to empty`() {
        val buf = PcmRingBuffer(capacity = 3)
        buf.write(shortArrayOf(1, 2))
        buf.clear()
        assertEquals(0, buf.size())
        assertArrayEquals(shortArrayOf(), buf.toShortArray())
    }

    @Test
    fun `write longer than capacity keeps only the last capacity samples`() {
        val buf = PcmRingBuffer(capacity = 2)
        buf.write(shortArrayOf(9, 8, 7, 6))
        assertArrayEquals(shortArrayOf(7, 6), buf.toShortArray())
    }
}
