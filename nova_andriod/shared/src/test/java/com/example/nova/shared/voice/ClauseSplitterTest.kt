package com.example.nova.shared.voice

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ClauseSplitterTest {
    @Test
    fun `flushes on terminal punct when long enough`() {
        val s = ClauseSplitter()
        assertNull(s.push("Hello there"))     // 11 chars, no terminal punct
        assertEquals("Hello there.", s.push("."))
    }

    @Test
    fun `does not flush short terminal clause`() {
        val s = ClauseSplitter(minChars = 8)
        assertNull(s.push("Hi."))             // 3 chars < 8
    }

    @Test
    fun `comma flushes only past comma threshold`() {
        val s = ClauseSplitter(commaMinChars = 10)
        assertNull(s.push("short,"))          // 6 < 10
        val s2 = ClauseSplitter(commaMinChars = 10)
        assertEquals("a long enough chunk,", s2.push("a long enough chunk,"))
    }

    @Test
    fun `buffer continues accumulating after no-flush`() {
        val s = ClauseSplitter()
        assertNull(s.push("Part one "))
        assertEquals("Part one and two.", s.push("and two."))
    }

    @Test
    fun `flush returns remaining buffer and empties it`() {
        val s = ClauseSplitter()
        s.push("trailing text")
        assertEquals("trailing text", s.flushRemaining())
        assertNull(s.flushRemaining())
    }
}
