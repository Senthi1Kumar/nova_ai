package com.example.nova.shared.voice

import org.junit.Assert.assertEquals
import org.junit.Test

class VoiceStateTest {
    @Test
    fun `listening event maps to Listening state`() {
        assertEquals(VoiceState.LISTENING, VoiceState.fromEvent(VoiceEvent.Listening))
    }

    @Test
    fun `speech end maps to Thinking state`() {
        assertEquals(VoiceState.THINKING, VoiceState.fromEvent(VoiceEvent.SpeechEnd))
    }

    @Test
    fun `audio chunk maps to Speaking state`() {
        val ev = VoiceEvent.AudioChunk(ByteArray(0), sampleRate = 24000)
        assertEquals(VoiceState.SPEAKING, VoiceState.fromEvent(ev))
    }

    @Test
    fun `done maps to Idle state`() {
        assertEquals(VoiceState.IDLE, VoiceState.fromEvent(VoiceEvent.Done))
    }
}
