package com.example.nova

import androidx.test.platform.app.InstrumentationRegistry
import com.example.nova.shared.voice.LlmEngine
import com.example.nova.shared.voice.SAMPLE_RATE_CAPTURE
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class LlmEngineAudioTest {
    @Test
    fun streamsReplyToSpokenAudio() = runBlocking {
        val inst = InstrumentationRegistry.getInstrumentation()
        val ctx = inst.targetContext
        val modelPath = LlmEngine.defaultModelPath(ctx)
        assumeTrue("Model not pushed", File(modelPath).exists())

        val wavBytes = inst.context.assets.open("sample_speech.wav").use { it.readBytes() }
        assumeTrue("No sample_speech.wav asset", wavBytes.size > 44)
        val pcmBytes = wavBytes.copyOfRange(44, wavBytes.size)
        val shorts = ShortArray(pcmBytes.size / 2)
        ByteBuffer.wrap(pcmBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts)

        val engine = LlmEngine(modelPath, ctx.cacheDir.absolutePath)
        try {
            engine.warmUp()
            val full = engine.replyToAudio(shorts, SAMPLE_RATE_CAPTURE).toList().joinToString("")
            android.util.Log.i("NovaTest", "Gemma audio reply: $full")
            assertTrue("Expected non-empty reply to audio", full.isNotBlank())
        } finally {
            engine.close()
        }
    }
}
