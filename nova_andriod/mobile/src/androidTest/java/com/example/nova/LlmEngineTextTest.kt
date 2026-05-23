package com.example.nova

import androidx.test.platform.app.InstrumentationRegistry
import com.example.nova.shared.voice.LlmEngine
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

class LlmEngineTextTest {
    @Test
    fun warmsAndStreamsTextReply() = runBlocking {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val modelPath = LlmEngine.defaultModelPath(ctx)
        assumeTrue("Model not pushed to $modelPath — see Task 4 Step 1", File(modelPath).exists())

        val engine = LlmEngine(modelPath = modelPath, cacheDir = ctx.cacheDir.absolutePath)
        try {
            engine.warmUp()
            val chunks = engine.replyToText("Say hello in one short sentence.").toList()
            val full = chunks.joinToString("")
            android.util.Log.i("NovaTest", "Gemma text reply: $full")
            assertTrue("Expected non-empty streamed reply", full.isNotBlank())
            assertTrue("Reply should contain letters, not object dumps", full.any { it.isLetter() })
        } finally {
            engine.close()
        }
    }
}
