package com.example.nova.shared.voice

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Owns the LiteRT-LM [Engine] + a single Conversation for gemma-4-E2B.
 * [warmUp] loads the model (GPU weights, CPU audio) once; reply* methods stream
 * tokens as a [Flow] of text. Not thread-safe; confine to one orchestration scope.
 */
class LlmEngine(
    private val modelPath: String,
    private val cacheDir: String,
) {
    private var engine: Engine? = null
    private var conversation: com.google.ai.edge.litertlm.Conversation? = null

    /** Which backend the engine actually loaded on, after [warmUp]. For UI/metrics. */
    var activeBackend: String = "none"
        private set

    /**
     * Try the [preferGpu] backend first; on engine-init failure fall back to CPU.
     * Some SoCs (e.g. Samsung Xclipse / Exynos) lack OpenCL and LiteRT's OpenGL
     * GPU delegate is unimplemented there, so GPU init throws — CPU still works.
     */
    suspend fun warmUp(preferGpu: Boolean = true, enableMtp: Boolean = false) = withContext(Dispatchers.Default) {
        if (engine != null) return@withContext
        val backends = if (preferGpu) listOf("GPU", "CPU") else listOf("CPU")
        var lastError: Throwable? = null
        for (name in backends) {
            try {
                // MTP / speculative decoding. On the Xclipse 530 GPU this produced
                // repeating multilingual gibberish (broken verify numerics + missing
                // GPU sampler lib), so it's OFF by default while we isolate the cause.
                @OptIn(ExperimentalApi::class)
                ExperimentalFlags.enableSpeculativeDecoding = (enableMtp && name == "GPU")
                val e = Engine(
                    EngineConfig(
                        modelPath = modelPath,
                        backend = if (name == "GPU") Backend.GPU() else Backend.CPU(),
                        audioBackend = Backend.CPU(),
                        cacheDir = cacheDir,
                    ),
                )
                e.initialize()
                engine = e
                conversation = e.createConversation(
                    ConversationConfig(
                        systemInstruction = Contents.of(SYSTEM_PROMPT),
                        samplerConfig = SamplerConfig(topK = 64, topP = 0.95, temperature = 1.0),
                    ),
                )
                activeBackend = name
                return@withContext
            } catch (t: Throwable) {
                lastError = t
                try { engine?.close() } catch (_: Throwable) {}
                engine = null
            }
        }
        throw IllegalStateException("Engine init failed on all backends ($backends)", lastError)
    }

    fun replyToText(prompt: String): Flow<String> {
        val convo = conversation ?: error("warmUp() must complete before replyToText()")
        return convo.sendMessageAsync(Contents.of(Content.Text(prompt)))
            .map { it.toString() }
    }

    /**
     * Stream a reply to a spoken utterance. [pcm] is mono PCM16 at [sampleRate]
     * (the capture rate). Audio is wrapped as WAV and placed before the text
     * instruction, per the gemma-4 modality-order guidance.
     */
    fun replyToAudio(pcm: ShortArray, sampleRate: Int = SAMPLE_RATE_CAPTURE): Flow<String> {
        val convo = conversation ?: error("warmUp() must complete before replyToAudio()")
        val wav = pcmToWav(pcm, sampleRate)
        return convo.sendMessageAsync(
            Contents.of(
                Content.AudioBytes(wav),
                Content.Text("Respond to what the user just said."),
            ),
        ).map { it.toString() }
    }

    fun close() {
        conversation?.close()
        engine?.close()
        conversation = null
        engine = null
    }

    companion object {
        const val MODEL_FILE = "gemma-4-E2B-it.litertlm"
        private const val SYSTEM_PROMPT =
            "You are Nova, a concise on-device voice assistant. Keep replies short and spoken-friendly."

        fun defaultModelPath(context: Context): String =
            com.example.nova.shared.model.modelManager(context).gemmaModelPath()
    }
}
