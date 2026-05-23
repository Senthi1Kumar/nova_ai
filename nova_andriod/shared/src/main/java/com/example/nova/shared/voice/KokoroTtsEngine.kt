package com.example.nova.shared.voice

import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig

/**
 * On-device Kokoro TTS via sherpa-onnx OfflineTts. [modelDir] holds the Kokoro
 * bundle: model.onnx, voices.bin, tokens.txt, and espeak-ng-data/ (dataDir).
 */
class KokoroTtsEngine(
    modelDir: String,
    private val speakerId: Int = 0,
    private val speed: Float = 1.0f,
) : TtsEngine {

    private val tts = OfflineTts(
        assetManager = null,
        config = OfflineTtsConfig(
            model = OfflineTtsModelConfig(
                kokoro = OfflineTtsKokoroModelConfig(
                    model = "$modelDir/model.onnx",
                    voices = "$modelDir/voices.bin",
                    tokens = "$modelDir/tokens.txt",
                    dataDir = "$modelDir/espeak-ng-data",
                ),
                numThreads = 2,
                provider = "cpu",
            ),
        ),
    )

    override val sampleRate: Int get() = tts.sampleRate()

    override fun speak(text: String, onPcm: (ShortArray) -> Unit, shouldContinue: () -> Boolean) {
        if (text.isBlank()) return
        // sherpa-onnx convention: callback returns 1 to CONTINUE, 0 to STOP.
        tts.generateWithCallback(text = text, sid = speakerId, speed = speed) { samples ->
            onPcm(floatToPcm16(samples))
            if (shouldContinue()) 1 else 0
        }
    }

    override fun close() {
        tts.free()
    }
}
