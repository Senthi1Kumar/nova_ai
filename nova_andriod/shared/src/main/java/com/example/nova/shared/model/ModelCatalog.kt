package com.example.nova.shared.model

/** How a downloaded asset is packaged on the server. */
enum class ArchiveType { NONE, TAR_BZ2 }

/**
 * One downloadable artifact. [sizeBytes] is the exact server file length (verified
 * 2026-06-01) and is used to decide whether a local copy is complete.
 */
data class ModelAsset(
    val id: String,
    val displayName: String,
    val subtitle: String,
    val url: String,
    val fileName: String,
    val sizeBytes: Long,
    val archive: ArchiveType,
)

/** Single source of truth for the public, no-token model downloads. */
object ModelCatalog {
    val GEMMA_E2B = ModelAsset(
        id = "gemma-e2b",
        displayName = "Gemma-4 E2B",
        subtitle = "2.6 GB · faster (recommended)",
        url = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
        fileName = "gemma-4-E2B-it.litertlm",
        sizeBytes = 2_588_147_712L,
        archive = ArchiveType.NONE,
    )

    val GEMMA_E4B = ModelAsset(
        id = "gemma-e4b",
        displayName = "Gemma-4 E4B",
        subtitle = "3.7 GB · smarter, slower",
        url = "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm",
        fileName = "gemma-4-E4B-it.litertlm",
        sizeBytes = 3_659_530_240L,
        archive = ArchiveType.NONE,
    )

    val KOKORO = ModelAsset(
        id = "kokoro-en",
        displayName = "Kokoro English TTS",
        subtitle = "305 MB",
        url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2",
        fileName = "kokoro-en-v0_19.tar.bz2",
        sizeBytes = 319_625_534L,
        archive = ArchiveType.TAR_BZ2,
    )

    fun gemmaOptions(): List<ModelAsset> = listOf(GEMMA_E2B, GEMMA_E4B)

    fun byId(id: String): ModelAsset? =
        (gemmaOptions() + KOKORO).firstOrNull { it.id == id }
}
