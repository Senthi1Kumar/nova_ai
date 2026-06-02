package com.example.nova.shared.model

import java.io.File

/**
 * Knows what is installed in [filesDir] and which Gemma the user picked.
 * Install = the target file exists AND its length matches [ModelAsset.sizeBytes]
 * (a half-downloaded multi-GB file must not count as ready). The Kokoro asset is a
 * downloaded archive; "installed" means its EXTRACTED layout is present.
 */
class ModelManager(
    private val filesDir: File,
    private val store: KeyValueStore,
) {
    var selectedGemmaId: String
        get() = store.getString(KEY_SELECTED_GEMMA, ModelCatalog.GEMMA_E2B.id)
        set(value) = store.putString(KEY_SELECTED_GEMMA, value)

    private fun selectedGemmaAsset(): ModelAsset =
        ModelCatalog.byId(selectedGemmaId) ?: ModelCatalog.GEMMA_E2B

    fun gemmaModelFile(): File = File(filesDir, selectedGemmaAsset().fileName)
    fun gemmaModelPath(): String = gemmaModelFile().absolutePath
    fun kokoroDir(): File = File(filesDir, "kokoro")

    fun isInstalled(asset: ModelAsset): Boolean = when (asset.archive) {
        ArchiveType.NONE -> {
            val f = File(filesDir, asset.fileName)
            f.isFile && f.length() == asset.sizeBytes
        }
        ArchiveType.TAR_BZ2 -> {
            // Kokoro: extracted bundle, not the archive.
            val k = kokoroDir()
            File(k, "model.onnx").isFile &&
                File(k, "voices.bin").isFile &&
                File(k, "tokens.txt").isFile &&
                File(k, "espeak-ng-data").isDirectory
        }
    }

    fun isReady(): Boolean =
        isInstalled(selectedGemmaAsset()) && isInstalled(ModelCatalog.KOKORO)

    private companion object {
        const val KEY_SELECTED_GEMMA = "selected_gemma_id"
    }
}
