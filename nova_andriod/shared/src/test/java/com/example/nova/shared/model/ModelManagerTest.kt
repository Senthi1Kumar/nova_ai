package com.example.nova.shared.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ModelManagerTest {
    @get:Rule val tmp = TemporaryFolder()

    private fun manager(): Pair<ModelManager, File> {
        val dir = tmp.newFolder("files")
        return ModelManager(dir, FakeKeyValueStore()) to dir
    }

    /** Write a file of exactly [len] bytes. */
    private fun writeSized(f: File, len: Long) {
        f.parentFile?.mkdirs()
        f.outputStream().use { out -> repeat((len / 1024).toInt()) { out.write(ByteArray(1024)) }; out.write(ByteArray((len % 1024).toInt())) }
    }

    @Test fun selectedGemma_defaultsToE2b() {
        val (m, _) = manager()
        assertEquals("gemma-e2b", m.selectedGemmaId)
    }

    @Test fun selectedGemma_persists() {
        val (m, _) = manager()
        m.selectedGemmaId = "gemma-e4b"
        assertEquals("gemma-e4b", m.selectedGemmaId)
        assertEquals(ModelCatalog.GEMMA_E4B.fileName, File(m.gemmaModelPath()).name)
    }

    @Test fun gemma_partialFileIsNotInstalled() {
        val (m, dir) = manager()
        writeSized(File(dir, ModelCatalog.GEMMA_E2B.fileName), 100L) // wrong size
        assertFalse(m.isInstalled(ModelCatalog.GEMMA_E2B))
    }

    @Test fun gemma_correctSizeIsInstalled() {
        val (m, dir) = manager()
        writeSized(File(dir, ModelCatalog.GEMMA_E2B.fileName), ModelCatalog.GEMMA_E2B.sizeBytes)
        assertTrue(m.isInstalled(ModelCatalog.GEMMA_E2B))
    }

    @Test fun kokoro_installedRequiresExtractedLayout() {
        val (m, dir) = manager()
        val k = File(dir, "kokoro")
        assertFalse(m.isInstalled(ModelCatalog.KOKORO))
        File(k, "model.onnx").also { it.parentFile?.mkdirs() }.writeText("x")
        File(k, "voices.bin").writeText("x")
        File(k, "tokens.txt").writeText("x")
        assertFalse(m.isInstalled(ModelCatalog.KOKORO)) // espeak-ng-data still missing
        File(k, "espeak-ng-data").mkdirs()
        assertTrue(m.isInstalled(ModelCatalog.KOKORO))
    }

    @Test fun isReady_needsSelectedGemmaAndKokoro() {
        val (m, dir) = manager()
        writeSized(File(dir, ModelCatalog.GEMMA_E2B.fileName), ModelCatalog.GEMMA_E2B.sizeBytes)
        val k = File(dir, "kokoro")
        File(k, "model.onnx").also { it.parentFile?.mkdirs() }.writeText("x")
        File(k, "voices.bin").writeText("x")
        File(k, "tokens.txt").writeText("x")
        File(k, "espeak-ng-data").mkdirs()
        assertTrue(m.isReady())
    }
}
