package com.example.nova.shared.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelCatalogTest {
    @Test fun gemmaOptions_areE2bThenE4b() {
        val ids = ModelCatalog.gemmaOptions().map { it.id }
        assertEquals(listOf("gemma-e2b", "gemma-e4b"), ids)
    }

    @Test fun e2b_hasVerifiedUrlAndSize() {
        val a = ModelCatalog.GEMMA_E2B
        assertEquals("gemma-4-E2B-it.litertlm", a.fileName)
        assertEquals(2_588_147_712L, a.sizeBytes)
        assertTrue(a.url.startsWith("https://huggingface.co/litert-community/"))
        assertEquals(ArchiveType.NONE, a.archive)
    }

    @Test fun kokoro_isTarBz2() {
        val k = ModelCatalog.KOKORO
        assertEquals(ArchiveType.TAR_BZ2, k.archive)
        assertEquals(319_625_534L, k.sizeBytes)
        assertEquals("kokoro-en-v0_19.tar.bz2", k.fileName)
    }

    @Test fun byId_returnsNullForUnknown() {
        assertEquals(null, ModelCatalog.byId("nope"))
    }
}
