package com.example.nova.shared.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DownloadMathTest {
    @Test fun aggregatePercent_sumsAcrossFiles() {
        // 50/100 and 50/100 -> 50%
        assertEquals(50, DownloadMath.aggregatePercent(listOf(50L to 100L, 50L to 100L)))
    }

    @Test fun aggregatePercent_zeroWhenNoTotalsKnown() {
        assertEquals(0, DownloadMath.aggregatePercent(listOf(0L to 0L)))
    }

    @Test fun isSizeValid_exactMatch() {
        assertTrue(DownloadMath.isSizeValid(actual = 100L, expected = 100L))
        assertFalse(DownloadMath.isSizeValid(actual = 99L, expected = 100L))
    }

    @Test fun requiredSpace_countsKokoroTwice() {
        // gemma (no archive) once + kokoro (archive) twice + margin
        val req = DownloadMath.requiredSpaceBytes(
            gemma = ModelCatalog.GEMMA_E2B,
            kokoro = ModelCatalog.KOKORO,
            marginBytes = 500L,
        )
        assertEquals(
            ModelCatalog.GEMMA_E2B.sizeBytes + 2 * ModelCatalog.KOKORO.sizeBytes + 500L,
            req,
        )
    }

    @Test fun hasEnoughSpace_comparesAvailable() {
        assertTrue(DownloadMath.hasEnoughSpace(available = 1000L, required = 999L))
        assertFalse(DownloadMath.hasEnoughSpace(available = 998L, required = 999L))
    }
}
