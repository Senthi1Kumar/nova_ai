package com.example.nova.shared.model

/** Pure arithmetic for the downloader so the decisions are unit-testable. */
object DownloadMath {
    /** Overall percent across multiple files: sum(bytes)/sum(totals). 0 if no totals yet. */
    fun aggregatePercent(items: List<Pair<Long, Long>>): Int {
        val total = items.sumOf { it.second }
        if (total <= 0L) return 0
        val done = items.sumOf { it.first }
        return ((done * 100) / total).toInt()
    }

    fun isSizeValid(actual: Long, expected: Long): Boolean = actual == expected

    /** Archive assets need room for the archive AND its extracted copy (≈2×). */
    fun requiredSpaceBytes(gemma: ModelAsset, kokoro: ModelAsset, marginBytes: Long): Long {
        val gemmaNeed = gemma.sizeBytes
        val kokoroNeed = if (kokoro.archive == ArchiveType.NONE) kokoro.sizeBytes else 2 * kokoro.sizeBytes
        return gemmaNeed + kokoroNeed + marginBytes
    }

    fun hasEnoughSpace(available: Long, required: Long): Boolean = available >= required
}
