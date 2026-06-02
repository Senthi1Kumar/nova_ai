package com.example.nova.shared.model

import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import java.io.BufferedInputStream
import java.io.File

/** Extracts a .tar.bz2 into [destDir], guarding against path-traversal entries. */
object ArchiveExtractor {
    fun extractTarBz2(archive: File, destDir: File) {
        destDir.mkdirs()
        val canonicalDest = destDir.canonicalFile
        TarArchiveInputStream(BZip2CompressorInputStream(BufferedInputStream(archive.inputStream()))).use { tar ->
            var entry = tar.nextEntry
            while (entry != null) {
                val outFile = File(destDir, entry.name)
                if (!outFile.canonicalFile.toPath().startsWith(canonicalDest.toPath())) {
                    throw SecurityException("Blocked path-traversal entry: ${entry.name}")
                }
                if (entry.isDirectory) {
                    outFile.mkdirs()
                } else {
                    outFile.parentFile?.mkdirs()
                    outFile.outputStream().use { tar.copyTo(it) }
                }
                entry = tar.nextEntry
            }
        }
    }
}
