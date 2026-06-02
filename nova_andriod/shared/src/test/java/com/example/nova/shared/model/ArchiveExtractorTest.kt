package com.example.nova.shared.model

import org.apache.commons.compress.archivers.tar.TarArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ArchiveExtractorTest {
    @get:Rule val tmp = TemporaryFolder()

    private fun writeEntry(tar: TarArchiveOutputStream, name: String, body: ByteArray) {
        val e = TarArchiveEntry(name)
        e.size = body.size.toLong()
        tar.putArchiveEntry(e)
        tar.write(body)
        tar.closeArchiveEntry()
    }

    private fun makeTarBz2(file: File, entries: Map<String, ByteArray>) {
        BZip2CompressorOutputStream(file.outputStream()).use { bz ->
            TarArchiveOutputStream(bz).use { tar ->
                tar.setLongFileMode(TarArchiveOutputStream.LONGFILE_POSIX)
                entries.forEach { (n, b) -> writeEntry(tar, n, b) }
            }
        }
    }

    @Test fun extracts_nestedFiles() {
        val archive = tmp.newFile("a.tar.bz2")
        makeTarBz2(archive, mapOf(
            "model.onnx" to "weights".toByteArray(),
            "espeak-ng-data/phontab" to "ph".toByteArray(),
        ))
        val dest = tmp.newFolder("out")
        ArchiveExtractor.extractTarBz2(archive, dest)
        assertEquals("weights", File(dest, "model.onnx").readText())
        assertTrue(File(dest, "espeak-ng-data/phontab").isFile)
    }

    @Test fun rejects_pathTraversal() {
        val archive = tmp.newFile("evil.tar.bz2")
        makeTarBz2(archive, mapOf("../escape.txt" to "x".toByteArray()))
        val dest = tmp.newFolder("out")
        assertThrows(SecurityException::class.java) {
            ArchiveExtractor.extractTarBz2(archive, dest)
        }
        assertFalse(File(dest.parentFile, "escape.txt").exists())
    }
}
