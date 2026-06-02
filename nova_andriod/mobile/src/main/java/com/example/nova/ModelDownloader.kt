package com.example.nova

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import com.example.nova.shared.model.ArchiveExtractor
import com.example.nova.shared.model.ArchiveType
import com.example.nova.shared.model.DownloadMath
import com.example.nova.shared.model.ModelAsset
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.File

data class DownloadUi(val percent: Int, val label: String, val failedReason: Int? = null)

/** Thin wrapper over Android's DownloadManager for the model assets. */
class ModelDownloader(private val context: Context) {

    private val dm = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager

    /** Enqueue one asset; returns the DownloadManager id. */
    fun enqueue(asset: ModelAsset, wifiOnly: Boolean): Long {
        val req = DownloadManager.Request(Uri.parse(asset.url))
            .setTitle(asset.displayName)
            .setDescription("Nova model download")
            .setAllowedOverMetered(!wifiOnly)
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
            .setDestinationInExternalFilesDir(context, null, asset.fileName)
        return dm.enqueue(req)
    }

    /**
     * Emit aggregate progress for [ids] until all reach SUCCESSFUL (emits percent 100)
     * or any reaches FAILED (emits failedReason). Polls ~2x/sec.
     */
    fun progress(ids: List<Long>): Flow<DownloadUi> = flow {
        while (true) {
            val per = ArrayList<Pair<Long, Long>>()
            var allDone = true
            var currentLabel = "Starting…"
            dm.query(DownloadManager.Query().setFilterById(*ids.toLongArray())).use { c ->
                while (c.moveToNext()) {
                    val status = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                    val done = c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                    val total = c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
                    val title = c.getString(c.getColumnIndexOrThrow(DownloadManager.COLUMN_TITLE)) ?: ""
                    per.add(done to total)
                    if (status == DownloadManager.STATUS_FAILED) {
                        val reason = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON))
                        emit(DownloadUi(percent = 0, label = "Download failed", failedReason = reason))
                        return@flow
                    }
                    if (status != DownloadManager.STATUS_SUCCESSFUL) {
                        allDone = false
                        currentLabel = "Downloading $title  ${done / 1_000_000} / ${total / 1_000_000} MB"
                    }
                }
            }
            emit(DownloadUi(percent = DownloadMath.aggregatePercent(per), label = if (allDone) "Verifying…" else currentLabel))
            if (allDone) return@flow
            delay(500)
        }
    }

    /** After SUCCESSFUL: verify size; extract Kokoro archive into kokoro/, then delete archive. */
    fun postProcess(asset: ModelAsset): Boolean {
        val filesDir = context.getExternalFilesDir(null) ?: return false
        val file = File(filesDir, asset.fileName)
        if (!DownloadMath.isSizeValid(file.length(), asset.sizeBytes)) {
            file.delete(); return false
        }
        if (asset.archive == ArchiveType.TAR_BZ2) {
            val dest = File(filesDir, "kokoro")
            return try {
                ArchiveExtractor.extractTarBz2(file, dest)
                file.delete()
                true
            } catch (t: Throwable) {
                dest.deleteRecursively(); false
            }
        }
        return true
    }
}
