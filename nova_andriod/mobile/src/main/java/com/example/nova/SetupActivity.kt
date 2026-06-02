package com.example.nova

import android.content.Intent
import android.os.Bundle
import android.os.StatFs
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.ProgressBar
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.nova.shared.model.DownloadMath
import com.example.nova.shared.model.ModelCatalog
import com.example.nova.shared.model.modelManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/** First-run gate: routes straight to MainActivity if models are ready, else downloads them. */
class SetupActivity : AppCompatActivity() {

    private val manager by lazy { modelManager(this) }
    private val downloader by lazy { ModelDownloader(this) }

    /** Reentrancy guard: a tapped Download enqueues instantly but downloads run for
     *  minutes; without this a second tap (or an Activity re-create) enqueues a second
     *  pair, and DownloadManager writes "-1" suffixed duplicate files that verification
     *  can't see. One in-flight download at a time. */
    private var downloading = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (manager.isReady()) { goToMain(); return }
        setContentView(R.layout.activity_setup)

        val group = findViewById<RadioGroup>(R.id.modelGroup)
        val wifiOnly = findViewById<CheckBox>(R.id.wifiOnly)
        val button = findViewById<Button>(R.id.downloadButton)
        val progress = findViewById<ProgressBar>(R.id.progress)
        val label = findViewById<TextView>(R.id.progressLabel)

        button.setOnClickListener {
            if (downloading) return@setOnClickListener

            val gemma = if (group.checkedRadioButtonId == R.id.optE4b)
                ModelCatalog.GEMMA_E4B else ModelCatalog.GEMMA_E2B

            // Pre-flight free space (~gemma + 2x kokoro + 500MB margin).
            val required = DownloadMath.requiredSpaceBytes(gemma, ModelCatalog.KOKORO, 500L * 1_000_000)
            val available = StatFs(getExternalFilesDir(null)!!.path).availableBytes
            if (!DownloadMath.hasEnoughSpace(available, required)) {
                Toast.makeText(this,
                    "Need %.1f GB free, have %.1f GB".format(required / 1e9, available / 1e9),
                    Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }

            downloading = true
            manager.selectedGemmaId = gemma.id
            button.isEnabled = false
            group.isEnabled = false
            progress.visibility = View.VISIBLE

            // Clean slate so DownloadManager never produces "-1" suffixed duplicates of a
            // prior partial/aborted attempt. Safe: we only reach here when models aren't ready.
            val assets = listOf(gemma, ModelCatalog.KOKORO)
            val dir = getExternalFilesDir(null)
            assets.forEach { File(dir, it.fileName).delete() }

            val ids = assets.map { downloader.enqueue(it, wifiOnly.isChecked) }

            lifecycleScope.launch {
                var failed = false
                downloader.progress(ids).collect { ui ->
                    progress.progress = ui.percent
                    label.text = ui.label
                    if (ui.failedReason != null) failed = true
                }
                if (failed) {
                    downloading = false
                    button.isEnabled = true
                    progress.visibility = View.GONE
                    Toast.makeText(this@SetupActivity, "Download failed — tap Download to retry", Toast.LENGTH_LONG).show()
                    return@launch
                }
                // Downloads complete → verify + extract off the main thread.
                label.text = "Verifying…"
                val ok = withContext(Dispatchers.IO) { assets.all { downloader.postProcess(it) } }
                if (ok && manager.isReady()) {
                    goToMain()
                } else {
                    downloading = false
                    button.isEnabled = true
                    progress.visibility = View.GONE
                    Toast.makeText(this@SetupActivity, "Verification failed — tap Download to retry", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun goToMain() {
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }
}
