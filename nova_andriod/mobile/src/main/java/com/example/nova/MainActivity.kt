package com.example.nova

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.example.nova.shared.voice.AudioCapture
import com.example.nova.shared.voice.LlmEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private val capture = AudioCapture()
    private val recorded = ArrayList<Short>()
    private var captureJob: Job? = null
    private lateinit var engine: LlmEngine
    private var ready = false
    private val player = com.example.nova.shared.voice.AudioPlayer(sampleRate = com.example.nova.shared.voice.SAMPLE_RATE_TTS)
    private var tts: com.example.nova.shared.voice.KokoroTtsEngine? = null
    private var speakJob: Job? = null

    @Suppress("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQ_MIC)

        val status = findViewById<TextView>(R.id.statusText)
        val reply = findViewById<TextView>(R.id.replyText)
        val button = findViewById<Button>(R.id.talkButton)

        val modelPath = LlmEngine.defaultModelPath(this)
        engine = LlmEngine(modelPath, cacheDir.absolutePath)

        val ttsDir = java.io.File(getExternalFilesDir(null), "kokoro").absolutePath
        if (java.io.File(ttsDir, "model.onnx").exists()) {
            tts = com.example.nova.shared.voice.KokoroTtsEngine(ttsDir)
        }

        if (!File(modelPath).exists()) {
            status.text = "Model missing — adb push to\n$modelPath"
        } else {
            status.text = "Loading model…"
            lifecycleScope.launch {
                try {
                    // Xclipse 530 GPU produces gibberish in LiteRT-LM 0.12.0 (broken
                    // OpenCL numerics), so force CPU on this device. Flip to true on
                    // SoCs with a correct GPU path (e.g. Snapdragon 8-gen + GPU sampler).
                    withContext(Dispatchers.Default) { engine.warmUp(preferGpu = false) }
                    ready = true
                    status.text = "Ready (${engine.activeBackend}). Hold to talk."
                } catch (t: Throwable) {
                    status.text = "Model load failed: ${t.message}"
                }
            }
        }

        button.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> { startCapture(status); true }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> { stopAndAsk(status, reply); true }
                else -> false
            }
        }
    }

    private fun hasMic() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    private fun startCapture(status: TextView) {
        if (!ready) { Toast.makeText(this, "Model still loading", Toast.LENGTH_SHORT).show(); return }
        if (!hasMic()) { Toast.makeText(this, "Mic permission needed", Toast.LENGTH_SHORT).show(); return }
        status.text = "Listening…"
        synchronized(recorded) { recorded.clear() }
        captureJob = lifecycleScope.launch(Dispatchers.IO) {
            capture.frames().collect { frame -> synchronized(recorded) { for (s in frame) recorded.add(s) } }
        }
    }

    private fun stopAndAsk(status: TextView, reply: TextView) {
        captureJob?.cancel(); captureJob = null
        speakJob?.cancel(); speakJob = null
        val pcm = synchronized(recorded) { recorded.toShortArray() }
        if (pcm.isEmpty()) { status.text = "Ready. Hold to talk."; return }
        status.text = "Thinking…"
        reply.text = ""
        val theTts = tts
        lifecycleScope.launch {
            try {
                val splitter = com.example.nova.shared.voice.ClauseSplitter()
                val clauses = kotlinx.coroutines.channels.Channel<String>(kotlinx.coroutines.channels.Channel.UNLIMITED)
                if (theTts != null) {
                    player.start()
                    speakJob = launch(Dispatchers.IO) {
                        for (clause in clauses) theTts.speak(clause, onPcm = { player.write(it) })
                    }
                }
                engine.replyToAudio(pcm).collect { tok ->
                    reply.append(tok)
                    splitter.push(tok)?.let { clauses.trySend(it) }
                }
                splitter.flushRemaining()?.let { clauses.trySend(it) }
                clauses.close()
                speakJob?.join()
                status.text = "Ready. Hold to talk."
            } catch (t: Throwable) {
                status.text = "Error: ${t.message}"
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        player.stop()
        tts?.close()
        engine.close()
    }

    private companion object {
        const val REQ_MIC = 1
    }
}
