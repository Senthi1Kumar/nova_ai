package com.example.nova.shared.voice

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlin.coroutines.coroutineContext

/**
 * Captures mono PCM16 from the mic as a cold [Flow] of frames at [sampleRate].
 * Collecting the flow starts recording; cancelling collection stops and
 * releases the [AudioRecord]. Emits the [level] (RMS) as it records.
 *
 * The caller MUST hold RECORD_AUDIO permission before collecting.
 */
class AudioCapture(private val sampleRate: Int = SAMPLE_RATE_CAPTURE) {

    private val _level = MutableStateFlow(0f)
    val level: StateFlow<Float> = _level.asStateFlow()

    @SuppressLint("MissingPermission") // caller guarantees permission; see MainActivity
    fun frames(): Flow<ShortArray> = flow {
        val minBytes = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(MIN_FRAME_BYTES)

        val record = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .build(),
            )
            .setBufferSizeInBytes(minBytes)
            .build()

        try {
            record.startRecording()
            val frame = ShortArray(minBytes / 2)
            while (coroutineContext.isActive) {
                val n = record.read(frame, 0, frame.size)
                if (n > 0) {
                    val out = frame.copyOf(n)
                    _level.value = rms(out)
                    emit(out)
                }
            }
        } finally {
            try { record.stop() } catch (_: IllegalStateException) {}
            record.release()
            _level.value = 0f
        }
    }.flowOn(Dispatchers.IO)

    private companion object {
        const val MIN_FRAME_BYTES = 2048
    }
}
