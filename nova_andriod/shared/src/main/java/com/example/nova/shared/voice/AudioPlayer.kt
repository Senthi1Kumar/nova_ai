package com.example.nova.shared.voice

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Streaming PCM16 playback via [AudioTrack] at [sampleRate].
 * Lifecycle: [start] once, [write] chunks (blocking; call off the main thread),
 * then [stop] to halt and release. [stop] is immediate — used for barge-in in
 * later plans. For a graceful end-of-stream, the caller waits out the audio
 * duration before calling [stop] (see MainActivity loopback).
 */
class AudioPlayer(private val sampleRate: Int = SAMPLE_RATE_TTS) {

    private val _level = MutableStateFlow(0f)
    val level: StateFlow<Float> = _level.asStateFlow()

    private var track: AudioTrack? = null

    fun start() {
        if (track != null) return
        val minBytes = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(MIN_BUFFER_BYTES)

        track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            )
            .setBufferSizeInBytes(minBytes)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
            .also { it.play() }
    }

    /** Blocking write of one PCM chunk. Call from a background coroutine. */
    fun write(pcm: ShortArray) {
        val t = track ?: return
        _level.value = rms(pcm)
        var off = 0
        while (off < pcm.size) {
            val n = t.write(pcm, off, pcm.size - off, AudioTrack.WRITE_BLOCKING)
            if (n < 0) break
            off += n
        }
    }

    /** Immediate stop (barge-in) + release. */
    fun stop() {
        track?.run {
            try { pause(); flush(); stop() } catch (_: IllegalStateException) {}
            release()
        }
        track = null
        _level.value = 0f
    }

    private companion object {
        const val MIN_BUFFER_BYTES = 4096
    }
}
