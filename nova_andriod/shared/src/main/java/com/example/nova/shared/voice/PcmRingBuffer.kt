package com.example.nova.shared.voice

/**
 * Fixed-capacity ring buffer of 16-bit PCM samples. When more than [capacity]
 * samples are written, the oldest are overwritten. Not thread-safe; callers
 * confine it to a single capture coroutine.
 */
class PcmRingBuffer(private val capacity: Int) {
    init { require(capacity > 0) { "capacity must be > 0" } }

    private val data = ShortArray(capacity)
    private var head = 0      // index of oldest sample
    private var count = 0     // number of valid samples

    fun size(): Int = count

    fun clear() { head = 0; count = 0 }

    fun write(samples: ShortArray) {
        for (s in samples) {
            val tail = (head + count) % capacity
            data[tail] = s
            if (count < capacity) count++ else head = (head + 1) % capacity
        }
    }

    fun toShortArray(): ShortArray {
        val out = ShortArray(count)
        for (i in 0 until count) out[i] = data[(head + i) % capacity]
        return out
    }
}
