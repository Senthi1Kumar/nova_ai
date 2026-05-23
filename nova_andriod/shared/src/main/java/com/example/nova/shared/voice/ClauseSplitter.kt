package com.example.nova.shared.voice

/**
 * Buffers streamed tokens and flushes speakable clauses. Kotlin port of the
 * webui app/clause_splitter.py.
 *
 * Rules:
 *  - Terminal punctuation (. ! ?) flushes once the buffer is >= [minChars].
 *  - A trailing comma flushes once the buffer is >= [commaMinChars].
 *  - Otherwise returns null and keeps accumulating.
 *
 * Not thread-safe; confined to the orchestration coroutine.
 */
class ClauseSplitter(
    private val minChars: Int = 8,
    private val commaMinChars: Int = 32,
) {
    private val buf = StringBuilder()

    /** Append [token]; return a clause if one should flush now, else null. */
    fun push(token: String): String? {
        buf.append(token)
        if (buf.isEmpty()) return null
        val joined = buf.toString()
        val last = joined.last()
        val flush = (last in TERMINAL && joined.length >= minChars) ||
            (last == ',' && joined.length >= commaMinChars)
        return if (flush) { buf.setLength(0); joined } else null
    }

    /** Flush whatever remains (end of turn); null if empty. */
    fun flushRemaining(): String? {
        if (buf.isEmpty()) return null
        val joined = buf.toString()
        buf.setLength(0)
        return joined
    }

    private companion object {
        val TERMINAL = setOf('.', '!', '?')
    }
}
