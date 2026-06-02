package com.example.nova.shared.model

import android.content.Context
import java.io.File

/** SharedPreferences-backed [KeyValueStore]. */
class SharedPrefsKeyValueStore(context: Context) : KeyValueStore {
    private val prefs = context.getSharedPreferences("nova_models", Context.MODE_PRIVATE)
    override fun getString(key: String, default: String): String = prefs.getString(key, default) ?: default
    override fun putString(key: String, value: String) { prefs.edit().putString(key, value).apply() }
}

/** Build a [ModelManager] backed by the app's external files dir + SharedPreferences. */
fun modelManager(context: Context): ModelManager {
    val filesDir: File = context.getExternalFilesDir(null)
        ?: throw IllegalStateException("External files dir unavailable")
    return ModelManager(filesDir, SharedPrefsKeyValueStore(context))
}
