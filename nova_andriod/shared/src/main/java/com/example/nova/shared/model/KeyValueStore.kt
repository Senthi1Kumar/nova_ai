package com.example.nova.shared.model

/**
 * Minimal persistence abstraction so [ModelManager] logic stays free of Android's
 * SharedPreferences and is unit-testable on the JVM.
 */
interface KeyValueStore {
    fun getString(key: String, default: String): String
    fun putString(key: String, value: String)
}
