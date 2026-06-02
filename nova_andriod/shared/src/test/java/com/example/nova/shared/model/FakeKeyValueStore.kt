package com.example.nova.shared.model

class FakeKeyValueStore : KeyValueStore {
    private val map = HashMap<String, String>()
    override fun getString(key: String, default: String): String = map[key] ?: default
    override fun putString(key: String, value: String) { map[key] = value }
}
