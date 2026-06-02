plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.example.nova.shared"
    compileSdk = 36

    defaultConfig {
        minSdk = 28
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.litertlm.android)
    // sherpa-onnx v1.13.2 prebuilt AAR (includes JNI libs + compiled Kotlin API classes)
    // Source: https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.2/sherpa-onnx-1.13.2.aar
    implementation(files("libs/sherpa-onnx-1.13.2.aar"))
    implementation(libs.commons.compress)

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}
