[app]
# Name of your application
title = Krdoku

# Package name (must be lowercase, no spaces, no special characters)
package.name = krdoku

# Your domain (reverse domain style, e.g., com.example)
package.domain = org.test

# Directory containing your application code
source.dir = .

# File extensions to include in the APK
source.include_exts = py,png,jpg,tflite,mp3,kv,atlas,txt,dm

# Python modules your app depends on
requirements = python3,kivy==2.3.1,numpy,pillow,opencv_extras,opencv,filetype,android,jnius,plyer

# Application version
version = 1.0

# Screen orientation
orientation = portrait

# Make the app fullscreen
fullscreen = 1

# Permissions (add more if needed)
android.permissions = CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE

# App icon and splash screen
icon.filename = %(source.dir)s/icon.png
presplash.filename = %(source.dir)s/splash.png

# Scale the splash screen
android.presplash_scale = 1

p4a.bootstrap = sdl2

android.ndk_api = 26
android.minapi = 26
android.api = 31



# Pokud potřebuješ kopírovat dodatečné knihovny do adresáře libs/armeabi, přidej je zde:

android.add_libs_arm64_v8a = %(source.dir)s/sdk/native/libs/arm64-v8a/libopencv_java4.so
android.add_libs_armeabi_v7a = %(source.dir)s/sdk/native/libs/armeabi-v7a/libopencv_java4.so

android.repositories = 
    "mavenCentral()"
    "google()"


android.gradle_dependencies = org.tensorflow:tensorflow-lite:2.15.0, org.tensorflow:tensorflow-lite-support:0.2.0



[buildozer]
# Logging level
log_level = 2

# Target Android API level
android.api = 31



# Packaging format (APK nebo AAB)
android.packaging_format = apk

# Additional Android architecture targets
android.archs = arm64-v8a, armeabi-v7a
