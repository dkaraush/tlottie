package com.example.tlottie;

final class NativeBridge {
    static {
        System.loadLibrary("tlottie_android");
    }

    private NativeBridge() {}

    static native long create(byte[] json);
    static native void destroy(long handle);
    static native int frameCount(long handle);
    static native float frameRate(long handle);
    static native String renderCpu(
            long handle,
            float frame,
            int width,
            int height,
            int[] pixels);
    static native String renderRlottie(
            long handle,
            int variant,
            float frame,
            int width,
            int height,
            int[] pixels);
    static native String renderThorvgCpu(
            long handle,
            float frame,
            int width,
            int height,
            int[] pixels);
    static native String benchmarkCpu(
            long handle, int warmupFrames, int measuredFrames, int size, boolean antialias);
}
