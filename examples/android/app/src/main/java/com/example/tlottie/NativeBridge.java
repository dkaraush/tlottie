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
    static native String setSurface(long handle, android.view.Surface surface, int width, int height);
    static native String setThorvgSurface(
            long handle, android.view.Surface surface, int width, int height);
    static native void clearSurface(long handle);
    static native String renderVulkan(
            long handle, float frame, boolean antialias, float curveTolerance);
    static native long lastVulkanGpuNs(long handle);
    static native String renderThorvgGpu(long handle, float frame);
    static native String benchmarkCpu(
            long handle, int warmupFrames, int measuredFrames, int size, boolean antialias);
    static native String benchmarkVulkan(
            long handle, int warmupFrames, int measuredFrames, boolean antialias,
            float curveTolerance);
    static native String benchmarkThorvgGpu(
            long handle, int warmupFrames, int measuredFrames);
}
