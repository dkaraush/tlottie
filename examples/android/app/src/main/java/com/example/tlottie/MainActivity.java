package com.example.tlottie;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.util.Log;
import android.view.Choreographer;
import android.view.Gravity;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

public final class MainActivity extends Activity {
    private static final int OPEN_LOTTIE = 7;
    private static final File LEGACY_FULL_FIXTURES = new File("/data/local/tmp/tgs_dump");
    private static final File SMALL_FIXTURES = new File("/data/local/tmp/fx");
    private LottieView lottieView;
    private TextView status;
    private Button backend;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        setShowWhenLocked(true);
        setTurnScreenOn(true);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(16), dp(16), dp(16));
        root.setBackgroundColor(Color.rgb(20, 23, 31));

        TextView title = new TextView(this);
        title.setText("tlottie Android");
        title.setTextColor(Color.WHITE);
        title.setTextSize(24);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(title, new LinearLayout.LayoutParams(-1, -2));

        status = new TextView(this);
        status.setTextColor(Color.rgb(190, 198, 215));
        status.setTextSize(14);
        status.setGravity(Gravity.CENTER_HORIZONTAL);
        status.setPadding(0, dp(4), 0, dp(10));
        root.addView(status, new LinearLayout.LayoutParams(-1, -2));

        lottieView = new LottieView();
        root.addView(lottieView, new LinearLayout.LayoutParams(-1, 0, 1f));

        LinearLayout actions = new LinearLayout(this);
        actions.setGravity(Gravity.CENTER);

        backend = new Button(this);
        backend.setText("tlottie CPU");
        backend.setOnClickListener(this::showBackendMenu);
        actions.addView(backend);

        Button fixtures = new Button(this);
        fixtures.setText("Fixtures");
        fixtures.setOnClickListener(v -> showFixturePacks());
        actions.addView(fixtures);

        Button open = new Button(this);
        open.setText("Open file");
        open.setOnClickListener(v -> openLottie());
        actions.addView(open);
        root.addView(actions, new LinearLayout.LayoutParams(-1, -2));

        LinearLayout sizeControls = new LinearLayout(this);
        sizeControls.setGravity(Gravity.CENTER_VERTICAL);
        TextView sizeLabel = new TextView(this);
        sizeLabel.setTextColor(Color.rgb(190, 198, 215));
        sizeLabel.setText("512 px");
        sizeControls.addView(sizeLabel, new LinearLayout.LayoutParams(dp(72), -2));
        SeekBar size = new SeekBar(this);
        size.setMax((1024 - 64) / 32);
        size.setProgress((512 - 64) / 32);
        size.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar bar, int progress, boolean fromUser) {
                int pixels = 64 + progress * 32;
                sizeLabel.setText(pixels + " px");
                if (fromUser) lottieView.setRenderSize(pixels);
            }

            @Override public void onStartTrackingTouch(SeekBar bar) {}
            @Override public void onStopTrackingTouch(SeekBar bar) {}
        });
        sizeControls.addView(size, new LinearLayout.LayoutParams(0, -2, 1f));
        root.addView(sizeControls, new LinearLayout.LayoutParams(-1, -2));

        setContentView(root);
        loadBundledSample();
        startAutomatedBenchmark(getIntent());
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void showBackendMenu(View anchor) {
        String[] names = {
                "tlottie CPU", "tlottie Vulkan", "rlottie",
                "rlottie 2019", "rlottie 2019 patched", "ThorVG CPU", "ThorVG GPU (GL)"
        };
        PopupMenu menu = new PopupMenu(this, anchor);
        for (int index = 0; index < names.length; index++) {
            menu.getMenu().add(0, index, index, names[index]);
        }
        menu.setOnMenuItemClickListener(item -> {
            int selected = item.getItemId();
            if (lottieView.setBackend(selected)) backend.setText(names[selected]);
            return true;
        });
        menu.show();
    }

    private void loadBundledSample() {
        try (InputStream input = getAssets().open("sample.json")) {
            lottieView.load(readAll(input), "sample.json");
        } catch (Exception e) {
            status.setText("Could not load sample: " + e.getMessage());
        }
    }

    private void startAutomatedBenchmark(Intent intent) {
        String fixture = intent.getStringExtra("benchmark_fixture");
        if (fixture == null) return;
        int selected = intent.getIntExtra("benchmark_backend", 0);
        int size = intent.getIntExtra("benchmark_size", 512);
        int seconds = intent.getIntExtra("benchmark_seconds", 3);
        boolean antialias = intent.getBooleanExtra("benchmark_aa", true);
        float curveTolerance = intent.getFloatExtra("benchmark_curve_tolerance", 0.5f);
        File fixtureFile = resolveFixture(fixture);
        try (InputStream input = new FileInputStream(fixtureFile)) {
            lottieView.load(readAll(input), fixtureFile.getAbsolutePath());
            lottieView.setRenderSize(Math.max(64, Math.min(1024, size)));
            lottieView.antialias = antialias;
            lottieView.curveTolerance = curveTolerance;
            if (!lottieView.setBackend(selected)) {
                throw new IllegalStateException("backend surface initialization failed");
            }
            backend.setText(lottieView.backendName());
            lottieView.beginBenchmark(Math.max(1, seconds));
        } catch (Exception error) {
            Log.e("TLottieBench", "ERROR fixture=" + fixture, error);
            status.setText("Benchmark error: " + error.getMessage());
        }
    }

    private void openLottie() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/json");
        startActivityForResult(intent, OPEN_LOTTIE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != OPEN_LOTTIE || resultCode != RESULT_OK || data == null) return;
        Uri uri = data.getData();
        if (uri == null) return;
        try (InputStream input = getContentResolver().openInputStream(uri)) {
            if (input == null) throw new IllegalStateException("file could not be opened");
            lottieView.load(readAll(input), displayName(uri));
        } catch (Exception e) {
            showLoadError(e);
        }
    }

    private String displayName(Uri uri) {
        try (android.database.Cursor cursor = getContentResolver().query(
                uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) return cursor.getString(0);
        } catch (Exception ignored) {}
        return "selected Lottie";
    }

    private File fixtureRoot() {
        File external = getExternalFilesDir(null);
        if (external != null) {
            File fixtures = new File(external, "tgs_dump");
            if (fixtures.isDirectory()) return fixtures;
        }
        if (LEGACY_FULL_FIXTURES.isDirectory()) return LEGACY_FULL_FIXTURES;
        return SMALL_FIXTURES;
    }

    private File resolveFixture(String path) {
        File requested = new File(path);
        if (requested.canRead()) return requested;
        String marker = File.separator + "tgs_dump" + File.separator;
        int markerIndex = path.indexOf(marker);
        if (markerIndex >= 0) {
            String relative = path.substring(markerIndex + marker.length());
            File external = getExternalFilesDir(null);
            if (external != null) {
                File relocated = new File(new File(external, "tgs_dump"), relative);
                if (relocated.canRead()) return relocated;
            }
        }
        return requested;
    }

    private void showFixturePacks() {
        File root = fixtureRoot();
        File[] packs = root.listFiles(File::isDirectory);
        if (packs == null || packs.length == 0) {
            new AlertDialog.Builder(this)
                    .setTitle("Fixtures are not readable")
                    .setMessage("Run examples/android/prepare-device-fixtures.sh once. "
                            + "It installs the corpus in this app's external files directory.")
                    .setPositiveButton("OK", null)
                    .show();
            return;
        }
        Arrays.sort(packs, Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
        String[] names = Arrays.stream(packs).map(File::getName).toArray(String[]::new);
        new AlertDialog.Builder(this)
                .setTitle("Fixture packs (" + packs.length + ")")
                .setItems(names, (dialog, which) -> showFixtures(packs[which]))
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void showFixtures(File pack) {
        List<File> fixtures = new ArrayList<>();
        collectJson(pack, fixtures);
        fixtures.sort(Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
        String[] names = fixtures.stream()
                .map(file -> relativeName(pack, file))
                .toArray(String[]::new);
        new AlertDialog.Builder(this)
                .setTitle(pack.getName() + " (" + fixtures.size() + ")")
                .setItems(names, (dialog, which) -> loadFixture(pack, fixtures.get(which)))
                .setNegativeButton("Back", (dialog, which) -> showFixturePacks())
                .show();
    }

    private static void collectJson(File directory, List<File> output) {
        File[] files = directory.listFiles();
        if (files == null) return;
        for (File file : files) {
            if (file.isDirectory()) collectJson(file, output);
            else if (!file.getName().startsWith("._")
                    && file.getName().toLowerCase(Locale.US).endsWith(".json")) output.add(file);
        }
    }

    private static String relativeName(File pack, File file) {
        String prefix = pack.getAbsolutePath() + File.separator;
        String path = file.getAbsolutePath();
        return path.startsWith(prefix) ? path.substring(prefix.length()) : file.getName();
    }

    private void loadFixture(File pack, File fixture) {
        try (InputStream input = new FileInputStream(fixture)) {
            lottieView.load(readAll(input), pack.getName() + "/" + relativeName(pack, fixture));
        } catch (Exception e) {
            showLoadError(e);
        }
    }

    private void showLoadError(Exception error) {
        Toast.makeText(this, "Could not load Lottie: " + error.getMessage(), Toast.LENGTH_LONG).show();
    }

    private static byte[] readAll(InputStream input) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[16 * 1024];
        int count;
        while ((count = input.read(buffer)) >= 0) output.write(buffer, 0, count);
        return output.toByteArray();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (lottieView != null) lottieView.start();
    }

    @Override
    protected void onPause() {
        if (lottieView != null) lottieView.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        if (lottieView != null) lottieView.release();
        super.onDestroy();
    }

    private final class LottieView extends FrameLayout
            implements SurfaceHolder.Callback, Choreographer.FrameCallback {
        private SurfaceView gpuView;
        private final CpuView cpuView;
        private final Rect source = new Rect();
        private final Rect target = new Rect();
        private int renderSize = 512;
        private int[] pixels = new int[renderSize * renderSize];
        private Bitmap bitmap = Bitmap.createBitmap(
                renderSize, renderSize, Bitmap.Config.ARGB_8888);
        private long handle;
        private int frames = 1;
        private float frameRate = 60f;
        private long animationStartNs;
        private long lastStatusNs;
        private long totalRenderNs;
        private int renderedFrames;
        private boolean running;
        private int backendMode;
        private boolean surfaceReady;
        private boolean antialias = true;
        private float curveTolerance = 0.5f;
        private String fileName = "";
        private boolean benchmarking;
        private long benchmarkDurationNs;
        private long benchmarkFirstNs;
        private long benchmarkMeasureStartNs;
        private long benchmarkLastFrameNs;
        private final List<Long> benchmarkIntervals = new ArrayList<>();
        private final List<Long> benchmarkRenders = new ArrayList<>();

        LottieView() {
            super(MainActivity.this);
            source.set(0, 0, renderSize, renderSize);
            setBackgroundColor(Color.rgb(34, 39, 51));
            gpuView = createGpuView();
            addView(gpuView, new FrameLayout.LayoutParams(-1, -1));
            cpuView = new CpuView();
            addView(cpuView, new FrameLayout.LayoutParams(-1, -1));
        }

        void load(byte[] json, String name) {
            long next = NativeBridge.create(json);
            if (next == 0) throw new IllegalArgumentException("tlottie rejected the JSON");
            if (handle != 0) {
                NativeBridge.clearSurface(handle);
                NativeBridge.destroy(handle);
            }
            handle = next;
            frames = Math.max(1, NativeBridge.frameCount(handle));
            frameRate = Math.max(1f, NativeBridge.frameRate(handle));
            fileName = name;
            if (isGpuBackend() && surfaceReady && !attachGpuSurface()) {
                setBackend(0);
                backend.setText("tlottie CPU");
            }
            resetClock();
            updateStatus(0);
        }

        boolean setBackend(int selected) {
            if (selected == backendMode) return true;
            int previous = backendMode;
            if (isGpuBackend() && handle != 0) NativeBridge.clearSurface(handle);
            if (selected == 1 || selected == 6) {
                backendMode = selected;
                if ((previous == 1 || previous == 6) && previous != selected) {
                    replaceGpuView();
                }
                gpuView.setVisibility(VISIBLE);
                cpuView.setVisibility(INVISIBLE);
                if (surfaceReady && handle != 0 && !attachGpuSurface()) {
                    backendMode = 0;
                    gpuView.setVisibility(INVISIBLE);
                    cpuView.setVisibility(VISIBLE);
                    return false;
                }
            } else {
                backendMode = selected;
                gpuView.setVisibility(INVISIBLE);
                cpuView.setVisibility(VISIBLE);
            }
            resetClock();
            updateStatus(0);
            return true;
        }

        private SurfaceView createGpuView() {
            SurfaceView view = new SurfaceView(MainActivity.this);
            view.setZOrderOnTop(true);
            view.getHolder().setFormat(PixelFormat.TRANSLUCENT);
            view.getHolder().setFixedSize(renderSize, renderSize);
            view.getHolder().addCallback(this);
            view.setVisibility(INVISIBLE);
            return view;
        }

        private void replaceGpuView() {
            surfaceReady = false;
            gpuView.getHolder().removeCallback(this);
            removeView(gpuView);
            gpuView = createGpuView();
            addView(gpuView, 0, new FrameLayout.LayoutParams(-1, -1));
        }

        void setRenderSize(int size) {
            if (size == renderSize) return;
            renderSize = size;
            source.set(0, 0, size, size);
            pixels = new int[size * size];
            bitmap.recycle();
            bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
            gpuView.getHolder().setFixedSize(size, size);
            if (isGpuBackend() && surfaceReady && !attachGpuSurface()) {
                setBackend(0);
                backend.setText("tlottie CPU");
            }
            resetClock();
        }

        void beginBenchmark(int seconds) {
            benchmarking = true;
            benchmarkDurationNs = seconds * 1_000_000_000L;
            benchmarkFirstNs = 0;
            benchmarkMeasureStartNs = 0;
            benchmarkLastFrameNs = 0;
            benchmarkIntervals.clear();
            benchmarkRenders.clear();
            start();
        }

        private boolean attachGpuSurface() {
            Surface surface = gpuView.getHolder().getSurface();
            if (!surface.isValid() || handle == 0) return false;
            String error;
            error = backendMode == 6
                    ? NativeBridge.setThorvgSurface(handle, surface, renderSize, renderSize)
                    : NativeBridge.setSurface(handle, surface, renderSize, renderSize);
            if (error != null) {
                Log.e("TLottieBackend", backendName() + " surface error: " + error);
                status.setText(backendName() + " surface error: " + error);
                return false;
            }
            return true;
        }

        private void resetClock() {
            animationStartNs = System.nanoTime();
            lastStatusNs = 0;
            totalRenderNs = 0;
            renderedFrames = 0;
        }

        void start() {
            if (running) return;
            running = true;
            animationStartNs = System.nanoTime();
            if (running) Choreographer.getInstance().postFrameCallback(this);
        }

        void stop() {
            running = false;
            Choreographer.getInstance().removeFrameCallback(this);
        }

        void release() {
            stop();
            if (handle != 0) {
                NativeBridge.clearSurface(handle);
                NativeBridge.destroy(handle);
                handle = 0;
            }
        }

        @Override
        public void doFrame(long frameTimeNanos) {
            if (!running) return;
            if (handle != 0 && (!isGpuBackend() || surfaceReady)) {
                float frame = ((frameTimeNanos - animationStartNs)
                        / 1_000_000_000f * frameRate) % frames;
                long before = System.nanoTime();
                long elapsed = 0;
                String error;
                if (backendMode == 1) {
                    error = NativeBridge.renderVulkan(
                            handle, frame, antialias, curveTolerance);
                } else if (backendMode == 6) {
                    error = NativeBridge.renderThorvgGpu(handle, frame);
                } else if (backendMode >= 2 && backendMode <= 4) {
                    error = NativeBridge.renderRlottie(
                            handle, backendMode - 2, frame, renderSize, renderSize, pixels);
                } else if (backendMode == 5) {
                    error = NativeBridge.renderThorvgCpu(
                            handle, frame, renderSize, renderSize, pixels);
                } else {
                    error = NativeBridge.renderCpu(
                            handle, frame, renderSize, renderSize, pixels);
                }
                if (error == null) {
                    if (!isGpuBackend()) drawCpuFrame();
                    elapsed = System.nanoTime() - before;
                    totalRenderNs += elapsed;
                    renderedFrames++;
                    collectBenchmark(frameTimeNanos, elapsed);
                } else {
                    Log.e("TLottieBackend", backendName() + " render error: " + error);
                    status.setText(backendName() + " error: " + error);
                    if (benchmarking) {
                        benchmarking = false;
                        Log.e("TLottieBench", "ERROR backend=" + backendName()
                                + " render=" + error);
                    }
                    setBackend(0);
                    backend.setText("tlottie CPU");
                }
                if (frameTimeNanos - lastStatusNs > 500_000_000L) {
                    updateStatus(elapsed);
                    lastStatusNs = frameTimeNanos;
                }
            }
            Choreographer.getInstance().postFrameCallback(this);
        }

        private void collectBenchmark(long frameTimeNs, long renderNs) {
            if (!benchmarking) return;
            if (benchmarkFirstNs == 0) benchmarkFirstNs = frameTimeNs;
            if (frameTimeNs - benchmarkFirstNs < 500_000_000L) return;
            if (benchmarkMeasureStartNs == 0) {
                benchmarkMeasureStartNs = frameTimeNs;
                benchmarkLastFrameNs = frameTimeNs;
                Log.i("TLottieBench", "MEASURE_START");
            } else {
                benchmarkIntervals.add(frameTimeNs - benchmarkLastFrameNs);
                benchmarkLastFrameNs = frameTimeNs;
            }
            benchmarkRenders.add(renderNs);
            if (frameTimeNs - benchmarkMeasureStartNs < benchmarkDurationNs) return;

            benchmarking = false;
            long refreshNs = Math.round(1_000_000_000.0 / getDisplay().getRefreshRate());
            long missed = 0;
            for (long interval : benchmarkIntervals) {
                missed += Math.max(0, Math.round((double) interval / refreshNs) - 1);
            }
            long delivered = benchmarkIntervals.size();
            double missedPercent = 100.0 * missed / Math.max(1, delivered + missed);
            String result = String.format(Locale.US,
                    "RESULT backend=%s size=%d refresh_hz=%.2f frames=%d "
                            + "missed_vsync=%d missed_pct=%.2f interval_p50_ns=%d "
                            + "interval_p90_ns=%d interval_p99_ns=%d render_p50_ns=%d render_p90_ns=%d "
                            + "render_p99_ns=%d",
                    backendName().replace(' ', '_'), renderSize, getDisplay().getRefreshRate(),
                    delivered, missed, missedPercent,
                    percentile(benchmarkIntervals, 50), percentile(benchmarkIntervals, 90),
                    percentile(benchmarkIntervals, 99),
                    percentile(benchmarkRenders, 50), percentile(benchmarkRenders, 90),
                    percentile(benchmarkRenders, 99));
            Log.i("TLottieBench", result);
            status.setText(result);
            stop();
            int measuredBackend = backendMode;
            boolean measuredAntialias = antialias;
            float measuredCurveTolerance = curveTolerance;
            if (measuredBackend == 1 || measuredBackend == 6) {
                postDelayed(() -> {
                    if (handle == 0) return;
                    String gpu = measuredBackend == 1
                            ? NativeBridge.benchmarkVulkan(
                                    handle, 10, 30, measuredAntialias,
                                    measuredCurveTolerance)
                            : NativeBridge.benchmarkThorvgGpu(handle, 10, 30);
                    Log.i("TLottieBench", "GPU " + gpu);
                }, 750);
            }
        }

        private long percentile(List<Long> values, int percent) {
            if (values.isEmpty()) return 0;
            ArrayList<Long> sorted = new ArrayList<>(values);
            sorted.sort(Long::compare);
            return sorted.get((sorted.size() - 1) * percent / 100);
        }

        private void drawCpuFrame() {
            bitmap.setPixels(pixels, 0, renderSize, 0, 0, renderSize, renderSize);
            cpuView.invalidate();
        }

        private void updateStatus(long lastNs) {
            double avgMs = renderedFrames == 0 ? 0
                    : totalRenderNs / renderedFrames / 1_000_000.0;
            if (backendMode == 1 && handle != 0) {
                double gpuMs = NativeBridge.lastVulkanGpuNs(handle) / 1_000_000.0;
                status.setText(String.format(Locale.US,
                        "%s  •  %s  •  GPU work %.2f ms  •  CPU submit %.2f ms",
                        fileName, backendName(), gpuMs, lastNs / 1_000_000.0));
            } else {
                status.setText(String.format(Locale.US,
                        "%s  •  %s  •  frame work %.2f ms avg (%.2f ms last)",
                        fileName, backendName(),
                        avgMs, lastNs / 1_000_000.0));
            }
        }

        private String backendName() {
            switch (backendMode) {
                case 1: return "tlottie Vulkan";
                case 2: return "rlottie";
                case 3: return "rlottie 2019";
                case 4: return "rlottie 2019 patched";
                case 5: return "ThorVG CPU";
                case 6: return "ThorVG GPU (OpenGL ES)";
                default: return "tlottie CPU";
            }
        }

        private boolean isGpuBackend() {
            return backendMode == 1 || backendMode == 6;
        }

        @Override
        protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
            super.onMeasure(widthMeasureSpec, heightMeasureSpec);
            int side = Math.min(getMeasuredWidth(), getMeasuredHeight());
            setMeasuredDimension(side, side);
            int square = MeasureSpec.makeMeasureSpec(side, MeasureSpec.EXACTLY);
            gpuView.measure(square, square);
            cpuView.measure(square, square);
        }

        @Override
        protected void onLayout(boolean changed, int left, int top, int right, int bottom) {
            int width = right - left;
            int height = bottom - top;
            int side = Math.min(width, height);
            int childLeft = (width - side) / 2;
            int childTop = (height - side) / 2;
            gpuView.layout(childLeft, childTop, childLeft + side, childTop + side);
            cpuView.layout(childLeft, childTop, childLeft + side, childTop + side);
        }

        @Override
        public void surfaceCreated(SurfaceHolder holder) {
            holder.setFixedSize(renderSize, renderSize);
            surfaceReady = true;
            if (isGpuBackend() && handle != 0 && !attachGpuSurface()) {
                setBackend(0);
                backend.setText("tlottie CPU");
            }
        }

        @Override
        public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {}

        @Override
        public void surfaceDestroyed(SurfaceHolder holder) {
            surfaceReady = false;
            if (handle != 0) NativeBridge.clearSurface(handle);
        }

        private final class CpuView extends View {
            CpuView() {
                super(MainActivity.this);
            }

            @Override
            protected void onDraw(Canvas canvas) {
                super.onDraw(canvas);
                int side = Math.min(getWidth(), getHeight());
                int left = (getWidth() - side) / 2;
                int top = (getHeight() - side) / 2;
                target.set(left, top, left + side, top + side);
                canvas.drawBitmap(bitmap, source, target, null);
            }
        }
    }
}
