/* Android-only tlottie performance harness using the public C API. */

#include "tlottie.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0;
    }
    return (uint64_t)value.tv_sec * 1000000000ull + (uint64_t)value.tv_nsec;
}

static uint8_t *read_file(const char *path, size_t *length) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot read %s: errno %d\n", path, errno);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    long size = ftell(file);
    if (size <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    uint8_t *bytes = malloc((size_t)size);
    if (bytes == NULL || fread(bytes, 1, (size_t)size, file) != (size_t)size) {
        free(bytes);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *length = (size_t)size;
    return bytes;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <file.json> <size> <frames> <curve-tolerance>\n", argv[0]);
        return 2;
    }
    const uint32_t size = (uint32_t)strtoul(argv[2], NULL, 10);
    const uint32_t frames = (uint32_t)strtoul(argv[3], NULL, 10);
    const float curve_tolerance = strtof(argv[4], NULL);
    if (size == 0 || size > 4096 || frames == 0 || !(curve_tolerance > 0.0f)) {
        fprintf(stderr, "invalid size, frame count, or curve tolerance\n");
        return 2;
    }

    const uint64_t first_frame_started = now_ns();
    size_t json_length = 0;
    uint8_t *json = read_file(argv[1], &json_length);
    if (json == NULL) {
        return 1;
    }
    TLottieInstance *renderer = tlottie_new(json, json_length);
    free(json);
    if (renderer == NULL) {
        fprintf(stderr, "tlottie failed to parse %s\n", argv[1]);
        return 1;
    }

    const uint32_t animation_frames = tlottie_frame_count(renderer);
    const size_t pixel_count = (size_t)size * (size_t)size;
    uint32_t *pixels = malloc(pixel_count * sizeof(uint32_t));
    if (pixels == NULL) {
        tlottie_drop(renderer);
        return 1;
    }
    for (uint32_t index = 0; index < frames; ++index) {
        const uint64_t started = now_ns();
        const int32_t status = tlottie_render_with_options(
            renderer,
            (float)(index % (animation_frames > 0 ? animation_frames : 1)),
            size,
            size,
            pixels,
            pixel_count,
            1,
            curve_tolerance,
            1);
        const uint64_t elapsed = now_ns() - started;
        if (status != TLOTTIE_OK) {
            fprintf(stderr, "tlottie render %u failed with status %d\n", index, status);
            free(pixels);
            tlottie_drop(renderer);
            return 1;
        }
        printf("F %u %llu\n", index, (unsigned long long)elapsed);
        if (index == 0) {
            printf("FMS %.6f\n", (double)(now_ns() - first_frame_started) / 1000000.0);
        }
    }

    free(pixels);
    tlottie_drop(renderer);
    return 0;
}
