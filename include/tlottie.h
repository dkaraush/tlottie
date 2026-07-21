#ifndef TLOTTIE_H
#define TLOTTIE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TLottieInstance TLottieInstance;

enum TlottieFitzModifier {
  TLOTTIE_FITZ_NONE = 0,
  TLOTTIE_FITZ_TYPE_12 = 1,
  TLOTTIE_FITZ_TYPE_3 = 2,
  TLOTTIE_FITZ_TYPE_4 = 3,
  TLOTTIE_FITZ_TYPE_5 = 4,
  TLOTTIE_FITZ_TYPE_6 = 5,
};

typedef struct TLottieLayerColorReplacement {
  const uint8_t *layer_name_prefix;
  size_t layer_name_prefix_len;
  /* Straight-alpha 0xAARRGGBB. */
  uint32_t color;
} TLottieLayerColorReplacement;

typedef struct TLottieColorReplacement {
  /* Android 0xAARRGGBB values are accepted. Alpha bytes are ignored and the
     animation's original alpha is preserved. */
  uint32_t source_color;
  uint32_t target_color;
} TLottieColorReplacement;

enum TlottieStatus {
  TLOTTIE_OK = 0,
  TLOTTIE_ERROR_INVALID_ARGUMENT = -1,
  TLOTTIE_ERROR_BUFFER_TOO_SMALL = -2,
  TLOTTIE_ERROR_RENDER = -3,
};

/* Parses JSON and creates a CPU renderer. Returns NULL on failure. */
TLottieInstance *tlottie_new(const uint8_t *json_ptr, size_t json_len);

/*
 * Like tlottie_new, with parse-time customization. Layer prefixes do not
 * include the legacy trailing "**"; matching and color replacement happen
 * once while the instance is created.
 */
TLottieInstance *tlottie_new_with_options(
    const uint8_t *json_ptr,
    size_t json_len,
    uint32_t fitz_modifier,
    const TLottieLayerColorReplacement *replacements,
    size_t replacements_len,
    const TLottieColorReplacement *color_replacements,
    size_t color_replacements_len);

/* NULL is accepted. Other pointers must have come from tlottie_new. */
void tlottie_drop(TLottieInstance *renderer);

uint32_t tlottie_width(const TLottieInstance *renderer);
uint32_t tlottie_height(const TLottieInstance *renderer);
float tlottie_frame_rate(const TLottieInstance *renderer);
uint32_t tlottie_frame_count(const TLottieInstance *renderer);

/*
 * Renders width * height premultiplied 0xAARRGGBB pixels into out.
 * out_len is measured in uint32_t pixels. Nonzero antialias enables AA.
 */
int32_t tlottie_render(
    TLottieInstance *renderer,
    float frame,
    uint32_t width,
    uint32_t height,
    uint32_t *out,
    size_t out_len,
    uint32_t antialias);

/*
 * Like tlottie_render, with an explicit maximum curve-flattening error in
 * device pixels. curve_tolerance must be finite and greater than zero.
 * Smaller values improve geometric accuracy at a performance cost.
 */
int32_t tlottie_render_with_options(
    TLottieInstance *renderer,
    float frame,
    uint32_t width,
    uint32_t height,
    uint32_t *out,
    size_t out_len,
    uint32_t antialias,
    float curve_tolerance);

/*
 * Renders width * height bytes directly into an Alpha8 bitmap. Unlike the
 * ARGB APIs above, out_len is measured in bytes and no color conversion is
 * performed.
 */
int32_t tlottie_render_alpha8(
    TLottieInstance *renderer,
    float frame,
    uint32_t width,
    uint32_t height,
    uint8_t *out,
    size_t out_len,
    uint32_t antialias);

int32_t tlottie_render_alpha8_with_options(
    TLottieInstance *renderer,
    float frame,
    uint32_t width,
    uint32_t height,
    uint8_t *out,
    size_t out_len,
    uint32_t antialias,
    float curve_tolerance);

#ifdef __cplusplus
}
#endif

#endif /* TLOTTIE_H */
