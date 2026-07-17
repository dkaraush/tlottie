#version 450

layout(set = 0, binding = 1) uniform sampler2D group_color;

layout(push_constant) uniform PushConstants {
    vec2 viewport;
    uint argb;
    uint point_offset;
    uint paint_kind;
    uint gradient_kind;
    uint lut_word_offset;
    uint padding;
    vec4 inverse0;
    vec4 inverse1;
    vec4 params0;
    vec4 params1;
    vec4 affine0;
    vec2 affine1;
} push;

layout(location = 0) out vec4 out_color;

void main() {
    float opacity = float((push.argb >> 24) & 255u) / 255.0;
    out_color = texelFetch(group_color, ivec2(gl_FragCoord.xy), 0) * opacity;
}
