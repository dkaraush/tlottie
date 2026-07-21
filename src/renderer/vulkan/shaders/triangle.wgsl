struct PushConstants {
    viewport: vec2<f32>,
    rgba: u32,
    point_offset: u32,
    paint_kind: u32,
    gradient_kind: u32,
    lut_word_offset: u32,
    padding: u32,
    inverse0: vec4<f32>,
    inverse1: vec4<f32>,
    params0: vec4<f32>,
    params1: vec4<f32>,
    affine0: vec4<f32>,
    affine1: vec2<f32>,
}

var<push_constant> push: PushConstants;

@group(0) @binding(0)
var<storage, read> point_words: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> @builtin(position) vec4<f32> {
    var point_offset = push.point_offset;
    var affine0 = push.affine0;
    var affine1 = push.affine1;
    if push.padding != 0u {
        let draw = push.point_offset + instance_index * 12u;
        point_offset = point_words[draw];
        affine0 = vec4<f32>(
            bitcast<f32>(point_words[draw + 6u]),
            bitcast<f32>(point_words[draw + 7u]),
            bitcast<f32>(point_words[draw + 8u]),
            bitcast<f32>(point_words[draw + 9u]),
        );
        affine1 = vec2<f32>(
            bitcast<f32>(point_words[draw + 10u]),
            bitcast<f32>(point_words[draw + 11u]),
        );
    }
    let local_point = vertex_index;
    let word = (point_offset + local_point) * 2u;
    let base = vec2<f32>(
        bitcast<f32>(point_words[word]),
        bitcast<f32>(point_words[word + 1u]),
    );
    var point = base + affine1;
    if any(affine0 != vec4<f32>(1.0, 0.0, 0.0, 1.0)) {
        point = vec2<f32>(
            affine0.x * base.x + affine0.z * base.y + affine1.x,
            affine0.y * base.x + affine0.w * base.y + affine1.y,
        );
    }
    let ndc = point / push.viewport * 2.0 - vec2<f32>(1.0, 1.0);
    return vec4<f32>(ndc, 0.0, 1.0);
}

@vertex
fn vs_cover(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(push.affine0.x, push.affine0.y),
        vec2<f32>(push.affine0.z, push.affine0.y),
        vec2<f32>(push.affine0.x, push.affine0.w),
        vec2<f32>(push.affine0.x, push.affine0.w),
        vec2<f32>(push.affine0.z, push.affine0.y),
        vec2<f32>(push.affine0.z, push.affine0.w),
    );
    let ndc = corners[vertex_index] / push.viewport * 2.0 - vec2<f32>(1.0, 1.0);
    return vec4<f32>(ndc, 0.0, 1.0);
}

@fragment
fn fs_stencil() {}

fn paint_color(position: vec4<f32>) -> vec4<f32> {
    var rgba = push.rgba;
    if push.paint_kind == 1u {
        let local = vec2<f32>(
            push.inverse0.x * position.x + push.inverse0.z * position.y + push.inverse1.x,
            push.inverse0.y * position.x + push.inverse0.w * position.y + push.inverse1.y,
        );
        var t = 0.0;
        if push.gradient_kind == 0u {
            let delta = local - push.params0.xy;
            t = dot(delta, push.params0.zw) * push.params1.x;
        } else if push.gradient_kind == 1u {
            t = length(local - push.params0.xy) * push.params0.z;
        } else {
            let g = local - push.params0.xy;
            let d = push.params0.zw;
            let a = push.params1.x;
            if abs(a) < 0.000000001 {
                return vec4<f32>(0.0);
            }
            let b = 2.0 * dot(g, d);
            let det = b * b + 4.0 * a * dot(g, g);
            if det < 0.0 {
                return vec4<f32>(0.0);
            }
            let root = sqrt(det);
            let inv2a = 1.0 / (2.0 * a);
            t = max((-b - root) * inv2a, (-b + root) * inv2a);
            if push.params1.y * t < 0.0 {
                return vec4<f32>(0.0);
            }
        }
        let lut_index = u32(clamp(t, 0.0, 1.0) * 1023.0 + 0.5);
        rgba = point_words[push.lut_word_offset + lut_index];
    }
    let scale = 1.0 / 255.0;
    return vec4<f32>(
        f32(rgba & 255u) * scale,
        f32((rgba >> 8u) & 255u) * scale,
        f32((rgba >> 16u) & 255u) * scale,
        f32((rgba >> 24u) & 255u) * scale,
    );
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return paint_color(position);
}

@fragment
fn fs_group(@builtin(position) position: vec4<f32>) -> @location(1) vec4<f32> {
    return paint_color(position);
}
