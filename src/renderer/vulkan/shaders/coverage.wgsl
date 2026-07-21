struct PushConstants {
    width: u32,
    height: u32,
    output_word: u32,
    point_word: u32,
    contour_word: u32,
    paint_word: u32,
    lut_word: u32,
    paint_count: u32,
    antialias: u32,
    tile_word: u32,
    tile_index_word: u32,
    tiles_x: u32,
    edge_bin_word: u32,
    edge_word: u32,
    tiles_y: u32,
    compact_flags: u32,
}

var<push_constant> push: PushConstants;

@group(0) @binding(0)
var<storage, read_write> words: array<u32>;

fn load_f32(word: u32) -> f32 {
    return bitcast<f32>(words[word]);
}

struct Segment {
    a: vec2<f32>,
    b: vec2<f32>,
    min_x: f32,
}

fn load_edge(index: u32) -> Segment {
    let word = push.edge_word + index * 5u;
    return Segment(
        vec2<f32>(
            bitcast<f32>(words[word]), bitcast<f32>(words[word + 1u]),
        ),
        vec2<f32>(
            bitcast<f32>(words[word + 2u]), bitcast<f32>(words[word + 3u]),
        ),
        bitcast<f32>(words[word + 4u]),
    );
}

fn load_tile_paint(index: u32) -> u32 {
    if (push.compact_flags & 1u) == 0u {
        return words[push.tile_index_word + index];
    }
    let packed = words[push.tile_index_word + index / 2u];
    return select(packed & 65535u, packed >> 16u, (index & 1u) != 0u);
}

fn unpack_rgba(rgba: u32) -> vec4<u32> {
    return vec4<u32>(
        rgba & 255u,
        (rgba >> 8u) & 255u,
        (rgba >> 16u) & 255u,
        (rgba >> 24u) & 255u,
    );
}

fn sample_paint(base: u32, position: vec2<f32>) -> vec4<u32> {
    if words[base + 2u] == 0u {
        return unpack_rgba(words[base + 4u]);
    }
    let local = vec2<f32>(
        load_f32(base + 8u) * position.x + load_f32(base + 10u) * position.y + load_f32(base + 12u),
        load_f32(base + 9u) * position.x + load_f32(base + 11u) * position.y + load_f32(base + 13u),
    );
    let params0 = vec4<f32>(
        load_f32(base + 14u), load_f32(base + 15u),
        load_f32(base + 16u), load_f32(base + 17u),
    );
    let params1 = vec2<f32>(load_f32(base + 18u), load_f32(base + 19u));
    var t = 0.0;
    let kind = words[base + 5u];
    if kind == 0u {
        t = dot(local - params0.xy, params0.zw) * params1.x;
    } else if kind == 1u {
        t = length(local - params0.xy) * params0.z;
    } else {
        let g = local - params0.xy;
        let a = params1.x;
        if abs(a) < 0.000000001 {
            return vec4<u32>(0u);
        }
        let b = 2.0 * dot(g, params0.zw);
        let det = b * b + 4.0 * a * dot(g, g);
        if det < 0.0 {
            return vec4<u32>(0u);
        }
        let root = sqrt(det);
        let inv2a = 1.0 / (2.0 * a);
        t = max((-b - root) * inv2a, (-b + root) * inv2a);
        if params1.y * t < 0.0 {
            return vec4<u32>(0u);
        }
    }
    let index = u32(clamp(t, 0.0, 1.0) * 1023.0 + 0.5);
    return unpack_rgba(words[push.lut_word + words[base + 6u] + index]);
}

fn edge_pixel_area(a: vec2<f32>, b: vec2<f32>, pixel: vec2<f32>) -> f32 {
    if abs(a.y - b.y) <= 0.000000001 {
        return 0.0;
    }
    var low = a;
    var high = b;
    var direction = 1.0;
    if a.y > b.y {
        low = b;
        high = a;
        direction = -1.0;
    }
    let y0 = max(low.y, pixel.y);
    let y1 = min(high.y, pixel.y + 1.0);
    let dy = y1 - y0;
    if dy <= 0.0 {
        return 0.0;
    }
    if max(low.x, high.x) <= pixel.x {
        return direction * dy;
    }
    let slope = (high.x - low.x) / (high.y - low.y);
    let x0 = low.x + (y0 - low.y) * slope;
    let x1 = low.x + (y1 - low.y) * slope;
    let dx = x1 - x0;
    let right = pixel.x + 1.0;
    var integral = 0.0;
    if max(x0, x1) <= pixel.x {
        integral = 1.0;
    } else if min(x0, x1) >= right {
        integral = 0.0;
    } else if abs(dx) <= 0.000001 {
        integral = clamp(right - 0.5 * (x0 + x1), 0.0, 1.0);
    } else {
        let t_left = (pixel.x - x0) / dx;
        let t_right = (right - x0) / dx;
        let ta = clamp(min(t_left, t_right), 0.0, 1.0);
        let tb = clamp(max(t_left, t_right), 0.0, 1.0);
        let v0 = clamp(right - x0, 0.0, 1.0);
        let va = clamp(right - (x0 + dx * ta), 0.0, 1.0);
        let vb = clamp(right - (x0 + dx * tb), 0.0, 1.0);
        let v1 = clamp(right - x1, 0.0, 1.0);
        integral = 0.5 * (
            (v0 + va) * ta +
            (va + vb) * (tb - ta) +
            (vb + v1) * (1.0 - tb)
        );
    }
    return direction * dy * integral;
}

fn analytic_coverage(paint_index: u32, paint: u32, tile_y: u32, pixel: vec2<f32>) -> f32 {
    var area = 0.0;
    let edge_bin = push.edge_bin_word + (paint_index * push.tiles_y + tile_y) * 2u;
    let edge_first = words[edge_bin];
    let edge_count = words[edge_bin + 1u];
    for (var edge_index = 0u; edge_index < edge_count; edge_index++) {
        let edge = load_edge(edge_first + edge_index);
        if edge.min_x >= pixel.x + 1.0 {
            continue;
        }
        area += edge_pixel_area(edge.a, edge.b, pixel);
    }
    if words[paint + 3u] == 0u {
        return min(abs(area), 1.0);
    }
    let wrapped = area - floor(area * 0.5) * 2.0;
    return select(wrapped, 2.0 - wrapped, wrapped > 1.0);
}

fn center_coverage(paint_index: u32, paint: u32, tile_y: u32, center: vec2<f32>) -> f32 {
    var winding = 0i;
    let edge_bin = push.edge_bin_word + (paint_index * push.tiles_y + tile_y) * 2u;
    let edge_first = words[edge_bin];
    let edge_count = words[edge_bin + 1u];
    for (var edge_index = 0u; edge_index < edge_count; edge_index++) {
        let edge = load_edge(edge_first + edge_index);
        if edge.min_x > center.x {
            continue;
        }
        let a = edge.a;
        let b = edge.b;
        let upward = a.y <= center.y && b.y > center.y;
        let downward = b.y <= center.y && a.y > center.y;
        if !upward && !downward {
            continue;
        }
        let crossing_x = a.x + (center.y - a.y) * (b.x - a.x) / (b.y - a.y);
        if crossing_x <= center.x {
            winding += select(-1i, 1i, upward);
        }
    }
    if words[paint + 3u] == 0u {
        return select(0.0, 1.0, winding != 0i);
    }
    return select(0.0, 1.0, (abs(winding) & 1i) != 0i);
}

fn source_over(source: vec4<u32>, destination: vec4<u32>) -> vec4<u32> {
    // Match the CPU span kernels' byte blend: destination attenuation uses
    // (256 - alpha) / 256 rather than rounded division by 255.
    let inverse = 256u - source.a;
    return min(
        source + (destination * inverse) / 256u,
        vec4<u32>(255u),
    );
}

fn matte_factor(matte: vec4<u32>, kind: u32, source_opacity: u32) -> u32 {
    let scaled_alpha = (matte.a * source_opacity + 127u) / 255u;
    if kind == 1u {
        return scaled_alpha;
    }
    if kind == 2u {
        return 255u - scaled_alpha;
    }
    var luma = 0u;
    if matte.a != 0u {
        var straight = matte.rgb;
        if matte.a != 255u {
            straight = (straight * 255u) / matte.a;
        }
        luma = min(
            (straight.r * 299u + straight.g * 587u + straight.b * 114u) / 1000u,
            255u,
        );
    }
    return select(255u - luma, luma, kind == 3u);
}

// FrameWalker accepts precomp depths 0 through 16 inclusive. An isolated
// precomp at every accepted depth can therefore have 17 saved destinations
// live at once. Keep this in sync with MAX_PRECOMP_DEPTH in the frame walker.
const MAX_LAYER_DEPTH = 17u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= push.width || gid.y >= push.height {
        return;
    }
    let center = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    var destination = vec4<u32>(0u);
    var layer_stack: array<vec4<u32>, 17>;
    var matte_stack: array<vec4<u32>, 17>;
    var layer_depth = 0u;
    var mask_accumulator = 0u;
    let tile_x = gid.x / 16u;
    let tile_y = gid.y / 16u;
    let tile = push.tile_word + (tile_y * push.tiles_x + tile_x) * 2u;
    let tile_first = words[tile];
    let tile_count = words[tile + 1u];
    for (var tile_paint = 0u; tile_paint < tile_count; tile_paint++) {
        let paint_index = load_tile_paint(tile_first + tile_paint);
        let paint = push.paint_word + paint_index * 24u;
        let paint_kind = words[paint + 2u];
        if paint_kind == 2u {
            if layer_depth < MAX_LAYER_DEPTH {
                layer_stack[layer_depth] = destination;
                layer_depth += 1u;
                destination = vec4<u32>(0u);
            }
            continue;
        }
        if paint_kind == 3u {
            if layer_depth > 0u {
                let opacity = (words[paint + 4u] >> 24u) & 255u;
                let source = (destination * opacity + vec4<u32>(127u)) / 255u;
                layer_depth -= 1u;
                destination = source_over(source, layer_stack[layer_depth]);
            }
            continue;
        }
        if paint_kind == 4u {
            if layer_depth < MAX_LAYER_DEPTH {
                layer_stack[layer_depth] = destination;
                layer_depth += 1u;
                destination = vec4<u32>(0u);
            }
            continue;
        }
        if paint_kind == 5u {
            if layer_depth > 0u {
                matte_stack[layer_depth - 1u] = destination;
                destination = vec4<u32>(0u);
            }
            continue;
        }
        if paint_kind == 6u {
            if layer_depth > 0u {
                let metadata = words[paint + 4u];
                let factor = matte_factor(
                    matte_stack[layer_depth - 1u],
                    metadata & 255u,
                    (metadata >> 8u) & 255u,
                );
                var source = (destination * factor + vec4<u32>(127u)) / 255u;
                let opacity = (metadata >> 24u) & 255u;
                source = (source * opacity + vec4<u32>(127u)) / 255u;
                layer_depth -= 1u;
                destination = source_over(source, layer_stack[layer_depth]);
            }
            continue;
        }
        if paint_kind == 7u {
            let metadata = words[paint + 4u];
            let mode = metadata & 255u;
            let inverted = (metadata & (1u << 8u)) != 0u;
            let first = (metadata & (1u << 9u)) != 0u;
            let last = (metadata & (1u << 10u)) != 0u;
            if first {
                mask_accumulator = select(255u, 0u, mode == 97u || mode == 102u);
            }
            let pixel_min = vec2<f32>(f32(gid.x), f32(gid.y));
            var mask_coverage = 0.0;
            if push.antialias == 0u {
                mask_coverage = center_coverage(paint_index, paint, tile_y, center);
            } else {
                mask_coverage = analytic_coverage(paint_index, paint, tile_y, pixel_min);
            }
            var contribution = u32(clamp(mask_coverage, 0.0, 1.0) * 255.0 + 0.5);
            contribution = select(contribution, 255u - contribution, inverted);
            let opacity = (metadata >> 24u) & 255u;
            contribution = (contribution * opacity + 127u) / 255u;
            if mode == 115u {
                mask_accumulator = (mask_accumulator * (255u - contribution) + 127u) / 255u;
            } else if mode == 105u {
                mask_accumulator = (mask_accumulator * contribution + 127u) / 255u;
            } else if mode == 102u {
                mask_accumulator = select(
                    contribution - mask_accumulator,
                    mask_accumulator - contribution,
                    mask_accumulator >= contribution,
                );
            } else {
                mask_accumulator = contribution +
                    ((255u - contribution) * mask_accumulator + 127u) / 255u;
            }
            if last {
                destination = (destination * mask_accumulator + vec4<u32>(127u)) / 255u;
            }
            continue;
        }
        let bounds_min = vec2<f32>(load_f32(paint + 20u), load_f32(paint + 21u));
        let bounds_max = vec2<f32>(load_f32(paint + 22u), load_f32(paint + 23u));
        let pixel_min = vec2<f32>(f32(gid.x), f32(gid.y));
        let pixel_max = pixel_min + vec2<f32>(1.0);
        if pixel_max.x <= bounds_min.x || pixel_min.x >= bounds_max.x ||
           pixel_max.y <= bounds_min.y || pixel_min.y >= bounds_max.y {
            continue;
        }
        var coverage = 0.0;
        if push.antialias == 0u {
            coverage = center_coverage(paint_index, paint, tile_y, center);
        } else {
            coverage = analytic_coverage(paint_index, paint, tile_y, pixel_min);
        }
        if coverage > 0.0 {
            let coverage_byte = u32(clamp(coverage, 0.0, 1.0) * 255.0 + 0.5);
            let source = (sample_paint(paint, center) * coverage_byte + vec4<u32>(127u)) / 255u;
            destination = source_over(source, destination);
        }
    }
    var packed = select(
        (destination.a << 24u) | (destination.r << 16u) |
            (destination.g << 8u) | destination.b,
        (destination.a << 24u) | (destination.b << 16u) |
            (destination.g << 8u) | destination.r,
        (push.compact_flags & 4u) != 0u,
    );
    if (push.compact_flags & 8u) != 0u {
        packed = destination.a << 24u;
    }
    words[push.output_word + gid.y * push.width + gid.x] = packed;
}

// Fast path for scenes without layer-isolation or matte commands. Keeping this
// as a separate entry point lets drivers discard both 16-deep color stacks.
@compute @workgroup_size(8, 8)
fn simple_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= push.width || gid.y >= push.height {
        return;
    }
    let center = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    var destination = vec4<u32>(0u);
    let tile_x = gid.x / 16u;
    let tile_y = gid.y / 16u;
    let tile = push.tile_word + (tile_y * push.tiles_x + tile_x) * 2u;
    let tile_first = words[tile];
    let tile_count = words[tile + 1u];
    for (var tile_paint = 0u; tile_paint < tile_count; tile_paint++) {
        let paint_index = load_tile_paint(tile_first + tile_paint);
        let paint = push.paint_word + paint_index * 24u;
        let bounds_min = vec2<f32>(load_f32(paint + 20u), load_f32(paint + 21u));
        let bounds_max = vec2<f32>(load_f32(paint + 22u), load_f32(paint + 23u));
        let pixel_min = vec2<f32>(f32(gid.x), f32(gid.y));
        let pixel_max = pixel_min + vec2<f32>(1.0);
        if pixel_max.x <= bounds_min.x || pixel_min.x >= bounds_max.x ||
           pixel_max.y <= bounds_min.y || pixel_min.y >= bounds_max.y {
            continue;
        }
        var coverage = 0.0;
        if push.antialias == 0u {
            coverage = center_coverage(paint_index, paint, tile_y, center);
        } else {
            coverage = analytic_coverage(paint_index, paint, tile_y, pixel_min);
        }
        if coverage > 0.0 {
            let coverage_byte = u32(clamp(coverage, 0.0, 1.0) * 255.0 + 0.5);
            let source = (sample_paint(paint, center) * coverage_byte + vec4<u32>(127u)) / 255u;
            destination = source_over(source, destination);
        }
    }
    var packed = select(
        (destination.a << 24u) | (destination.r << 16u) |
            (destination.g << 8u) | destination.b,
        (destination.a << 24u) | (destination.b << 16u) |
            (destination.g << 8u) | destination.r,
        (push.compact_flags & 4u) != 0u,
    );
    if (push.compact_flags & 8u) != 0u {
        packed = destination.a << 24u;
    }
    words[push.output_word + gid.y * push.width + gid.x] = packed;
}
