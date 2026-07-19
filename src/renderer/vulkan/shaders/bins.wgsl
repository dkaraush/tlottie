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

var<workgroup> scan_values: array<u32, 64>;

fn load_f32(word: u32) -> f32 {
    return bitcast<f32>(words[word]);
}

fn load_raw_point(index: u32) -> vec2<f32> {
    let word = push.point_word + index * 2u;
    return vec2<f32>(load_f32(word), load_f32(word + 1u));
}

struct Segment {
    a: vec2<f32>,
    b: vec2<f32>,
}

fn load_segment(a_index: u32, b_index: u32, contour_index: u32) -> Segment {
    let contour = push.contour_word + contour_index * 8u;
    let a = load_raw_point(a_index);
    let b = load_raw_point(b_index);
    let affine = vec4<f32>(
        load_f32(contour + 2u), load_f32(contour + 3u),
        load_f32(contour + 4u), load_f32(contour + 5u),
    );
    let translation = vec2<f32>(load_f32(contour + 6u), load_f32(contour + 7u));
    if all(affine == vec4<f32>(1.0, 0.0, 0.0, 1.0)) {
        return Segment(a + translation, b + translation);
    }
    return Segment(
        vec2<f32>(
            affine.x * a.x + affine.z * a.y + translation.x,
            affine.y * a.x + affine.w * a.y + translation.y,
        ),
        vec2<f32>(
            affine.x * b.x + affine.z * b.y + translation.x,
            affine.y * b.x + affine.w * b.y + translation.y,
        ),
    );
}

fn store_tile_paint(index: u32, paint: u32) {
    if (push.compact_flags & 1u) == 0u {
        words[push.tile_index_word + index] = paint;
        return;
    }
    let word = push.tile_index_word + index / 2u;
    if (index & 1u) == 0u {
        words[word] = paint;
    } else {
        words[word] = (words[word] & 65535u) | (paint << 16u);
    }
}

fn store_edge(index: u32, a: vec2<f32>, b: vec2<f32>) {
    let word = push.edge_word + index * 5u;
    words[word] = bitcast<u32>(a.x);
    words[word + 1u] = bitcast<u32>(a.y);
    words[word + 2u] = bitcast<u32>(b.x);
    words[word + 3u] = bitcast<u32>(b.y);
    words[word + 4u] = bitcast<u32>(min(a.x, b.x));
}

fn tile_paint_count(tile_index: u32) -> u32 {
    let tile_x = tile_index % push.tiles_x;
    let tile_y = tile_index / push.tiles_x;
    let min_x = f32(tile_x * 16u);
    let min_y = f32(tile_y * 16u);
    let max_x = f32(min((tile_x + 1u) * 16u, push.width));
    let max_y = f32(min((tile_y + 1u) * 16u, push.height));
    var count = 0u;
    for (var paint_index = 0u; paint_index < push.paint_count; paint_index++) {
        let paint = push.paint_word + paint_index * 24u;
        let bounds_min = vec2<f32>(load_f32(paint + 20u), load_f32(paint + 21u));
        let bounds_max = vec2<f32>(load_f32(paint + 22u), load_f32(paint + 23u));
        if max_x <= bounds_min.x || min_x >= bounds_max.x ||
           max_y <= bounds_min.y || min_y >= bounds_max.y {
            continue;
        }
        count += 1u;
    }
    return count;
}

fn edge_bin_count(bin_index: u32) -> u32 {
    let paint_index = bin_index / push.tiles_y;
    let tile_y = bin_index % push.tiles_y;
    let paint = push.paint_word + paint_index * 24u;
    let row_min = f32(tile_y * 16u);
    let row_max = f32(min((tile_y + 1u) * 16u, push.height));
    let contour_start = words[paint];
    let contour_count = words[paint + 1u];
    var count = 0u;
    for (var contour_index = 0u; contour_index < contour_count; contour_index++) {
        let contour_id = contour_start + contour_index;
        let contour = push.contour_word + contour_id * 8u;
        let point_first = words[contour];
        let point_count = words[contour + 1u];
        if point_count < 2u {
            continue;
        }
        for (var edge_index = 0u; edge_index < point_count; edge_index++) {
            let a_index = point_first + edge_index;
            let b_index = point_first + (edge_index + 1u) % point_count;
            let segment = load_segment(a_index, b_index, contour_id);
            let a = segment.a;
            let b = segment.b;
            if max(a.y, b.y) <= row_min || min(a.y, b.y) >= row_max {
                continue;
            }
            count += 1u;
        }
    }
    return count;
}

fn scatter_tile(tile_index: u32) {
    let tile_x = tile_index % push.tiles_x;
    let tile_y = tile_index / push.tiles_x;
    let min_x = f32(tile_x * 16u);
    let min_y = f32(tile_y * 16u);
    let max_x = f32(min((tile_x + 1u) * 16u, push.width));
    let max_y = f32(min((tile_y + 1u) * 16u, push.height));
    let record = push.tile_word + tile_index * 2u;
    let first = words[record];
    var count = 0u;
    for (var paint_index = 0u; paint_index < push.paint_count; paint_index++) {
        let paint = push.paint_word + paint_index * 24u;
        let bounds_min = vec2<f32>(load_f32(paint + 20u), load_f32(paint + 21u));
        let bounds_max = vec2<f32>(load_f32(paint + 22u), load_f32(paint + 23u));
        if max_x <= bounds_min.x || min_x >= bounds_max.x ||
           max_y <= bounds_min.y || min_y >= bounds_max.y {
            continue;
        }
        store_tile_paint(first + count, paint_index);
        count += 1u;
    }
}

fn scatter_edge_bin(bin_index: u32) {
    let paint_index = bin_index / push.tiles_y;
    let tile_y = bin_index % push.tiles_y;
    let paint = push.paint_word + paint_index * 24u;
    let row_min = f32(tile_y * 16u);
    let row_max = f32(min((tile_y + 1u) * 16u, push.height));
    let contour_start = words[paint];
    let contour_count = words[paint + 1u];
    let record = push.edge_bin_word + bin_index * 2u;
    let first = words[record];
    var count = 0u;
    for (var contour_index = 0u; contour_index < contour_count; contour_index++) {
        let contour_id = contour_start + contour_index;
        let contour = push.contour_word + contour_id * 8u;
        let point_first = words[contour];
        let point_count = words[contour + 1u];
        if point_count < 2u {
            continue;
        }
        for (var edge_index = 0u; edge_index < point_count; edge_index++) {
            let a_index = point_first + edge_index;
            let b_index = point_first + (edge_index + 1u) % point_count;
            let segment = load_segment(a_index, b_index, contour_id);
            let a = segment.a;
            let b = segment.b;
            if max(a.y, b.y) <= row_min || min(a.y, b.y) >= row_max {
                continue;
            }
            store_edge(first + count, a, b);
            count += 1u;
        }
    }
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tile_count = push.tiles_x * push.tiles_y;
    let bin_count = push.paint_count * push.tiles_y;
    let tile_blocks = (tile_count + 63u) / 64u;
    let edge_blocks = (bin_count + 63u) / 64u;
    if push.antialias == 0u {
        if gid.x < tile_count {
            let record = push.tile_word + gid.x * 2u;
            words[record + 1u] = tile_paint_count(gid.x);
        }
        if gid.x < bin_count {
            let record = push.edge_bin_word + gid.x * 2u;
            words[record + 1u] = edge_bin_count(gid.x);
        }
        return;
    }
    if push.antialias == 1u {
        let block = workgroup_id.x;
        let is_tile = block < tile_blocks;
        let local_block = select(block - tile_blocks, block, is_tile);
        let count = select(bin_count, tile_count, is_tile);
        let index = local_block * 64u + local_id.x;
        var value = 0u;
        if index < count {
            let record = select(push.edge_bin_word, push.tile_word, is_tile) + index * 2u;
            value = words[record + 1u];
            if is_tile && (push.compact_flags & 1u) != 0u {
                value = (value + 1u) & 4294967294u;
            }
        }
        scan_values[local_id.x] = value;
        workgroupBarrier();
        for (var offset = 1u; offset < 64u; offset *= 2u) {
            var addend = 0u;
            if local_id.x >= offset {
                addend = scan_values[local_id.x - offset];
            }
            workgroupBarrier();
            scan_values[local_id.x] += addend;
            workgroupBarrier();
        }
        if index < count {
            let record = select(push.edge_bin_word, push.tile_word, is_tile) + index * 2u;
            words[record] = scan_values[local_id.x] - value;
            let valid_in_block = min(64u, count - local_block * 64u);
            if local_id.x + 1u == valid_in_block {
                words[push.tile_index_word + block] = scan_values[local_id.x];
            }
        }
        return;
    }
    if push.antialias == 2u {
        if gid.x == 0u {
            var cursor = 0u;
            for (var block = 0u; block < tile_blocks; block++) {
                let sum = words[push.tile_index_word + block];
                words[push.tile_index_word + block] = cursor;
                cursor += sum;
            }
            cursor = 0u;
            for (var block = 0u; block < edge_blocks; block++) {
                let index = tile_blocks + block;
                let sum = words[push.tile_index_word + index];
                words[push.tile_index_word + index] = cursor;
                cursor += sum;
            }
        }
        return;
    }
    if push.antialias == 3u {
        if gid.x < tile_count {
            words[push.tile_word + gid.x * 2u] += words[push.tile_index_word + gid.x / 64u];
        }
        if gid.x < bin_count {
            words[push.edge_bin_word + gid.x * 2u] +=
                words[push.tile_index_word + tile_blocks + gid.x / 64u];
        }
        return;
    }
    if gid.x < tile_count {
        scatter_tile(gid.x);
    }
    if gid.x < bin_count {
        scatter_edge_bin(gid.x);
    }
}
