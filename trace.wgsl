// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The response, SH, intersection and blending equations are ported from
// threedgrt_tracer/include/3dgrt/kernels/cuda/gaussianParticles.cuh and
// threedgrt_tracer/src/kernels/cuda/referenceOptix.cu. The BVH traversal is a
// browser implementation: no CUDA, OptiX or hardware ray-tracing API is used.

struct Camera {
    origin: vec4<f32>,       // xyz origin; w tan(vertical FOV / 2)
    right: vec4<f32>,
    down: vec4<f32>,
    forward: vec4<f32>,
    size: vec4<u32>,         // width, height, active SH degree, node count
    thresholds: vec4<f32>,  // min response, min alpha, min transmittance, unused
    background: vec4<f32>,
};
struct Gaussian {
    centerDensity: vec4<f32>,
    row0Radius: vec4<f32>,   // world-to-unscaled-canonical; w = support radius
    row1: vec4<f32>,
    row2: vec4<f32>,
};
struct Node {
    lower: vec3<f32>,
    left: u32,
    upper: vec3<f32>,
    right: u32,              // high bit = leaf; low 31 bits = count
};
struct HitBatch {
    ids: array<u32, 16>,
    distances: array<f32, 16>,
    count: u32,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> sh: array<f32>;
@group(0) @binding(3) var<storage, read> nodes: array<Node>;
@group(0) @binding(4) var outputImage: texture_storage_2d<rgba8unorm, write>;

const FAR: f32 = 1e20;
const EPS_T: f32 = 1e-9;

// Explicit parallel-ray cases avoid 0 * infinity / NaN at an AABB boundary.
fn boxInterval(origin: vec3<f32>, direction: vec3<f32>, lower: vec3<f32>, upper: vec3<f32>, tMin: f32, tMax: f32) -> vec2<f32> {
    var near = tMin;
    var far = tMax;
    for (var axis = 0u; axis < 3u; axis++) {
        if (direction[axis] == 0.0) {
            if (origin[axis] < lower[axis] || origin[axis] > upper[axis]) {
                return vec2<f32>(FAR, -FAR);
            }
        } else {
            let a = (lower[axis] - origin[axis]) / direction[axis];
            let b = (upper[axis] - origin[axis]) / direction[axis];
            near = max(near, min(a, b));
            far = min(far, max(a, b));
        }
    }
    return vec2<f32>(near, far);
}

fn canonicalVector(g: Gaussian, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(g.row0Radius.xyz, v), dot(g.row1.xyz, v), dot(g.row2.xyz, v));
}

// Preserve the native instances path's custom primitive, including its
// extra inverse squared ray length in the distance test. It is NOT an
// analytic ellipsoid surface intersection and NOT a screen-space splat.
fn intersectGaussian(id: u32, origin: vec3<f32>, direction: vec3<f32>, tMin: f32, tMax: f32) -> f32 {
    let g = gaussians[id];
    let o = canonicalVector(g, origin - g.centerDensity.xyz);
    let d = canonicalVector(g, direction);
    let radius = g.row0Radius.w;
    let gate = boxInterval(o, d, vec3<f32>(-radius), vec3<f32>(radius), tMin, tMax);
    if (gate.x > gate.y) { return FAR; }
    let d2 = dot(d, d);
    let t = -dot(o, d) / d2;
    if (t <= tMin || t >= tMax) { return FAR; }
    let perpendicular = cross(d * inverseSqrt(d2), o);
    // Instance rays are divided by radius in CUDA. That factor cancels
    // between numerator and denominator in this specific test.
    if (dot(perpendicular, perpendicular) / d2 >= 9.0) { return FAR; }
    return t;
}

// Closest 16 hits, followed by another BVH traversal if needed. There is
// deliberately no per-ray total hit cap: transparent layers are not dropped.
fn traceBatch(origin: vec3<f32>, direction: vec3<f32>, tMin: f32, tMax: f32) -> HitBatch {
    var hits: HitBatch;
    hits.count = 0u;
    for (var i = 0u; i < 16u; i++) { hits.distances[i] = FAR; }
    var stack: array<u32, 64>;
    var stackSize = 1u;
    stack[0] = 0u;
    loop {
        if (stackSize == 0u) { break; }
        stackSize--;
        let node = nodes[stack[stackSize]];
        let searchMax = min(tMax, hits.distances[15]);
        let interval = boxInterval(origin, direction, node.lower, node.upper, tMin, searchMax);
        if (interval.x > interval.y) { continue; }
        if ((node.right & 0x80000000u) != 0u) {
            let end = node.left + (node.right & 0x7fffffffu);
            for (var id = node.left; id < end; id++) {
                var t = intersectGaussian(id, origin, direction, tMin, min(tMax, hits.distances[15]));
                if (t >= hits.distances[15]) { continue; }
                var insertedId = id;
                for (var slot = 0u; slot < 16u; slot++) {
                    if (t < hits.distances[slot]) {
                        let oldT = hits.distances[slot];
                        let oldId = hits.ids[slot];
                        hits.distances[slot] = t;
                        hits.ids[slot] = insertedId;
                        t = oldT;
                        insertedId = oldId;
                    }
                }
                hits.count = min(hits.count + 1u, 16u);
            }
        } else {
            let a = nodes[node.left];
            let b = nodes[node.right];
            let ia = boxInterval(origin, direction, a.lower, a.upper, tMin, searchMax);
            let ib = boxInterval(origin, direction, b.lower, b.upper, tMin, searchMax);
            let validA = ia.x <= ia.y;
            let validB = ib.x <= ib.y;
            // Export/loader validate max depth < 64; no silently overflowing
            // stack, and no per-ray allocations or recursion are required.
            if (validA && validB) {
                stack[stackSize] = select(node.left, node.right, ia.x <= ib.x);
                stack[stackSize + 1u] = select(node.right, node.left, ia.x <= ib.x);
                stackSize += 2u;
            } else if (validA) {
                stack[stackSize] = node.left;
                stackSize++;
            } else if (validB) {
                stack[stackSize] = node.right;
                stackSize++;
            }
        }
    }
    return hits;
}

fn coefficient(id: u32, index: u32) -> vec3<f32> {
    let offset = id * 48u + index * 3u;
    return vec3<f32>(sh[offset], sh[offset + 1u], sh[offset + 2u]);
}

fn rayColor(id: u32, d: vec3<f32>) -> vec3<f32> {
    var c = 0.28209479177387814 * coefficient(id, 0u);
    if (camera.size.z > 0u) {
        let x = d.x; let y = d.y; let z = d.z;
        c = c - 0.4886025119029199 * y * coefficient(id, 1u)
              + 0.4886025119029199 * z * coefficient(id, 2u)
              - 0.4886025119029199 * x * coefficient(id, 3u);
        if (camera.size.z > 1u) {
            let xx = x*x; let yy = y*y; let zz = z*z;
            let xy = x*y; let yz = y*z; let xz = x*z;
            c = c + 1.0925484305920792 * xy * coefficient(id, 4u)
                  - 1.0925484305920792 * yz * coefficient(id, 5u)
                  + 0.31539156525252005 * (2.0*zz - xx - yy) * coefficient(id, 6u)
                  - 1.0925484305920792 * xz * coefficient(id, 7u)
                  + 0.5462742152960396 * (xx - yy) * coefficient(id, 8u);
            if (camera.size.z > 2u) {
                c = c - 0.5900435899266435 * y * (3.0*xx - yy) * coefficient(id, 9u)
                      + 2.890611442640554 * xy * z * coefficient(id, 10u)
                      - 0.4570457994644658 * y * (4.0*zz - xx - yy) * coefficient(id, 11u)
                      + 0.3731763325901154 * z * (2.0*zz - 3.0*xx - 3.0*yy) * coefficient(id, 12u)
                      - 0.4570457994644658 * x * (4.0*zz - xx - yy) * coefficient(id, 13u)
                      + 1.445305721320277 * z * (xx - yy) * coefficient(id, 14u)
                      - 0.5900435899266435 * x * (xx - 3.0*yy) * coefficient(id, 15u);
            }
        }
    }
    return max(c + vec3<f32>(0.5), vec3<f32>(0.0));
}

@compute @workgroup_size(8, 8)
fn trace(@builtin(global_invocation_id) pixel: vec3<u32>) {
    if (pixel.x >= camera.size.x || pixel.y >= camera.size.y) { return; }
    let xy = (vec2<f32>(pixel.xy) + vec2<f32>(0.5) - vec2<f32>(camera.size.xy) * 0.5)
        * (2.0 * camera.origin.w / f32(camera.size.y));
    // Normalize camera-space direction before its rotation, as V2 does.
    let cameraDirection = normalize(vec3<f32>(xy, 1.0));
    let direction = camera.right.xyz * cameraDirection.x + camera.down.xyz * cameraDirection.y
        + camera.forward.xyz * cameraDirection.z;
    let origin = camera.origin.xyz;
    let scene = boxInterval(origin, direction, nodes[0].lower, nodes[0].upper, 0.0, FAR);
    var lastT = max(0.0, scene.x - EPS_T);
    var transmittance = 1.0;
    var color = vec3<f32>(0.0);
    loop {
        if (lastT > scene.y || transmittance <= camera.thresholds.z) { break; }
        let hits = traceBatch(origin, direction, lastT + EPS_T, scene.y + EPS_T);
        if (hits.count == 0u) { break; }
        for (var h = 0u; h < hits.count; h++) {
            if (transmittance <= camera.thresholds.z) { break; }
            let id = hits.ids[h];
            let g = gaussians[id];
            let o = canonicalVector(g, origin - g.centerDensity.xyz);
            let d = normalize(canonicalVector(g, direction));
            let perpendicular = cross(d, o);
            let distanceSquared = dot(perpendicular, perpendicular);
            let response = exp(-0.0555555555556 * distanceSquared * distanceSquared);
            let alpha = min(0.99, response * g.centerDensity.w);
            if (response > camera.thresholds.x && alpha > camera.thresholds.y) {
                color += rayColor(id, direction) * (alpha * transmittance);
                transmittance *= (1.0 - alpha);
            }
            lastT = max(lastT, hits.distances[h]);
        }
    }
    // V2 displays clamp(pred_rgb, 0, 1) without applying an sRGB transfer.
    color = clamp(color + transmittance * camera.background.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(outputImage, vec2<i32>(pixel.xy), vec4<f32>(color, 1.0));
}
