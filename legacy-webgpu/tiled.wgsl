// SPDX-License-Identifier: Apache-2.0
// Concatenated after trace.wgsl's shared reference-rendering functions.
// Tiles only filter candidates. Every retained pixel hit uses the same native
// intersection, quartic response, world-ray SH and ray-local depth ordering.

@group(0) @binding(5) var<storage, read_write> tileStatistics: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> projectedBounds: array<vec4<f32>>;

// 8 KiB IDs + 4 KiB geometry + flags stay below WebGPU's 16 KiB minimum.
var<workgroup> candidateIds: array<u32, 2048>;
var<workgroup> candidateCount: u32;
var<workgroup> overflow: u32;
var<workgroup> sharedGeometry: array<Gaussian, 64>;
var<workgroup> sharedRects: array<vec4<f32>, 64>;
var<workgroup> activeFlags: array<u32, 64>;
var<workgroup> activeRemaining: u32;

fn cameraCoordinates(v: vec3<f32>) -> vec3<f32> {
    // Inverse of the actual FP32 camera basis used by pixelDirection().
    let a = cross(camera.down.xyz, camera.forward.xyz);
    let b = cross(camera.forward.xyz, camera.right.xyz);
    let c = cross(camera.right.xyz, camera.down.xyz);
    let determinant = dot(camera.right.xyz, a);
    return vec3<f32>(dot(a, v), dot(b, v), dot(c, v)) / determinant;
}

fn projectedRectangle(g: Gaussian) -> vec4<f32> {
    let full = vec4<f32>(0.0, 0.0, f32(camera.size.x), f32(camera.size.y));
    let s = vec3<f32>(max(max(abs(g.row0Radius.x), abs(g.row0Radius.y)), abs(g.row0Radius.z)),
        max(max(abs(g.row1.x), abs(g.row1.y)), abs(g.row1.z)),
        max(max(abs(g.row2.x), abs(g.row2.y)), abs(g.row2.z)));
    if (any(s <= vec3<f32>(0.0))) { return full; }
    let r0 = g.row0Radius.xyz / s.x;
    let r1 = g.row1.xyz / s.y;
    let r2 = g.row2.xyz / s.z;
    let determinant = dot(r0, cross(r1, r2));
    if (abs(determinant) < 1e-6) { return full; }
    let c0 = cameraCoordinates(cross(r1, r2) / determinant / s.x * g.row0Radius.w);
    let c1 = cameraCoordinates(cross(r2, r0) / determinant / s.y * g.row0Radius.w);
    let c2 = cameraCoordinates(cross(r0, r1) / determinant / s.z * g.row0Radius.w);
    let center = cameraCoordinates(g.centerDensity.xyz - camera.origin.xyz);
    // Malformed/extreme transforms must broaden filtering, never omit pixels.
    if (any(abs(c0) > vec3<f32>(1e30)) || any(abs(c1) > vec3<f32>(1e30))
        || any(abs(c2) > vec3<f32>(1e30)) || any(abs(center) > vec3<f32>(1e30))) { return full; }
    let extentZ = abs(c0.z) + abs(c1.z) + abs(c2.z);
    let depthTolerance = 1e-5 * (1.0 + abs(center.z) + extentZ);
    if (center.z + extentZ < -depthTolerance) { return vec4<f32>(1.0, 1.0, 0.0, 0.0); }
    if (center.z - extentZ <= depthTolerance) { return full; }
    var minimum = vec2<f32>(FAR);
    var maximum = vec2<f32>(-FAR);
    let focal = f32(camera.size.y) / (2.0 * camera.origin.w);
    for (var corner = 0u; corner < 8u; corner++) {
        let p = center + c0 * select(-1.0, 1.0, (corner & 1u) != 0u)
            + c1 * select(-1.0, 1.0, (corner & 2u) != 0u)
            + c2 * select(-1.0, 1.0, (corner & 4u) != 0u);
        let xy = p.xy / p.z * focal + vec2<f32>(camera.size.xy) * 0.5;
        minimum = min(minimum, xy);
        maximum = max(maximum, xy);
    }
    // Two pixels of outward padding cover FP32 projection and inverse errors.
    return vec4<f32>(max(vec2<f32>(0.0), floor(minimum - 2.0)),
        min(vec2<f32>(camera.size.xy), ceil(maximum + 2.0)));
}

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if (invocation.x < camera.dispatch.y) {
        projectedBounds[invocation.x] = projectedRectangle(gaussians[invocation.x]);
    }
}

fn intersectsCone(node: Node, planes: array<vec3<f32>, 5>) -> bool {
    let center = node.lower * 0.5 + node.upper * 0.5;
    let extent = node.upper * 0.5 - node.lower * 0.5;
    for (var i = 0u; i < 5u; i++) {
        let n = planes[i];
        let tolerance = 1e-5 * (dot(abs(n), abs(center) + abs(camera.origin.xyz) + extent) + 1.0);
        if (dot(n, center - camera.origin.xyz) + dot(abs(n), extent) < -tolerance) { return false; }
    }
    return true;
}

fn collectTile(tileOrigin: vec2<u32>) {
    candidateCount = 0u;
    overflow = 0u;
    let lo = vec2<f32>(tileOrigin);
    let hi = vec2<f32>(min(tileOrigin + vec2<u32>(8u), camera.size.xy));
    let scale = 2.0 * camera.origin.w / f32(camera.size.y);
    let limits = (vec4<f32>(lo - 1.0, hi + 1.0) - vec4<f32>(vec2<f32>(camera.size.xy) * 0.5, vec2<f32>(camera.size.xy) * 0.5)) * scale;
    let right = camera.right.xyz; let down = camera.down.xyz; let forward = camera.forward.xyz;
    let planes = array<vec3<f32>, 5>(
        cross(down, forward + limits.x * right),
        cross(forward + limits.z * right, down),
        cross(forward + limits.y * down, right),
        cross(right, forward + limits.w * down),
        cross(right, down));
    var stack: array<u32, 64>;
    var stackSize = 1u;
    stack[0] = 0u;
    loop {
        if (stackSize == 0u || overflow != 0u) { break; }
        stackSize--;
        let node = nodes[stack[stackSize]];
        if (!intersectsCone(node, planes)) { continue; }
        if ((node.right & 0x80000000u) != 0u) {
            let end = node.left + (node.right & 0x7fffffffu);
            for (var id = node.left; id < end; id++) {
                let rect = projectedBounds[id];
                if (rect.x >= hi.x || rect.y >= hi.y || rect.z <= lo.x || rect.w <= lo.y) { continue; }
                if (candidateCount >= camera.dispatch.z) { overflow = 1u; break; }
                candidateIds[candidateCount] = id;
                candidateCount++;
            }
        } else {
            stack[stackSize] = node.left;
            stack[stackSize + 1u] = node.right;
            stackSize += 2u;
        }
    }
    atomicAdd(&tileStatistics[0], 1u);
    atomicAdd(&tileStatistics[1], overflow);
    atomicAdd(&tileStatistics[2], candidateCount);
    atomicMax(&tileStatistics[3], candidateCount);
}

@compute @workgroup_size(8, 8)
fn traceTiled(@builtin(global_invocation_id) invocation: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
    let pixel = invocation.xy + vec2<u32>(0u, camera.dispatch.x);
    let tileOrigin = group.xy * 8u + vec2<u32>(0u, camera.dispatch.x);
    let valid = pixel.x < camera.size.x && pixel.y < camera.size.y;
    if (lane == 0u) { collectTile(tileOrigin); }
    let count = workgroupUniformLoad(&candidateCount);
    let fallback = workgroupUniformLoad(&overflow);
    let origin = camera.origin.xyz;
    let direction = pixelDirection(pixel);
    if (fallback != 0u) {
        if (valid) { textureStore(outputImage, vec2<i32>(pixel), vec4<f32>(traceRadiance(origin, direction), 1.0)); }
        return;
    }
    let scene = boxInterval(origin, direction, nodes[0].lower, nodes[0].upper, 0.0, FAR);
    var lastT = max(0.0, scene.x - EPS_T);
    var transmittance = 1.0;
    var color = vec3<f32>(0.0);
    var done = !valid || lastT > scene.y || count == 0u;
    var tiedDistance = false;
    loop {
        var hits: HitBatch;
        hits.count = 0u;
        for (var h = 0u; h < 16u; h++) { hits.distances[h] = FAR; }
        for (var start = 0u; start < count; start += 64u) {
            let blockCount = min(64u, count - start);
            if (lane < blockCount) {
                let id = candidateIds[start + lane];
                sharedGeometry[lane] = gaussians[id];
                sharedRects[lane] = projectedBounds[id];
            }
            workgroupBarrier();
            if (!done) {
                let center = vec2<f32>(pixel) + 0.5;
                for (var j = 0u; j < blockCount; j++) {
                    let rect = sharedRects[j];
                    if (center.x < rect.x || center.y < rect.y || center.x > rect.z || center.y > rect.w) { continue; }
                    let intersection = intersectRecordDetailed(sharedGeometry[j], origin, direction, lastT + EPS_T, scene.y + EPS_T);
                    var distance = intersection.x;
                    if (distance == FAR || distance > hits.distances[15]) { continue; }
                    // Equal local distances have traversal-order-dependent
                    // blending in the original BVH. Preserve that exact path
                    // on the affected pixels instead of imposing a new tie
                    // ordering or silently dropping a boundary tie.
                    if (distance == hits.distances[15]) { tiedDistance = true; }
                    if (distance >= hits.distances[15]) { continue; }
                    // Keep the original shrinking AABB interval as well as
                    // the reported t limit; only tie detection uses the wider
                    // interval above.
                    if (intersection.y > hits.distances[15]) {
                        for (var previous = 0u; previous < hits.count; previous++) {
                            if (distance == hits.distances[previous]) { tiedDistance = true; }
                        }
                        continue;
                    }
                    var id = candidateIds[start + j];
                    for (var slot = 0u; slot < 16u; slot++) {
                        if (distance == FAR) { break; }
                        if (distance == hits.distances[slot]) { tiedDistance = true; }
                        if (distance < hits.distances[slot]) {
                            let oldDistance = hits.distances[slot];
                            let oldId = hits.ids[slot];
                            hits.distances[slot] = distance;
                            hits.ids[slot] = id;
                            distance = oldDistance;
                            id = oldId;
                        }
                    }
                    hits.count = min(hits.count + 1u, 16u);
                }
            }
            workgroupBarrier();
        }
        if (!done) {
            if (hits.count == 0u) { done = true; }
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
                    transmittance *= 1.0 - alpha;
                }
                lastT = max(lastT, hits.distances[h]);
            }
            // An incomplete batch means every remaining candidate was seen;
            // another whole-list scan could only return an empty batch.
            done = done || hits.count < 16u || transmittance <= camera.thresholds.z || lastT > scene.y;
        }
        activeFlags[lane] = select(0u, 1u, !done);
        workgroupBarrier();
        if (lane == 0u) {
            activeRemaining = 0u;
            for (var i = 0u; i < 64u; i++) { activeRemaining += activeFlags[i]; }
        }
        let remaining = workgroupUniformLoad(&activeRemaining);
        if (remaining == 0u) { break; }
    }
    if (valid) {
        if (tiedDistance) {
            atomicAdd(&tileStatistics[4], 1u);
            textureStore(outputImage, vec2<i32>(pixel), vec4<f32>(traceRadiance(origin, direction), 1.0));
            return;
        }
        color = clamp(color + transmittance * camera.background.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
        textureStore(outputImage, vec2<i32>(pixel), vec4<f32>(color, 1.0));
    }
}
