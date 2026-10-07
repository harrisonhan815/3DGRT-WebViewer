// Browser-only FP32 port of the reference/instances quartic 3DGRT renderer.
// See trace.wgsl for the upstream equation attribution and license.

const PRESENT_SHADER = `
@group(0) @binding(0) var image: texture_2d<f32>;
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let vertices = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(vertices[index], 0.0, 1.0);
}
@fragment fn fragmentMain(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(image, vec2<i32>(position.xy), 0);
}`;

function exactInteger(value, label, minimum = 1) {
    if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`${label} must be an integer >= ${minimum}.`);
    return value;
}

function finiteVector(value, size, label) {
    if (!value || value.length !== size || !Array.from(value).every(Number.isFinite)) {
        throw new Error(`${label} must contain ${size} finite numbers.`);
    }
    return Array.from(value);
}

function bytesOf(value, label) {
    if (value instanceof ArrayBuffer) return new Uint8Array(value);
    if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    throw new Error(`${label} must be an ArrayBuffer or typed array.`);
}

async function checkedShader(device, code, label) {
    const module = device.createShaderModule({code, label});
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === "error");
    if (errors.length) throw new Error(`${label}: ${errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")}`);
    return module;
}

export class WebGpuRayRenderer {
    static async create(canvas) {
        if (!globalThis.isSecureContext || !navigator.gpu) {
            throw new Error("WebGPU is unavailable. Use an up-to-date WebGPU browser with GPU acceleration, over HTTPS or localhost.");
        }
        let adapter = null;
        // Some browsers report a transient null while their GPU process is
        // starting. Keep retries short, bounded and visible as initialization.
        for (let attempt = 0; attempt < 3 && !adapter; attempt++) {
            if (attempt) await new Promise(resolve => setTimeout(resolve, attempt * 250));
            adapter = await navigator.gpu.requestAdapter({powerPreference: "high-performance"});
        }
        if (!adapter) throw new Error("No WebGPU adapter is available. Check browser GPU acceleration and graphics drivers.");
        // Ask for the adapter's storage-buffer limits; browser defaults are
        // often smaller than the hardware supports. Actual allocations still
        // follow the exported model sizes and are checked before uploading.
        const device = await adapter.requestDevice({
            label: "3DGRT WebGPU viewer",
            requiredFeatures: adapter.features.has("timestamp-query") ? ["timestamp-query"] : [],
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
            },
        });
        const renderer = new WebGpuRayRenderer(canvas, adapter, device);
        try {
            await renderer.initialize();
            return renderer;
        } catch (error) {
            renderer.dispose();
            throw error;
        }
    }

    constructor(canvas, adapter, device) {
        this.canvas = canvas;
        this.device = device;
        this.adapterInfo = adapter.info ?? {};
        this.context = canvas.getContext("webgpu");
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.modelBuffers = [];
        this.uniformData = new ArrayBuffer(128);
        this.uniformFloats = new Float32Array(this.uniformData);
        this.uniformUints = new Uint32Array(this.uniformData);
        this.width = 0;
        this.height = 0;
        this.model = null;
        this.modelRevision = 0;
        this.previewTarget = null;
        this.camera = null;
        this.disposed = false;
        this.closing = false;
        this.shutdownPromise = null;
        this.rendering = false;
        this.lostReason = null;
        this.frameCount = 0;
        this.lastGpuMs = null;
        this.lastTileStats = null;
        this.lastMode = null;
        device.lost.then(info => { this.lostReason = info.message || "The graphics device was lost."; });
    }

    async initialize() {
        if (!this.context) throw new Error("Unable to create a WebGPU canvas context.");
        const response = await fetch(new URL("./trace.wgsl", import.meta.url));
        if (!response.ok) throw new Error(`Cannot load trace.wgsl: HTTP ${response.status}.`);
        const traceSource = await response.text();
        const tiledResponse = await fetch(new URL("./tiled.wgsl", import.meta.url));
        if (!tiledResponse.ok) throw new Error(`Cannot load tiled.wgsl: HTTP ${tiledResponse.status}.`);
        const tiledSource = traceSource.slice(0, traceSource.indexOf("@compute @workgroup_size")) + await tiledResponse.text();
        const traceModule = await checkedShader(this.device, traceSource, "3DGRT trace.wgsl");
        const tiledModule = await checkedShader(this.device, tiledSource, "3DGRT tiled.wgsl");
        const presentModule = await checkedShader(this.device, PRESENT_SHADER, "3DGRT present");
        this.computePipeline = await this.device.createComputePipelineAsync({
            label: "3DGRT quartic reference tracing", layout: "auto",
            compute: {module: traceModule, entryPoint: "trace"},
        });
        this.tiledPipeline = await this.device.createComputePipelineAsync({
            label: "3DGRT tiled reference tracing", layout: "auto",
            compute: {module: tiledModule, entryPoint: "traceTiled"},
        });
        this.projectPipeline = await this.device.createComputePipelineAsync({
            label: "3DGRT conservative support projection", layout: "auto",
            compute: {module: tiledModule, entryPoint: "project"},
        });
        this.presentPipeline = await this.device.createRenderPipelineAsync({
            label: "3DGRT present without additional gamma", layout: "auto",
            vertex: {module: presentModule, entryPoint: "vertexMain"},
            fragment: {module: presentModule, entryPoint: "fragmentMain", targets: [{format: this.format}]},
            primitive: {topology: "triangle-list"},
        });
        this.uniformBuffer = this.device.createBuffer({
            label: "3DGRT camera", size: this.uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.tileStatsBuffer = this.device.createBuffer({size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
        this.tileStatsReadback = this.device.createBuffer({size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
        if (this.device.features.has("timestamp-query")) {
            this.timestampQuery = this.device.createQuerySet({label: "3DGRT compute timing", type: "timestamp", count: 2});
            this.timestampResolve = this.device.createBuffer({size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC});
            this.timestampReadback = this.device.createBuffer({size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
        }
        this.context.configure({device: this.device, format: this.format, alphaMode: "opaque", colorSpace: "srgb"});
    }

    assertAvailable() {
        if (this.disposed) throw new Error("Renderer has been disposed.");
        if (this.closing) throw new DOMException("Renderer is shutting down.", "AbortError");
        if (this.lostReason) throw new Error(`WebGPU device lost: ${this.lostReason} Reload the page to reconnect.`);
    }

    async loadModel({geometry, sh, bvh, metadata}) {
        this.assertAvailable();
        if (this.rendering) throw new Error("Wait for the current render before loading a model.");
        const count = exactInteger(metadata.gaussian_count, "gaussian_count");
        const nodeCount = exactInteger(metadata.bvh_node_count, "bvh_node_count");
        const depth = exactInteger(metadata.bvh_max_depth, "bvh_max_depth", 0);
        if (depth > 62) throw new Error("This BVH exceeds the supported traversal depth (62). Re-export it using the bundled exporter.");
        const render = metadata.render;
        if (!render || render.kernel_degree !== 4 || render.primitive_type !== "instances" || render.pipeline !== "reference") {
            throw new Error("This renderer requires a reference/instances model with kernel degree 4. Re-export a supported checkpoint.");
        }
        exactInteger(render.sh_degree, "render.sh_degree", 0);
        if (render.sh_degree > 3) throw new Error("Only SH degrees 0 through 3 are supported.");
        for (const name of ["min_response", "min_alpha", "min_transmittance"]) {
            if (!Number.isFinite(render[name]) || render[name] < 0 || render[name] >= 1) {
                throw new Error(`render.${name} must be in [0, 1).`);
            }
        }
        if (render.max_alpha !== undefined && render.max_alpha !== 0.99) throw new Error("The native maximum alpha must be 0.99.");
        const background = finiteVector(render.background, 3, "render.background");
        const arrays = [bytesOf(geometry, "geometry"), bytesOf(sh, "sh"), bytesOf(bvh, "bvh")];
        const sizes = [count * 64, count * 192, nodeCount * 32];
        const labels = ["Gaussian geometry", "SH coefficients", "BVH"];
        const limit = Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
        for (let i = 0; i < arrays.length; i++) {
            if (arrays[i].byteLength !== sizes[i]) throw new Error(`${labels[i]} size is ${arrays[i].byteLength}; expected ${sizes[i]} bytes.`);
            if (sizes[i] > limit) {
                throw new Error(`${labels[i]} needs ${(sizes[i] / 1048576).toFixed(1)} MiB, but this browser/GPU allows ${(limit / 1048576).toFixed(1)} MiB per storage buffer. File splitting only solves download limits; use a device with larger WebGPU buffer limits.`);
            }
        }
        const nextBuffers = [];
        let nextProjected;
        this.device.pushErrorScope("out-of-memory");
        this.device.pushErrorScope("validation");
        let uploadError;
        try {
            for (let i = 0; i < arrays.length; i++) {
                const buffer = this.device.createBuffer({label: labels[i], size: sizes[i], usage: GPUBufferUsage.STORAGE, mappedAtCreation: true});
                nextBuffers.push(buffer);
                new Uint8Array(buffer.getMappedRange()).set(arrays[i]);
                buffer.unmap();
            }
            nextProjected = this.device.createBuffer({label: "Conservative projected support", size: count * 16, usage: GPUBufferUsage.STORAGE});
        } catch (error) {
            uploadError = error;
        }
        const validationError = await this.device.popErrorScope();
        const memoryError = await this.device.popErrorScope();
        if (uploadError || validationError || memoryError || this.disposed || this.lostReason) {
            nextBuffers.forEach(buffer => buffer.destroy());
            nextProjected?.destroy();
            this.assertAvailable();
            throw new Error(`Model GPU upload failed: ${(uploadError || validationError || memoryError).message}`);
        }
        try {
            await this.device.queue.onSubmittedWorkDone();
            this.assertAvailable();
        } catch (error) {
            nextBuffers.forEach(buffer => buffer.destroy());
            nextProjected?.destroy();
            throw error;
        }
        this.modelBuffers.forEach(buffer => buffer.destroy());
        this.projectedBuffer?.destroy();
        this.modelBuffers = nextBuffers;
        this.projectedBuffer = nextProjected;
        this.model = {...metadata, render: {...render, background}};
        this.modelRevision++;
        this.lastGpuMs = null;
        this.lastTileStats = null;
        this.frameCount = 0;
        this.updateBindGroups();
    }

    setCamera({position, right, down, forward, tanHalfFovY}) {
        this.assertAvailable();
        if (!Number.isFinite(tanHalfFovY) || tanHalfFovY <= 0) throw new Error("tanHalfFovY must be positive and finite.");
        this.camera = {
            position: finiteVector(position, 3, "camera.position"),
            right: finiteVector(right, 3, "camera.right"),
            down: finiteVector(down, 3, "camera.down"),
            forward: finiteVector(forward, 3, "camera.forward"),
            tanHalfFovY,
        };
    }

    updateBindGroups() {
        if (!this.outputTexture || this.modelBuffers.length !== 3) return;
        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: this.uniformBuffer}},
                ...this.modelBuffers.map((buffer, i) => ({binding: i + 1, resource: {buffer}})),
                {binding: 4, resource: this.outputTexture.createView()},
            ],
        });
        this.tiledBindGroup = this.device.createBindGroup({
            layout: this.tiledPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: this.uniformBuffer}},
                ...this.modelBuffers.map((buffer, i) => ({binding: i + 1, resource: {buffer}})),
                {binding: 4, resource: this.outputTexture.createView()},
                {binding: 5, resource: {buffer: this.tileStatsBuffer}},
                {binding: 6, resource: {buffer: this.projectedBuffer}},
            ],
        });
        this.projectBindGroup = this.device.createBindGroup({
            layout: this.projectPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: this.uniformBuffer}},
                {binding: 1, resource: {buffer: this.modelBuffers[0]}},
                {binding: 6, resource: {buffer: this.projectedBuffer}},
            ],
        });
        this.presentBindGroup = this.device.createBindGroup({
            layout: this.presentPipeline.getBindGroupLayout(0),
            entries: [{binding: 0, resource: this.outputTexture.createView()}],
        });
    }

    resize(width, height) {
        exactInteger(width, "width");
        exactInteger(height, "height");
        const limit = this.device.limits.maxTextureDimension2D;
        if (width > limit || height > limit) throw new Error(`Render dimensions exceed this GPU's ${limit}px texture limit.`);
        if (width === this.width && height === this.height) return;
        this.outputTexture?.destroy();
        this.width = width;
        this.height = height;
        this.lastGpuMs = null;
        this.lastTileStats = null;
        this.frameCount = 0;
        this.canvas.width = width;
        this.canvas.height = height;
        this.outputTexture = this.device.createTexture({
            label: "3DGRT display image", size: [width, height], format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.updateBindGroups();
    }

    async render({width, height, measureGpu, mode = "tiled", tileCapacity = 2048}) {
        this.assertAvailable();
        if (!this.model || !this.camera) throw new Error("Load a model and set a camera before rendering.");
        if (this.rendering) throw new Error("A frame is already rendering. Await render() before scheduling another.");
        if (mode !== "tiled" && mode !== "bvh") throw new Error("Rendering mode must be tiled or bvh.");
        exactInteger(tileCapacity, "tileCapacity");
        if (tileCapacity > 2048) throw new Error("tileCapacity must not exceed 2048.");
        this.rendering = true;
        const start = performance.now();
        // Sample GPU time once per 30 completed frames by default. Reading a
        // query requires a small extra map; it is included in renderMs, and
        // callers doing benchmarks may explicitly request every frame.
        let queryGpu = false;
        let sampleStats = false;
        let stripCount = 0;
        let gpuMilliseconds = 0;
        this.device.pushErrorScope("validation");
        try {
            this.resize(width, height);
            if (mode !== this.lastMode) {
                this.lastGpuMs = null;
                this.lastTileStats = null;
                this.frameCount = 0;
                this.lastMode = mode;
            }
            sampleStats = measureGpu ?? (this.frameCount % 30 === 0);
            queryGpu = Boolean(this.timestampQuery && sampleStats);
            const camera = this.camera;
            const render = this.model.render;
            this.uniformFloats.fill(0);
            this.uniformFloats.set([...camera.position, camera.tanHalfFovY], 0);
            this.uniformFloats.set(camera.right, 4);
            this.uniformFloats.set(camera.down, 8);
            this.uniformFloats.set(camera.forward, 12);
            this.uniformUints.set([width, height, render.sh_degree, this.model.bvh_node_count], 16);
            this.uniformFloats.set([render.min_response, render.min_alpha, render.min_transmittance, 0], 20);
            this.uniformFloats.set(render.background, 24);
            // Each submission covers at most about 128K pixels. Submit/await
            // strips sequentially so navigation/disposal can stop between
            // them. This bounds queued work, not worst-case ray complexity.
            const stripRows = Math.max(8, Math.floor(Math.min(height, 131072 / width) / 8) * 8);
            for (let row = 0; row < height; row += stripRows) {
                this.assertAvailable();
                const rows = Math.min(stripRows, height - row);
                const last = row + rows === height;
                this.uniformUints.set([row, this.model.gaussian_count, tileCapacity, 0], 28);
                this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
                const encoder = this.device.createCommandEncoder({label: `3DGRT ${mode} strip ${stripCount}`});
                if (row === 0 && mode === "tiled") encoder.clearBuffer(this.tileStatsBuffer);
                const computeDescriptor = {label: `${mode} projection and ray tracing`};
                if (queryGpu) computeDescriptor.timestampWrites = {
                    querySet: this.timestampQuery, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
                };
                const compute = encoder.beginComputePass(computeDescriptor);
                if (row === 0 && mode === "tiled") {
                    compute.setPipeline(this.projectPipeline);
                    compute.setBindGroup(0, this.projectBindGroup);
                    compute.dispatchWorkgroups(Math.ceil(this.model.gaussian_count / 64));
                }
                compute.setPipeline(mode === "tiled" ? this.tiledPipeline : this.computePipeline);
                compute.setBindGroup(0, mode === "tiled" ? this.tiledBindGroup : this.computeBindGroup);
                compute.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(rows / 8));
                compute.end();
                if (queryGpu) {
                    encoder.resolveQuerySet(this.timestampQuery, 0, 2, this.timestampResolve, 0);
                    encoder.copyBufferToBuffer(this.timestampResolve, 0, this.timestampReadback, 0, 16);
                }
                if (last && sampleStats && mode === "tiled") encoder.copyBufferToBuffer(this.tileStatsBuffer, 0, this.tileStatsReadback, 0, 32);
                if (last) {
                    const present = encoder.beginRenderPass({
                        label: "Present completed frame",
                        colorAttachments: [{view: this.context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1]}],
                    });
                    present.setPipeline(this.presentPipeline);
                    present.setBindGroup(0, this.presentBindGroup);
                    present.draw(3);
                    present.end();
                }
                this.device.queue.submit([encoder.finish()]);
                await this.device.queue.onSubmittedWorkDone();
                this.assertAvailable();
                if (queryGpu) {
                    await this.timestampReadback.mapAsync(GPUMapMode.READ);
                    try {
                        const stamps = new BigUint64Array(this.timestampReadback.getMappedRange());
                        gpuMilliseconds += Number(stamps[1] - stamps[0]) / 1e6;
                    } finally {
                        this.timestampReadback.unmap();
                    }
                }
                stripCount++;
            }
            if (queryGpu) this.lastGpuMs = gpuMilliseconds;
            if (sampleStats && mode === "tiled") {
                await this.tileStatsReadback.mapAsync(GPUMapMode.READ);
                try {
                    const values = new Uint32Array(this.tileStatsReadback.getMappedRange());
                    this.lastTileStats = {tileCount: values[0], overflowTileCount: values[1], fallbackTileCount: values[1],
                        fallbackTiles: values[1], candidateCountSum: values[2], maxCandidates: values[3], tiePixelCount: values[4], tileCapacity};
                } finally {
                    this.tileStatsReadback.unmap();
                }
            }
        } finally {
            this.rendering = false;
            const error = await this.device.popErrorScope();
            if (error) throw new Error(`WebGPU frame failed: ${error.message}`);
        }
        this.assertAvailable();
        this.frameCount++;
        return {renderMs: performance.now() - start, width, height, gpuMs: this.lastGpuMs, gpuMeasuredThisFrame: queryGpu,
            mode, stripCount, tileStats: mode === "tiled" ? this.lastTileStats : null};
    }

    ensurePreviewTarget(canvas, width, height) {
        exactInteger(width, "preview width");
        exactInteger(height, "preview height");
        const limit = this.device.limits.maxTextureDimension2D;
        if (width > limit || height > limit) throw new Error(`Preview dimensions exceed this GPU's ${limit}px texture limit.`);
        if (canvas === this.canvas) throw new Error("The preview needs a separate canvas.");
        if (this.previewTarget && this.previewTarget.canvas !== canvas) this.destroyPreviewTarget();
        if (!this.previewTarget) {
            const context = canvas.getContext("webgpu");
            if (!context) throw new Error("Unable to create the WebGPU preview canvas context.");
            context.configure({device: this.device, format: this.format, alphaMode: "opaque", colorSpace: "srgb"});
            this.previewTarget = {
                canvas, context, width: 0, height: 0, boundRevision: -1, renderedRevision: -1,
                uniformData: new ArrayBuffer(128),
                uniformBuffer: this.device.createBuffer({
                    label: "3DGRT preview camera", size: 128,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                }),
            };
        }
        const target = this.previewTarget;
        if (target.width !== width || target.height !== height) {
            target.texture?.destroy();
            target.width = canvas.width = width;
            target.height = canvas.height = height;
            target.texture = this.device.createTexture({
                label: "3DGRT fisheye preview image", size: [width, height], format: "rgba8unorm",
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
            });
            target.boundRevision = -1;
            target.renderedRevision = -1;
        }
        if (target.boundRevision !== this.modelRevision) {
            target.computeBindGroup = this.device.createBindGroup({
                layout: this.computePipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: target.uniformBuffer}},
                    ...this.modelBuffers.map((buffer, i) => ({binding: i + 1, resource: {buffer}})),
                    {binding: 4, resource: target.texture.createView()},
                ],
            });
            target.presentBindGroup = this.device.createBindGroup({
                layout: this.presentPipeline.getBindGroupLayout(0),
                entries: [{binding: 0, resource: target.texture.createView()}],
            });
            target.boundRevision = this.modelRevision;
        }
        return target;
    }

    async renderPreview(canvas, {width = 192, height = 192, camera = this.camera, measureGpu = false} = {}) {
        this.assertAvailable();
        if (!this.model || !camera) throw new Error("Load a model and provide a camera before rendering a preview.");
        if (this.rendering) throw new Error("Wait for the current render before rendering a preview.");
        const pose = {
            position: finiteVector(camera.position, 3, "preview camera.position"),
            right: finiteVector(camera.right, 3, "preview camera.right"),
            down: finiteVector(camera.down, 3, "preview camera.down"),
            forward: finiteVector(camera.forward, 3, "preview camera.forward"),
        };
        this.rendering = true;
        const started = performance.now();
        const queryGpu = Boolean(measureGpu && this.timestampQuery);
        let gpuMilliseconds = 0;
        let stripCount = 0;
        this.device.pushErrorScope("validation");
        try {
            const target = this.ensurePreviewTarget(canvas, width, height);
            target.renderedRevision = -1;
            const floats = new Float32Array(target.uniformData);
            const uints = new Uint32Array(target.uniformData);
            const settings = this.model.render;
            floats.fill(0);
            floats.set([...pose.position, 1], 0);
            floats.set(pose.right, 4);
            floats.set(pose.down, 8);
            floats.set(pose.forward, 12);
            uints.set([width, height, settings.sh_degree, this.model.bvh_node_count], 16);
            floats.set([settings.min_response, settings.min_alpha, settings.min_transmittance, 0], 20);
            floats.set(settings.background, 24);
            const stripRows = Math.max(8, Math.floor(Math.min(height, 131072 / width) / 8) * 8);
            for (let row = 0; row < height; row += stripRows) {
                this.assertAvailable();
                const rows = Math.min(stripRows, height - row);
                const last = row + rows === height;
                // Perspective tile bounds are invalid for a 180-degree lens.
                // The preview deliberately uses the exact BVH reference path.
                uints.set([row, this.model.gaussian_count, 0, 1], 28);
                this.device.queue.writeBuffer(target.uniformBuffer, 0, target.uniformData);
                const encoder = this.device.createCommandEncoder({label: "3DGRT fisheye preview"});
                const descriptor = {label: "180-degree equidistant BVH tracing"};
                if (queryGpu) descriptor.timestampWrites = {
                    querySet: this.timestampQuery, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
                };
                const compute = encoder.beginComputePass(descriptor);
                compute.setPipeline(this.computePipeline);
                compute.setBindGroup(0, target.computeBindGroup);
                compute.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(rows / 8));
                compute.end();
                if (queryGpu) {
                    encoder.resolveQuerySet(this.timestampQuery, 0, 2, this.timestampResolve, 0);
                    encoder.copyBufferToBuffer(this.timestampResolve, 0, this.timestampReadback, 0, 16);
                }
                if (last) {
                    const present = encoder.beginRenderPass({
                        label: "Present completed fisheye preview",
                        colorAttachments: [{view: target.context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1]}],
                    });
                    present.setPipeline(this.presentPipeline);
                    present.setBindGroup(0, target.presentBindGroup);
                    present.draw(3);
                    present.end();
                }
                this.device.queue.submit([encoder.finish()]);
                await this.device.queue.onSubmittedWorkDone();
                this.assertAvailable();
                if (queryGpu) {
                    await this.timestampReadback.mapAsync(GPUMapMode.READ);
                    try {
                        const stamps = new BigUint64Array(this.timestampReadback.getMappedRange());
                        gpuMilliseconds += Number(stamps[1] - stamps[0]) / 1e6;
                    } finally {
                        this.timestampReadback.unmap();
                    }
                }
                stripCount++;
            }
            target.renderedRevision = this.modelRevision;
        } finally {
            this.rendering = false;
            const error = await this.device.popErrorScope();
            if (error) throw new Error(`WebGPU preview failed: ${error.message}`);
        }
        this.assertAvailable();
        return {renderMs: performance.now() - started, gpuMs: queryGpu ? gpuMilliseconds : null,
            width, height, stripCount, projection: "fisheye-equidistant", mode: "bvh"};
    }

    async readPixels() {
        this.assertAvailable();
        if (this.rendering) throw new Error("Wait for the current render before reading pixels.");
        if (!this.outputTexture) throw new Error("Render a frame before reading pixels.");
        return this.readTexturePixels(this.outputTexture, this.width, this.height);
    }

    async readPreviewPixels() {
        this.assertAvailable();
        if (this.rendering) throw new Error("Wait for the current render before reading preview pixels.");
        const target = this.previewTarget;
        if (!target || target.renderedRevision !== this.modelRevision) throw new Error("Render a preview for the current model before reading preview pixels.");
        return this.readTexturePixels(target.texture, target.width, target.height);
    }

    async readTexturePixels(texture, width, height) {
        const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
        const buffer = this.device.createBuffer({
            label: "3DGRT validation readback", size: bytesPerRow * height,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        try {
            const encoder = this.device.createCommandEncoder();
            encoder.copyTextureToBuffer({texture}, {buffer, bytesPerRow, rowsPerImage: height}, [width, height]);
            this.device.queue.submit([encoder.finish()]);
            await buffer.mapAsync(GPUMapMode.READ);
            const source = new Uint8Array(buffer.getMappedRange());
            const rgba = new Uint8Array(width * height * 4);
            for (let y = 0; y < height; y++) rgba.set(source.subarray(y * bytesPerRow, y * bytesPerRow + width * 4), y * width * 4);
            buffer.unmap();
            return {width, height, rgba};
        } finally {
            buffer.destroy();
        }
    }

    shutdown({timeoutMs = 8000} = {}) {
        if (this.shutdownPromise) return this.shutdownPromise;
        if (this.disposed) return Promise.resolve();
        // Stop asynchronous render/upload continuations before placing a
        // queue fence. A running strip may finish, but no next strip starts.
        this.closing = true;
        this.shutdownPromise = (async () => {
            let timer;
            try {
                if (this.lostReason) throw new Error(this.lostReason);
                await Promise.race([
                    this.device.queue.onSubmittedWorkDone(),
                    new Promise((_, reject) => {
                        timer = setTimeout(() => reject(new Error("Timed out waiting for the GPU to finish.")), timeoutMs);
                    }),
                ]);
                if (this.lostReason) throw new Error(this.lostReason);
            } finally {
                clearTimeout(timer);
                this.dispose();
            }
        })();
        return this.shutdownPromise;
    }

    destroyPreviewTarget() {
        const target = this.previewTarget;
        if (!target) return;
        target.uniformBuffer.destroy();
        target.texture?.destroy();
        target.context.unconfigure();
        this.previewTarget = null;
    }

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.modelBuffers.forEach(buffer => buffer.destroy());
        this.modelBuffers = [];
        this.uniformBuffer?.destroy();
        this.outputTexture?.destroy();
        this.timestampQuery?.destroy();
        this.timestampResolve?.destroy();
        this.timestampReadback?.destroy();
        this.projectedBuffer?.destroy();
        this.tileStatsBuffer?.destroy();
        this.tileStatsReadback?.destroy();
        this.destroyPreviewTarget();
        this.context?.unconfigure();
        this.device.destroy();
        this.model = null;
        this.camera = null;
        this.computeBindGroup = null;
        this.presentBindGroup = null;
        this.tiledBindGroup = null;
        this.projectBindGroup = null;
        this.outputTexture = null;
        this.computePipeline = null;
        this.tiledPipeline = null;
        this.projectPipeline = null;
        this.presentPipeline = null;
    }
}
