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
        this.uniformData = new ArrayBuffer(112);
        this.uniformFloats = new Float32Array(this.uniformData);
        this.uniformUints = new Uint32Array(this.uniformData);
        this.width = 0;
        this.height = 0;
        this.model = null;
        this.camera = null;
        this.disposed = false;
        this.rendering = false;
        this.lostReason = null;
        this.frameCount = 0;
        this.lastGpuMs = null;
        device.lost.then(info => { this.lostReason = info.message || "The graphics device was lost."; });
    }

    async initialize() {
        if (!this.context) throw new Error("Unable to create a WebGPU canvas context.");
        const response = await fetch(new URL("./trace.wgsl", import.meta.url));
        if (!response.ok) throw new Error(`Cannot load trace.wgsl: HTTP ${response.status}.`);
        const traceSource = await response.text();
        const traceModule = await checkedShader(this.device, traceSource, "3DGRT trace.wgsl");
        const presentModule = await checkedShader(this.device, PRESENT_SHADER, "3DGRT present");
        this.computePipeline = await this.device.createComputePipelineAsync({
            label: "3DGRT quartic reference tracing", layout: "auto",
            compute: {module: traceModule, entryPoint: "trace"},
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
        if (this.device.features.has("timestamp-query")) {
            this.timestampQuery = this.device.createQuerySet({label: "3DGRT compute timing", type: "timestamp", count: 2});
            this.timestampResolve = this.device.createBuffer({size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC});
            this.timestampReadback = this.device.createBuffer({size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
        }
        this.context.configure({device: this.device, format: this.format, alphaMode: "opaque", colorSpace: "srgb"});
    }

    assertAvailable() {
        if (this.disposed) throw new Error("Renderer has been disposed.");
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
        } catch (error) {
            uploadError = error;
        }
        const validationError = await this.device.popErrorScope();
        const memoryError = await this.device.popErrorScope();
        if (uploadError || validationError || memoryError) {
            nextBuffers.forEach(buffer => buffer.destroy());
            throw new Error(`Model GPU upload failed: ${(uploadError || validationError || memoryError).message}`);
        }
        await this.device.queue.onSubmittedWorkDone();
        this.modelBuffers.forEach(buffer => buffer.destroy());
        this.modelBuffers = nextBuffers;
        this.model = {...metadata, render: {...render, background}};
        this.lastGpuMs = null;
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
        this.frameCount = 0;
        this.canvas.width = width;
        this.canvas.height = height;
        this.outputTexture = this.device.createTexture({
            label: "3DGRT display image", size: [width, height], format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.updateBindGroups();
    }

    async render({width, height, measureGpu}) {
        this.assertAvailable();
        if (!this.model || !this.camera) throw new Error("Load a model and set a camera before rendering.");
        if (this.rendering) throw new Error("A frame is already rendering. Await render() before scheduling another.");
        this.rendering = true;
        const start = performance.now();
        // Sample GPU time once per 30 completed frames by default. Reading a
        // query requires a small extra map; it is included in renderMs, and
        // callers doing benchmarks may explicitly request every frame.
        let queryGpu = false;
        this.device.pushErrorScope("validation");
        try {
            this.resize(width, height);
            queryGpu = Boolean(this.timestampQuery && (measureGpu ?? (this.frameCount % 30 === 0)));
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
            this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
            const encoder = this.device.createCommandEncoder({label: "3DGRT frame"});
            const computeDescriptor = {label: "BVH ray tracing"};
            if (queryGpu) computeDescriptor.timestampWrites = {
                querySet: this.timestampQuery, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
            };
            const compute = encoder.beginComputePass(computeDescriptor);
            compute.setPipeline(this.computePipeline);
            compute.setBindGroup(0, this.computeBindGroup);
            compute.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
            compute.end();
            if (queryGpu) {
                encoder.resolveQuerySet(this.timestampQuery, 0, 2, this.timestampResolve, 0);
                encoder.copyBufferToBuffer(this.timestampResolve, 0, this.timestampReadback, 0, 16);
            }
            const present = encoder.beginRenderPass({
                label: "Present frame",
                colorAttachments: [{view: this.context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1]}],
            });
            present.setPipeline(this.presentPipeline);
            present.setBindGroup(0, this.presentBindGroup);
            present.draw(3);
            present.end();
            this.device.queue.submit([encoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            if (queryGpu) {
                await this.timestampReadback.mapAsync(GPUMapMode.READ);
                try {
                    const stamps = new BigUint64Array(this.timestampReadback.getMappedRange());
                    this.lastGpuMs = Number(stamps[1] - stamps[0]) / 1e6;
                } finally {
                    this.timestampReadback.unmap();
                }
            }
        } finally {
            this.rendering = false;
            const error = await this.device.popErrorScope();
            if (error) throw new Error(`WebGPU frame failed: ${error.message}`);
        }
        this.assertAvailable();
        this.frameCount++;
        return {renderMs: performance.now() - start, width, height, gpuMs: this.lastGpuMs, gpuMeasuredThisFrame: queryGpu};
    }

    async readPixels() {
        this.assertAvailable();
        if (this.rendering) throw new Error("Wait for the current render before reading pixels.");
        if (!this.outputTexture) throw new Error("Render a frame before reading pixels.");
        const {width, height} = this;
        const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
        const buffer = this.device.createBuffer({
            label: "3DGRT validation readback", size: bytesPerRow * height,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        try {
            const encoder = this.device.createCommandEncoder();
            encoder.copyTextureToBuffer({texture: this.outputTexture}, {buffer, bytesPerRow, rowsPerImage: height}, [width, height]);
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

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.modelBuffers.forEach(buffer => buffer.destroy());
        this.uniformBuffer?.destroy();
        this.outputTexture?.destroy();
        this.timestampQuery?.destroy();
        this.timestampResolve?.destroy();
        this.timestampReadback?.destroy();
        this.context?.unconfigure();
        this.device.destroy();
        this.model = null;
    }
}
