import {loadModelBuffers, validateModelFiles} from './model-loader.js?v=20261016-model-cache-1';

export const VARIANT_LABELS = {sog: 'SOG', bin: '二进制'};
const FIELDS = new Set(['files', 'storage', 'precision', 'gaussian_order', 'decoded_bounds', 'known_differences', 'native_support']);
const cancelled = () => new DOMException('Model loading was cancelled.', 'AbortError');

export function modelVariants(root) {
  if (root.variants === undefined) {
    const key = root.storage?.format === 'sog-v2' ? 'sog' : 'bin';
    validateModelFiles(root);
    return {default: key, models: new Map([[key, root]])};
  }
  const variants = root.variants;
  if (!variants || !['bin', 'sog'].includes(variants.default) || Object.keys(variants).some(key => !['default', 'bin', 'sog'].includes(key))) throw new Error('无效模型版本列表');
  const common = Object.fromEntries(Object.entries(root).filter(([key]) => key !== 'variants' && !FIELDS.has(key)));
  const models = new Map();
  for (const key of ['sog', 'bin']) {
    const spec = variants[key];
    if (!spec || Object.keys(spec).some(name => !FIELDS.has(name))) throw new Error('模型版本必须共用相机、渲染参数和校正身份');
    const metadata = {...common, ...spec};
    if ((key === 'bin' && metadata.storage !== undefined) || (key === 'sog' && metadata.storage?.format !== 'sog-v2')) throw new Error('模型版本与编码格式不符');
    validateModelFiles(metadata);
    models.set(key, metadata);
  }
  return {default: variants.default, models};
}

// Only requested encodings of the CURRENT logical model are cached. Each
// encoding has its own abort controller; unused versions are never prefetched.
// Typed arrays stay on CPU; the renderer owns a single active GPU model.
export class ModelVariants {
  constructor(root, base, {onChange = () => {}, loader = loadModelBuffers} = {}) {
    const specs = modelVariants(root);
    this.default = specs.default; this.base = String(base); this.onChange = onChange; this.loader = loader; this.disposed = false;
    this.entries = new Map([...specs.models].map(([key, metadata]) => [key, {
      key, metadata, status: 'idle', received: 0, total: validateModelFiles(metadata).downloadBytes,
      buffers: null, promise: null, controller: null, error: null,
    }]));
  }
  load(key, onProgress = () => {}) {
    if (this.disposed) return Promise.reject(cancelled());
    const entry = this.entries.get(key);
    if (!entry) return Promise.reject(new Error('该模型没有此版本'));
    if (entry.status === 'ready') return Promise.resolve(entry.buffers);
    if (entry.promise) return entry.promise;
    entry.controller = new AbortController();
    entry.status = 'loading'; entry.received = 0; entry.error = null; entry.lastProgressUpdate = 0;
    this.onChange();
    const task = Promise.resolve().then(() => this.loader(entry.metadata, this.base, entry.controller.signal, n => {
      if (this.disposed) return;
      entry.received += n; onProgress(n);
      // Background downloads should not trigger hundreds of DOM updates
      // while the user is navigating the already displayed version.
      const now = performance.now();
      if (now - entry.lastProgressUpdate >= 100) { entry.lastProgressUpdate = now; this.onChange(); }
    })).then(buffers => {
      if (this.disposed) throw cancelled();
      entry.buffers = buffers; entry.status = 'ready'; return buffers;
    }).catch(error => {
      entry.controller?.abort(); // also abort sibling geometry/SH file loads
      if (!this.disposed) { entry.status = 'error'; entry.error = error; }
      throw error;
    }).finally(() => {
      entry.promise = null; entry.controller = null;
      if (!this.disposed) this.onChange();
    });
    entry.promise = task;
    task.catch(() => {}); // cancellation can outlive its UI consumer
    return task;
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    for (const entry of this.entries.values()) { entry.controller?.abort(); entry.buffers = null; }
    this.entries.clear();
    this.onChange = () => {};
  }
}
