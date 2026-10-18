// file.byteLength describes the GPU layout, including packed half SH. Only
// the gzip/byte-shuffle transport is decoded here, not the numeric dtype.
// The renderer validates dtype/stride; workers restore and check those bytes.
const MAX_PART_BYTES = 100 * 1024 * 1024;

export function modelPartUrl(part, base) {
  const url = new URL(part.path, base);
  if (part.sha256 !== undefined) {
    // A model can be re-exported at the same path. Bind the HTTP cache entry
    // to this manifest's content, including the binary transport layout:
    // binary SHA256 describes decoded bytes; SOG SHA256 describes the ZIP.
    url.searchParams.set('__fc_asset', [part.sha256, part.downloadByteLength ?? part.byteLength,
      part.byteLength, part.compression ?? 'none', part.transform ?? 'none'].join('.'));
  }
  return url;
}

export function validateModelFiles(manifest) {
  if (manifest.storage === undefined) {
    const sizes = ['geometry', 'sh'].map(key => validateFile(manifest.files?.[key]));
    return sizes.reduce((a, b) => ({decodedBytes: a.decodedBytes + b.decodedBytes, downloadBytes: a.downloadBytes + b.downloadBytes}));
  }
  const storage = manifest.storage, count = manifest.gaussian_count;
  if (!storage || storage.format !== 'sog-v2' || typeof storage.path !== 'string' || !storage.path
      || !Number.isSafeInteger(storage.byteLength) || storage.byteLength < 22 || storage.byteLength > 512 * 1024 * 1024
      || !/^[0-9a-f]{64}$/.test(storage.sha256 ?? '') || !Number.isSafeInteger(count) || count < 1 || count > 10_000_000) throw new Error('无效 SOG 下载元数据');
  const dtype = manifest.files?.sh?.dtype, stride = dtype === 'float16' ? 96 : dtype === 'float32' ? 192 : 0;
  if (!stride || manifest.files?.geometry?.byteLength !== count * 48 || manifest.files.geometry.stride !== 48
      || manifest.files.sh.byteLength !== count * stride || manifest.files.sh.stride !== stride) throw new Error('SOG 解码缓冲区大小不符');
  return {downloadBytes: storage.byteLength, decodedBytes: count * (48 + stride)};
}

export function loadModelBuffers(manifest, base, signal, onProgress = () => {}) {
  validateModelFiles(manifest);
  if (manifest.storage === undefined) {
    return Promise.all(['geometry', 'sh'].map(key => loadBuffer(manifest.files[key], base, signal, onProgress)))
      .then(([geometry, sh]) => ({geometry, sh}));
  }
  return new Promise((resolve, reject) => {
    let worker, finished = false;
    const cleanup = () => { signal?.removeEventListener('abort', abort); worker?.terminate(); };
    const fail = error => { if (!finished) { finished = true; cleanup(); reject(error); } };
    const abort = () => fail(new DOMException('Model loading was cancelled.', 'AbortError'));
    try {
      if (signal?.aborted) { abort(); return; }
      worker = new Worker(new URL('./model-loader.worker.js?v=20261016-model-cache-1', import.meta.url), {type: 'module'});
      signal?.addEventListener('abort', abort, {once: true});
      worker.onmessage = event => {
        if (finished) return;
        const message = event.data;
        if (message.type === 'progress') {
          try { onProgress(message.bytes); } catch (error) { fail(error); }
        } else if (message.type === 'done') {
          for (const key of ['geometry', 'sh']) {
            if (!(message[key] instanceof ArrayBuffer) || message[key].byteLength !== manifest.files[key].byteLength) {
              fail(new Error('SOG 后台解码长度错误')); return;
            }
          }
          finished = true; cleanup(); resolve({geometry: message.geometry, sh: message.sh});
        } else if (message.type === 'error') fail(new Error(message.message));
      };
      worker.onerror = event => { event.preventDefault(); fail(new Error(event.message || 'SOG 解码进程失败')); };
      worker.postMessage({manifest, base: String(base)});
    } catch (error) { fail(error); }
  });
}

export function validateFile(file) {
  if (!file || !Number.isSafeInteger(file.byteLength) || file.byteLength <= 0 || !Array.isArray(file.parts) || !file.parts.length) throw new Error('无效模型缓冲区');
  let decoded = 0, download = 0;
  for (const part of file.parts) {
    if (!part || typeof part.path !== 'string' || !part.path || !Number.isSafeInteger(part.byteLength) || part.byteLength <= 0 || part.byteLength > MAX_PART_BYTES) throw new Error('无效模型分片长度或地址');
    const stored = part.downloadByteLength ?? part.byteLength;
    if (!Number.isSafeInteger(stored) || stored <= 0 || stored > MAX_PART_BYTES) throw new Error('无效模型分片下载长度');
    const compression = part.compression ?? 'none';
    const transform = part.transform ?? 'none';
    if (!['none','gzip'].includes(compression) || !['none','byte-shuffle-2','byte-shuffle-4'].includes(transform)) throw new Error('不支持的模型压缩格式');
    if (compression === 'none' && (transform !== 'none' || stored !== part.byteLength)) throw new Error('未压缩模型分片的元数据不一致');
    const wordBytes = transform === 'byte-shuffle-2' ? 2 : 4;
    if (transform !== 'none' && part.byteLength % wordBytes !== 0) throw new Error(`字节重排分片必须对齐到 ${wordBytes} 字节`);
    if ((part.sha256 !== undefined || compression !== 'none') && (typeof part.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(part.sha256))) throw new Error('模型分片缺少有效的 SHA256');
    decoded += part.byteLength; download += stored;
  }
  if (!Number.isSafeInteger(decoded) || decoded !== file.byteLength || !Number.isSafeInteger(download)) throw new Error('模型分片总长度不一致');
  return {decodedBytes:decoded,downloadBytes:download};
}

export function loadBuffer(file, base, signal, onProgress = () => {}) {
  return new Promise((resolve,reject) => {
    let worker, finished = false;
    const cleanup = () => {
      signal?.removeEventListener('abort',abort);
      worker?.terminate();
    };
    const fail = error => {
      if (finished) return;
      finished = true; cleanup(); reject(error);
    };
    const abort = () => fail(new DOMException('Model loading was cancelled.','AbortError'));
    try {
      validateFile(file);
      if (signal?.aborted) { abort(); return; }
      if (typeof Worker === 'undefined') throw new Error('此浏览器不支持模型后台加载，请更新浏览器');
      worker = new Worker(new URL('./model-loader.worker.js?v=20261016-model-cache-1',import.meta.url),{type:'module'});
      signal?.addEventListener('abort',abort,{once:true});
      worker.onmessage = event => {
        if (finished) return;
        const message = event.data;
        if (message.type === 'progress') {
          try { onProgress(message.bytes); } catch (error) { fail(error); }
        } else if (message.type === 'done') {
          if (!(message.buffer instanceof ArrayBuffer) || message.buffer.byteLength !== file.byteLength) { fail(new Error('模型后台加载返回了错误长度')); return; }
          finished = true; cleanup(); resolve(message.buffer);
        } else if (message.type === 'error') {
          fail(new Error(message.message));
        }
      };
      worker.onerror = event => { event.preventDefault(); fail(new Error(event.message || '模型解压进程失败')); };
      worker.postMessage({file,base:String(base)});
    } catch (error) { fail(error); }
  });
}
