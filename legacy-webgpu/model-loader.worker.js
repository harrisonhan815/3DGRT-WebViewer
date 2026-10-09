import {validateFile} from './model-loader.js';

const gzipMagic = data => data.length >= 3 && data[0] === 0x1f && data[1] === 0x8b && data[2] === 8;
const expectedDownload = part => part.downloadByteLength ?? part.byteLength;

async function readPart(part, base, progress) {
  const response = await fetch(new URL(part.path,base),{credentials:'omit'});
  if (!response.ok || !response.body) throw new Error(`模型分片下载失败：${part.path}（HTTP ${response.status}）`);
  // Fetch may already have removed a server's Content-Encoding layer. The
  // JS body can therefore be either stored gzip bytes or decoded bytes.
  // Bound both cases with validated manifest sizes, not Content-Length.
  const capacity = Math.max(expectedDownload(part),part.byteLength);
  const buffer = new Uint8Array(capacity);
  const reader = response.body.getReader();
  let received = 0;
  try {
    while (true) {
      const {value,done} = await reader.read();
      if (done) break;
      if (received + value.byteLength > capacity) throw new Error(`模型分片下载长度超出声明：${part.path}`);
      buffer.set(value,received); received += value.byteLength;
      const anticipated = part.compression === 'gzip' && received >= 3 && gzipMagic(buffer.subarray(0,received)) ? expectedDownload(part) : part.byteLength;
      // This is asset-budget progress, not a claim about physical wire bytes.
      // Reserve the last 10% for decompression, unshuffle and checksum.
      progress(0.9 * expectedDownload(part) * Math.min(1,received/anticipated));
    }
  } catch (error) {
    await reader.cancel().catch(()=>{});
    throw error;
  } finally { reader.releaseLock(); }
  if (received !== expectedDownload(part) && received !== part.byteLength) throw new Error(`模型分片下载不完整：${part.path}`);
  progress(0.9 * expectedDownload(part));
  return buffer.subarray(0,received);
}

async function gunzip(bytes, expected) {
  if (typeof DecompressionStream === 'undefined') throw new Error('此浏览器不支持 gzip 解压，请更新浏览器');
  let position = 0;
  const input = new ReadableStream({pull(controller) {
    if (position === bytes.length) { controller.close(); return; }
    const next = Math.min(position+65536,bytes.length);
    controller.enqueue(bytes.subarray(position,next)); position = next;
  }});
  const reader = input.pipeThrough(new DecompressionStream('gzip')).getReader();
  const decoded = new Uint8Array(expected);
  let offset = 0;
  try {
    while (true) {
      const {value,done} = await reader.read();
      if (done) break;
      if (offset + value.byteLength > expected) throw new Error('模型解压长度超出声明');
      decoded.set(value,offset); offset += value.byteLength;
    }
    // Consume through done, even after expected bytes: gzip CRC/trailer and
    // truncation errors can surface only when the decompressor is flushed.
    if (offset !== expected) throw new Error('模型解压长度与声明不符');
    return decoded;
  } catch (error) {
    await reader.cancel().catch(()=>{});
    throw error;
  } finally { reader.releaseLock(); }
}

async function restoreAndCheck(bytes,part,destination) {
  if (bytes.byteLength !== part.byteLength) throw new Error('模型解码长度与声明不符');
  if (part.transform === 'byte-shuffle-4' || part.transform === 'byte-shuffle-2') {
    // Half SH coefficients stay packed after decoding. The shader expands
    // them to f32 only when read, so CPU and GPU buffers retain the saving.
    const wordBytes = part.transform === 'byte-shuffle-2' ? 2 : 4;
    const words = bytes.length/wordBytes;
    for (let lane=0;lane<wordBytes;lane++) {
      const start = lane*words;
      for (let i=0;i<words;i++) destination[wordBytes*i+lane] = bytes[start+i];
    }
  } else destination.set(bytes);
  if (part.sha256 !== undefined) {
    if (!globalThis.crypto?.subtle) throw new Error('模型校验需要 HTTPS 或 localhost');
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256',destination));
    const hex = Array.from(digest,v=>v.toString(16).padStart(2,'0')).join('');
    if (hex !== part.sha256) throw new Error('模型解压后 SHA256 校验失败');
  }
}

async function decodePart(bytes,part,destination) {
  if ((part.compression ?? 'none') === 'none') {
    await restoreAndCheck(bytes,part,destination);
    return;
  }
  let compressedError;
  if (gzipMagic(bytes) && bytes.byteLength === expectedDownload(part)) {
    try {
      await restoreAndCheck(await gunzip(bytes,part.byteLength),part,destination);
      return;
    } catch (error) { compressedError = error; }
  }
  // Some hosts set Content-Encoding:gzip directly on the .gz asset, so
  // Fetch delivers its already-decompressed payload. Accept that only when
  // inverse transformation and the original SHA256 both match. This also
  // disambiguates valid raw data that happens to start with gzip magic.
  if (bytes.byteLength === part.byteLength) {
    await restoreAndCheck(bytes,part,destination);
    return;
  }
  throw compressedError ?? new Error('压缩分片头部或长度不正确');
}

self.onmessage = async event => {
  try {
    const {file,base} = event.data;
    validateFile(file);
    const result = new Uint8Array(file.byteLength);
    let offset = 0;
    for (const part of file.parts) {
      let reported = 0;
      const progress = count => {
        const target = Math.max(reported,Math.min(expectedDownload(part),Math.floor(count)));
        if (target > reported) self.postMessage({type:'progress',bytes:target-reported});
        reported = target;
      };
      const received = await readPart(part,base,progress);
      await decodePart(received,part,result.subarray(offset,offset+part.byteLength));
      progress(expectedDownload(part));
      offset += part.byteLength;
    }
    self.postMessage({type:'done',buffer:result.buffer},[result.buffer]);
  } catch (error) {
    self.postMessage({type:'error',message:error.message || String(error)});
  }
};
