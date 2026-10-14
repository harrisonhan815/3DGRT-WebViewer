// SOG v2 numeric WebP textures, decoded without canvas/color/alpha conversion.
// https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/
import decodeWebp from './vendor/webp/decode.js';

const MAX_MEMBER = 128 * 1024 * 1024;
const fail = message => { throw new Error(`SOG: ${message}`); };
const filename = name => typeof name === 'string' && name.length > 0 && !/[\\/\x00]/.test(name) && !['.', '..'].includes(name);

async function inflate(bytes, size) {
  const reader = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw')).getReader();
  const output = new Uint8Array(size);
  let offset = 0;
  try {
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      if (offset + value.length > size) fail('ZIP 解压超出声明长度');
      output.set(value, offset); offset += value.length;
    }
    if (offset !== size) fail('ZIP 解压长度不符');
    return output;
  } catch (error) { await reader.cancel().catch(() => {}); throw error; }
  finally { reader.releaseLock(); }
}

export function openSogZip(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const check = (offset, size) => {
    if (offset < 0 || size < 0 || offset + size > bytes.length) fail('ZIP 数据不完整');
  };
  // Use the central directory: streaming ZIP writers put zero sizes in the
  // local headers and append data descriptors (including Insta360 files).
  let end = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65557); i--) {
    if (view.getUint32(i, true) === 0x06054b50 && i + 22 + view.getUint16(i + 20, true) === bytes.length) { end = i; break; }
  }
  if (end < 0) fail('缺少 ZIP 目录');
  const count = view.getUint16(end + 10, true), size = view.getUint32(end + 12, true), start = view.getUint32(end + 16, true);
  if (view.getUint32(end + 4, true) !== 0 || view.getUint16(end + 8, true) !== count || !count || count > 64 || start + size !== end) fail('不支持此 ZIP 目录');
  const entries = new Map(), text = new TextDecoder('utf-8', {fatal: true});
  let offset = start;
  for (let i = 0; i < count; i++) {
    check(offset, 46);
    if (view.getUint32(offset, true) !== 0x02014b50) fail('无效 ZIP 目录项');
    const flags = view.getUint16(offset + 8, true), method = view.getUint16(offset + 10, true);
    const packed = view.getUint32(offset + 20, true), unpacked = view.getUint32(offset + 24, true);
    const nameSize = view.getUint16(offset + 28, true), extra = view.getUint16(offset + 30, true), comment = view.getUint16(offset + 32, true);
    const local = view.getUint32(offset + 42, true);
    check(offset + 46, nameSize + extra + comment);
    const name = text.decode(bytes.subarray(offset + 46, offset + 46 + nameSize));
    if (!filename(name) || entries.has(name) || flags & 1 || ![0, 8].includes(method) || view.getUint16(offset + 34, true) !== 0 || unpacked > MAX_MEMBER || packed > MAX_MEMBER) fail('不支持或过大的 ZIP 文件项');
    check(local, 30);
    if (view.getUint32(local, true) !== 0x04034b50 || view.getUint16(local + 6, true) !== flags || view.getUint16(local + 8, true) !== method) fail('ZIP 头部不一致');
    const localName = view.getUint16(local + 26, true), localExtra = view.getUint16(local + 28, true);
    check(local + 30, localName + localExtra);
    if (text.decode(bytes.subarray(local + 30, local + 30 + localName)) !== name) fail('ZIP 文件名不一致');
    const data = local + 30 + localName + localExtra;
    check(data, packed);
    if (data + packed > start || (method === 0 && packed !== unpacked)) fail('无效 ZIP 文件长度');
    entries.set(name, {data, packed, unpacked, method});
    offset += 46 + nameSize + extra + comment;
  }
  if (offset !== end) fail('ZIP 目录长度不符');
  return async name => {
    const entry = entries.get(name);
    if (!filename(name) || !entry) fail(`缺少 ${name}`);
    if (name === 'meta.json' && entry.unpacked > 1048576) fail('元数据过大');
    const data = bytes.subarray(entry.data, entry.data + entry.packed);
    return entry.method === 0 ? data : inflate(data, entry.unpacked);
  };
}

function webpDimensions(bytes) {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 25 || v.getUint32(0, true) !== 0x46464952 || v.getUint32(8, true) !== 0x50424557 || v.getUint32(4, true) + 8 !== bytes.length) fail('无效 WebP 数据');
  for (let offset = 12; offset + 8 <= bytes.length;) {
    const type = v.getUint32(offset, true), size = v.getUint32(offset + 4, true), start = offset + 8;
    if (start + size > bytes.length) fail('WebP 块长度不符');
    // SOG must use lossless VP8L; reject animation and lossy VP8 payloads.
    if (type === 0x4c385056 && size >= 5 && bytes[start] === 0x2f) {
      const bits = v.getUint32(start + 1, true);
      return [(bits & 0x3fff) + 1, ((bits >>> 14) & 0x3fff) + 1];
    }
    if ([0x20385056, 0x4d494e41, 0x464d4e41].includes(type)) fail('需要静态无损 WebP');
    offset = start + size + (size & 1);
  }
  fail('缺少无损 WebP 图像');
}

export function floatToHalf(value) {
  // Round a float32 codebook value to IEEE binary16, ties to even. Doing
  // this for 256-entry codebooks avoids a large intermediate FP32 SH array.
  halfFloat[0] = value;
  const bits = halfBits[0], sign = (bits >>> 16) & 0x8000;
  const exponent = ((bits >>> 23) & 255) - 127;
  if (!Number.isFinite(value) || Math.abs(value) > 65504) fail('SH 超出 FP16 范围，请用 float32 导出');
  if (exponent < -25) return sign;
  const shift = exponent < -14 ? -exponent - 1 : 13;
  const mantissa = (bits & 0x7fffff) | (exponent < -14 ? 0x800000 : 0);
  const truncated = mantissa >>> shift, remainder = mantissa & ((1 << shift) - 1), midpoint = 1 << (shift - 1);
  const rounded = truncated + (remainder > midpoint || (remainder === midpoint && (truncated & 1)) ? 1 : 0);
  return sign | ((exponent < -14 ? 0 : (exponent + 15) << 10) + rounded);
}
const halfFloat = new Float32Array(1), halfBits = new Uint32Array(halfFloat.buffer);

export async function decodeSog(bytes, manifest) {
  const read = openSogZip(bytes);
  const meta = JSON.parse(new TextDecoder().decode(await read('meta.json')));
  const count = meta.count;
  if (meta.version !== 2 || count !== manifest.gaussian_count || !Number.isSafeInteger(count) || count < 1 || count > 10_000_000) fail('版本或高斯数量不符');
  if (![undefined, 'antialiased'].includes(meta.model) || (meta.model === 'antialiased') !== manifest.render.antialiasing) fail('SOG 渲染模型不符');
  const files = (section, size) => {
    const names = meta[section]?.files;
    if (!Array.isArray(names) || names.length !== size || !names.every(filename)) fail(`${section} 图像列表无效`);
    return names;
  };
  const pixels = async name => {
    const data = await read(name), [width, height] = webpDimensions(data);
    if (width * height > 10_000_000 + 65536) fail('WebP 像素数量过大');
    // Canvas 2D premultiplies alpha and loses color-code indices, especially
    // at alpha=0. This WASM decoder returns the actual numeric RGBA bytes.
    const result = await decodeWebp(data);
    if (result.width !== width || result.height !== height || result.data.length !== width * height * 4) fail('WebP 解码尺寸不符');
    return result;
  };
  const book = section => {
    const values = meta[section]?.codebook;
    if (!Array.isArray(values) || values.length !== 256 || !values.every(Number.isFinite)) fail('无效的 256 项码本');
    const result = Float32Array.from(values);
    if (!result.every(Number.isFinite)) fail('码本超出 FP32 范围');
    return result;
  };
  const [lowName, highName] = files('means', 2);
  // Sequential decode keeps the WASM instance serialized and peak memory low.
  const low = await pixels(lowName), high = await pixels(highName);
  const quats = await pixels(files('quats', 1)[0]), scales = await pixels(files('scales', 1)[0]), dc = await pixels(files('sh0', 1)[0]);
  const matches = image => image.width === low.width && image.height === low.height;
  if (low.width * low.height < count || ![high, quats, scales, dc].every(matches)) fail('属性纹理尺寸不符');
  const lower = meta.means.mins, upper = meta.means.maxs;
  if (![lower, upper].every(a => Array.isArray(a) && a.length === 3 && a.every(v => Number.isFinite(v) && Math.abs(v) <= 80)) || lower.some((v, i) => v > upper[i])) fail('坐标范围无效');
  const scaleBook = book('scales');
  if (Math.max(...scaleBook) > 40) fail('协方差超出 FP32 范围');
  const packed = manifest.files.sh.dtype === 'float16';
  const shBook = section => packed ? Uint16Array.from(book(section), floatToHalf) : book(section);
  const dcBook = shBook('sh0'), linearScale = Float64Array.from(scaleBook, Math.exp);
  let bands = 0, coefficients = 0, palette, labels, higherBook, paletteCount;
  if (meta.shN != null) {
    bands = meta.shN.bands; paletteCount = meta.shN.count;
    if (![1, 2, 3].includes(bands) || !Number.isInteger(paletteCount) || paletteCount < 1 || paletteCount > 65536) fail('SH 码本尺寸无效');
    coefficients = [0, 3, 8, 15][bands];
    const [centroids, indices] = files('shN', 2);
    palette = await pixels(centroids); labels = await pixels(indices); higherBook = shBook('shN');
    if (palette.width !== 64 * coefficients || palette.height * 64 < paletteCount || !matches(labels)) fail('SH 纹理尺寸不符');
  }
  if (bands !== manifest.render.sh_degree) fail('SH 阶数与 manifest 不符');
  const geometry = new Float32Array(count * 12);
  const sh = packed ? new Uint16Array(count * 48) : new Float32Array(count * 48);
  const q = new Float64Array(4), factor = new Float64Array(9);
  for (let i = 0; i < count; i++) {
    const p = i * 4, g = i * 12, s = i * 48;
    for (let c = 0; c < 3; c++) {
      const code = low.data[p + c] | high.data[p + c] << 8;
      const log = lower[c] + (upper[c] - lower[c]) * (code / 65535);
      geometry[g + c] = Math.sign(log) * Math.expm1(Math.abs(log));
      sh[s + c] = dcBook[dc.data[p + c]];
    }
    geometry[g + 3] = dc.data[p + 3] / 255;
    const omitted = quats.data[p + 3] - 252;
    if (omitted < 0 || omitted > 3) fail('无效四元数编码');
    let sum = 0, source = 0;
    for (let c = 0; c < 4; c++) if (c !== omitted) { q[c] = (quats.data[p + source++] / 255 - .5) * Math.SQRT2; sum += q[c] ** 2; }
    q[omitted] = Math.sqrt(Math.max(0, 1 - sum));
    const norm = Math.hypot(...q);
    for (let c = 0; c < 4; c++) q[c] /= norm;
    const [w, x, y, z] = q;
    factor.set([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]);
    for (let row = 0; row < 3; row++) for (let c = 0; c < 3; c++) factor[row * 3 + c] *= linearScale[scales.data[p + c]];
    let target = g + 4;
    for (let a = 0; a < 3; a++) for (let b = a; b < 3; b++) {
      geometry[target++] = factor[a*3]*factor[b*3] + factor[a*3+1]*factor[b*3+1] + factor[a*3+2]*factor[b*3+2];
    }
    if (palette) {
      const label = labels.data[p] | labels.data[p + 1] << 8;
      if (label >= paletteCount) fail('SH 索引超出码本');
      for (let c = 0; c < coefficients; c++) for (let channel = 0; channel < 3; channel++) sh[s + 3 + c * 3 + channel] = higherBook[palette.data[(label * coefficients + c) * 4 + channel]];
    }
  }
  if (!geometry.every(Number.isFinite)) fail('解码几何超出 FP32 范围');
  return {geometry: geometry.buffer, sh: sh.buffer};
}
