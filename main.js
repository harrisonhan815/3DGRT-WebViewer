import { WebGpuRayRenderer } from './renderer.js';
import { AdaptiveResolution } from './adaptive-resolution.js';

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const previewCanvas = $('fisheye-view');
let renderer, manifest, camera, worldUp, ready = false, loading = false, dirty = true, rendered = false;
let catalog = [], currentId = '', loadController, busy = false, capturing = false, baseSpeed = 1;
let frameTimes = [], previousFrame = 0, lastUiUpdate = 0;
let stopped = false, animationFrame = null;
let rendererInitialization = null, exitPromise = null;
let cameraRevision = 0, mainRevision = -1, previewRevision = -1;
let lastMainCamera = null, lastInteraction = -Infinity;
const keys = new Set();
const adaptive = new AdaptiveResolution();

function haltViewer() {
  if (stopped) return;
  stopped = true;
  ready = false;
  keys.clear();
  if (animationFrame !== null) cancelAnimationFrame(animationFrame);
  animationFrame = null;
  loadController?.abort();
}

function stopViewer() {
  haltViewer();
  // pagehide can arrive while the explicit exit button is waiting for a
  // fence; disposal must still run even if scheduling has already stopped.
  renderer?.dispose();
}

async function safeExit() {
  if (exitPromise) return exitPromise;
  haltViewer();
  const button = $('safe-exit');
  button.disabled = true;
  button.textContent = '正在退出…';
  for (const input of document.querySelectorAll('#panel button, #panel input, #panel select, #toggle')) input.disabled = true;
  $('loading').hidden = true;
  $('error').hidden = true;
  const panel = $('exit-panel');
  panel.hidden = false;
  panel.dataset.state = 'closing';
  panel.setAttribute('aria-busy', 'true');
  $('exit-title').textContent = '正在安全退出';
  $('exit-status').textContent = '已停止渲染和下载，正在等待显卡完成当前工作…';
  panel.focus();
  exitPromise = (async () => {
    let initializationTimer;
    try {
      // The button is also available while the first device is initializing.
      // main() disposes a late-created device when it sees stopped=true.
      const activeRenderer = renderer ?? await Promise.race([
        rendererInitialization?.catch(() => null) ?? Promise.resolve(null),
        new Promise((_, reject) => {
          initializationTimer = setTimeout(() => reject(new Error('GPU initialization did not finish.')), 8000);
        }),
      ]);
      clearTimeout(initializationTimer);
      await activeRenderer?.shutdown();
      renderer = null;
      rendererInitialization = null;
      manifest = null;
      camera = null;
      worldUp = null;
      catalog = [];
      frameTimes = [];
      $('model').replaceChildren();
      $('camera').replaceChildren();
      canvas.width = canvas.height = 1;
      canvas.hidden = true;
      $('fisheye-panel').hidden = true;
      previewCanvas.width = previewCanvas.height = 1;
      lastMainCamera = null;
      panel.dataset.state = 'closed';
      panel.setAttribute('aria-busy', 'false');
      $('exit-title').textContent = '已安全退出';
      $('exit-status').textContent = '资源已释放，可以手动关闭此标签页。';
      button.textContent = '已退出';
      // Normal user-opened tabs may reject close(). Keep the cleared page
      // and an honest manual-close message; do not use close-policy tricks.
      try { window.close(); } catch { /* Browser requires manual closing. */ }
    } catch (error) {
      renderer?.dispose();
      panel.dataset.state = 'error';
      panel.setAttribute('aria-busy', 'false');
      $('exit-title').textContent = '渲染已停止';
      $('exit-status').textContent = '已请求清理资源，但未能确认显卡完成当前工作。请手动关闭此标签页。';
      button.textContent = '已停止';
      console.warn('Safe exit could not confirm GPU completion:', error);
    } finally {
      clearTimeout(initializationTimer);
    }
  })();
  return exitPromise;
}

$('safe-exit').addEventListener('click', safeExit);

// pagehide covers navigation and back/forward-cache entry without adding an
// unload listener. Restoring a cached page must create a new WebGPU device.
window.addEventListener('pagehide', stopViewer);
window.addEventListener('pageshow', (event) => {
  if (event.persisted && stopped) location.reload();
});

function scheduleFrame() {
  if (!stopped) animationFrame = requestAnimationFrame(frame);
}
const length = (v) => Math.hypot(...v);
const norm = (v) => { const l = length(v); return v.map((x) => x / l); };
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
function rotate(v, axis, angle) {
  const c = Math.cos(angle), s = Math.sin(angle), av = dot(axis, v), axv = cross(axis, v);
  return v.map((x, i) => x*c + axv[i]*s + axis[i]*av*(1-c));
}
function message(error) {
  if (stopped) return;
  console.error(error);
  $('error').textContent = error instanceof Error ? error.message : String(error);
  $('error').hidden = false;
}

function markCameraChanged(moving = false) {
  cameraRevision++;
  dirty = true;
  if (moving) lastInteraction = performance.now();
  if ($('fisheye-enabled').checked) {
    $('fisheye-status').textContent = moving ? '移动中，停止后更新' : '等待当前视角';
  }
}

function copyCamera(value) {
  return {
    position: [...value.position], right: [...value.right], down: [...value.down],
    forward: [...value.forward], tanHalfFovY: value.tanHalfFovY,
  };
}

function invalidatePreview() {
  previewRevision = -1;
  mainRevision = -1;
  lastMainCamera = null;
  previewCanvas.hidden = true;
  $('fisheye-status').textContent = '等待当前视角';
}
function applyPreset(index) {
  const p = manifest.cameras[index];
  const r = p.rotation;
  camera = {
    position: [...p.position],
    right: [r[0][0], r[1][0], r[2][0]],
    down: [r[0][1], r[1][1], r[2][1]],
    forward: [r[0][2], r[1][2], r[2][2]],
    tanHalfFovY: Math.tan(p.fov_y / 2),
  };
  worldUp = camera.down.map((x) => -x);
  $('fov').value = (p.fov_y * 180 / Math.PI).toFixed(1);
  adaptive.reset();
  lastInteraction = -Infinity;
  markCameraChanged();
}

async function fetchJson(url, signal) {
  const res = await fetch(url, { signal, credentials: 'omit' });
  if (!res.ok) throw new Error(`${res.status}: ${url}`);
  return res.json();
}

// Content-Length can be missing or compressed on CDNs: the manifest describes
// decoded byte lengths, and the stream is checked before upload to WebGPU.
async function loadBuffer(file, base, signal, onBytes) {
  const result = new Uint8Array(file.byteLength);
  let offset = 0;
  for (const part of file.parts) {
    const url = new URL(part.path, base);
    const res = await fetch(url, { signal, credentials: 'omit' });
    if (!res.ok) throw new Error(`${res.status}: ${url}`);
    let received = 0;
    const reader = res.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (received + value.byteLength > part.byteLength || offset + value.byteLength > result.length) {
        await reader.cancel();
        throw new Error(`模型分片大小与 manifest 不符：${part.path}`);
      }
      result.set(value, offset);
      offset += value.byteLength;
      received += value.byteLength;
      onBytes(value.byteLength);
    }
    if (received !== part.byteLength) throw new Error(`模型分片不完整：${part.path}`);
  }
  if (offset !== result.length) throw new Error('模型缓冲区大小与 manifest 不符');
  return result.buffer;
}

function validateManifest(m) {
  if (m.schema !== 'fullcircle-webgpu-reference-v1') throw new Error('不支持此模型格式，请使用配套 export_model.py 导出');
  if (!Array.isArray(m.cameras) || !m.cameras.length) throw new Error('模型缺少初始相机');
  for (const key of ['geometry', 'sh', 'bvh']) {
    const f = m.files?.[key];
    if (!f || !Number.isSafeInteger(f.byteLength) || f.byteLength <= 0 || !Array.isArray(f.parts)) throw new Error(`无效模型缓冲区：${key}`);
    if (f.byteLength > renderer.device.limits.maxStorageBufferBindingSize || f.byteLength > renderer.device.limits.maxBufferSize) {
      throw new Error(`模型的 ${key} 缓冲区超过当前浏览器显卡限制。需要支持更大 storage buffer 的设备。`);
    }
    if (f.parts.some((p) => typeof p.path !== 'string' || !Number.isSafeInteger(p.byteLength) || p.byteLength <= 0) || f.parts.reduce((s, p) => s+p.byteLength, 0) !== f.byteLength) throw new Error(`无效模型分片：${key}`);
  }
}

async function switchModel(entry) {
  if (loading || stopped) return;
  loading = true;
  ready = false;
  rendered = false;
  invalidatePreview();
  keys.clear();
  $('model').disabled = $('camera').disabled = $('save').disabled = $('reset').disabled = true;
  $('error').hidden = true;
  $('loading').hidden = false;
  $('progress').removeAttribute('value');
  $('load-title').textContent = `载入 ${entry.name || entry.id}`;
  $('load-detail').textContent = '正在下载模型信息';
  loadController?.abort();
  loadController = new AbortController();
  try {
    // A previous frame may still be on the GPU when a dropdown is changed.
    while ((busy || capturing) && !stopped) await new Promise((resolve) => setTimeout(resolve, 10));
    if (stopped) return;
    const base = new URL(entry.url, document.baseURI);
    const m = await fetchJson(base, loadController.signal);
    if (stopped) return;
    validateManifest(m);
    const total = Object.values(m.files).reduce((s, f) => s + f.byteLength, 0);
    let received = 0;
    $('progress').max = total;
    $('progress').value = 0;
    const onBytes = (n) => {
      if (stopped) return;
      received += n;
      $('progress').value = received;
      $('load-detail').textContent = `${(received / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MiB`;
    };
    const [geometry, sh, bvh] = await Promise.all(['geometry', 'sh', 'bvh'].map((key) => loadBuffer(m.files[key], base, loadController.signal, onBytes)));
    if (stopped) return;
    $('load-detail').textContent = '上传到本机显卡';
    await renderer.loadModel({ geometry, sh, bvh, metadata: m });
    if (stopped) return;
    manifest = m;
    currentId = entry.id;
    $('camera').replaceChildren(...m.cameras.map((p, i) => new Option(p.name || `相机 ${i}`, i)));
    $('camera').value = String(m.default_camera || 0);
    applyPreset(Number($('camera').value));
    // This only affects the control speed, never the Gaussian parameters.
    const spans = m.bounds.max.map((v, i) => v-m.bounds.min[i]);
    baseSpeed = Math.max(0.05, Math.min(...spans.filter((v) => v > 0)) * 0.08);
    $('details').textContent = `${m.gaussian_count.toLocaleString()} 个高斯 · SH ${m.render.sh_degree} · ${(total / 1048576).toFixed(1)} MiB`;
    $('model').value = entry.id;
    const address = new URL(location.href);
    if (!address.searchParams.has('url')) {
      address.searchParams.set('model', entry.id);
      history.replaceState(null, '', address);
    }
    frameTimes = [];
    ready = true;
  } catch (error) {
    loadController.abort();
    if (stopped) return;
    message(error);
    $('stats').textContent = '载入失败，请刷新重试或选择其他模型';
  } finally {
    loading = false;
    if (!stopped) {
      $('loading').hidden = true;
      $('model').disabled = false;
      $('camera').disabled = $('reset').disabled = !ready;
      $('save').disabled = !ready || !rendered;
    }
  }
}

function move(direction, distance) {
  if (distance === 0) return;
  camera.position = camera.position.map((v, i) => v + direction[i]*distance);
  adaptive.markMotion(performance.now());
  markCameraChanged(true);
}
function turn(dx, dy) {
  if (dx === 0 && dy === 0) return;
  // Match the existing pan interaction: dragging right moves the scene right.
  const yaw = dx * 0.003, pitch = dy * 0.003;
  for (const key of ['right', 'down', 'forward']) camera[key] = rotate(camera[key], worldUp, yaw);
  const right = [...camera.right];
  for (const key of ['down', 'forward']) camera[key] = rotate(camera[key], right, pitch);
  camera.forward = norm(camera.forward);
  camera.right = norm(cross(camera.down, camera.forward));
  camera.down = norm(cross(camera.forward, camera.right));
  adaptive.markMotion(performance.now());
  markCameraChanged(true);
}

let drag;
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (e) => {
  if (!ready || drag) return;
  canvas.focus();
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey, id: e.pointerId };
});
canvas.addEventListener('pointermove', (e) => {
  if (!drag || !ready || e.pointerId !== drag.id) return;
  const dx = e.clientX-drag.x, dy = e.clientY-drag.y;
  if (drag.pan) {
    const scale = baseSpeed * Math.pow(10, Number($('speed').value)) * 0.006;
    move(camera.right, -dx*scale); move(camera.down, -dy*scale);
  } else turn(dx, dy);
  drag.x = e.clientX; drag.y = e.clientY;
});
for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => { drag = null; });
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (ready) move(camera.forward, -Math.sign(e.deltaY)*baseSpeed*Math.pow(10, Number($('speed').value))*0.2);
}, { passive: false });
window.addEventListener('keydown', (e) => {
  if (e.target !== canvas) return;
  if (['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ShiftLeft','ShiftRight'].includes(e.code)) { e.preventDefault(); keys.add(e.code); }
});
window.addEventListener('keyup', (e) => keys.delete(e.code));
window.addEventListener('blur', () => { keys.clear(); drag = null; });
document.addEventListener('visibilitychange', () => { keys.clear(); frameTimes = []; dirty = true; });
window.addEventListener('resize', () => { adaptive.markMotion(performance.now()); markCameraChanged(true); });
$('resolution').addEventListener('change', () => { frameTimes = []; dirty = true; });
$('adaptive').addEventListener('change', () => { frameTimes = []; dirty = true; });
$('target-fps').addEventListener('change', () => { adaptive.setTarget(Number($('target-fps').value)); dirty = true; });
$('render-mode').addEventListener('change', () => { adaptive.reset(); frameTimes = []; dirty = true; });
$('fov').addEventListener('change', () => {
  if (!ready) return;
  const value = Math.min(120, Math.max(20, Number($('fov').value) || 70));
  $('fov').value = value;
  camera.tanHalfFovY = Math.tan(value*Math.PI/360);
  adaptive.markMotion(performance.now());
  markCameraChanged(true);
});
$('fisheye-enabled').addEventListener('change', () => {
  $('fisheye-panel').hidden = !$('fisheye-enabled').checked;
  if ($('fisheye-enabled').checked) {
    previewRevision = -1;
    $('fisheye-status').textContent = '等待当前视角';
  }
});
$('fisheye-hide').addEventListener('click', () => {
  $('fisheye-enabled').checked = false;
  $('fisheye-panel').hidden = true;
});
$('model').addEventListener('change', () => switchModel(catalog.find((e) => e.id === $('model').value)));
$('camera').addEventListener('change', () => { keys.clear(); applyPreset(Number($('camera').value)); });
$('reset').addEventListener('click', () => applyPreset(Number($('camera').value)));
$('toggle').addEventListener('click', () => document.body.classList.toggle('collapsed'));
$('continuous').addEventListener('change', () => { frameTimes = []; dirty = true; });
$('save').addEventListener('click', async () => {
  if (!ready || !rendered || loading || capturing) return;
  capturing = true;
  $('save').disabled = true;
  try {
    while (busy && !stopped) await new Promise((resolve) => setTimeout(resolve, 10));
    if (stopped) return;
    // readPixels queues after the most recent completed frame; capture the
    // renderer's RGBA texture rather than the optionally discarded canvas.
    const { width, height, rgba } = await renderer.readPixels();
    if (stopped) return;
    const output = document.createElement('canvas'); output.width = width; output.height = height;
    output.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);
    const blob = await new Promise((resolve) => output.toBlob(resolve, 'image/png'));
    if (stopped) return;
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = `${currentId}-webgpu.png`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { message(error); }
  finally { capturing = false; $('save').disabled = !ready || loading; }
});

async function frame(time) {
  animationFrame = null;
  if (stopped) return;
  const dt = Math.min((time-previousFrame)/1000 || 0, 0.1); previousFrame = time;
  if (ready && !loading && !capturing && !document.hidden) {
    const speed = baseSpeed * Math.pow(10, Number($('speed').value)) * dt * (keys.has('ShiftLeft') || keys.has('ShiftRight') ? 3 : 1);
    if (keys.has('KeyW')) move(camera.forward, speed);
    if (keys.has('KeyS')) move(camera.forward, -speed);
    if (keys.has('KeyD')) move(camera.right, speed);
    if (keys.has('KeyA')) move(camera.right, -speed);
    if (keys.has('KeyE')) move(worldUp, speed);
    if (keys.has('KeyQ')) move(worldUp, -speed);
    const choice = adaptive.select(Number($('resolution').value), performance.now(), $('adaptive').checked);
    const width = choice.width;
    const height = Math.max(1, Math.min(1920, Math.round(width*canvas.clientHeight/canvas.clientWidth)));
    // After motion settles, render one full-resolution frame even if no
    // further input event occurred. Continuing idle frames remains optional.
    const restoreFull = !choice.moving && canvas.dataset.quality === 'moving';
    if (dirty || $('continuous').checked || renderer.width !== width || renderer.height !== height || restoreFull) {
      dirty = false; busy = true;
      try {
        const frameCamera = copyCamera(camera);
        const frameRevision = cameraRevision;
        renderer.setCamera(frameCamera);
        const mode = $('render-mode').value;
        const modeLabel = $('render-mode').selectedOptions[0].textContent;
        const result = await renderer.render({ width, height, mode });
        if (stopped) return;
        lastMainCamera = frameCamera;
        mainRevision = frameRevision;
        adaptive.record(result, choice.moving);
        canvas.dataset.quality = choice.moving ? 'moving' : 'full';
        rendered = true;
        $('save').disabled = loading || capturing;
        const done = performance.now();
        frameTimes.push(done); frameTimes = frameTimes.filter((t) => t >= done-1000);
        const fps = frameTimes.length > 1 ? (frameTimes.length-1)*1000/(done-frameTimes[0]) : null;
        if (done-lastUiUpdate > 200 || !$('continuous').checked) {
          $('stats').textContent = `${width} × ${height} · ${result.renderMs.toFixed(1)} ms / 完成帧${Number.isFinite(result.gpuMs) ? ` · GPU ≈ ${result.gpuMs.toFixed(1)} ms` : ''}${fps !== null ? ` · ${fps.toFixed(1)} FPS` : ''}`;
          $('quality').textContent = `${choice.moving ? '交互分辨率' : '完整分辨率'} · ${modeLabel}`;
          lastUiUpdate = done;
        }
      } catch (error) { ready = false; message(error); }
      finally { busy = false; }
    } else if (frameTimes.length && time-frameTimes.at(-1) > 1000) {
      frameTimes = [];
      $('stats').textContent = $('stats').textContent.replace(/ · [\d.]+ FPS$/, '') + ' · 静止';
    }
  }
  // The comparison uses the exact pose of the completed main frame. Never
  // add a fisheye render to every moving frame on a resource-limited device.
  if (ready && !loading && !capturing && !stopped && !document.hidden && !dirty
      && $('fisheye-enabled').checked && lastMainCamera && mainRevision === cameraRevision
      && previewRevision !== mainRevision && performance.now() - lastInteraction >= 200) {
    busy = true;
    const revision = mainRevision;
    $('fisheye-status').textContent = '正在更新鱼眼视角…';
    // Keep the canvas in layout before acquiring its presentation surface.
    previewCanvas.hidden = false;
    try {
      const result = await renderer.renderPreview(previewCanvas, {width:192, height:192, camera:lastMainCamera});
      if (stopped) return;
      if (!loading && revision === cameraRevision && $('fisheye-enabled').checked) {
        previewRevision = revision;
        previewCanvas.hidden = false;
        previewCanvas.dataset.revision = String(revision);
        $('fisheye-status').textContent = `与主视角同位姿 · ${result.renderMs.toFixed(1)} ms`;
      } else if ($('fisheye-enabled').checked) {
        $('fisheye-status').textContent = '移动中，停止后更新';
      }
    } catch (error) {
      if (!stopped) {
        $('fisheye-enabled').checked = false;
        previewCanvas.hidden = true;
        $('fisheye-status').textContent = '鱼眼预览不可用，可重新开启';
        console.warn('Fisheye preview stopped:', error);
      }
    } finally {
      busy = false;
    }
  }
  scheduleFrame();
}

async function main() {
  if (!isSecureContext || !navigator.gpu) throw new Error('此页面需要支持 WebGPU 的浏览器，并通过 HTTPS 或 localhost 打开。请使用启用了硬件加速的 Chrome / Edge。');
  rendererInitialization = WebGpuRayRenderer.create(canvas);
  renderer = await rendererInitialization;
  if (stopped) { renderer.dispose(); return; }
  const info = renderer.adapterInfo || {};
  $('adapter').textContent = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' · ') || 'WebGPU · 本机显卡';
  renderer.device.lost.then((info) => {
    if (stopped) return;
    if (info.reason !== 'destroyed') message(`显卡连接中断：${info.message}。请降低分辨率后刷新。`);
    stopViewer();
  });
  const params = new URLSearchParams(location.search);
  let defaultId;
  if (params.has('url')) {
    catalog = [{ id: 'external', name: '外部模型', url: params.get('url') }];
    defaultId = 'external';
  } else {
    loadController = new AbortController();
    const data = await fetchJson(new URL('./models.json', document.baseURI), loadController.signal);
    if (stopped) return;
    catalog = data.models;
    if (!Array.isArray(catalog) || !catalog.length) throw new Error('models.json 中没有模型，请先运行 export_model.py 并生成模型列表');
    defaultId = params.get('model') || data.default || catalog[0].id;
  }
  $('model').replaceChildren(...catalog.map((entry) => new Option(entry.name || entry.id, entry.id)));
  const entry = catalog.find((e) => e.id === defaultId);
  if (!entry) throw new Error(`模型列表中没有 ${defaultId}`);
  scheduleFrame();
  await switchModel(entry);
}
main().catch((error) => {
  if (stopped) return;
  $('loading').hidden = true;
  message(error);
  stopViewer();
});
