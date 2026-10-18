import { WebGlSplatRenderer } from './renderer.js?v=20261013-lazy-hud-1';
import { AdaptiveResolution } from './adaptive-resolution.js?v=20261013-lazy-hud-1';
import { createFrameAlignment, frameFromAlignment, defaultFrame, applyAlignment, alignmentUp, loadAlignment, ALIGNMENT_SCHEMA } from './alignment.js?v=20261013-lazy-hud-1';
import { AlignmentGizmo } from './alignment-gizmo.js?v=20261013-lazy-hud-1';
import { validateModelFiles } from './model-loader.js?v=20261016-model-cache-1';
import { ModelVariants, VARIANT_LABELS } from './model-variants.js?v=20261016-model-cache-1';
import { FlightControls } from './flight-controls.js?v=20261013-lazy-hud-1';

const $ = (id) => document.getElementById(id);
const DEFAULT_FOV_DEG = 90;
const DEFAULT_PROJECTION = 'fisheye-perspective';
const canvas = $('view');
const previewCanvas = $('fisheye-view');
let renderer, manifest, camera, worldUp, ready = false, loading = false, dirty = true, rendered = false;
let catalog = [], currentId = '', loadController, busy = false, capturing = false, baseSpeed = 1;
let activeProjection = DEFAULT_PROJECTION;
let activeEntry = null;
const activeCompatibility = false;
let variants = null, activeVariant = '';
let flightControls = null;
let frameTimes = [], previousFrame = 0, lastUiUpdate = 0;
let stopped = false, animationFrame = null;
let rendererInitialization = null, exitPromise = null;
let cameraRevision = 0, mainRevision = -1, previewRevision = -1;
let lastMainCamera = null, lastInteraction = -Infinity;
let alignmentState = null, modelManifestUrl = null, alignmentSaving = false, alignmentSaveController = null;
let gizmoFrame = null, calibrationDragging = false;
const keys = new Set();
let drag = null;
const touchPointers = new Map();
const adaptive = new AdaptiveResolution();
if (matchMedia('(pointer: coarse)').matches && matchMedia('(max-width: 900px), (max-height: 600px)').matches) {
  $('fisheye-enabled').checked = false;
  $('fisheye-panel').hidden = true;
  document.body.classList.add('collapsed');
}
const gizmo = new AlignmentGizmo($('alignment-overlay'), {
  onChange: (frame,event) => {
    if (stopped || !ready || loading || alignmentSaving) return;
    gizmoFrame = {origin:[...frame.origin],rotation:frame.rotation.map(row=>[...row])};
    if (event.phase === 'end') {
      commitGizmoFrame();
    } else if (event.phase === 'change') {
      alignmentStatus('editing','正在拖动参考地面；松开后应用观察中心与上方向');
    } else if (event.phase === 'cancel') {
      alignmentStatus('modified','已取消本次拖动');
    }
  },
  onDragState: (active) => {
    calibrationDragging = active;
    if (active) { keys.clear(); clearPointerState(); }
    if (!stopped) updateAlignmentControls();
  },
});

function haltViewer() {
  if (stopped) return;
  stopped = true;
  ready = false;
  updateProjectionControls();
  updateFpsHud(null, 'stopped', '已停止');
  keys.clear();
  clearPointerState();
  flightControls?.dispose();
  if (animationFrame !== null) cancelAnimationFrame(animationFrame);
  animationFrame = null;
  loadController?.abort();
  variants?.dispose(); variants = null;
  alignmentSaveController?.abort();
  gizmo.dispose();
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
      // The button is also available while the graphics context is initializing.
      // main() disposes a late-created context when it sees stopped=true.
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
      activeEntry = null;
      activeVariant = '';
      frameTimes = [];
      $('model').replaceChildren();
      $('camera').replaceChildren();
      canvas.width = canvas.height = 1;
      canvas.hidden = true;
      $('fisheye-panel').hidden = true;
      previewCanvas.width = previewCanvas.height = 1;
      lastMainCamera = null;
      alignmentState = null;
      modelManifestUrl = null;
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
// unload listener. Restoring a cached page must create a new graphics context.
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

function updateFpsHud(fps, state = 'rendering', label = '主画面') {
  const value = Number.isFinite(fps) ? `${fps.toFixed(1)} FPS` : '— FPS';
  if ($('fps-value').textContent !== value) $('fps-value').textContent = value;
  if ($('fps-state').textContent !== label) $('fps-state').textContent = label;
  $('fps-hud').dataset.state = state;
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

function alignmentStatus(state,text) {
  if (stopped) return;
  $('align-status').dataset.state = state;
  $('align-status').textContent = text;
}

function updateAlignmentControls() {
  $('align-enabled').checked = Boolean(alignmentState?.enabled);
  for (const id of ['align-tool','align-enabled','align-center','align-reset','align-save']) {
    $(id).disabled = !ready || loading || stopped || alignmentSaving || calibrationDragging;
  }
  updateVariantControls();
  updateGizmo();
}

function updateGizmo() {
  if (stopped) return;
  gizmo.setState({camera,frame:gizmoFrame,visible:Boolean(ready && !loading && $('align-tool').checked),
    enabled:Boolean(ready && !loading && !alignmentSaving),sizePx:105});
}

function commitGizmoFrame(enabled = true) {
  if (!ready || loading || stopped || alignmentSaving || !gizmoFrame) return;
  try {
    alignmentState = createFrameAlignment(manifest,currentId,gizmoFrame,enabled);
    if (enabled) {
      worldUp = alignmentUp(alignmentState);
      camera = applyAlignment(camera,alignmentState);
      adaptive.markMotion(performance.now());
      markCameraChanged(true);
    } else {
      applyPreset(Number($('camera').value));
    }
    updateAlignmentControls();
    alignmentStatus('modified','校正已修改，尚未保存');
  } catch (error) {
    alignmentStatus('warning',error.message);
  }
}
function applyPreset(index) {
  clearPointerState();
  const p = manifest.cameras[index];
  const r = p.rotation;
  camera = {
    position: [...p.position],
    right: [r[0][0], r[1][0], r[2][0]],
    down: [r[0][1], r[1][1], r[2][1]],
    forward: [r[0][2], r[1][2], r[2][2]],
    // Viewer defaults are independent of the preset's exported image FOV.
    // Presets retain their position/orientation and reset the viewer to 90°.
    tanHalfFovY: Math.tan(DEFAULT_FOV_DEG * Math.PI / 360),
  };
  worldUp = camera.down.map((x) => -x);
  if (alignmentState?.enabled) {
    worldUp = alignmentUp(alignmentState);
    camera = applyAlignment(camera,alignmentState);
  }
  $('fov').value = DEFAULT_FOV_DEG;
  adaptive.reset();
  lastInteraction = -Infinity;
  markCameraChanged();
  updateGizmo();
}

async function fetchJson(url, signal, cache = 'default') {
  const res = await fetch(url, { signal, credentials: 'omit', cache });
  if (!res.ok) throw new Error(`${res.status}: ${url}`);
  return res.json();
}

function shStorageFormat(m) {
  const dtype = m.files?.sh?.dtype ?? 'float32';
  if (!['float16', 'float32'].includes(dtype)) throw new Error('不支持此 SH 存储格式');
  return { dtype, stride: dtype === 'float16' ? 96 : 192, packed: dtype === 'float16' };
}

function validateManifest(m) {
  if (m.schema !== 'fullcircle-webgl2-ewa-v1') {
    throw new Error(m.schema === 'fullcircle-webgpu-reference-v1'
      ? '这是旧版 3DGRT 光追模型，不能由此 WebGL2 页面正确显示。请选择使用 WebGL2 导出工具生成的 3DGS 模型。'
      : '不支持此模型格式，请使用当前 WebGL2 导出工具生成模型。');
  }
  if (!Array.isArray(m.cameras) || !m.cameras.length) throw new Error('模型缺少初始相机');
  const storage = shStorageFormat(m);
  validateModelFiles(m);
  for (const key of ['geometry', 'sh']) {
    const file = m.files?.[key];
    const stride = key === 'geometry' ? 48 : storage.stride;
    if (!Number.isSafeInteger(m.gaussian_count) || m.gaussian_count < 1
        || file.byteLength !== m.gaussian_count * stride
        || (file.stride !== undefined && file.stride !== stride)) {
      throw new Error(`模型数量、步长与缓冲区长度不一致：${key}`);
    }
  }
  renderer.validateModel(m);
}

function modelProjection(entry) {
  const projection = entry.output_projection === undefined ? DEFAULT_PROJECTION : entry.output_projection;
  if (!['pinhole', 'fisheye-perspective'].includes(projection)) {
    throw new Error(`模型 ${entry.id} 的 output_projection 无效：${String(projection)}；只能使用 pinhole 或 fisheye-perspective。`);
  }
  return projection;
}

function updateCompatibilityControls() {
  $('compatibility-control').hidden = true;
  $('compatibility').checked = false;
  $('compatibility').disabled = true;
}

function updateProjectionControls() {
  const control = $('fisheye-reproject');
  control.checked = activeProjection === 'fisheye-perspective';
  control.disabled = !ready || loading || stopped || capturing || alignmentSaving || calibrationDragging;
  $('projection-status').textContent = !ready ? (stopped ? '渲染已停止' : '等待模型')
    : control.checked ? '主画面：等距鱼眼渲染后重投影，中心视线与当前相机一致'
      : '主画面：直接针孔渲染';
}

function updateVariantControls() {
  updateProjectionControls();
  $('variant-controls').hidden = !variants || variants.entries.size < 2;
  const notes = [];
  for (const key of ['sog', 'bin']) {
    const entry = variants?.entries.get(key), button = $(`variant-${key}`);
    button.setAttribute('aria-pressed', String(activeVariant === key));
    button.disabled = !ready || loading || stopped || capturing || alignmentSaving || calibrationDragging || !entry
      || entry.status === 'loading' || key === activeVariant;
    button.textContent = VARIANT_LABELS[key] + (entry?.status === 'loading' ? ' · 加载中' : entry?.status === 'error' ? ' · 重试' : '');
    if (entry?.status === 'loading') notes.push(`${VARIANT_LABELS[key]} 下载及解码 ${Math.min(100, Math.floor(100*entry.received/entry.total))}%`);
    if (entry?.status === 'error') notes.push(`${VARIANT_LABELS[key]} 加载失败：${entry.error.message}`);
    if (entry?.status === 'idle') notes.push(`点击${VARIANT_LABELS[key]}按钮后加载`);
  }
  $('variant-status').textContent = notes.join('；') || (variants?.entries.size > 1
    ? [...variants.entries.values()].every(entry => entry.status === 'ready') ? '两个版本已就绪，切换保持当前视角与校正' : '正在准备模型版本…' : '');
}

function updateModelDetails() {
  const m = manifest, sizes = validateModelFiles(m);
  $('details').textContent = `${m.gaussian_count.toLocaleString()} 个高斯 · SH ${m.render.sh_degree}（${shStorageFormat(m).packed ? 'FP16' : 'FP32'}） · ${VARIANT_LABELS[activeVariant]} · 下载 ${(sizes.downloadBytes/1048576).toFixed(1)} MiB / 解码后 ${(sizes.decodedBytes/1048576).toFixed(1)} MiB`;
}

function updateVariantAddress() {
  const address = new URL(location.href);
  if (variants?.entries.size > 1) address.searchParams.set('variant', activeVariant);
  else address.searchParams.delete('variant');
  history.replaceState(null, '', address);
}

async function switchVariant(key) {
  if (!ready || loading || stopped || capturing || calibrationDragging || alignmentSaving || key === activeVariant) return;
  const pool = variants, previous = activeVariant;
  if (!pool?.entries.has(key) || pool.entries.get(key).status === 'loading') return;
  // Download only after an explicit button click. Keep the current version
  // interactive while the requested one downloads and decodes on a worker.
  let buffers;
  try { buffers = await pool.load(key); }
  catch { if (!stopped && variants === pool) updateVariantControls(); return; }
  while ((capturing || calibrationDragging || alignmentSaving) && !stopped && variants === pool) await new Promise(resolve => setTimeout(resolve, 10));
  if (stopped || variants !== pool) return;
  loading = true; ready = false; clearPointerState(); keys.clear();
  frameTimes = []; updateFpsHud(null, 'loading', '切换中');
  $('model').disabled = $('camera').disabled = $('save').disabled = $('reset').disabled = true;
  updateAlignmentControls(); updateCompatibilityControls(); updateVariantControls();
  try {
    while ((busy || capturing) && !stopped) await new Promise(resolve => setTimeout(resolve, 10));
    if (stopped || variants !== pool) return;
    const metadata = pool.entries.get(key).metadata;
    // Wait for the frame fence, free the previous model, then upload ONE
    // cached version. Do not allocate both models on a mobile GPU at once.
    renderer.unloadModel();
    await renderer.loadModel({...buffers, metadata});
    if (stopped) return;
    manifest = metadata; activeVariant = key;
    invalidatePreview(); frameTimes = []; rendered = false;
    updateModelDetails(); updateVariantAddress(); markCameraChanged();
    ready = true;
  } catch (error) {
    if (stopped) return;
    // A download failure leaves the active GPU model intact. If upload
    // failed after release, restore its already decoded CPU buffers.
    if (!renderer.model) {
      try { await renderer.loadModel({...pool.entries.get(previous).buffers, metadata: pool.entries.get(previous).metadata}); }
      catch (restoreError) { message(restoreError); return; }
    }
    ready = true; markCameraChanged();
    const entry = pool.entries.get(key);
    entry.status = 'error'; entry.error = error;
  } finally {
    loading = false;
    if (!stopped) {
      $('model').disabled = false;
      $('camera').disabled = $('reset').disabled = !ready;
      $('save').disabled = !ready || !rendered;
      updateAlignmentControls(); updateCompatibilityControls(); updateVariantControls();
    }
  }
}
for (const key of ['sog', 'bin']) $(`variant-${key}`).addEventListener('click', () => switchVariant(key));

async function switchModel(entry) {
  clearPointerState();
  if (loading || stopped) return;
  variants?.dispose(); variants = null; activeVariant = '';
  loading = true;
  ready = false;
  frameTimes = []; updateFpsHud(null, 'loading', '加载中');
  rendered = false;
  canvas.style.visibility = 'hidden';
  alignmentSaveController?.abort();
  alignmentState = null;
  gizmoFrame = null;
  modelManifestUrl = null;
  updateAlignmentControls();
  alignmentStatus('loading','正在检查模型校正文件…');
  invalidatePreview();
  keys.clear();
  $('model').disabled = $('camera').disabled = $('save').disabled = $('reset').disabled = true;
  $('compatibility').disabled = true;
  updateVariantControls();
  $('error').hidden = true;
  $('loading').hidden = false;
  $('progress').removeAttribute('value');
  $('load-title').textContent = `载入 ${entry.name || entry.id}`;
  $('load-detail').textContent = '正在下载模型信息';
  loadController?.abort();
  loadController = new AbortController();
  try {
    const projection = modelProjection(entry);
    // A previous frame may still be on the GPU when a dropdown is changed.
    while ((busy || capturing) && !stopped) await new Promise((resolve) => setTimeout(resolve, 10));
    if (stopped) return;
    renderer.unloadModel();
    const base = new URL(entry.url, document.baseURI);
    const m = await fetchJson(base, loadController.signal, 'no-cache');
    if (stopped) return;
    validateManifest(m);
    const pool = new ModelVariants(m, base, {onChange: () => { if (variants === pool && !stopped) updateVariantControls(); }});
    variants = pool;
    for (const version of pool.entries.values()) validateManifest(version.metadata);
    // A paired model always starts with SOG. Neither a remembered selection,
    // a URL hint nor a failed SOG request may implicitly download the binary.
    const selected = pool.entries.has('sog') ? 'sog' : pool.default;
    const total = pool.entries.get(selected).total;
    let received = 0;
    $('progress').max = total;
    $('progress').value = 0;
    const onBytes = (n) => {
      if (stopped) return;
      received += n;
      $('progress').value = received;
      $('load-detail').textContent = `下载及解码 ${(received / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MiB`;
    };
    const [{geometry, sh}, calibration] = await Promise.all([
      pool.load(selected, onBytes),
      loadAlignment(base,m,loadController.signal),
    ]);
    if (stopped) return;
    $('load-detail').textContent = '上传到本机显卡';
    const shown = pool.entries.get(selected).metadata;
    await renderer.loadModel({ geometry, sh, metadata: shown });
    if (stopped) return;
    activeEntry = entry;
    activeProjection = projection;
    manifest = shown;
    activeVariant = selected;
    currentId = entry.id;
    modelManifestUrl = base;
    alignmentState = calibration.alignment ?? createFrameAlignment(m,entry.id);
    gizmoFrame = frameFromAlignment(alignmentState,m);
    alignmentStatus(calibration.state,calibration.message);
    $('camera').replaceChildren(...m.cameras.map((p, i) => new Option(p.name || `相机 ${i}`, i)));
    $('camera').value = String(m.default_camera || 0);
    applyPreset(Number($('camera').value));
    // This only affects the control speed, never the Gaussian parameters.
    const navigationBounds = m.navigation_bounds ?? m.bounds;
    const spans = navigationBounds.max.map((v, i) => v-navigationBounds.min[i]).filter(v => Number.isFinite(v) && v > 0);
    baseSpeed = spans.length ? Math.max(0.05, Math.min(...spans) * 0.08) : 0.05;
    updateModelDetails();
    $('model').value = entry.id;
    const address = new URL(location.href);
    if (!address.searchParams.has('url')) {
      address.searchParams.set('model', entry.id);
      history.replaceState(null, '', address);
    }
    frameTimes = [];
    ready = true;
    updateVariantAddress();
  } catch (error) {
    loadController.abort();
    variants?.dispose(); variants = null;
    if (stopped) return;
    message(error);
    updateFpsHud(null, 'error', '加载失败');
    $('stats').textContent = '载入失败，请刷新重试或选择其他模型';
  } finally {
    loading = false;
    if (!stopped) {
      $('loading').hidden = true;
      $('model').disabled = false;
      $('camera').disabled = $('reset').disabled = !ready;
      $('save').disabled = !ready || !rendered;
      updateAlignmentControls();
      updateCompatibilityControls();
      updateVariantControls();
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
  rotateCamera(dx * 0.003, dy * 0.003);
}
function rotateCamera(yaw, pitch) {
  if (yaw === 0 && pitch === 0) return;
  for (const key of ['right', 'down', 'forward']) camera[key] = rotate(camera[key], worldUp, yaw);
  const right = [...camera.right];
  for (const key of ['down', 'forward']) camera[key] = rotate(camera[key], right, pitch);
  camera.forward = norm(camera.forward);
  camera.right = norm(cross(camera.down, camera.forward));
  camera.down = norm(cross(camera.forward, camera.right));
  adaptive.markMotion(performance.now());
  markCameraChanged(true);
}

function clearPointerState() {
  flightControls?.reset();
  const ids = [...touchPointers.keys(), ...(drag ? [drag.id] : [])];
  touchPointers.clear();
  drag = null;
  for (const id of ids) {
    if (canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
  }
}
function touchPair() {
  const [a, b] = touchPointers.values();
  return { y: (a.y+b.y)/2 };
}
function pan(dx, dy) {
  const scale = baseSpeed * Math.pow(10, Number($('speed').value)) * 0.006;
  move(camera.right, -dx*scale); move(camera.down, -dy*scale);
}
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (e) => {
  if (!ready || loading || stopped || calibrationDragging) return;
  if (e.pointerType === 'touch') {
    if (drag || touchPointers.size >= 2) return;
    canvas.focus();
    canvas.setPointerCapture(e.pointerId);
    // Joining/leaving fingers changes gesture mode without changing the camera.
    touchPointers.set(e.pointerId, {x:e.clientX,y:e.clientY});
    return;
  }
  if (drag || touchPointers.size) return;
  canvas.focus();
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey, id: e.pointerId };
});
canvas.addEventListener('pointermove', (e) => {
  if (!ready || loading || stopped || calibrationDragging) return;
  if (e.pointerType === 'touch') {
    const previous = touchPointers.get(e.pointerId);
    if (!previous) return;
    const pair = touchPointers.size === 2 ? touchPair() : null;
    touchPointers.set(e.pointerId, {x:e.clientX,y:e.clientY});
    if (pair) {
      const next = touchPair();
      const scale = baseSpeed * Math.pow(10, Number($('speed').value)) * 0.006;
      // Only the pair's vertical midpoint changes calibrated height;
      // horizontal movement and pinch distance are ignored.
      move(norm(worldUp), (pair.y-next.y)*scale);
    } else turn(e.clientX-previous.x, e.clientY-previous.y);
    return;
  }
  if (!drag || e.pointerId !== drag.id) return;
  const dx = e.clientX-drag.x, dy = e.clientY-drag.y;
  if (drag.pan) pan(dx, dy);
  else turn(dx, dy);
  drag.x = e.clientX; drag.y = e.clientY;
});
canvas.addEventListener('pointerup', (e) => {
  touchPointers.delete(e.pointerId);
  if (drag?.id === e.pointerId) drag = null;
});
for (const name of ['pointercancel', 'lostpointercapture']) canvas.addEventListener(name, (e) => {
  if (touchPointers.has(e.pointerId) || drag?.id === e.pointerId) clearPointerState();
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (ready && !calibrationDragging) move(camera.forward, -Math.sign(e.deltaY)*baseSpeed*Math.pow(10, Number($('speed').value))*0.2);
}, { passive: false });
window.addEventListener('keydown', (e) => {
  if (e.target !== canvas) return;
  if (['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ShiftLeft','ShiftRight'].includes(e.code)) { e.preventDefault(); keys.add(e.code); }
});
window.addEventListener('keyup', (e) => keys.delete(e.code));
window.addEventListener('blur', () => { keys.clear(); clearPointerState(); });
document.addEventListener('visibilitychange', () => {
  keys.clear(); clearPointerState(); frameTimes = []; dirty = true;
  if (!stopped) updateFpsHud(null, document.hidden ? 'paused' : 'rendering', document.hidden ? '后台暂停' : '待渲染');
});
window.addEventListener('resize', () => { clearPointerState(); adaptive.markMotion(performance.now()); markCameraChanged(true); updateGizmo(); });
$('resolution').addEventListener('change', () => { frameTimes = []; dirty = true; });
$('adaptive').addEventListener('change', () => { frameTimes = []; dirty = true; });
$('target-fps').addEventListener('change', () => { adaptive.setTarget(Number($('target-fps').value)); dirty = true; });
$('fov').addEventListener('change', () => {
  if (!ready) return;
  const value = Math.min(120, Math.max(20, Number($('fov').value) || DEFAULT_FOV_DEG));
  $('fov').value = value;
  camera.tanHalfFovY = Math.tan(value*Math.PI/360);
  adaptive.markMotion(performance.now());
  markCameraChanged(true);
});
$('fisheye-reproject').addEventListener('change', () => {
  if (!ready || loading || stopped || capturing || alignmentSaving || calibrationDragging) {
    updateProjectionControls();
    return;
  }
  activeProjection = $('fisheye-reproject').checked ? 'fisheye-perspective' : 'pinhole';
  // Keep the current pose/FOV and GPU model. Only the projection path changes.
  frameTimes = []; lastUiUpdate = 0; rendered = false;
  $('save').disabled = true;
  updateFpsHud(null, 'rendering', '切换模式');
  markCameraChanged();
  updateProjectionControls();
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
$('align-enabled').addEventListener('change', () => commitGizmoFrame($('align-enabled').checked));
$('align-tool').addEventListener('change', updateGizmo);
$('align-center').addEventListener('click', () => {
  if (!ready || loading || stopped || alignmentSaving || calibrationDragging) return;
  const placed = defaultFrame(manifest,camera);
  if (gizmoFrame) placed.rotation = gizmoFrame.rotation.map(row=>[...row]);
  gizmoFrame = placed;
  $('align-tool').checked = true;
  commitGizmoFrame();
});
$('align-reset').addEventListener('click', () => {
  if (!ready || loading || stopped || alignmentSaving || calibrationDragging) return;
  alignmentState = createFrameAlignment(manifest,currentId);
  applyPreset(Number($('camera').value));
  gizmoFrame = defaultFrame(manifest,camera);
  alignmentState = createFrameAlignment(manifest,currentId,gizmoFrame,false);
  updateAlignmentControls();
  alignmentStatus('modified','已恢复当前预设原始姿态；保存后才会覆盖旧校正文件');
});
$('align-save').addEventListener('click', async () => {
  if (!ready || loading || stopped || alignmentSaving || calibrationDragging || !modelManifestUrl || !gizmoFrame) return;
  let correction;
  try {
    correction = createFrameAlignment(manifest,currentId,gizmoFrame,$('align-enabled').checked);
  } catch (error) { alignmentStatus('warning',error.message); return; }
  alignmentSaving = true;
  alignmentSaveController = new AbortController();
  const signal = alignmentSaveController.signal;
  const target = new URL(modelManifestUrl);
  const savingId = currentId;
  updateAlignmentControls();
  alignmentStatus('saving','正在保存校正…');
  try {
    const pageBase = new URL('.',document.baseURI);
    let writable = false;
    if (target.origin === pageBase.origin && target.pathname.startsWith(pageBase.pathname)) {
      try {
        const response = await fetch(new URL('__viewer__/capabilities',pageBase),{signal,credentials:'omit',cache:'no-store'});
        if (response.ok && response.headers.get('content-type')?.includes('application/json')) {
          const capabilities = await response.json();
          writable = capabilities.alignment_write === true && (capabilities.alignment_schema === ALIGNMENT_SCHEMA || capabilities.alignment_schemas?.includes(ALIGNMENT_SCHEMA));
        }
      } catch (error) { if (signal.aborted) throw error; }
    }
    if (stopped || signal.aborted || savingId !== currentId || loading) return;
    if (writable) {
      const response = await fetch(new URL('__viewer__/alignment',pageBase),{
        method:'POST',signal,credentials:'omit',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({model:decodeURIComponent(target.pathname.slice(pageBase.pathname.length)),alignment:correction}),
      });
      const result = await response.json();
      if (!response.ok || result.saved !== true) throw new Error(result.error || `服务器未确认保存（HTTP ${response.status}）`);
      if (stopped || signal.aborted || savingId !== currentId || loading) return;
      alignmentStatus('saved','已保存到模型目录的 alignment.json；下次加载自动读取');
    } else {
      const blob = new Blob([JSON.stringify(correction,null,2)+'\n'],{type:'application/json'});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url; link.download = 'alignment.json';
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url),1000);
      alignmentStatus('downloaded',`已导出 alignment.json，请放入 ${currentId} 模型目录并发布；当前服务未提供此版本的写入接口`);
    }
  } catch (error) {
    if (!stopped && !signal.aborted && savingId === currentId && !loading) alignmentStatus('warning',`保存失败：${error.message}`);
  } finally {
    alignmentSaving = false;
    if (!stopped) updateAlignmentControls();
  }
});
$('toggle').addEventListener('click', () => { clearPointerState(); document.body.classList.toggle('collapsed'); });
$('continuous').addEventListener('change', () => { frameTimes = []; dirty = true; });
$('save').addEventListener('click', async () => {
  if (!ready || !rendered || loading || capturing) return;
  capturing = true;
  updateProjectionControls();
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
    a.href = url; a.download = `${currentId}-webgl2.png`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { message(error); }
  finally { capturing = false; $('save').disabled = !ready || loading; updateProjectionControls(); }
});

async function frame(time) {
  animationFrame = null;
  if (stopped) return;
  const dt = Math.min((time-previousFrame)/1000 || 0, 0.1); previousFrame = time;
  if (ready && !loading && !capturing && !calibrationDragging && !document.hidden) {
    const speed = baseSpeed * Math.pow(10, Number($('speed').value)) * dt * (keys.has('ShiftLeft') || keys.has('ShiftRight') ? 3 : 1);
    if (keys.has('KeyW')) move(camera.forward, speed);
    if (keys.has('KeyS')) move(camera.forward, -speed);
    if (keys.has('KeyD')) move(camera.right, speed);
    if (keys.has('KeyA')) move(camera.right, -speed);
    if (keys.has('KeyE')) move(worldUp, speed);
    if (keys.has('KeyQ')) move(worldUp, -speed);
    const sticks = flightControls.sample();
    if (sticks.forward || sticks.strafe) {
      const up = norm(worldUp);
      const vertical = dot(camera.forward,up);
      let forward = camera.forward.map((v,i)=>v-vertical*up[i]);
      if (length(forward)<1e-6) forward=cross(up,camera.right);
      forward=norm(forward);
      const right=norm(cross(forward,up));
      // Use the saved ground-plane normal, not a hard-coded world axis.
      // This is the forward view direction projected onto that plane.
      move(forward,sticks.forward*speed); move(right,sticks.strafe*speed);
    }
    const choice = adaptive.select(Number($('resolution').value), performance.now(), $('adaptive').checked);
    const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    const width = aspect >= 1 ? choice.width : Math.max(1, Math.round(choice.width * aspect));
    const height = aspect >= 1 ? Math.max(1, Math.round(choice.width / aspect)) : choice.width;
    // After motion settles, render one full-resolution frame even if no
    // further input event occurred. Continuing idle frames remains optional.
    const restoreFull = !choice.moving && canvas.dataset.quality === 'moving';
    if (dirty || $('continuous').checked || renderer.width !== width || renderer.height !== height || restoreFull) {
      dirty = false; busy = true;
      try {
        const frameCamera = copyCamera(camera);
        const frameRevision = cameraRevision;
        const frameProjection = activeProjection;
        const frameSortMode = activeCompatibility ? 'camera' : 'radial';
        renderer.setCamera(frameCamera);
        const result = await renderer.render({ width, height, projection: frameProjection, sortMode: frameSortMode });
        if (stopped) return;
        lastMainCamera = frameCamera;
        mainRevision = frameRevision;
        updateGizmo();
        // Renderer timing has no output-width field. Supply the chosen long
        // edge so manual re-enabling also adapts correctly in portrait views.
        adaptive.record({ width: choice.width, renderMs: result.renderMs }, choice.moving);
        canvas.dataset.quality = choice.moving ? 'moving' : 'full';
        canvas.dataset.variant = activeVariant;
        canvas.dataset.projection = frameProjection;
        canvas.dataset.compatibility = frameSortMode === 'camera' ? 'insta360' : 'native';
        canvas.style.visibility = 'visible';
        // A mode switch can happen while the previous GPU frame is in flight.
        // Do not offer that stale frame as the newly selected mode's screenshot.
        rendered = frameProjection === activeProjection;
        $('save').disabled = loading || capturing || !rendered;
        const done = performance.now();
        frameTimes.push(done); frameTimes = frameTimes.filter((t) => t >= done-1000);
        const fps = frameTimes.length > 1 ? (frameTimes.length-1)*1000/(done-frameTimes[0]) : null;
        if (done-lastUiUpdate > 200 || !$('continuous').checked) {
          updateFpsHud(fps, 'rendering', fps === null ? '等待连续帧' : '主画面');
          $('stats').textContent = `${width} × ${height} · ${result.renderMs.toFixed(1)} ms / 完成帧${Number.isFinite(result.gpuMs) ? ` · GPU ≈ ${result.gpuMs.toFixed(1)} ms` : ''}${fps !== null ? ` · ${fps.toFixed(1)} FPS` : ''}`;
          $('quality').textContent = `${choice.moving ? '交互分辨率' : '完整分辨率'} · WebGL2 Gaussian Splatting · ${frameProjection === 'pinhole' ? '直接针孔' : '鱼眼渲染 → 针孔重投影'}${frameSortMode === 'camera' ? ' · 影石兼容' : ''}${result.accumulation === 'unorm8' ? ' · 兼容色彩精度' : ''}`;
          lastUiUpdate = done;
        }
      } catch (error) { ready = false; updateProjectionControls(); updateFpsHud(null, 'error', '渲染失败'); message(error); }
      finally { busy = false; }
    } else if (frameTimes.length && time-frameTimes.at(-1) > 1000) {
      frameTimes = [];
      updateFpsHud(0, 'idle', '静止');
      $('stats').textContent = $('stats').textContent.replace(/ · [\d.]+ FPS$/, '') + ' · 静止';
    }
  }
  // The comparison uses the exact pose of the completed main frame. Never
  // add a fisheye render to every moving frame on a resource-limited device.
  if (ready && !loading && !capturing && !calibrationDragging && !stopped && !document.hidden && !dirty
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
  rendererInitialization = WebGlSplatRenderer.create(canvas);
  renderer = await rendererInitialization;
  if (stopped) { renderer.dispose(); return; }
  const info = renderer.adapterInfo || {};
  $('adapter').textContent = info.label || 'WebGL2 · 本机显卡';
  renderer.lostPromise.then((info) => {
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
    const catalogPath = params.get('catalog') || document.querySelector('meta[name="viewer-model-catalog"]')?.content || './models.json';
    const catalogUrl = new URL(catalogPath, document.baseURI);
    const data = await fetchJson(catalogUrl, loadController.signal, 'no-store');
    if (stopped) return;
    catalog = data.models;
    if (!Array.isArray(catalog) || !catalog.length) throw new Error(`${catalogUrl.pathname} 中没有模型，请先导出 WebGL2 模型并生成模型列表`);
    catalog = catalog.map(entry => {
      if (!entry || typeof entry.id !== 'string' || !entry.id.trim() || typeof entry.url !== 'string' || !entry.url.trim()) {
        throw new Error('模型列表中的每项都需要 id 和 url');
      }
      return {...entry, url: new URL(entry.url, catalogUrl).href};
    });
    defaultId = params.get('model') || data.default || catalog[0].id;
  }
  $('model').replaceChildren(...catalog.map((entry) => new Option(entry.name || entry.id, entry.id)));
  const entry = catalog.find((e) => e.id === defaultId || e.aliases?.includes(defaultId));
  if (!entry) throw new Error(`模型列表中没有 ${defaultId}`);
  scheduleFrame();
  await switchModel(entry);
}
flightControls = new FlightControls($('flight-controls'), () =>
  ready && !loading && !stopped && !capturing && !calibrationDragging);
main().catch((error) => {
  if (stopped) return;
  $('loading').hidden = true;
  message(error);
  stopViewer();
});
