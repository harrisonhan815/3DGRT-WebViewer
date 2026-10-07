// Perspective SVG transform controls. Geometry remains in model coordinates;
// this module changes only the caller's calibration frame, never GPU models.
const NS = 'http://www.w3.org/2000/svg';
const COLORS = ['#ff6262', '#66db8b', '#67adff'];
const AXIS_NAMES = ['X', 'Y', 'Z'];
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, s) => a.map(v => v * s);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = a => { const n = Math.hypot(...a); return n > 1e-12 ? mul(a, 1 / n) : null; };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const axis = (rotation, i) => rotation.map(row => row[i]);
const cloneFrame = frame => ({origin: [...frame.origin], rotation: frame.rotation.map(row => [...row])});
const finiteVector = (v, length) => Array.isArray(v) && v.length === length && v.every(Number.isFinite);

function checkedFrame(frame) {
  if (!finiteVector(frame.origin, 3) || !Array.isArray(frame.rotation) || frame.rotation.length !== 3
      || !frame.rotation.every(row => finiteVector(row, 3))) throw new Error('Gizmo frame must contain a finite origin and 3×3 rotation.');
  const cols = [0, 1, 2].map(i => axis(frame.rotation, i));
  if (cols.some((a, i) => cols.some((b, j) => Math.abs(dot(a, b) - (i === j ? 1 : 0)) > 1e-4))
      || Math.abs(dot(cols[0], cross(cols[1], cols[2])) - 1) > 1e-4) {
    throw new Error('Gizmo rotation must be orthonormal and right-handed.');
  }
  return cloneFrame(frame);
}

function checkedCamera(camera) {
  if (!['position', 'right', 'down', 'forward'].every(key => finiteVector(camera[key], 3))
      || !Number.isFinite(camera.tanHalfFovY) || camera.tanHalfFovY <= 0) throw new Error('Gizmo camera must be a finite perspective camera.');
  return Object.fromEntries(Object.entries(camera).map(([key, value]) => [key, Array.isArray(value) ? [...value] : value]));
}

function rotate(vector, rotationAxis, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return add(add(mul(vector, c), mul(cross(rotationAxis, vector), s)), mul(rotationAxis, dot(rotationAxis, vector) * (1 - c)));
}

function rotatedFrame(frame, rotationAxis, angle) {
  const columns = [0, 1, 2].map(i => rotate(axis(frame.rotation, i), rotationAxis, angle));
  return {origin: [...frame.origin], rotation: [0, 1, 2].map(row => columns.map(col => col[row]))};
}

function project(point, view) {
  const p = sub(point, view.camera.position), z = dot(p, view.camera.forward);
  if (!(z >= view.near)) return null;
  const result = [view.width / 2 + dot(p, view.camera.right) * view.focal / z,
    view.height / 2 + dot(p, view.camera.down) * view.focal / z];
  return result.every(Number.isFinite) ? result : null;
}

function rayAt(point, view) {
  const x = (point[0] - view.width / 2) / view.focal, y = (point[1] - view.height / 2) / view.focal;
  return {origin: view.camera.position, direction: norm(add(add(mul(view.camera.right, x), mul(view.camera.down, y)), view.camera.forward))};
}

function axisParameter(ray, center, direction) {
  const b = dot(direction, ray.direction), denominator = 1 - b*b;
  if (denominator < 1e-4) return null;
  const delta = sub(ray.origin, center);
  const parameter = (dot(direction, delta) - b * dot(ray.direction, delta)) / denominator;
  return Number.isFinite(parameter) ? parameter : null;
}

function planeVector(ray, center, normal) {
  const denominator = dot(ray.direction, normal);
  if (Math.abs(denominator) < 1e-4) return null;
  const distance = dot(sub(center, ray.origin), normal) / denominator;
  if (!(distance > 0) || !Number.isFinite(distance)) return null;
  return norm(sub(add(ray.origin, mul(ray.direction, distance)), center));
}

function clipScreen(a, b, width, height, margin = 12) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const p = [-dx, dx, -dy, dy], q = [a[0]+margin, width+margin-a[0], a[1]+margin, height+margin-a[1]];
  let low = 0, high = 1;
  for (let i = 0; i < 4; i++) {
    if (Math.abs(p[i]) < 1e-14) { if (q[i] < 0) return null; }
    else {
      const t = q[i] / p[i];
      if (p[i] < 0) low = Math.max(low, t); else high = Math.min(high, t);
      if (low > high) return null;
    }
  }
  return [[a[0]+dx*low, a[1]+dy*low], [a[0]+dx*high, a[1]+dy*high]];
}

function projectSegment(a, b, view) {
  let first = a, second = b;
  const za = dot(sub(a, view.camera.position), view.camera.forward);
  const zb = dot(sub(b, view.camera.position), view.camera.forward);
  if (za < view.near && zb < view.near) return null;
  if (za < view.near) first = add(a, mul(sub(b, a), (view.near - za) / (zb - za)));
  if (zb < view.near) second = add(a, mul(sub(b, a), (view.near - za) / (zb - za)));
  const pa = project(first, view), pb = project(second, view);
  return pa && pb ? clipScreen(pa, pb, view.width, view.height) : null;
}

function segmentDistance(point, segment) {
  const [a, b] = segment, dx = b[0]-a[0], dy = b[1]-a[1];
  const length2 = dx*dx+dy*dy;
  const t = length2 ? clamp(((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length2, 0, 1) : 0;
  return Math.hypot(point[0]-a[0]-t*dx, point[1]-a[1]-t*dy);
}

const pathFor = segments => segments.map(([a, b]) => `M${a[0].toFixed(2)},${a[1].toFixed(2)}L${b[0].toFixed(2)},${b[1].toFixed(2)}`).join('');
function svgElement(name, attributes = {}, text) {
  const element = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (text !== undefined) element.textContent = text;
  return element;
}

export class AlignmentGizmo {
  constructor(container, {onChange = () => {}, onDragState = () => {}} = {}) {
    this.container = container;
    this.previousPointerEvents = container.style.pointerEvents;
    container.style.pointerEvents = 'none';
    this.onChange = onChange;
    this.onDragState = onDragState;
    this.state = {camera: null, frame: null, visible: false, enabled: false, sizePx: 120};
    this.drag = null;
    this.hover = null;
    this.handles = new Map();
    this.disposed = false;
    this.drawKey = '';
    this.svg = svgElement('svg', {'aria-label': 'Ground plane and XYZ translation / rotation controls'});
    Object.assign(this.svg.style, {position: 'absolute', inset: '0', width: '100%', height: '100%',
      overflow: 'hidden', pointerEvents: 'none', touchAction: 'none', userSelect: 'none'});
    container.append(this.svg);
    this.pointerDown = e => this.startDrag(e);
    this.pointerHover = e => {
      if (this.drag) return;
      const next = e.target.closest?.('[data-handle]')?.getAttribute('data-handle') || null;
      if (next !== this.hover) { this.hover = next; this.draw(); }
    };
    this.pointerLeave = () => { if (!this.drag && this.hover) { this.hover = null; this.draw(); } };
    this.pointerMove = e => this.moveDrag(e);
    this.pointerUp = e => {
      if (this.drag && e.pointerId === this.drag.pointerId) {
        e.preventDefault(); e.stopPropagation(); this.finishDrag('end');
      }
    };
    this.pointerCancel = e => { if (this.drag && e.pointerId === this.drag.pointerId) this.finishDrag('cancel'); };
    this.keyDown = e => {
      if (this.drag && e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); this.finishDrag('cancel'); }
    };
    this.svg.addEventListener('pointerdown', this.pointerDown);
    this.svg.addEventListener('pointermove', this.pointerHover);
    this.svg.addEventListener('pointerleave', this.pointerLeave);
    this.svg.addEventListener('lostpointercapture', this.pointerCancel);
    window.addEventListener('pointermove', this.pointerMove, {passive: false});
    window.addEventListener('pointerup', this.pointerUp, {passive: false});
    window.addEventListener('pointercancel', this.pointerCancel);
    window.addEventListener('keydown', this.keyDown);
    this.resizeObserver = new ResizeObserver(() => { this.drawKey = ''; this.draw(); });
    this.resizeObserver.observe(container);
  }

  setState(next) {
    if (this.disposed) return;
    if ('camera' in next) this.state.camera = next.camera == null ? null : checkedCamera(next.camera);
    if ('frame' in next) this.state.frame = next.frame == null ? null : checkedFrame(next.frame);
    if ('visible' in next) this.state.visible = Boolean(next.visible);
    if ('enabled' in next) this.state.enabled = Boolean(next.enabled);
    if ('sizePx' in next) {
      if (!Number.isFinite(next.sizePx)) throw new Error('Gizmo sizePx must be finite.');
      this.state.sizePx = clamp(next.sizePx, 48, 240);
    }
    if (this.drag && (!this.state.visible || !this.state.enabled || !this.state.camera || !this.state.frame)) this.finishDrag('cancel');
    this.draw();
  }

  draw() {
    if (this.disposed) return;
    const bounds = this.container.getBoundingClientRect();
    const {camera, frame, visible, enabled, sizePx} = this.state;
    const key = JSON.stringify([bounds.width, bounds.height, camera, frame, visible, enabled, sizePx, this.hover, this.drag?.handle]);
    if (key === this.drawKey) return;
    this.drawKey = key;
    this.svg.hidden = !visible;
    this.svg.style.display = visible ? '' : 'none';
    this.handles.clear();
    if (!visible || !camera || !frame || bounds.width < 1 || bounds.height < 1) { this.svg.replaceChildren(); return; }
    const depth = dot(sub(frame.origin, camera.position), camera.forward);
    const distance = Math.hypot(...sub(frame.origin, camera.position));
    const view = {camera, width: bounds.width, height: bounds.height,
      focal: bounds.height / (2 * camera.tanHalfFovY), near: Math.max(1e-7, distance * 1e-5)};
    const worldSize = sizePx * Math.max(view.near * 10, depth, distance * 0.15) / view.focal;
    this.view = view;
    this.worldSize = worldSize;
    this.svg.setAttribute('viewBox', `0 0 ${view.width} ${view.height}`);
    const children = [], axes = [0, 1, 2].map(i => norm(axis(frame.rotation, i)));
    const gridSegments = [];
    for (let i = -12; i <= 12; i++) {
      const offset = i * worldSize / 4;
      const lineX = add(frame.origin, mul(axes[0], offset));
      const lineZ = add(frame.origin, mul(axes[2], offset));
      for (const segment of [projectSegment(add(lineX, mul(axes[2], -3*worldSize)), add(lineX, mul(axes[2], 3*worldSize)), view),
        projectSegment(add(lineZ, mul(axes[0], -3*worldSize)), add(lineZ, mul(axes[0], 3*worldSize)), view)]) {
        if (segment) gridSegments.push(segment);
      }
    }
    children.push(svgElement('path', {d: pathFor(gridSegments), fill: 'none', stroke: '#bccfd7', 'stroke-width': 1,
      opacity: 0.3, 'pointer-events': 'none', 'data-ground-grid': 'true'}));
    const center = project(frame.origin, view);
    if (!center) { this.svg.replaceChildren(...children); return; }

    // Build rings before arrows so translation stems win intentional overlaps.
    for (let index = 0; index < 3; index++) {
      const u = axes[(index+1)%3], v = axes[(index+2)%3], radius = worldSize * 0.82;
      const samples = [], segments = [];
      let previous = null;
      for (let step = 0; step <= 96; step++) {
        const angle = step * 2*Math.PI/96;
        const vector = add(mul(u, Math.cos(angle)), mul(v, Math.sin(angle)));
        const world = add(frame.origin, mul(vector, radius)), screen = project(world, view);
        if (screen) samples.push({screen, vector});
        if (previous) { const segment = projectSegment(previous, world, view); if (segment) segments.push(segment); }
        previous = world;
      }
      if (segments.length) this.handles.set(`r${'xyz'[index]}`, {index, kind: 'rotate', direction: axes[index], radius, samples, segments});
    }
    for (let index = 0; index < 3; index++) {
      const endpoint = project(add(frame.origin, mul(axes[index], worldSize)), view);
      const segment = projectSegment(frame.origin, add(frame.origin, mul(axes[index], worldSize)), view);
      const length = endpoint ? Math.hypot(endpoint[0]-center[0], endpoint[1]-center[1]) : 0;
      if (!segment && !endpoint) continue;
      const endOn = length < 8;
      this.handles.set(`t${'xyz'[index]}`, {index, kind: 'translate', direction: axes[index], endOn,
        endpoint: endpoint || segment[1], center, segments: segment ? [segment] : [], samples: []});
    }

    const endOn = [];
    for (const [name, handle] of this.handles) {
      const active = this.drag?.handle === name || this.hover === name;
      const color = COLORS[handle.index], opacity = enabled ? (active ? 1 : 0.86) : 0.35;
      let pick;
      if (handle.kind === 'rotate') {
        const candidates = handle.samples.filter(({screen: p}, index) => index % 8 === 0
          && p[0] > 14 && p[1] > 14 && p[0] < view.width-14 && p[1] < view.height-14);
        let best = -1;
        for (const sample of candidates.length ? candidates : handle.samples) {
          let score = Infinity;
          for (const [otherName, other] of this.handles) if (otherName !== name) {
            for (const segment of other.segments) score = Math.min(score, segmentDistance(sample.screen, segment));
            if (other.endOn) score = Math.min(score, Math.hypot(sample.screen[0]-center[0], sample.screen[1]-center[1]) - 10);
          }
          if (score > best) { best = score; pick = sample.screen; }
        }
        pick ||= handle.samples[0]?.screen || center;
      } else if (handle.endOn) pick = center;
      else pick = [center[0] + (handle.endpoint[0]-center[0])*0.96, center[1] + (handle.endpoint[1]-center[1])*0.96];
      handle.pick = pick;
      const group = svgElement('g', {opacity});
      if (handle.endOn) {
        group.append(svgElement('circle', {cx:center[0], cy:center[1], r:7, fill:'#172027', stroke:color, 'stroke-width':active?3:2, 'pointer-events':'none'}));
        group.append(svgElement('circle', {cx:center[0], cy:center[1], r:2, fill:color, 'pointer-events':'none'}));
      } else {
        group.append(svgElement('path', {d:pathFor(handle.segments), fill:'none', stroke:color, 'stroke-width':active?3:2,
          'stroke-linecap':'round', 'pointer-events':'none'}));
        if (handle.kind === 'translate') {
          const endpoint = handle.endpoint, delta = sub(endpoint, center), length = Math.hypot(...delta);
          const d = mul(delta, 1/length), side = [-d[1], d[0]];
          const a = add(sub(endpoint, mul(d, 11)), mul(side, 4.5));
          const b = sub(sub(endpoint, mul(d, 11)), mul(side, 4.5));
          group.append(svgElement('polygon', {points:[endpoint,a,b].map(p=>p.join(',')).join(' '), fill:color, 'pointer-events':'none'}));
        }
      }
      const hitAttributes = {'data-handle':name, 'data-handle-kind':handle.kind, 'data-pick-x':pick[0], 'data-pick-y':pick[1],
        'aria-label':`${handle.kind === 'translate'?'Translate':'Rotate'} ${AXIS_NAMES[handle.index]} axis`,
        style:`pointer-events:${enabled ? (handle.endOn?'all':'stroke') : 'none'};cursor:${this.drag?'grabbing':'grab'};touch-action:none`};
      const hit = handle.endOn ? svgElement('circle', {...hitAttributes,cx:center[0],cy:center[1],r:12,fill:'transparent'})
        : svgElement('path', {...hitAttributes,d:pathFor(handle.segments),fill:'none',stroke:'transparent','stroke-width':18,'stroke-linecap':'round'});
      group.append(hit);
      const label = handle.kind === 'rotate' ? `R${AXIS_NAMES[handle.index].toLowerCase()}` : AXIS_NAMES[handle.index];
      group.append(svgElement('text', {x:pick[0]+7,y:pick[1]-7,fill:color,stroke:'#0b1016','stroke-width':3,
        'paint-order':'stroke','font-size':11,'font-family':'sans-serif','font-weight':600,'pointer-events':'none'}, label));
      (handle.endOn ? endOn : children).push(group);
    }
    children.push(...endOn);
    this.svg.replaceChildren(...children);
  }

  localPoint(event, rectangle = this.container.getBoundingClientRect()) {
    return [event.clientX - rectangle.left, event.clientY - rectangle.top];
  }

  startDrag(event) {
    if (this.disposed || this.drag || !this.state.visible || !this.state.enabled || (event.pointerType === 'mouse' && event.button !== 0)) return;
    const name = event.target.closest?.('[data-handle]')?.getAttribute('data-handle');
    const handle = this.handles.get(name);
    if (!handle || !this.state.frame || !this.view) return;
    event.preventDefault(); event.stopPropagation();
    const rectangle = this.container.getBoundingClientRect(), point = this.localPoint(event, rectangle);
    const frame = cloneFrame(this.state.frame), view = {...this.view, camera:checkedCamera(this.view.camera)};
    const ray = rayAt(point, view), direction = [...handle.direction];
    const drag = {handle:name, pointerId:event.pointerId, frame, view, rectangle, start:point, direction,
      kind:handle.kind, worldSize:this.worldSize, currentAngle:0, lastAngle:0, accumulatedAngle:0};
    if (handle.kind === 'translate') {
      const projected = project(add(frame.origin, mul(direction, this.worldSize)), view), center = project(frame.origin, view);
      const delta = projected && center ? sub(projected, center) : [0,0], length = Math.hypot(...delta);
      drag.screenDirection = length > 8 ? mul(delta, 1/length) : [0, dot(direction, view.camera.forward) >= 0 ? -1 : 1];
      drag.unitsPerPixel = this.worldSize / Math.max(length, this.state.sizePx * 0.6);
      drag.startParameter = axisParameter(ray, frame.origin, direction);
      drag.useRay = length > 20 && 1-dot(direction,ray.direction)**2 > 0.04 && drag.startParameter !== null;
    } else {
      drag.startVector = planeVector(ray, frame.origin, direction);
      drag.usePlane = Math.abs(dot(ray.direction, direction)) > 0.08 && drag.startVector !== null;
      const sample = handle.samples.reduce((best, candidate) => !best || Math.hypot(...sub(candidate.screen,point)) < Math.hypot(...sub(best.screen,point)) ? candidate : best, null);
      const vector = drag.startVector || sample?.vector || axis(frame.rotation, (handle.index+1)%3);
      const plus = project(add(frame.origin, mul(rotate(vector,direction,0.02),handle.radius)),view);
      const minus = project(add(frame.origin, mul(rotate(vector,direction,-0.02),handle.radius)),view);
      let tangent = plus && minus ? mul(sub(plus,minus),25) : [0,0];
      let pixelsPerRadian = Math.hypot(...tangent);
      if (pixelsPerRadian < this.state.sizePx * 0.15) {
        const center = project(frame.origin,view), tip = project(add(frame.origin,mul(direction,this.worldSize)),view);
        const projectedAxis = center && tip ? norm(sub(tip,center)) : null;
        tangent = projectedAxis ? [-projectedAxis[1],projectedAxis[0]] : [1,0];
        pixelsPerRadian = this.state.sizePx * 0.7;
      }
      drag.screenDirection = norm(tangent);
      drag.radiansPerPixel = 1 / Math.max(this.state.sizePx * 0.35,pixelsPerRadian);
    }
    this.drag = drag;
    this.hover = name;
    try { this.svg.setPointerCapture(event.pointerId); } catch { /* Window listeners also cover uncaptured synthetic input. */ }
    this.onDragState(true);
    this.onChange(cloneFrame(frame), {phase:'start',handle:name});
    this.draw();
  }

  moveDrag(event) {
    const drag = this.drag;
    if (this.disposed || !drag || event.pointerId !== drag.pointerId) return;
    event.preventDefault(); event.stopPropagation();
    const point = this.localPoint(event,drag.rectangle), ray = rayAt(point,drag.view);
    let next;
    if (drag.kind === 'translate') {
      let amount;
      if (drag.useRay) {
        const parameter = axisParameter(ray,drag.frame.origin,drag.direction);
        if (parameter === null) return;
        amount = parameter-drag.startParameter;
      } else amount = dot(sub(point,drag.start),drag.screenDirection)*drag.unitsPerPixel;
      next = {origin:add(drag.frame.origin,mul(drag.direction,amount)),rotation:drag.frame.rotation.map(row=>[...row])};
    } else {
      let angle;
      if (drag.usePlane) {
        const vector = planeVector(ray,drag.frame.origin,drag.direction);
        if (!vector) return;
        const current = Math.atan2(dot(drag.direction,cross(drag.startVector,vector)),dot(drag.startVector,vector));
        let delta = current-drag.lastAngle;
        if (delta > Math.PI) delta -= 2*Math.PI;
        if (delta < -Math.PI) delta += 2*Math.PI;
        drag.accumulatedAngle += delta;
        drag.lastAngle = current;
        angle = drag.accumulatedAngle;
      } else angle = dot(sub(point,drag.start),drag.screenDirection)*drag.radiansPerPixel;
      next = rotatedFrame(drag.frame,drag.direction,angle);
    }
    if (!finiteVector(next.origin,3) || !next.rotation.every(row=>finiteVector(row,3))) return;
    this.state.frame = next;
    this.onChange(cloneFrame(next), {phase:'change',handle:drag.handle});
    this.draw();
  }

  finishDrag(phase, notify = true) {
    const drag = this.drag;
    if (!drag) return;
    this.drag = null;
    if (phase === 'cancel') this.state.frame = cloneFrame(drag.frame);
    try { if (this.svg.hasPointerCapture(drag.pointerId)) this.svg.releasePointerCapture(drag.pointerId); } catch { /* Canvas may already be detached. */ }
    if (notify) this.onChange(cloneFrame(this.state.frame || drag.frame), {phase,handle:drag.handle});
    this.onDragState(false);
    this.draw();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.finishDrag('cancel',false);
    this.resizeObserver.disconnect();
    this.svg.removeEventListener('pointerdown',this.pointerDown);
    this.svg.removeEventListener('pointermove',this.pointerHover);
    this.svg.removeEventListener('pointerleave',this.pointerLeave);
    this.svg.removeEventListener('lostpointercapture',this.pointerCancel);
    window.removeEventListener('pointermove',this.pointerMove);
    window.removeEventListener('pointerup',this.pointerUp);
    window.removeEventListener('pointercancel',this.pointerCancel);
    window.removeEventListener('keydown',this.keyDown);
    this.svg.remove();
    this.container.style.pointerEvents = this.previousPointerEvents;
    this.handles.clear();
  }
}
