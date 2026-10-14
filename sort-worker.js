// Stable float32 radix ordering: radial distance for native fisheye models,
// camera depth for standard 3DGS compatibility. No quantized depth bins.
let positions = null;
let generation = 0;

export function radialOrder(points, camera) {
  const count = points.length / 3;
  const keys = new Uint32Array(count);
  const distances = new Float32Array(keys.buffer);
  const f = Math.fround;
  const p = camera.map(f);
  for (let i = 0; i < count; i++) {
    const x = f(points[3*i] - p[0]);
    const y = f(points[3*i+1] - p[1]);
    const z = f(points[3*i+2] - p[2]);
    // Float32 inputs and their squares fit comfortably in JS float64, even
    // at the finite float32 extremes. sqrt avoids Math.hypot's generic
    // rescaling overhead while preserving the two rounded hypotf stages.
    const xy = f(Math.sqrt(x*x + y*y));
    distances[i] = f(Math.sqrt(xy*xy + z*z));
  }
  return radixOrder(keys);
}

export function cameraOrder(points, camera, forward) {
  if (!forward || forward.length !== 3 || !forward.every(Number.isFinite)) throw new Error('Invalid sort direction.');
  const count=points.length/3,keys=new Uint32Array(count),depth=new Float32Array(keys.buffer);
  const f=Math.fround,p=camera.map(f),direction=forward.map(f);
  for(let i=0;i<count;i++) {
    const x=f(points[3*i]-p[0]),y=f(points[3*i+1]-p[1]),z=f(points[3*i+2]-p[2]);
    const value=f(f(f(x*direction[0])+f(y*direction[1]))+f(z*direction[2]));
    if(Number.isNaN(value)) throw new Error('Non-finite camera sort depth.');
    depth[i]=value===0?0:value;
    // Monotonic unsigned keys for signed IEEE754 depths. Behind-camera
    // splats remain in the index buffer and are rejected by projection.
    const bits=keys[i];keys[i]=(bits & 0x80000000)?~bits:bits^0x80000000;
  }
  return radixOrder(keys);
}

function radixOrder(keys) {
  const count=keys.length;
  let order=new Uint32Array(count),scratch=new Uint32Array(count);
  for(let i=0;i<count;i++) order[i]=i;
  const histogram = new Uint32Array(256);
  for (let shift = 0; shift < 32; shift += 8) {
    histogram.fill(0);
    for (let i = 0; i < count; i++) histogram[(keys[order[i]] >>> shift) & 255]++;
    let prefix = 0;
    for (let b = 0; b < 256; b++) {
      const size = histogram[b];
      histogram[b] = prefix;
      prefix += size;
    }
    for (let i = 0; i < count; i++) {
      const index = order[i];
      scratch[histogram[(keys[index] >>> shift) & 255]++] = index;
    }
    [order, scratch] = [scratch, order];
  }
  return order;
}

if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
  self.onmessage = ({data}) => {
    try {
      if (data.type === 'init') {
        positions = new Float32Array(data.positions);
        generation = data.generation;
      } else if (data.type === 'sort') {
        if (!positions || data.generation !== generation) throw new Error('Stale sort model.');
        const start = performance.now();
        if (!['radial','camera',undefined].includes(data.mode)) throw new Error('Unsupported depth sort.');
        const order = data.mode==='camera' ? cameraOrder(positions,data.position,data.forward) : radialOrder(positions, data.position);
        self.postMessage({type:'sorted',generation,requestId:data.requestId,
          order:order.buffer,sortMs:performance.now()-start}, [order.buffer]);
      }
    } catch (error) {
      self.postMessage({type:'error',generation:data.generation,requestId:data.requestId,
        message:error instanceof Error ? error.message : String(error)});
    }
  };
}
