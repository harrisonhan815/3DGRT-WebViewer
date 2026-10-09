/*
 * EWA projection, Gaussian evaluation and SH basis follow the 3DGS reference:
 * Copyright (C) 2023, Inria / GRAPHDECO. All rights reserved.
 * Research, evaluation and non-commercial use under the reference LICENSE.md:
 * See LICENSE-3DGS.md (from thirdparty/DirectFisheye-GS/LICENSE.md).
 * The browser implementation below retains the reference's 16-pixel tile
 * bounds and 0.3-pixel covariance filter. Fixed-function blending cannot
 * reproduce the CUDA kernel's per-pixel early transmittance termination.
 */

const EWA_VERTEX = `#version 300 es
precision highp float;
precision highp int;
layout(location=0) in uint a_index;
uniform highp sampler2D u_geometry;
uniform highp sampler2D u_sh;
uniform int u_geometryWidth;
uniform int u_shWidth;
uniform vec3 u_position;
uniform vec3 u_right;
uniform vec3 u_down;
uniform vec3 u_forward;
uniform vec2 u_size;
uniform vec2 u_focal;
uniform vec2 u_principal;
uniform vec2 u_tanFov;
uniform float u_near;
uniform float u_lowPass;
uniform float u_minAlpha;
uniform int u_projection;
uniform int u_degree;
uniform bool u_antialias;
flat out vec2 v_center;
flat out vec3 v_conic;
flat out vec4 v_colorOpacity;

vec4 geom(int offset) {
  int n = int(a_index)*3 + offset;
  return texelFetch(u_geometry, ivec2(n % u_geometryWidth, n / u_geometryWidth), 0);
}
vec4 shTexel(int n) {
  return texelFetch(u_sh, ivec2(n % u_shWidth, n / u_shWidth), 0);
}
vec3 coefficient(int k) {
  int scalar = int(a_index)*48 + 3*k;
  int lane = scalar % 4;
  int texel = scalar / 4;
  vec4 a = shTexel(texel);
  if (lane == 0) return a.xyz;
  if (lane == 1) return a.yzw;
  vec4 b = shTexel(texel+1);
  if (lane == 2) return vec3(a.zw,b.x);
  return vec3(a.w,b.xy);
}
vec3 shColor(vec3 direction) {
  float x=direction.x, y=direction.y, z=direction.z;
  vec3 rgb = 0.28209479177387814 * coefficient(0);
  if (u_degree > 0) {
    rgb += -0.4886025119029199*y*coefficient(1)
           +0.4886025119029199*z*coefficient(2)
           -0.4886025119029199*x*coefficient(3);
  }
  if (u_degree > 1) {
    float xx=x*x, yy=y*y, zz=z*z;
    rgb += 1.0925484305920792*x*y*coefficient(4)
           -1.0925484305920792*y*z*coefficient(5)
           +0.31539156525252005*(2.0*zz-xx-yy)*coefficient(6)
           -1.0925484305920792*x*z*coefficient(7)
           +0.5462742152960396*(xx-yy)*coefficient(8);
    if (u_degree > 2) {
      rgb += -0.5900435899266435*y*(3.0*xx-yy)*coefficient(9)
             +2.890611442640554*x*y*z*coefficient(10)
             -0.4570457994644658*y*(4.0*zz-xx-yy)*coefficient(11)
             +0.3731763325901154*z*(2.0*zz-3.0*xx-3.0*yy)*coefficient(12)
             -0.4570457994644658*x*(4.0*zz-xx-yy)*coefficient(13)
             +1.445305721320277*z*(xx-yy)*coefficient(14)
             -0.5900435899266435*x*(xx-3.0*yy)*coefficient(15);
    }
  }
  return max(rgb+0.5,vec3(0.0));
}
bool finiteVec(vec3 v) { return !any(isnan(v)) && !any(isinf(v)); }
void main() {
  gl_Position=vec4(0.0,0.0,2.0,1.0);
  v_center=vec2(0.0); v_conic=vec3(1.0,0.0,1.0); v_colorOpacity=vec4(0.0);
  vec4 g0=geom(0);
  // With AA compensation disabled, opacity is an upper bound on every
  // pixel's alpha. This skips only splats the reference would discard at
  // every pixel; the stored model and the sort order remain unchanged.
  if (!u_antialias && g0.w < u_minAlpha) return;
  vec4 g1=geom(1), g2=geom(2);
  vec3 relative=g0.xyz-u_position;
  vec3 p=vec3(dot(relative,u_right),dot(relative,u_down),dot(relative,u_forward));
  float distanceToCamera=length(p);
  vec2 center;
  vec3 ju,jv;
  if (u_projection == 0) {
    if (p.z <= u_near) return;
    // The homogeneous reference adds 1e-7 to w before ndc2Pix.
    center = u_focal*p.xy/(p.z+1e-7)+u_principal-0.5;
    vec2 bounded=clamp(p.xy/p.z,-1.3*u_tanFov,1.3*u_tanFov);
    ju=vec3(u_focal.x/p.z,0.0,-u_focal.x*bounded.x/p.z);
    jv=vec3(0.0,u_focal.y/p.z,-u_focal.y*bounded.y/p.z);
  } else {
    if (distanceToCamera <= u_near) return;
    float q=dot(p.xy,p.xy), r=sqrt(q);
    float theta=atan(r,p.z);
    if (theta > 1.5707963267948966) return;
    float h,hq,hz;
    if (p.z > 0.0 && q < 1e-4*p.z*p.z) {
      float w=q/(p.z*p.z), zi=1.0/p.z;
      float poly=1.0+w*(-1.0/3.0+w*(1.0/5.0+w*(-1.0/7.0+w/9.0)));
      float dp=-1.0/3.0+w*(2.0/5.0+w*(-3.0/7.0+w*4.0/9.0));
      h=poly*zi; hq=dp*zi*zi*zi; hz=-(poly+2.0*w*dp)*zi*zi;
    } else {
      float inv=1.0/dot(p,p);
      h=theta/r; hq=(p.z*inv-h)/(2.0*q); hz=-inv;
    }
    center=u_focal*(p.xy*h)+u_principal-0.5;
    ju=u_focal.x*vec3(h+2.0*p.x*p.x*hq,2.0*p.x*p.y*hq,p.x*hz);
    jv=u_focal.y*vec3(2.0*p.x*p.y*hq,h+2.0*p.y*p.y*hq,p.y*hz);
  }
  vec3 gu=u_right*ju.x+u_down*ju.y+u_forward*ju.z;
  vec3 gv=u_right*jv.x+u_down*jv.y+u_forward*jv.z;
  mat3 covariance=mat3(g1.x,g1.y,g1.z, g1.y,g1.w,g2.x, g1.z,g2.x,g2.y);
  float a=dot(gu,covariance*gu), b=dot(gu,covariance*gv), c=dot(gv,covariance*gv);
  float rawDet=a*c-b*b;
  a+=u_lowPass; c+=u_lowPass;
  float det=a*c-b*b;
  if (det == 0.0 || isnan(det) || isinf(det)) return;
  vec3 conic=vec3(c,-b,a)/det;
  float mid=0.5*(a+c);
  float radius=ceil(3.0*sqrt(mid+sqrt(max(0.1,mid*mid-det))));
  if (isnan(radius) || isinf(radius) || !finiteVec(vec3(center,0.0)) || !finiteVec(conic)) return;
  // The CUDA rasterizer visits complete 16x16 tiles, including Gaussian tails
  // outside the nominal 3-sigma ellipse. Float truncation avoids integer
  // overflow for enormous but finite projected radii before clipping.
  vec2 grid=ceil(u_size/16.0);
  vec2 lo=clamp(trunc((center-radius)/16.0),vec2(0.0),grid)*16.0;
  vec2 hi=clamp(trunc((center+radius+15.0)/16.0),vec2(0.0),grid)*16.0;
  if (any(lessThanEqual(hi,lo))) return;
  vec2 corner=vec2(float(gl_VertexID & 1),float((gl_VertexID >> 1) & 1));
  vec2 edge=mix(lo,hi,corner);
  gl_Position=vec4(2.0*edge.x/u_size.x-1.0,1.0-2.0*edge.y/u_size.y,0.0,1.0);
  v_center=center;
  v_conic=conic;
  float opacity=g0.w;
  if (u_antialias) opacity*=sqrt(max(0.000025,rawDet/det));
  vec3 direction=relative/max(length(relative),1e-30);
  v_colorOpacity=vec4(shColor(direction),opacity);
}`;

const EWA_FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
uniform vec2 u_size;
uniform float u_minAlpha;
uniform float u_maxAlpha;
uniform int u_projection;
flat in vec2 v_center;
flat in vec3 v_conic;
flat in vec4 v_colorOpacity;
out vec4 outColor;
void main() {
  vec2 pixel=vec2(gl_FragCoord.x-0.5,u_size.y-gl_FragCoord.y-0.5);
  if (u_projection == 1 && length(pixel+0.5-u_size*0.5)>min(u_size.x,u_size.y)*0.5) discard;
  vec2 d=v_center-pixel;
  float power=-0.5*(v_conic.x*d.x*d.x+v_conic.z*d.y*d.y)-v_conic.y*d.x*d.y;
  if (power > 0.0) discard;
  float alpha=min(u_maxAlpha,v_colorOpacity.a*exp(power));
  if (alpha < u_minAlpha) discard;
  outColor=vec4(v_colorOpacity.rgb*alpha,alpha);
}`;

const QUAD_VERTEX = `#version 300 es
precision highp float;
out vec2 uv;
void main() {
  vec2 p=vec2(float((gl_VertexID << 1) & 2),float(gl_VertexID & 2));
  uv=p; gl_Position=vec4(p*2.0-1.0,0.0,1.0);
}`;
const RESOLVE_FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_image;
uniform vec3 u_background;
uniform vec2 u_size;
uniform int u_projection;
uniform vec2 u_sourceSize;
uniform vec2 u_sourcePrincipal;
uniform float u_focal;
in vec2 uv;
out vec4 color;
vec4 sourcePixel(ivec2 pixel) {
  return texelFetch(u_image,clamp(pixel,ivec2(0),ivec2(u_sourceSize)-1),0);
}
vec4 perspectiveSample() {
  // A perspective output ray samples the continuous equidistant image.
  // Crop coordinates and principal points use pixel-edge conventions; GL
  // texture rows run upwards, while the camera's y axis points downwards.
  vec2 ray=vec2(uv.x-0.5,0.5-uv.y)*u_size/u_focal;
  float radius=length(ray);
  float scale=radius>1e-8?atan(radius)/radius:1.0;
  vec2 edge=u_focal*scale*ray+u_sourcePrincipal;
  vec2 texel=vec2(edge.x,u_sourceSize.y-edge.y)-0.5;
  ivec2 lo=ivec2(floor(texel));
  vec2 fraction=fract(texel);
  // Explicit filtering also works for RGBA32F without requiring
  // OES_texture_float_linear. Interpolate premultiplied color and alpha
  // before the background and the single final UNORM conversion.
  return mix(mix(sourcePixel(lo),sourcePixel(lo+ivec2(1,0)),fraction.x),
             mix(sourcePixel(lo+ivec2(0,1)),sourcePixel(lo+ivec2(1,1)),fraction.x),fraction.y);
}
void main() {
  if (u_projection == 1 && length((uv-0.5)*u_size)>min(u_size.x,u_size.y)*0.5) {
    color=vec4(0.0,0.0,0.0,1.0); return;
  }
  vec4 splats=u_projection==2?perspectiveSample():texture(u_image,uv);
  color=vec4(splats.rgb+(1.0-splats.a)*u_background,1.0);
}`;

function abortError() { return new DOMException('Renderer stopped.', 'AbortError'); }
function vector(value, name) {
  if (!value || value.length !== 3 || !Array.from(value).every(Number.isFinite)) throw new Error(`Invalid camera ${name}.`);
  return Array.from(value);
}
function copyCamera(camera) {
  if (!camera || !(camera.tanHalfFovY > 0) || !Number.isFinite(camera.tanHalfFovY)) throw new Error('Invalid camera field of view.');
  return {position:vector(camera.position,'position'),right:vector(camera.right,'right'),
    down:vector(camera.down,'down'),forward:vector(camera.forward,'forward'),tanHalfFovY:camera.tanHalfFovY};
}
function renderSource(width,height,camera,projection) {
  const focal=projection===1?Math.min(width,height)/Math.PI:height/(2*camera.tanHalfFovY);
  if(!Number.isFinite(focal)||focal<=0||focal>3.402823466e38) throw new Error('Camera focal length is outside the supported range.');
  let sourceWidth=width,sourceHeight=height;
  if(projection===2) {
    // The extrema of each equidistant coordinate over the perspective
    // rectangle lie on the corresponding axis. Two pixels on each side
    // retain both bilinear taps even at the output image's outer edge.
    sourceWidth=Math.ceil(2*focal*Math.atan(width/(2*focal)))+4;
    sourceHeight=Math.ceil(2*focal*Math.atan(height/(2*focal)))+4;
  }
  return {width:sourceWidth,height:sourceHeight,focal,
    principal:[sourceWidth/2,sourceHeight/2],projection};
}
export function shStorageFormat(metadata = {}) {
  const file=metadata.files?.sh;
  const dtype=file?.dtype ?? 'float32';
  if (!['float16','float32'].includes(dtype)) throw new Error(`Unsupported SH dtype: ${dtype}.`);
  const stride=dtype==='float16'?96:192;
  if (file?.stride !== undefined && file.stride !== stride) throw new Error('SH stride does not match its dtype.');
  if (metadata.byte_order !== undefined && metadata.byte_order !== 'little-endian') throw new Error('Only little-endian model data is supported.');
  return {dtype,stride,packed:dtype==='float16'};
}
function program(gl, vertexSource, fragmentSource) {
  const shaders=[];
  let result;
  try {
    for (const [type, source] of [[gl.VERTEX_SHADER,vertexSource],[gl.FRAGMENT_SHADER,fragmentSource]]) {
      const shader=gl.createShader(type);
      shaders.push(shader);
      gl.shaderSource(shader,source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader)||'WebGL shader compilation failed.');
    }
    result=gl.createProgram();
    for(const shader of shaders) gl.attachShader(result,shader);
    gl.linkProgram(result);
    if (!gl.getProgramParameter(result,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(result)||'WebGL program link failed.');
    const uniforms={};
    for(let i=0;i<gl.getProgramParameter(result,gl.ACTIVE_UNIFORMS);i++) {
      const info=gl.getActiveUniform(result,i);
      uniforms[info.name]=gl.getUniformLocation(result,info.name);
    }
    return {program:result,uniforms};
  } catch(error) {
    if(result) gl.deleteProgram(result);
    throw error;
  } finally { for(const shader of shaders) gl.deleteShader(shader); }
}

export class WebGlSplatRenderer {
  static async create(canvas) {
    const gl=canvas.getContext('webgl2',{alpha:false,antialias:false,depth:false,stencil:false,
      premultipliedAlpha:false,preserveDrawingBuffer:false,powerPreference:'high-performance'});
    if (!gl) throw new Error('此浏览器无法创建 WebGL2 上下文；请启用硬件加速或更换支持 WebGL2 的浏览器。');
    const renderer=new WebGlSplatRenderer(canvas,gl);
    try { renderer._initialize(); return renderer; }
    catch(error) { renderer.dispose(); throw error; }
  }
  constructor(canvas,gl) {
    this.canvas=canvas; this.gl=gl;
    this.disposed=false; this.closing=false; this.rendering=false;
    this.width=0; this.height=0; this.generation=0; this.requestId=0;
    this.model=null; this.target=null; this.previewTarget=null; this.sortPending=null;
    this._syncWaits=new Set(); this._queries=new Set();
    this.lostPromise=new Promise(resolve=>{this._resolveLost=resolve;});
    this._lostHandler=event=>{
      event.preventDefault();
      this._resolveLost({reason:'context-lost',message:'WebGL context lost. Reload this page to recreate it.'});
      this.dispose(false);
    };
    canvas.addEventListener('webglcontextlost',this._lostHandler);
  }
  _initialize() {
    const gl=this.gl;
    this.limits={maxTextureSize:gl.getParameter(gl.MAX_TEXTURE_SIZE),
      maxRenderbufferSize:gl.getParameter(gl.MAX_RENDERBUFFER_SIZE)};
    this.limits.maxTexturePixels=this.limits.maxTextureSize**2;
    const debug=gl.getExtension('WEBGL_debug_renderer_info');
    this.adapterInfo={label:debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER)};
    this.floatColor=gl.getExtension('EXT_color_buffer_float');
    this.halfColor=gl.getExtension('EXT_color_buffer_half_float');
    this.floatBlend=this.floatColor && gl.getExtension('EXT_float_blend');
    this.accumulation=this.floatBlend?'float32':(this.floatColor||this.halfColor?'float16':'unorm8');
    this.timer=gl.getExtension('EXT_disjoint_timer_query_webgl2');
    this.ewa=program(gl,EWA_VERTEX,EWA_FRAGMENT);
    this.resolve=program(gl,QUAD_VERTEX,RESOLVE_FRAGMENT);
    this.emptyVao=gl.createVertexArray();
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE); gl.disable(gl.DITHER);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT,4);
    this._checkError('initialization');
  }
  _assertAlive() { if(this.disposed||this.closing) throw abortError(); }
  _checkError(action) {
    const error=this.gl.getError();
    if(error!==this.gl.NO_ERROR) throw new Error(`WebGL ${action} failed (0x${error.toString(16)}).`);
  }
  validateModel(metadata) {
    this._assertAlive();
    if(metadata?.schema!=='fullcircle-webgl2-ewa-v1') throw new Error('This renderer requires a fullcircle-webgl2-ewa-v1 model.');
    const count=metadata.gaussian_count;
    if(!Number.isSafeInteger(count)||count<=0||count>0x7fffffff/48) throw new Error('Invalid Gaussian count.');
    const format=shStorageFormat(metadata);
    const geometryBytes=count*48,shBytes=count*format.stride;
    if(metadata.files?.geometry?.stride!==undefined&&metadata.files.geometry.stride!==48) throw new Error('Expected 48-byte covariance geometry records.');
    for(const [name,expected] of [['geometry',geometryBytes],['sh',shBytes]]) {
      const declared=metadata.files?.[name]?.byteLength;
      if(declared!==undefined&&declared!==expected) throw new Error(`${name} byteLength does not match Gaussian count.`);
    }
    if(count*12>this.limits.maxTexturePixels) throw new Error(`SH texture exceeds this device's ${this.limits.maxTextureSize}² texture limit.`);
    const render=metadata.render||{};
    if(render.depth_sort!==undefined&&render.depth_sort!=='radial') throw new Error('This model requires unsupported non-radial sorting.');
    if(render.sh_degree!==undefined&&(!Number.isInteger(render.sh_degree)||render.sh_degree<0||render.sh_degree>3)) throw new Error('SH degree must be between 0 and 3.');
    if(render.kernel_degree!==undefined&&render.kernel_degree!==2) throw new Error('Only the quadratic 3DGS Gaussian kernel is supported.');
    if(render.sh_direction!==undefined&&render.sh_direction!=='camera_to_gaussian') throw new Error('Unsupported SH direction.');
    for(const name of ['low_pass_variance','near_plane','min_alpha','max_alpha']) {
      if(render[name]!==undefined&&(!Number.isFinite(render[name])||render[name]<0)) throw new Error(`Invalid render parameter ${name}.`);
    }
    if(render.background!==undefined) vector(render.background,'background');
    return {geometryBytes,shBytes,totalBytes:geometryBytes+shBytes,shDtype:format.dtype};
  }
  _dataTexture(values,texels,half=false) {
    const gl=this.gl;
    const width=Math.min(this.limits.maxTextureSize,Math.max(1,Math.min(texels,4096),Math.ceil(texels/this.limits.maxTextureSize)));
    const height=Math.ceil(texels/width);
    if(height>this.limits.maxTextureSize) throw new Error('Model texture exceeds device limits.');
    let data=values;
    if(values.length!==width*height*4) { data=new values.constructor(width*height*4); data.set(values); }
    const texture=gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D,texture);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D,0,half?gl.RGBA16F:gl.RGBA32F,width,height,0,gl.RGBA,half?gl.HALF_FLOAT:gl.FLOAT,data);
    try { this._checkError('model texture upload'); return {texture,width,height}; }
    catch(error) { gl.deleteTexture(texture); throw error; }
  }
  async loadModel({geometry,sh,metadata}) {
    this._assertAlive();
    if(this.rendering) throw new Error('Wait for the current frame before loading a model.');
    const sizes=this.validateModel(metadata),format=shStorageFormat(metadata);
    if(!(geometry instanceof ArrayBuffer)||geometry.byteLength!==sizes.geometryBytes||
       !(sh instanceof ArrayBuffer)||sh.byteLength!==sizes.shBytes) throw new Error('Model buffer sizes do not match the manifest.');
    const gl=this.gl,count=metadata.gaussian_count,values=new Float32Array(geometry);
    const positions=new Float32Array(count*3);
    for(let i=0;i<count;i++) {
      for(let j=0;j<10;j++) if(!Number.isFinite(values[12*i+j])) throw new Error('Geometry contains a non-finite value.');
      positions.set(values.subarray(12*i,12*i+3),3*i);
    }
    const next={metadata,count,format};
    let worker;
    try {
      next.geometry=this._dataTexture(values,count*3);
      next.sh=this._dataTexture(format.packed?new Uint16Array(sh):new Float32Array(sh),count*12,format.packed);
      next.indexBuffer=gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER,next.indexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER,count*4,gl.DYNAMIC_DRAW);
      next.vao=gl.createVertexArray(); gl.bindVertexArray(next.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER,next.indexBuffer);
      gl.enableVertexAttribArray(0); gl.vertexAttribIPointer(0,1,gl.UNSIGNED_INT,0,0); gl.vertexAttribDivisor(0,1);
      gl.bindVertexArray(null);
      this._checkError('model upload');
      worker=new Worker(new URL('./sort-worker.js',import.meta.url),{type:'module'});
      const generation=this.generation+1;
      worker.onmessage=event=>this._sortMessage(event.data);
      worker.onerror=event=>{
        if(this.worker!==worker) return;
        this.sortFailure=new Error(event.message||'Depth sorting worker failed.');
        this._rejectSort(this.sortFailure);
      };
      worker.postMessage({type:'init',generation,positions:positions.buffer},[positions.buffer]);
      this._assertAlive();
      this._rejectSort(abortError()); this.worker?.terminate();
      this._destroyModel(this.model); this.model=next; this.worker=worker; this.generation=generation;
      this.sortedPosition=null; this.lastSortMs=0; this.sortFailure=null;
      this._destroyTarget(this.previewTarget); this.previewTarget=null;
    } catch(error) { worker?.terminate(); this._destroyModel(next); throw error; }
  }
  setCamera(camera) { this._assertAlive(); this.camera=copyCamera(camera); }
  _rejectSort(error) { if(this.sortPending) { this.sortPending.reject(error); this.sortPending=null; } }
  _sortMessage(data) {
    const pending=this.sortPending;
    if(!pending||data.generation!==this.generation||data.requestId!==pending.requestId) return;
    this.sortPending=null;
    if(this.disposed||this.closing) { pending.reject(abortError()); return; }
    if(data.type==='error') { pending.reject(new Error(data.message)); return; }
    if(data.type!=='sorted'||data.order.byteLength!==this.model.count*4) { pending.reject(new Error('Invalid sort worker response.')); return; }
    const gl=this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER,this.model.indexBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER,0,new Uint32Array(data.order));
    this.sortedPosition=pending.position;
    this.lastSortMs=data.sortMs;
    pending.resolve(data.sortMs);
  }
  async _sort(position) {
    if(this.sortFailure) throw this.sortFailure;
    const p=position.map(Math.fround);
    if(this.sortedPosition?.every((value,i)=>value===p[i])) return 0;
    if(this.sortPending) throw new Error('Another depth sort is already pending.');
    const requestId=++this.requestId;
    return new Promise((resolve,reject)=>{
      this.sortPending={requestId,position:p,resolve,reject};
      this.worker.postMessage({type:'sort',generation:this.generation,requestId,position:p});
    });
  }
  _attachment(width,height,internalFormat,type) {
    const gl=this.gl,texture=gl.createTexture(),fbo=gl.createFramebuffer();
    gl.bindTexture(gl.TEXTURE_2D,texture);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D,0,internalFormat,width,height,0,gl.RGBA,type,null);
    gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,texture,0);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE) {
      gl.deleteFramebuffer(fbo); gl.deleteTexture(texture);
      throw new Error('This device cannot allocate the requested render target.');
    }
    return {texture,fbo};
  }
  _target(width,height,previous,source) {
    if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||
      width>this.limits.maxTextureSize||height>this.limits.maxTextureSize) throw new Error('Invalid render dimensions.');
    if(!Number.isInteger(source.width)||!Number.isInteger(source.height)||source.width<1||source.height<1||
      source.width>this.limits.maxTextureSize||source.height>this.limits.maxTextureSize) throw new Error('The fisheye render crop exceeds this device\'s texture limit.');
    if(previous?.width===width&&previous?.height===height&&previous.source.width===source.width&&
      previous.source.height===source.height&&previous.accumulation===this.accumulation) {
      previous.source=source; return previous;
    }
    const gl=this.gl,next={width,height,source,accumulation:this.accumulation};
    try {
      next.accum=this._attachment(source.width,source.height,this.accumulation==='float32'?gl.RGBA32F:this.accumulation==='float16'?gl.RGBA16F:gl.RGBA8,
        this.accumulation==='float32'?gl.FLOAT:this.accumulation==='float16'?gl.HALF_FLOAT:gl.UNSIGNED_BYTE);
      next.resolved=this._attachment(width,height,gl.RGBA8,gl.UNSIGNED_BYTE);
      this._checkError('render target allocation');
    } catch(error) { this._destroyTarget(next); throw error; }
    this._destroyTarget(previous); return next;
  }
  _draw(target,camera,projection,present) {
    const gl=this.gl,{width,height,source}=target,r=this.model.metadata.render||{},u=this.ewa.uniforms;
    gl.bindFramebuffer(gl.FRAMEBUFFER,target.accum.fbo);
    gl.viewport(0,0,source.width,source.height); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD); gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA,gl.ONE,gl.ONE_MINUS_DST_ALPHA,gl.ONE);
    gl.useProgram(this.ewa.program); gl.bindVertexArray(this.model.vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,this.model.geometry.texture);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D,this.model.sh.texture);
    gl.uniform1i(u.u_geometry,0); gl.uniform1i(u.u_sh,1);
    gl.uniform1i(u.u_geometryWidth,this.model.geometry.width); gl.uniform1i(u.u_shWidth,this.model.sh.width);
    for(const name of ['position','right','down','forward']) gl.uniform3fv(u[`u_${name}`],camera[name]);
    const fy=source.focal;
    gl.uniform2f(u.u_size,source.width,source.height); gl.uniform2f(u.u_focal,fy,fy);
    gl.uniform2fv(u.u_principal,source.principal); gl.uniform2f(u.u_tanFov,source.width/(2*fy),source.height/(2*fy));
    gl.uniform1f(u.u_near,r.near_plane??.01); gl.uniform1f(u.u_lowPass,r.low_pass_variance??.3);
    gl.uniform1f(u.u_minAlpha,r.min_alpha??1/255); gl.uniform1f(u.u_maxAlpha,r.max_alpha??.99);
    gl.uniform1i(u.u_projection,projection); gl.uniform1i(u.u_degree,r.sh_degree??3);
    gl.uniform1i(u.u_antialias,Boolean(r.antialiasing));
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,this.model.count);
    gl.disable(gl.BLEND); gl.bindVertexArray(this.emptyVao);
    gl.bindFramebuffer(gl.FRAMEBUFFER,target.resolved.fbo);
    gl.viewport(0,0,width,height);
    gl.useProgram(this.resolve.program);
    const v=this.resolve.uniforms;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,target.accum.texture);
    gl.uniform1i(v.u_image,0); gl.uniform3fv(v.u_background,r.background??[0,0,0]);
    gl.uniform2f(v.u_size,width,height); gl.uniform1i(v.u_projection,projection);
    gl.uniform2f(v.u_sourceSize,source.width,source.height); gl.uniform2fv(v.u_sourcePrincipal,source.principal);
    gl.uniform1f(v.u_focal,source.focal);
    gl.drawArrays(gl.TRIANGLES,0,3);
    if(present) {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER,target.resolved.fbo);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,null);
      gl.blitFramebuffer(0,0,width,height,0,0,width,height,gl.COLOR_BUFFER_BIT,gl.NEAREST);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.bindVertexArray(null);
    this._checkError('render');
  }
  _fence() {
    const gl=this.gl,sync=gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE,0);
    if(!sync) return Promise.reject(new Error('Could not create a WebGL completion fence.'));
    gl.flush();
    return new Promise((resolve,reject)=>{
      const wait={sync,timer:null,reject};
      this._syncWaits.add(wait);
      const finish=(error)=>{
        if(!this._syncWaits.delete(wait)) return;
        clearTimeout(wait.timer); gl.deleteSync(sync);
        error?reject(error):resolve();
      };
      wait.cancel=()=>finish(abortError());
      const poll=()=>{
        if(this.disposed) { finish(abortError()); return; }
        const status=gl.clientWaitSync(sync,0,0);
        if(status===gl.WAIT_FAILED) finish(new Error('WebGL completion fence failed.'));
        else if(status===gl.ALREADY_SIGNALED||status===gl.CONDITION_SATISFIED) finish();
        else wait.timer=setTimeout(poll,1);
      };
      poll();
    });
  }
  async _frame({width,height,camera,previewCanvas=null,measureGpu=true,projection='fisheye-perspective'}) {
    this._assertAlive();
    if(!this.model) throw new Error('Load a model before rendering.');
    if(this.rendering) throw new Error('Only one WebGL frame may be in flight.');
    if(!['fisheye-perspective','pinhole'].includes(projection)) throw new Error(`Unsupported output projection: ${projection}.`);
    const projectionCode=previewCanvas?1:projection==='pinhole'?0:2;
    const source=renderSource(width,height,camera,projectionCode);
    this.rendering=true;
    const start=performance.now();
    let query=null;
    try {
      const sortMs=await this._sort(camera.position);
      this._assertAlive();
      let target;
      if(previewCanvas) {
        this.previewTarget=this._target(width,height,this.previewTarget,source); target=this.previewTarget;
      } else {
        this.target=this._target(width,height,this.target,source); target=this.target;
        if(this.canvas.width!==width||this.canvas.height!==height) { this.canvas.width=width; this.canvas.height=height; }
        this.width=width; this.height=height;
      }
      const gl=this.gl;
      if(this.timer&&measureGpu) { query=gl.createQuery(); this._queries.add(query); gl.beginQuery(this.timer.TIME_ELAPSED_EXT,query); }
      try { this._draw(target,camera,projectionCode,!previewCanvas); }
      finally { if(query) gl.endQuery(this.timer.TIME_ELAPSED_EXT); }
      await this._fence();
      this._assertAlive();
      let gpuMs=null;
      if(query&&gl.getQueryParameter(query,gl.QUERY_RESULT_AVAILABLE)&&!gl.getParameter(this.timer.GPU_DISJOINT_EXT)) gpuMs=gl.getQueryParameter(query,gl.QUERY_RESULT)/1e6;
      if(previewCanvas) {
        const {rgba}=this._readTarget(target);
        const context=previewCanvas.getContext('2d',{alpha:false});
        if(!context) throw new Error('The preview canvas requires a 2D context.');
        if(previewCanvas.width!==width||previewCanvas.height!==height) { previewCanvas.width=width; previewCanvas.height=height; }
        context.putImageData(new ImageData(new Uint8ClampedArray(rgba.buffer),width,height),0,0);
      }
      return {renderMs:performance.now()-start,sortMs,gpuMs,mode:'webgl2-ewa',accumulation:this.accumulation,
        projection:previewCanvas?'fisheye':projection,sourceWidth:source.width,sourceHeight:source.height,
        sourceFocal:source.focal,sourcePrincipal:[...source.principal]};
    } finally {
      if(query&&this._queries.delete(query)) this.gl.deleteQuery(query);
      this.rendering=false;
    }
  }
  render({width,height,measureGpu=true,projection='fisheye-perspective'}) {
    this._assertAlive();
    if(this.rendering) throw new Error('Only one WebGL frame may be in flight.');
    const camera=copyCamera(this.camera);
    const result=this._frame({width,height,camera,measureGpu,projection});
    this.pendingFrame=result; return result;
  }
  renderPreview(canvas,{width=192,height=192,camera,measureGpu=true}) {
    this._assertAlive();
    if(this.rendering) throw new Error('Only one WebGL frame may be in flight.');
    const result=this._frame({width,height,camera:copyCamera(camera??this.camera),previewCanvas:canvas,measureGpu});
    this.pendingFrame=result; return result;
  }
  _readTarget(target) {
    if(!target) throw new Error('No frame has been rendered for this target.');
    const gl=this.gl,{width,height}=target,raw=new Uint8Array(width*height*4),rgba=new Uint8Array(raw.length);
    gl.bindFramebuffer(gl.FRAMEBUFFER,target.resolved.fbo);
    gl.readPixels(0,0,width,height,gl.RGBA,gl.UNSIGNED_BYTE,raw);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null); this._checkError('frame readback');
    const row=width*4;
    for(let y=0;y<height;y++) rgba.set(raw.subarray((height-1-y)*row,(height-y)*row),y*row);
    return {width,height,rgba};
  }
  async readPixels() { this._assertAlive(); if(this.rendering) throw new Error('Wait for rendering before reading pixels.'); return this._readTarget(this.target); }
  async readPreviewPixels() { this._assertAlive(); if(this.rendering) throw new Error('Wait for rendering before reading pixels.'); return this._readTarget(this.previewTarget); }
  _destroyModel(model) {
    if(!model) return;
    const gl=this.gl;
    if(model.geometry) gl.deleteTexture(model.geometry.texture);
    if(model.sh) gl.deleteTexture(model.sh.texture);
    if(model.indexBuffer) gl.deleteBuffer(model.indexBuffer);
    if(model.vao) gl.deleteVertexArray(model.vao);
  }
  _destroyTarget(target) {
    if(!target) return;
    for(const part of [target.accum,target.resolved]) if(part) { this.gl.deleteFramebuffer(part.fbo); this.gl.deleteTexture(part.texture); }
  }
  async shutdown({timeoutMs=8000}={}) {
    if(this.disposed) return;
    this.closing=true; this._rejectSort(abortError()); this.worker?.terminate();
    let timer;
    try {
      await Promise.race([this.pendingFrame?.catch(error=>{if(error.name!=='AbortError') throw error;}),
        new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('GPU completion timed out.')),timeoutMs);})]);
    } finally { clearTimeout(timer); this.dispose(); }
  }
  dispose(loseContext=true) {
    if(this.disposed) return;
    this.disposed=true; this.closing=true;
    this.canvas.removeEventListener('webglcontextlost',this._lostHandler);
    this._rejectSort(abortError()); this.worker?.terminate(); this.worker=null;
    for(const wait of [...this._syncWaits]) wait.cancel();
    for(const query of this._queries) this.gl.deleteQuery(query);
    this._queries.clear();
    this._destroyModel(this.model); this.model=null;
    this._destroyTarget(this.target); this.target=null;
    this._destroyTarget(this.previewTarget); this.previewTarget=null;
    if(this.ewa) this.gl.deleteProgram(this.ewa.program);
    if(this.resolve) this.gl.deleteProgram(this.resolve.program);
    if(this.emptyVao) this.gl.deleteVertexArray(this.emptyVao);
    this.camera=null; this.sortedPosition=null;
    this._resolveLost({reason:'destroyed',message:'Renderer disposed.'});
    if(loseContext&&!this.gl.isContextLost()) this.gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
