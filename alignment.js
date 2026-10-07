// Calibration stays in the model's original coordinate system. Only the
// viewing camera and navigation up axis change; Gaussian/SH/BVH stay intact.
export const ALIGNMENT_SCHEMA = 'fullcircle-viewer-alignment-v1';
const dot = (a,b) => a.reduce((sum,value,i) => sum+value*b[i],0);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const vector = (value,label) => {
  if (!Array.isArray(value) || value.length !== 3 || !value.every(Number.isFinite)) throw new Error(`${label} 必须是三个有限数值`);
  return [...value];
};
const normalize = (value) => {
  const length = Math.hypot(...value);
  if (!(length > 1e-12)) throw new Error('校正方向不能为零');
  return value.map(v=>v/length);
};
function rotate(value,axis,angle) {
  const c=Math.cos(angle),s=Math.sin(angle),axv=cross(axis,value),av=dot(axis,value);
  return value.map((v,i)=>v*c+axv[i]*s+axis[i]*av*(1-c));
}

export function upFromAngles(manifest,roll,pitch) {
  if (!Number.isFinite(roll) || Math.abs(roll)>180 || !Number.isFinite(pitch) || Math.abs(pitch)>89) throw new Error('左右校正范围为 ±180°，前后校正范围为 ±89°');
  const rotation=manifest.cameras?.[0]?.rotation;
  if (!Array.isArray(rotation) || rotation.length!==3) throw new Error('模型缺少校正参考相机');
  rotation.forEach(row=>vector(row,'参考相机旋转'));
  const right=normalize(rotation.map(row=>row[0]));
  const up=normalize(rotation.map(row=>-row[1]));
  const forward=normalize(rotation.map(row=>row[2]));
  return normalize(rotate(rotate(up,forward,roll*Math.PI/180),right,pitch*Math.PI/180));
}

export function createAlignment(manifest,modelId,roll=0,pitch=0,enabled=false) {
  const value={
    schema:ALIGNMENT_SCHEMA, model_id:modelId, source:manifest.source,
    gaussian_count:manifest.gaussian_count,
    model_bounds:{min:[...manifest.bounds.min],max:[...manifest.bounds.max]},
    enabled, reference_camera:0, angles_deg:{roll,pitch},
    up:upFromAngles(manifest,roll,pitch),
  };
  return validateAlignment(value,manifest);
}

export function validateAlignment(value,manifest) {
  if (!value || value.schema!==ALIGNMENT_SCHEMA) throw new Error('不支持此校正文件格式');
  if (typeof value.model_id!=='string' || !value.model_id.trim() || value.model_id.length>200) throw new Error('校正文件缺少模型标识');
  if (typeof value.source!=='string' || value.source!==manifest.source
      || !Number.isSafeInteger(value.gaussian_count) || value.gaussian_count!==manifest.gaussian_count) throw new Error('校正文件与当前模型不匹配');
  if (typeof value.enabled!=='boolean' || value.reference_camera!==0) throw new Error('校正文件的开关或参考相机无效');
  for (const key of ['min','max']) {
    const saved=vector(value.model_bounds?.[key],'校正模型范围');
    const actual=vector(manifest.bounds?.[key],'模型范围');
    if (saved.some((v,i)=>Math.abs(v-actual[i])>1e-6*Math.max(1,Math.abs(actual[i])))) throw new Error('校正文件的模型坐标范围不匹配');
  }
  const up=vector(value.up,'竖直方向');
  if (Math.abs(Math.hypot(...up)-1)>1e-4) throw new Error('校正竖直方向必须为单位向量');
  const expected=upFromAngles(manifest,value.angles_deg?.roll,value.angles_deg?.pitch);
  if (up.some((v,i)=>Math.abs(v-expected[i])>1e-4)) throw new Error('校正方向与角度不一致');
  return {...value, up:[...up],angles_deg:{...value.angles_deg},model_bounds:{min:[...value.model_bounds.min],max:[...value.model_bounds.max]}};
}

export function levelCamera(camera,sceneUp) {
  const forward=normalize(vector(camera.forward,'相机前方向'));
  let right=cross(forward,normalize(vector(sceneUp,'场景竖直方向')));
  if (Math.hypot(...right)<1e-6) {
    // Looking straight up/down does not determine a unique roll. Preserve
    // the previous heading in its perpendicular plane instead of producing NaN.
    const old=vector(camera.right,'相机右方向');
    right=old.map((v,i)=>v-dot(old,forward)*forward[i]);
    if (Math.hypot(...right)<1e-6) right=cross(forward,Math.abs(forward[0])<0.9?[1,0,0]:[0,0,1]);
  }
  right=normalize(right);
  return {...camera,position:[...camera.position],forward:[...camera.forward],right,down:normalize(cross(forward,right))};
}

export async function loadAlignment(manifestUrl,manifest,signal) {
  try {
    const response=await fetch(new URL('alignment.json',manifestUrl),{signal,credentials:'omit',cache:'no-store'});
    if (response.status===404) return {alignment:null,state:'absent',message:'未发现校正文件，使用原始相机'};
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const alignment=validateAlignment(await response.json(),manifest);
    return {alignment,state:'loaded',message:alignment.enabled?'已读取模型目录中的校正文件':'已读取校正文件：校正已关闭'};
  } catch (error) {
    if (signal?.aborted || error.name==='AbortError') throw error;
    return {alignment:null,state:'warning',message:`校正文件无法应用，已跳过：${error.message}`};
  }
}
