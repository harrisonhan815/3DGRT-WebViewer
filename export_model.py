#!/usr/bin/env python3
"""Export a trusted local, radial-sorted 2D-EWA 3DGS checkpoint for WebGL2.

Only CPU PyTorch, NumPy and OmegaConf are needed. No CUDA/model import occurs.
Every Gaussian is retained in checkpoint order. Geometry is FP32; the default
SH float16 is an explicit storage quantization, followed by lossless transport.

Geometry: 48 little-endian bytes per Gaussian, three float32 vec4s:
  [center.xyz, sigmoid opacity], [cov_xx,cov_xy,cov_xz,cov_yy],
  [cov_yz,cov_zz,0,0], with covariance R*diag(exp(scale)^2)*R^T.
SH: 48 RGB-interleaved values, DC first; 96 bytes float16 or 192 bytes float32.
No BVH, ray-tracing kernel conversion, Gaussian pruning or reordering is used.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import zlib

import numpy as np

SCHEMA = "fullcircle-webgl2-ewa-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
COMPRESSION_MODES = ("none", "gzip", "gzip-shuffle")
MAX_PART_BYTES = 100 * 1024 * 1024
SH_DTYPES = {"float32": np.dtype("<f4"), "float16": np.dtype("<f2")}

def _register_resolvers():
    from omegaconf import OmegaConf

    for name, callback in {
        "div": lambda a, b: a / b,
        "eq": lambda a, b: a == b,
        "int_list": lambda values: [int(x) for x in values],
    }.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, callback)


def _config_value(config, key, default=None):
    from omegaconf import OmegaConf

    if OmegaConf.is_config(config):
        return OmegaConf.select(config, key, default=default)
    value = config
    for part in key.split("."):
        if isinstance(value, dict):
            value = value.get(part, default)
        else:
            value = getattr(value, part, default)
        if value is default:
            return default
    return value


def load_checkpoint(path):
    import torch

    _register_resolvers()
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False, mmap=True)
    required = {
        "config", "positions", "rotation", "scale", "density",
        "features_albedo", "features_specular", "n_active_features", "max_n_features",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Expected a FullCircle .pt model checkpoint with Gaussian parameters and config")
    return checkpoint


def _cpu_float32(checkpoint, name, shape):
    import torch

    value = checkpoint[name]
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError(f"{name} must be a CPU float32 tensor; load the checkpoint with map_location='cpu'")
    if tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return value.detach()


def convert_sh_bytes(data, source_dtype, target_dtype):
    """Convert only the SH storage values, rejecting non-finite/overflow data."""
    if source_dtype not in SH_DTYPES or target_dtype not in SH_DTYPES:
        raise ValueError("SH dtype must be float16 or float32")
    if len(data) % (48 * SH_DTYPES[source_dtype].itemsize):
        raise ValueError("SH chunks must contain 48 coefficients per particle")
    values = np.frombuffer(data, dtype=SH_DTYPES[source_dtype])
    if not np.isfinite(values).all():
        raise ValueError("SH coefficients must be finite")
    if source_dtype == target_dtype:
        return bytes(data)
    if target_dtype == "float16" and np.any(np.abs(values) > np.finfo(np.float16).max):
        raise ValueError("SH coefficients exceed the finite float16 range")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = values.astype(SH_DTYPES[target_dtype])
    if not np.isfinite(converted).all():
        raise ValueError("SH conversion overflowed its finite target dtype")
    return converted.tobytes()


def _shuffle_bytes(data, width, inverse=False):
    if width not in (2, 4):
        raise ValueError("Byte shuffle width must be 2 or 4")
    if len(data) % width:
        raise ValueError(f"byte-shuffle-{width} requires a byte length divisible by {width}")
    array = np.frombuffer(data, dtype=np.uint8)
    return (array.reshape(width, -1).T if inverse else array.reshape(-1, width).T).copy().tobytes()


def shuffle_bytes2(data, inverse=False):
    """Transpose the two byte lanes of packed IEEE binary16 values losslessly."""
    return _shuffle_bytes(data, 2, inverse)


def shuffle_bytes4(data, inverse=False):
    """Transpose byte lanes without interpreting float/int values or rounding.

    Four-byte words [a0 a1 a2 a3][b0 b1 b2 b3] become
    [a0 b0][a1 b1][a2 b2][a3 b3]. Each file part is transformed independently.
    This operates on bytes only and preserves every float32 bit pattern.
    """
    return _shuffle_bytes(data, 4, inverse)


def encode_binary_part(data, compression="none", *, shuffle_bytes=4):
    """Return encoded bytes and additive V1 manifest transport metadata."""
    if compression not in COMPRESSION_MODES:
        raise ValueError(f"Unsupported compression: {compression}")
    if shuffle_bytes not in (2, 4):
        raise ValueError("Byte shuffle width must be 2 or 4")
    if not 0 < len(data) < MAX_PART_BYTES:
        raise ValueError("Decoded parts must be nonempty and below 100 MiB")
    raw = bytes(data)
    metadata = {"byteLength": len(raw), "downloadByteLength": len(raw),
                "compression": "none", "transform": "none",
                "sha256": hashlib.sha256(raw).hexdigest()}
    encoded = raw
    if compression != "none":
        transformed = _shuffle_bytes(raw, shuffle_bytes) if compression == "gzip-shuffle" else raw
        candidate = gzip.compress(transformed, compresslevel=6, mtime=0)
        if len(candidate) < len(raw):
            encoded = candidate
            metadata.update(compression="gzip", transform=f"byte-shuffle-{shuffle_bytes}" if compression == "gzip-shuffle" else "none",
                            downloadByteLength=len(encoded))
        else:
            # Preserve the raw filename and actual encoding explicitly. A
            # compressed chunk must not exceed the original/Git upload budget.
            metadata["compressionFallback"] = "not-smaller"
    return encoded, metadata


def decode_binary_part(encoded, part):
    """Read legacy raw or new compressed parts, with bounded decompression."""
    expected = part.get("byteLength")
    if type(expected) is not int or not 0 < expected < MAX_PART_BYTES:
        raise ValueError("Part byteLength must be positive and below 100 MiB")
    compression = part.get("compression", "none")
    transform = part.get("transform", "none")
    downloaded = part.get("downloadByteLength", expected)
    if type(downloaded) is not int or downloaded != len(encoded) or not 0 < downloaded < MAX_PART_BYTES:
        raise ValueError("Part downloadByteLength does not match the encoded file")
    if compression == "none":
        if transform != "none":
            raise ValueError("Raw parts cannot declare a byte transform")
        raw = bytes(encoded)
    elif compression == "gzip":
        if transform not in ("none", "byte-shuffle-2", "byte-shuffle-4"):
            raise ValueError(f"Unsupported byte transform: {transform}")
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        raw = decoder.decompress(encoded, expected + 1)
        if len(raw) > expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Invalid, oversized, truncated or multi-member gzip part")
        if transform != "none":
            raw = _shuffle_bytes(raw, 2 if transform == "byte-shuffle-2" else 4, inverse=True)
    else:
        raise ValueError(f"Unsupported compression: {compression}")
    if len(raw) != expected:
        raise ValueError("Decoded part does not match byteLength")
    if "sha256" in part and part["sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Decoded part SHA256 does not match the manifest")
    return raw


def write_binary_part(directory, name, number, data, compression="none", *, shuffle_bytes=4):
    if not name or Path(name).name != name or any(char in name for char in "\\/%?#"):
        raise ValueError("Buffer name must be a simple filename component")
    encoded, part = encode_binary_part(data, compression, shuffle_bytes=shuffle_bytes)
    suffix = ".shuf.gz" if part["transform"] in ("byte-shuffle-2", "byte-shuffle-4") else ".gz" if part["compression"] == "gzip" else ""
    path = f"{name}.{number:03d}.bin{suffix}"
    (Path(directory) / path).write_bytes(encoded)
    return {"path": path, **part}


def write_parts(directory, name, array, stride, max_bytes, compression="none", *, shuffle_bytes=4):
    """Write record-aligned parts; byteLength always means decoded GPU bytes."""
    if not stride <= max_bytes < 100 * 1024 * 1024:
        raise ValueError("Chunk budget must hold a record and remain below 100 MiB")
    records_per_part = max_bytes // stride
    array = np.ascontiguousarray(array)
    if array.nbytes != len(array) * stride:
        raise ValueError("Array does not match its declared record stride")
    parts = []
    for number, start in enumerate(range(0, len(array), records_per_part)):
        end = min(start + records_per_part, len(array))
        parts.append(write_binary_part(directory, name, number, memoryview(array[start:end]).cast("B"), compression, shuffle_bytes=shuffle_bytes))
    return {"byteLength": array.nbytes, "downloadByteLength": sum(part["downloadByteLength"] for part in parts),
            "stride": stride, "parts": parts}


def set_sh_precision_metadata(manifest, target_dtype, source_spec=None):
    source_spec = source_spec or {}
    spec = manifest["files"]["sh"]
    spec["dtype"] = target_dtype
    rounded = (target_dtype == "float16" or source_spec.get("dtype") == "float16"
               or source_spec.get("quantization_dtype") == "float16")
    if rounded:
        spec["quantization_dtype"] = "float16"
    description = ("float16 quantized" if target_dtype == "float16" else
                   "float32 expanded from float16; earlier rounding retained" if rounded else "float32")
    manifest["precision"] = f"FP32 centers, opacity and covariance; SH {description}; no pruning or reordering"


def validate_config(checkpoint):
    """Require actual quadratic raster training; never relabel a 3DGRT model."""
    _register_resolvers()
    conf = checkpoint["config"]
    for key, expected in {
        "render.method": "3dgs", "render.depth_sort": "radial",
        "render.particle_kernel_degree": 2, "model.density_activation": "sigmoid",
        "model.scale_activation": "exp",
    }.items():
        actual = _config_value(conf, key)
        if actual != expected:
            raise ValueError(f"Unsupported {key}={actual!r}; expected {expected!r}")
    if _config_value(conf, "post_processing.method") is not None:
        raise ValueError("Checkpoint image post-processing cannot be discarded")
    if _config_value(conf, "model.progressive_training.feature_type", "sh") != "sh":
        raise ValueError("Only SH radiance is supported")
    active, maximum = checkpoint["n_active_features"], checkpoint["max_n_features"]
    if (isinstance(active, bool) or isinstance(maximum, bool) or int(active) != active
            or int(maximum) != maximum or not 0 <= int(active) <= int(maximum) <= 3):
        raise ValueError("Expected integer SH degrees 0 <= active <= maximum <= 3")
    if int(_config_value(conf, "render.particle_radiance_sph_degree", 3)) < maximum:
        raise ValueError("Checkpoint SH degree exceeds the configured renderer")
    antialiasing = _config_value(conf, "render.antialiasing", False)
    if antialiasing is not False:
        raise ValueError("This WebGL2 exporter currently requires render.antialiasing=false")
    near = float(_config_value(conf, "render.near_plane", .01))
    if not math.isfinite(near) or near <= 0:
        raise ValueError("The native near plane must be finite and positive")
    name = _config_value(conf, "model.background.name", "background-color")
    color = _config_value(conf, "model.background.color", "black")
    if name == "skip-background":
        background = [0., 0., 0.]
    elif name == "background-color" and color in ("black", "white"):
        background = [float(color == "white")] * 3
    else:
        raise ValueError("Only black, white or skip backgrounds can be preserved")
    return {"method": "3dgs", "pipeline": "2d-ewa", "kernel_degree": 2,
            "depth_sort": "radial", "sh_degree": int(active), "sh_direction": "camera_to_gaussian",
            "min_alpha": 1 / 255, "max_alpha": .99, "min_transmittance": .0001,
            "low_pass_variance": .3, "antialiasing": False, "near_plane": near,
            "background": background}


def prepare_gaussians(checkpoint, render=None):
    """Activate and precompute native world covariance using FP32 CPU math."""
    import torch

    if render is None:
        render = validate_config(checkpoint)
    positions = checkpoint["positions"]
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be shaped [N,3]")
    count = len(positions)
    if not 0 < count < 0x80000000:
        raise ValueError("Expected 1..2^31-1 Gaussian records")
    positions = _cpu_float32(checkpoint, "positions", (count, 3)).numpy()
    scales = torch.exp(_cpu_float32(checkpoint, "scale", (count, 3))).numpy()
    opacity = torch.sigmoid(_cpu_float32(checkpoint, "density", (count, 1))).numpy()
    quaternion = torch.nn.functional.normalize(_cpu_float32(checkpoint, "rotation", (count, 4)), dim=1).numpy()
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Activated scales must remain positive and finite")
    w, x, y, z = quaternion.T
    rotation = np.empty((count, 3, 3), dtype=np.float32)
    rotation[:, 0, 0], rotation[:, 0, 1], rotation[:, 0, 2] = 1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)
    rotation[:, 1, 0], rotation[:, 1, 1], rotation[:, 1, 2] = 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)
    rotation[:, 2, 0], rotation[:, 2, 1], rotation[:, 2, 2] = 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)
    with np.errstate(over="ignore", invalid="ignore"):
        factor = rotation * scales[:, None, :]
        covariance = factor @ factor.transpose(0, 2, 1)
    if not np.isfinite(covariance).all():
        raise ValueError("Gaussian covariance overflows FP32")
    geometry = np.zeros((count, 12), dtype="<f4")
    geometry[:, :3], geometry[:, 3] = positions, opacity[:, 0]
    geometry[:, 4:10] = covariance[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    coefficients = (int(checkpoint["max_n_features"]) + 1) ** 2
    dc = _cpu_float32(checkpoint, "features_albedo", (count, 3)).numpy()
    rest = _cpu_float32(checkpoint, "features_specular", (count, (coefficients - 1) * 3)).numpy()
    sh = np.zeros((count, 16, 3), dtype="<f4")
    sh[:, 0], sh[:, 1:coefficients] = dc, rest.reshape(count, coefficients - 1, 3)
    return geometry, sh.reshape(count, 48)


def _camera(name, position, rotation, fov_y, aspect=4 / 3):
    position, rotation = np.asarray(position, np.float64), np.asarray(rotation, np.float64)
    if (position.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(position).all()
            or not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError(f"Invalid camera pose: {name}")
    if not math.isfinite(fov_y) or not 0 < fov_y < math.pi or not math.isfinite(aspect) or aspect <= 0:
        raise ValueError(f"Invalid camera projection: {name}")
    return {"name": str(name), "position": position.tolist(), "rotation": rotation.tolist(),
            "fov_y": float(fov_y), "aspect": float(aspect)}


def _colmap_camera(name, quaternion, translation, fov_y):
    q = np.asarray(quaternion, np.float64)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid COLMAP quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    return _camera(name, -rotation.T @ np.asarray(translation), rotation.T, fov_y)


def export_cameras(checkpoint, positions, camera_json=None, data_path=None, fov_y=math.radians(70)):
    """Self-contained CPU COLMAP extrinsics reader; no V2/WebGPU dependency."""
    if camera_json:
        entries = json.loads(Path(camera_json).read_text())
        if not isinstance(entries, list) or not entries:
            raise ValueError("Camera JSON must contain a nonempty list")
        cameras = []
        for index, entry in enumerate(entries):
            if "fov_y" in entry:
                fov, aspect = float(entry["fov_y"]), float(entry.get("aspect", 4/3))
            else:
                width, height, fy = (float(entry[key]) for key in ("width", "height", "fy"))
                if not all(math.isfinite(v) and v > 0 for v in (width, height, fy)):
                    raise ValueError("Camera width/height/fy must be positive and finite")
                fov, aspect = 2 * math.atan(height/(2*fy)), width/height
            cameras.append(_camera(entry.get("name", entry.get("img_name", f"camera_{index}")),
                                   entry["position"], entry["rotation"], fov, aspect))
        return cameras, "camera JSON"
    path = Path(data_path or str(_config_value(checkpoint["config"], "path", ""))).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd()/path, REPO_ROOT/path]
    suffix = str(_config_value(checkpoint["config"], "dataset.test_frame_suffix", "_test"))
    for candidate in candidates:
        sparse = candidate / "sparse" / "0"
        binary, text = sparse / "images.bin", sparse / "images.txt"
        cameras = []
        if binary.is_file():
            with binary.open("rb") as stream:
                count = struct.unpack("<Q", stream.read(8))[0]
                size = binary.stat().st_size
                for _ in range(count):
                    record = struct.unpack("<idddddddi", stream.read(64))
                    name = bytearray()
                    while True:
                        char = stream.read(1)
                        if char == b"\0":
                            break
                        if not char:
                            raise ValueError("Truncated COLMAP image name")
                        name.extend(char)
                    points = struct.unpack("<Q", stream.read(8))[0]
                    if points > (size - stream.tell()) // 24:
                        raise ValueError("Truncated COLMAP observations")
                    stream.seek(points * 24, 1)
                    name = name.decode("utf-8")
                    if not suffix or not Path(name).stem.endswith(suffix):
                        cameras.append(_colmap_camera(name, record[1:5], record[5:8], fov_y))
        elif text.is_file():
            with text.open() as stream:
                for line in stream:
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    values = line.split()
                    if len(values) < 10:
                        raise ValueError("Invalid COLMAP camera record")
                    name = " ".join(values[9:])
                    if not suffix or not Path(name).stem.endswith(suffix):
                        cameras.append(_colmap_camera(name, list(map(float, values[1:5])),
                                                       list(map(float, values[5:8])), fov_y))
                    if next(stream, None) is None:
                        raise ValueError("Missing COLMAP observations line")
        if cameras:
            return sorted(cameras, key=lambda c: (Path(c["name"]).name, c["name"])), "COLMAP poses; virtual pinhole preview"
    lower, upper = np.quantile(positions, [.05, .95], axis=0)
    center, radius = (lower+upper)/2, max(float(np.linalg.norm(upper-lower))/2, .1)
    position = center - [0, 0, radius/math.tan(fov_y/2)]
    return [_camera("Scene overview", position, np.eye(3), fov_y)], "robust center bounds fallback"


def export_checkpoint(checkpoint, output, *, source_name="checkpoint.pt", chunk_bytes=64*1024*1024,
                      camera_json=None, data_path=None, fov_y=math.radians(70),
                      overwrite=False, compression="gzip-shuffle", sh_dtype="float16"):
    output = Path(output).expanduser()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or not overwrite)):
        raise FileExistsError(f"Output exists: {output}; choose a new directory or explicitly --overwrite")
    if not math.isfinite(fov_y) or not 0 < fov_y < math.pi:
        raise ValueError("Preview vertical field of view must be between 0 and pi")
    if compression not in COMPRESSION_MODES or sh_dtype not in SH_DTYPES:
        raise ValueError("Invalid compression or SH dtype")
    render = validate_config(checkpoint)
    geometry, sh = prepare_gaussians(checkpoint, render)
    cameras, camera_source = export_cameras(checkpoint, geometry[:, :3], camera_json, data_path, fov_y)
    if sh_dtype != "float32":
        sh = np.frombuffer(convert_sh_bytes(sh.tobytes(), "float32", sh_dtype), dtype=SH_DTYPES[sh_dtype]).reshape(-1, 48)
    positions = geometry[:, :3]
    navigation = np.quantile(positions, [.05, .95], axis=0)
    manifest = {
        "schema": SCHEMA, "source": Path(source_name).name, "gaussian_count": len(geometry),
        "bounds": {"min": positions.min(0).tolist(), "max": positions.max(0).tolist()},
        "bounds_definition": "complete Gaussian center bounds; Gaussian support is not finite",
        "navigation_bounds": {"min": navigation[0].tolist(), "max": navigation[1].tolist()},
        "render": render, "cameras": cameras, "default_camera": 0, "camera_source": camera_source,
        "byte_order": "little-endian", "gaussian_order": "checkpoint order; no pruning",
        "native_support": {"covariance": "R diag(exp(scale)^2) R^T in original model coordinates",
                           "projection": "pinhole Jacobian with native 1.3*tan(FOV) clamp and 0.3 pixel variance",
                           "color": "camera-to-Gaussian normalized direction SH; +0.5 then lower clamp at zero",
                           "ordering": "radial distance; preserve checkpoint index for equal keys"},
        "known_differences": [
            "CPU FP32 activation/covariance and browser math can round differently from CUDA.",
            "WebGL fixed-function blending does not implement CUDA's per-pixel early transmittance termination; the crossing contribution can differ.",
            "Framebuffer accumulation uses the best available float32/float16/unorm8 format; unorm8 is a lower-precision fallback.",
            "World-space radial sort keys can differ from CUDA camera-space keys at near-equal float32 depths.",
            "Float16 SH is optional quantization; geometry remains FP32 and no Gaussians are pruned.",
        ],
    }
    step = checkpoint.get("global_step")
    if step is not None:
        manifest["training_step"] = int(step)
    output.parent.mkdir(parents=True, exist_ok=True)
    old_parts = set()
    if overwrite and (output / "manifest.json").is_file():
        previous = json.loads((output / "manifest.json").read_text())
        if previous.get("schema") != SCHEMA:
            raise ValueError("Refusing to overwrite a model with a different schema")
        for spec in previous.get("files", {}).values():
            for part in spec.get("parts", []):
                name = part.get("path", "")
                if name and Path(name).name == name and name not in ("alignment.json", "cameras.json", "manifest.json"):
                    old_parts.add(name)
    with tempfile.TemporaryDirectory(prefix=".webgl2-export-", dir=output.parent) as temporary:
        stage = Path(temporary)
        manifest["files"] = {
            "geometry": write_parts(stage, "geometry", geometry, 48, chunk_bytes, compression),
            "sh": write_parts(stage, "sh", sh, 48*SH_DTYPES[sh_dtype].itemsize, chunk_bytes, compression,
                              shuffle_bytes=SH_DTYPES[sh_dtype].itemsize),
        }
        set_sh_precision_metadata(manifest, sh_dtype)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
        # Existing sidecars (including manual alignment) are untouched.
        output.mkdir(exist_ok=overwrite)
        new_parts = {p["path"] for spec in manifest["files"].values() for p in spec["parts"]}
        for name in sorted(new_parts):
            os.replace(stage/name, output/name)
        for name in old_parts-new_parts:
            (output/name).unlink(missing_ok=True)
        os.replace(stage/"manifest.json", output/"manifest.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--fov-deg", type=float, default=70)
    parser.add_argument("--chunk-mib", type=float, default=64)
    parser.add_argument("--compression", choices=COMPRESSION_MODES, default="gzip-shuffle")
    parser.add_argument("--sh-dtype", choices=tuple(SH_DTYPES), default="float16",
                        help="float16 quantizes SH only; float32 keeps full SH precision")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.chunk_mib < 100:
        parser.error("chunk-mib must be positive and less than 100")
    manifest = export_checkpoint(load_checkpoint(args.checkpoint), args.out_dir, source_name=args.checkpoint.name,
                                 chunk_bytes=int(args.chunk_mib*1024*1024), camera_json=args.camera_json,
                                 data_path=args.data_path, fov_y=math.radians(args.fov_deg), overwrite=args.overwrite,
                                 compression=args.compression, sh_dtype=args.sh_dtype)
    decoded = sum(spec["byteLength"] for spec in manifest["files"].values())
    stored = sum(spec["downloadByteLength"] for spec in manifest["files"].values())
    print(json.dumps({"manifest": str(args.out_dir/"manifest.json"), "gaussian_count": manifest["gaussian_count"],
                      "sh_dtype": manifest["files"]["sh"]["dtype"], "decoded_bytes": decoded,
                      "stored_bytes": stored, "compression_saved_percent": 100*(1-stored/decoded)}, indent=2))


if __name__ == "__main__":
    main()
