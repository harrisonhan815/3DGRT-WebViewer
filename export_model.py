#!/usr/bin/env python3
"""Export a trusted local FullCircle .pt checkpoint for static WebGPU inference.

Only CPU PyTorch, NumPy and OmegaConf are required; this module never imports
the training model, CUDA extensions, OptiX, or a web server. Checkpoints use
Python pickle and must be trusted local files. Exporting does not prune,
downsample or modify the source checkpoint. SH defaults to float32 and can
explicitly be quantized to float16; geometry and BVH retain their original bits.

Geometry: 64 bytes/particle, four little-endian float32 vec4 records:
  [center.xyz, activated_density], [world_to_local_row0.xyz, kernel_radius],
  [world_to_local_row1.xyz, 0], [world_to_local_row2.xyz, 0].
SH: 192 bytes/particle (float32) or 96 bytes/particle (float16),
  16 RGB coefficients with DC first, in the same order for both types.
BVH: 32 bytes/node, [min.xyz, left:u32], [max.xyz, right_or_count:u32].
  Internal nodes hold two node indices. Leaves set count's high bit and use
  left as the first particle index. All particle buffers follow leaf order.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import zlib

import numpy as np


SCHEMA = "fullcircle-webgpu-reference-v1"
LEAF_BIT = np.uint32(0x80000000)
BVH_DTYPE = np.dtype([
    ("minimum", "<f4", 3), ("left", "<u4"),
    ("maximum", "<f4", 3), ("right_or_count", "<u4"),
])
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def validate_config(checkpoint):
    """Reject variants whose native forward semantics this format cannot encode."""
    _register_resolvers()
    conf = checkpoint["config"]
    expected = {
        "render.method": "3dgrt",
        "render.pipeline_type": "reference",
        "render.primitive_type": "instances",
        "render.particle_kernel_degree": 4,
        "model.density_activation": "sigmoid",
        "model.scale_activation": "exp",
    }
    for key, expected_value in expected.items():
        actual = _config_value(conf, key)
        if actual != expected_value:
            raise ValueError(f"Unsupported {key}={actual!r}; expected {expected_value!r}")
    if _config_value(conf, "post_processing.method") is not None:
        raise ValueError("Checkpoint post-processing is not supported by this exporter")
    if _config_value(conf, "model.progressive_training.feature_type", "sh") != "sh":
        raise ValueError("Only spherical-harmonic radiance is supported")
    maximum = int(checkpoint["max_n_features"])
    active = int(checkpoint["n_active_features"])
    if not 0 <= active <= maximum <= 3:
        raise ValueError("Expected SH degrees 0 <= active <= maximum <= 3")
    if _config_value(conf, "render.particle_radiance_sph_degree", 3) != 3:
        raise ValueError("This exporter requires the native 16-coefficient SH pipeline")

    # referenceOptix hard-codes these in optixTracer.cpp/processHit(). Do not
    # export changed config values that would misleadingly imply they took effect.
    alpha_min = float(_config_value(conf, "render.particle_kernel_min_alpha", 1 / 255))
    alpha_max = float(_config_value(conf, "render.particle_kernel_max_alpha", 0.99))
    if not math.isclose(alpha_min, 1 / 255, rel_tol=1e-7) or not math.isclose(alpha_max, 0.99, rel_tol=1e-7):
        raise ValueError("reference pipeline uses fixed min alpha 1/255 and max alpha 0.99")
    response = float(_config_value(conf, "render.particle_kernel_min_response", 0.0113))
    transmittance = float(_config_value(conf, "render.min_transmittance", 0.001))
    if not 0 < response < 1 or not 0 < transmittance < 1:
        raise ValueError("Response and transmittance thresholds must be finite and between 0 and 1")
    clamped = _config_value(conf, "render.particle_kernel_density_clamping", True)
    if not isinstance(clamped, bool):
        raise ValueError("particle_kernel_density_clamping must be a boolean")
    background_name = _config_value(conf, "model.background.name", "background-color")
    background_color = _config_value(conf, "model.background.color", "black")
    if background_name == "skip-background":
        background = [0.0, 0.0, 0.0]
    elif background_name == "background-color" and background_color in ("black", "white", "random"):
        # Native random background is black during inference.
        background = [1.0, 1.0, 1.0] if background_color == "white" else [0.0, 0.0, 0.0]
    else:
        raise ValueError(f"Unsupported background {background_name!r}/{background_color!r}")
    return {
        "pipeline": "reference", "primitive_type": "instances", "kernel_degree": 4,
        "density_clamping": clamped, "min_response": response,
        "min_alpha": 1 / 255, "max_alpha": 0.99,
        "min_transmittance": transmittance, "sh_degree": active,
        "sh_direction": "world_ray", "background": background,
    }


def _cpu_float32(checkpoint, name, shape):
    import torch

    value = checkpoint[name]
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError(f"{name} must be a CPU float32 tensor; load the checkpoint with map_location='cpu'")
    if tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return value.detach()


def prepare_gaussians(checkpoint, render):
    """Activate parameters on CPU, preserving all particles and FP32 SH data."""
    import torch

    positions = checkpoint["positions"]
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    count = positions.shape[0]
    if not 0 < count < 0x80000000:
        raise ValueError("Expected a nonempty model with fewer than 2^31 particles")
    positions = _cpu_float32(checkpoint, "positions", (count, 3)).numpy()
    raw_scale = _cpu_float32(checkpoint, "scale", (count, 3))
    raw_density = _cpu_float32(checkpoint, "density", (count, 1))
    raw_rotation = _cpu_float32(checkpoint, "rotation", (count, 4))
    scales = torch.exp(raw_scale).numpy()
    density = torch.sigmoid(raw_density).numpy()[:, 0]
    # Native get_rotation() uses torch.nn.functional.normalize (eps=1e-12).
    quaternions = torch.nn.functional.normalize(raw_rotation, dim=1).numpy()
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Activated scales must be positive and finite")
    w, x, y, z = quaternions.T
    rotation = np.empty((count, 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = 1 - 2 * (y*y + z*z)
    rotation[:, 0, 1] = 2 * (x*y - w*z)
    rotation[:, 0, 2] = 2 * (x*z + w*y)
    rotation[:, 1, 0] = 2 * (x*y + w*z)
    rotation[:, 1, 1] = 1 - 2 * (x*x + z*z)
    rotation[:, 1, 2] = 2 * (y*z - w*x)
    rotation[:, 2, 0] = 2 * (x*z - w*y)
    rotation[:, 2, 1] = 2 * (y*z + w*x)
    rotation[:, 2, 2] = 1 - 2 * (x*x + y*y)
    local_matrix = rotation.transpose(0, 2, 1) / scales[:, :, None]
    if not np.isfinite(local_matrix).all():
        raise ValueError("World-to-local matrices overflow float32")
    modulation = density if render["density_clamping"] else np.ones(count, dtype=np.float32)
    with np.errstate(divide="ignore"):
        threshold = np.minimum(np.float32(render["min_response"]) / modulation, np.float32(0.97))
    radius = np.power(np.log(threshold) / np.float32(-4.5 / 81), np.float32(0.25))
    geometry = np.zeros((count, 16), dtype="<f4")
    geometry[:, :3], geometry[:, 3] = positions, density
    for row in range(3):
        geometry[:, 4 + row * 4:7 + row * 4] = local_matrix[:, row]
    geometry[:, 7] = radius

    # Use the exact packed matrix's inverse for conservative software-BVH
    # bounds. Its tiny FP32 difference from R*S is recorded in the manifest.
    # Compute in chunks to keep export memory bounded for larger models.
    minimum, maximum = np.empty_like(positions), np.empty_like(positions)
    for start in range(0, count, 65536):
        end = min(start + 65536, count)
        inverse = np.linalg.inv(local_matrix[start:end].astype(np.float64))
        extent = np.sum(np.abs(inverse), axis=2) * radius[start:end, None]
        minimum[start:end] = np.nextafter((positions[start:end] - extent).astype(np.float32), -np.inf)
        maximum[start:end] = np.nextafter((positions[start:end] + extent).astype(np.float32), np.inf)
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("Particle bounds overflow float32")

    maximum_degree = int(checkpoint["max_n_features"])
    coefficient_count = (maximum_degree + 1) ** 2
    dc = _cpu_float32(checkpoint, "features_albedo", (count, 3)).numpy()
    rest = _cpu_float32(checkpoint, "features_specular", (count, (coefficient_count - 1) * 3)).numpy()
    sh = np.zeros((count, 16, 3), dtype="<f4")
    sh[:, 0] = dc
    sh[:, 1:coefficient_count] = rest.reshape(count, coefficient_count - 1, 3)
    return geometry, sh.reshape(count, 48), minimum, maximum


def _surface_area(minimum, maximum):
    """Evaluate areas in float64, including large but finite float32 bounds."""
    extent = np.maximum(np.asarray(maximum, dtype=np.float64) - minimum, 0)
    return 2 * (extent[..., 0] * extent[..., 1] + extent[..., 1] * extent[..., 2] + extent[..., 2] * extent[..., 0])


def _sah_partition(points, minimum, maximum, bins):
    """Return a deterministic valid binned-SAH left mask, or None to fall back."""
    best_cost, best_mask = math.inf, None
    count = len(points)
    for axis in range(3):
        coordinates = points[:, axis]
        low, high = float(coordinates.min()), float(coordinates.max())
        if not high > low:
            continue
        bin_ids = np.minimum(((coordinates - low) / (high - low) * bins).astype(np.int64), bins - 1)
        counts = np.bincount(bin_ids, minlength=bins)
        bin_min = np.full((bins, 3), np.inf, dtype=np.float64)
        bin_max = np.full((bins, 3), -np.inf, dtype=np.float64)
        np.minimum.at(bin_min, bin_ids, minimum)
        np.maximum.at(bin_max, bin_ids, maximum)
        left_counts = np.cumsum(counts)[:-1]
        right_counts = count - left_counts
        valid = (left_counts > 0) & (right_counts > 0)
        if not valid.any():
            continue
        left_min = np.minimum.accumulate(bin_min, axis=0)[:-1]
        left_max = np.maximum.accumulate(bin_max, axis=0)[:-1]
        right_min = np.minimum.accumulate(bin_min[::-1], axis=0)[::-1][1:]
        right_max = np.maximum.accumulate(bin_max[::-1], axis=0)[::-1][1:]
        costs = np.full(bins - 1, np.inf)
        costs[valid] = (_surface_area(left_min[valid], left_max[valid]) * left_counts[valid]
                        + _surface_area(right_min[valid], right_max[valid]) * right_counts[valid])
        split = int(np.argmin(costs))
        # Fixed axis/bin iteration and strict comparison make equal costs
        # deterministic; no random sampling participates in tree construction.
        if costs[split] < best_cost:
            best_cost = float(costs[split])
            best_mask = bin_ids <= split
    return best_mask


def build_bvh(minimum, maximum, leaf_size=8, method="sah", bins=12, max_depth=62):
    """Build a depth-bounded BVH and return nodes, permutation, actual depth.

    SAH uses 12 bins per axis by default. Median splits remain available for
    reproducible comparison and are mandatory when centroid bins degenerate
    or when remaining depth must be reserved for a balanced subtree.
    """
    minimum, maximum = np.asarray(minimum, dtype=np.float32), np.asarray(maximum, dtype=np.float32)
    count = len(minimum)
    if minimum.shape != (count, 3) or maximum.shape != minimum.shape or count < 1:
        raise ValueError("Bounds must be nonempty matching [N,3] arrays")
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or np.any(minimum > maximum):
        raise ValueError("Invalid particle bounds")
    if not 1 <= leaf_size <= 64:
        raise ValueError("leaf_size must be between 1 and 64")
    if method not in ("sah", "median") or not 2 <= bins <= 64:
        raise ValueError("Expected bvh method 'sah'/'median' and 2..64 SAH bins")
    minimum_depth = ((count + leaf_size - 1) // leaf_size - 1).bit_length()
    if not minimum_depth <= max_depth <= 62:
        raise ValueError("BVH depth budget must fit a balanced tree and be at most 62")
    centroids = (minimum.astype(np.float64) + maximum.astype(np.float64)) * 0.5
    order = np.arange(count, dtype=np.uint32)
    # Unlike median trees, a useful SAH split can isolate a single outlier.
    # The full binary-tree bound is necessary even when leaf_size is eight.
    capacity = 2 * count - 1
    nodes = np.empty(capacity, dtype=BVH_DTYPE)
    node_count, actual_depth = 0, 0

    def visit(start, end, depth):
        nonlocal node_count, actual_depth
        index = node_count
        node_count += 1
        actual_depth = max(actual_depth, depth)
        indices = order[start:end]
        node = nodes[index]
        node["minimum"] = minimum[indices].min(axis=0)
        node["maximum"] = maximum[indices].max(axis=0)
        if end - start <= leaf_size:
            node["left"] = start
            node["right_or_count"] = LEAF_BIT | np.uint32(end - start)
            return index
        points = centroids[indices]
        size = end - start
        required_depth = ((size + leaf_size - 1) // leaf_size - 1).bit_length()
        mask = None
        if method == "sah" and depth + required_depth < max_depth:
            mask = _sah_partition(points, minimum[indices], maximum[indices], bins)
        if mask is not None and 0 < int(mask.sum()) < size:
            middle = int(mask.sum())
            order[start:end] = np.concatenate((indices[mask], indices[~mask]))
        else:
            axis = int(np.argmax(points.max(axis=0) - points.min(axis=0)))
            middle = size // 2
            # Stable ordering also handles identical centroids predictably.
            permutation = np.argsort(points[:, axis], kind="stable")
            order[start:end] = indices[permutation]
        split = start + middle
        node["left"] = visit(start, split, depth + 1)
        node["right_or_count"] = visit(split, end, depth + 1)
        return index

    visit(0, count, 0)
    if node_count >= 0x80000000:
        raise ValueError("BVH node indices exceed the reserved leaf-bit format")
    return nodes[:node_count].copy(), order, actual_depth


def bvh_statistics(nodes):
    """Return a geometric cost estimate; this is not a measured GPU frame time."""
    leaf = (nodes["right_or_count"] & LEAF_BIT) != 0
    counts = (nodes["right_or_count"][leaf] & np.uint32(0x7fffffff)).astype(np.float64)
    area = _surface_area(nodes["minimum"], nodes["maximum"])
    weighted_area = float(area[~leaf].sum() + (area[leaf] * counts).sum())
    return {
        "leaf_count": int(leaf.sum()), "mean_leaf_size": float(counts.mean()),
        "max_leaf_size": int(counts.max()),
        "estimated_sah_cost": weighted_area / float(area[0]) if area[0] > 0 else 0.0,
    }


def reference_instance_hit(record, origin, direction, t_min=0.0, t_max=1e20):
    """CPU scalar oracle for native instance support, hit order, and quartic alpha.

    Returns (distance, kernel_response, alpha) for a supported native candidate.
    It does not apply response/alpha thresholds, SH, or compositing. Tests can
    distinguish the instance candidate test from response acceptance.
    """
    record = np.asarray(record)
    matrix = record.reshape(4, 4)[1:, :3].astype(np.float64)
    origin = matrix @ (np.asarray(origin, dtype=np.float64) - record[:3])
    direction = matrix @ np.asarray(direction, dtype=np.float64)
    radius = float(record[7])
    entry, exit = float(t_min), float(t_max)
    for axis in range(3):
        if direction[axis] == 0:
            if abs(origin[axis]) > radius:
                return None
        else:
            a = (-radius - origin[axis]) / direction[axis]
            b = (radius - origin[axis]) / direction[axis]
            entry, exit = max(entry, min(a, b)), min(exit, max(a, b))
    if entry > exit:
        return None
    norm2 = float(np.dot(direction, direction))
    if norm2 <= 0:
        return None
    distance = -float(np.dot(origin, direction)) / norm2
    if not t_min < distance < t_max:
        return None
    squared_distance = float(np.dot(np.cross(direction / math.sqrt(norm2), origin),
                                    np.cross(direction / math.sqrt(norm2), origin)))
    # Exact reference CUDA instance predicate, rather than a replacement
    # ellipsoid-surface intersection. kernel_radius cancels from this ratio.
    if squared_distance / norm2 >= 9:
        return None
    response = math.exp(-squared_distance * squared_distance / 18)
    return distance, response, min(0.99, response * float(record[3]))


def _load_camera_module():
    name = "fullcircle_webgpu_export_cameras"
    if name not in sys.modules:
        path = REPO_ROOT / "WebViewer-V2" / "cameras.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def export_cameras(checkpoint, positions, camera_json=None, data_path=None, fov_y=math.radians(70)):
    module = _load_camera_module()
    if camera_json:
        presets = module.load_camera_json(camera_json)
        source = "camera JSON"
    else:
        path = Path(data_path or str(_config_value(checkpoint["config"], "path", ""))).expanduser()
        candidates = [path] if path.is_absolute() else [Path.cwd() / path, REPO_ROOT / path]
        presets = []
        for candidate in candidates:
            presets = module.load_colmap_cameras(candidate, fov_y)
            if presets:
                break
        source = "COLMAP training poses, perspective preview"
        if not presets:
            presets = [module.camera_from_bounds(positions, fov_y)]
            source = "Gaussian bounds fallback"
    return [
        {"name": preset.name, "position": preset.position.tolist(),
         "rotation": preset.c2w[:3, :3].tolist(), "fov_y": preset.fov_y, "aspect": preset.aspect}
        for preset in presets
    ], source


COMPRESSION_MODES = ("none", "gzip", "gzip-shuffle")
MAX_PART_BYTES = 100 * 1024 * 1024
SH_DTYPES = {"float32": np.dtype("<f4"), "float16": np.dtype("<f2")}


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


def set_sh_precision_metadata(manifest, target_dtype, source_spec=None):
    """Record half rounding even if a later repack expands storage to float32."""
    source_spec = source_spec or {}
    output_spec = manifest["files"]["sh"]
    output_spec["dtype"] = target_dtype
    had_half_rounding = (target_dtype == "float16" or source_spec.get("dtype") == "float16"
                         or source_spec.get("quantization_dtype") == "float16")
    if had_half_rounding:
        output_spec["quantization_dtype"] = "float16"
        description = "float16 (quantized)" if target_dtype == "float16" else "float32 expanded from float16; earlier rounding is retained"
        manifest["precision"] = f"geometry storage float32; SH storage {description}; BVH float32/u32; no additional pruning"


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
    This works equally for float32 geometry/SH and the mixed float/u32 BVH.
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


def export_checkpoint(checkpoint, output, *, source_name="checkpoint.pt", leaf_size=8,
                      bvh_method="sah", sah_bins=12,
                      chunk_bytes=64 * 1024 * 1024, camera_json=None, data_path=None,
                      fov_y=math.radians(70), overwrite=False, compression="none", sh_dtype="float32"):
    output = Path(output)
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not overwrite)):
        raise FileExistsError(f"Output already exists: {output}; choose another directory or pass --overwrite")
    if not math.isfinite(fov_y) or not 0 < fov_y < math.pi:
        raise ValueError("Preview vertical FOV must be between 0 and pi")
    if compression not in COMPRESSION_MODES:
        raise ValueError(f"Unsupported compression: {compression}")
    if sh_dtype not in SH_DTYPES:
        raise ValueError("SH dtype must be float16 or float32")
    render = validate_config(checkpoint)
    geometry, sh, minimum, maximum = prepare_gaussians(checkpoint, render)
    cameras, camera_source = export_cameras(checkpoint, geometry[:, :3], camera_json, data_path, fov_y)
    nodes, order, depth = build_bvh(minimum, maximum, leaf_size, method=bvh_method, bins=sah_bins)
    geometry, sh = geometry[order], sh[order]
    if sh_dtype == "float16":
        sh = np.frombuffer(convert_sh_bytes(sh.tobytes(), "float32", sh_dtype), dtype=SH_DTYPES[sh_dtype]).reshape(-1, 48)
    manifest = {
        "schema": SCHEMA, "source": Path(source_name).name,
        "gaussian_count": len(geometry), "bvh_node_count": len(nodes), "bvh_max_depth": depth,
        "bvh_root": 0, "bvh_leaf_size": leaf_size,
        "bvh_method": bvh_method, "bvh_sah_bins": sah_bins if bvh_method == "sah" else None,
        "bvh_stats": bvh_statistics(nodes),
        "bounds": {"min": nodes[0]["minimum"].tolist(), "max": nodes[0]["maximum"].tolist()},
        "render": render, "cameras": cameras, "default_camera": 0, "camera_source": camera_source,
        "precision": "float32, no quantization or pruning", "byte_order": "little-endian",
        "native_support": {
            "bound": "density-clamped transformed cube (native instances)",
            "kernel_radius": "pow(log(min(min_response/density,0.97))/(-1/18),1/4) when clamped",
            "hit_distance": "-dot(local_origin,local_direction)/dot(local_direction,local_direction)",
            "instance_predicate": "cross(normalize(local_direction),local_origin)^2 / dot(local_direction,local_direction) < 9",
            "color": "per-world-ray SH with native +0.5 and lower clamp, front-to-back alpha compositing",
            "ordering": "native reference gathers the nearest 16 candidates per trace, continuing while T>threshold",
        },
        "known_differences": [
            "CPU activation/normalization and precomputed FP32 inverse transforms can differ by rounding from CUDA/OptiX.",
            "Software BVH traversal and equal-distance tie ordering can differ from OptiX; bitwise identity is not promised.",
            "Browser transcendentals and final display/color conversion require comparison with V2 on fixed cameras.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    old_parts = set()
    old_manifest = output / "manifest.json"
    if overwrite and old_manifest.is_file():
        old = json.loads(old_manifest.read_text())
        for buffer in old.get("files", {}).values():
            for part in buffer.get("parts", []):
                name = part.get("path", "")
                if name and Path(name).name == name and name not in ("alignment.json", "cameras.json", "manifest.json"):
                    old_parts.add(name)
    # Complete the export in a sibling staging directory before touching the
    # requested destination. Manifest is installed last to mark completion.
    with tempfile.TemporaryDirectory(prefix=".webgpu-export-", dir=output.parent) as temporary:
        stage = Path(temporary)
        manifest["files"] = {
            "geometry": write_parts(stage, "geometry", geometry, 64, chunk_bytes, compression),
            "sh": write_parts(stage, "sh", sh, 48 * SH_DTYPES[sh_dtype].itemsize, chunk_bytes, compression,
                              shuffle_bytes=SH_DTYPES[sh_dtype].itemsize),
            "bvh": write_parts(stage, "bvh", nodes, 32, chunk_bytes, compression),
        }
        set_sh_precision_metadata(manifest, sh_dtype)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        output.mkdir(exist_ok=True)
        new_parts = {part["path"] for buffer in manifest["files"].values() for part in buffer["parts"]}
        for name in sorted(new_parts):
            shutil.move(str(stage / name), str(output / name))
        for name in old_parts - new_parts:
            (output / name).unlink(missing_ok=True)
        shutil.move(str(stage / "manifest.json"), str(output / "manifest.json"))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local .pt checkpoint")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, help="Optional existing WebViewer/3DGS camera JSON")
    parser.add_argument("--data-path", type=Path, help="Override dataset path used to read COLMAP poses")
    parser.add_argument("--fov-deg", type=float, default=70)
    parser.add_argument("--leaf-size", type=int, default=8)
    parser.add_argument("--bvh-method", choices=("sah", "median"), default="sah")
    parser.add_argument("--sah-bins", type=int, default=12)
    parser.add_argument("--chunk-mib", type=float, default=64)
    parser.add_argument("--compression", choices=COMPRESSION_MODES, default="gzip-shuffle",
                        help="Lossless transport encoding (default: gzip-shuffle), applied after any requested SH dtype conversion")
    parser.add_argument("--sh-dtype", choices=tuple(SH_DTYPES), default="float32",
                        help="SH storage dtype (default: float32); float16 explicitly quantizes SH only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.chunk_mib < 100:
        parser.error("chunk-mib must be greater than 0 and below 100")
    checkpoint = load_checkpoint(args.checkpoint)
    manifest = export_checkpoint(
        checkpoint, args.out_dir, source_name=args.checkpoint.name,
        leaf_size=args.leaf_size, chunk_bytes=int(args.chunk_mib * 1024 * 1024),
        bvh_method=args.bvh_method, sah_bins=args.sah_bins,
        camera_json=args.camera_json, data_path=args.data_path,
        fov_y=math.radians(args.fov_deg), overwrite=args.overwrite,
        compression=args.compression, sh_dtype=args.sh_dtype,
    )
    summary = {key: manifest[key] for key in ("gaussian_count", "bvh_node_count", "bvh_max_depth")}
    summary["manifest"] = str(args.out_dir / "manifest.json")
    summary["total_binary_bytes"] = sum(item["byteLength"] for item in manifest["files"].values())
    summary["total_download_bytes"] = sum(item["downloadByteLength"] for item in manifest["files"].values())
    summary["decoded_bytes"] = summary["total_binary_bytes"]
    summary["stored_bytes"] = summary["total_download_bytes"]
    summary["compression_ratio"] = summary["decoded_bytes"] / summary["stored_bytes"]
    summary["saved_percent"] = 100 * (1 - summary["stored_bytes"] / summary["decoded_bytes"])
    summary["sh_dtype"] = manifest["files"]["sh"]["dtype"]
    summary["bvh_method"] = manifest["bvh_method"]
    summary["bvh_stats"] = manifest["bvh_stats"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
