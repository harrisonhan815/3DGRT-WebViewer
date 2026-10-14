#!/usr/bin/env python3
"""Build a WebGL2 cache from an Insta360 SOG v2 and INRIA cameras JSON.

Requires NumPy and Pillow with WebP support; no training environment or GPU.
Keeps every Gaussian, source order, all available SH, and original coordinates.
SH stays FP32 by default, with lossless gzip/byte-shuffle transport. Original
SOG, camera/voxel files and an existing alignment.json are never overwritten.

Example:
    python import_insta_sog.py --model-dir models/zj-office1-insta

Format reference: https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/
Raw SOG and INRIA/3DGS PLY coordinates agree (SplatTransform reads and writes
SOG with Transform.PLY); no unilateral axis flip or SH rotation is applied.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from export_model import SCHEMA, _camera, set_sh_precision_metadata, write_parts
from sog_codec import decode_sog

PROFILE = "insta360-sog-v2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cameras(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a nonempty INRIA/3DGS camera JSON array")
    result = []
    for index, item in enumerate(data):
        fx, fy, width, height = (float(item[key]) for key in ("fx", "fy", "width", "height"))
        if not all(math.isfinite(value) and value > 0 for value in (fx, fy, width, height)):
            raise ValueError("Camera focal lengths and dimensions must be positive")
        camera = _camera(item.get("img_name", f"Camera {index}"), item["position"], item["rotation"],
                         2 * math.atan(height / (2 * fy)), width / height)
        result.append(camera)
    return result


def import_model(directory, *, sog=None, cameras=None, source_dir=None, overwrite=False, default_camera=0):
    directory = Path(directory).expanduser().resolve(strict=True)
    source_root = Path(source_dir).expanduser().resolve(strict=True) if source_dir else directory / "org" if (directory / "org").is_dir() else directory
    def select(explicit, pattern):
        if explicit is not None:
            return Path(explicit).expanduser().resolve(strict=True)
        matches = sorted(source_root.glob(pattern))
        if len(matches) != 1:
            raise ValueError(f"Specify the input explicitly: expected one {pattern} in {directory}")
        return matches[0]
    sog, cameras = select(sog, "*.sog"), select(cameras, "*cameras.json")
    manifest_path = directory / "manifest.json"
    previous = None
    if manifest_path.exists():
        if not overwrite:
            raise FileExistsError("Viewer cache already exists; use --overwrite to regenerate it")
        previous = json.loads(manifest_path.read_text())
        if previous.get("compatibility", {}).get("profile") != PROFILE:
            raise ValueError("Refusing to replace a different model's manifest")
    if manifest_path.is_symlink() or any(path.is_symlink() for path in directory.glob("insta-*.bin*")):
        raise ValueError("Cache targets must not be symbolic links")
    old_parts = {part["path"] for file in previous.get("files", {}).values() for part in file["parts"]} if previous else set()
    if any(Path(name).name != name or not name.startswith("insta-") for name in old_parts):
        raise ValueError("Invalid previous cache file paths")
    if any(path.name not in old_parts for path in directory.glob("insta-*.bin*")):
        raise FileExistsError("Untracked cache files exist; refusing to replace them")
    fingerprints = {path.name: sha256(path) for path in (sog, cameras)}
    print(f"Decoding {sog.name} ...", flush=True)
    geometry, sh, bands, meta = decode_sog(sog)
    positions = geometry[:, :3]
    navigation = np.quantile(positions, [.05, .95], axis=0)
    camera_list = read_cameras(cameras)
    if type(default_camera) is not int or not 0 <= default_camera < len(camera_list):
        raise ValueError("default-camera must index the imported camera list")
    manifest = {
        "schema": SCHEMA, "source": sog.name, "gaussian_count": len(geometry),
        "bounds": {"min": positions.min(0).tolist(), "max": positions.max(0).tolist()},
        "navigation_bounds": {"min": navigation[0].tolist(), "max": navigation[1].tolist()},
        "cameras": camera_list, "default_camera": default_camera,
        "camera_source": "Insta360 INRIA/3DGS cameras JSON; original camera-to-world poses",
        "render": {"method": "3dgs", "pipeline": "2d-ewa", "kernel_degree": 2,
                   "depth_sort": "camera", "sh_degree": bands, "sh_direction": "camera_to_gaussian",
                   "min_alpha": 1/255, "max_alpha": .99, "min_transmittance": .0001,
                   "low_pass_variance": .3, "antialiasing": meta.get("model") == "antialiased",
                   "near_plane": .01, "background": [0, 0, 0]},
        "byte_order": "little-endian", "gaussian_order": "source SOG pixel order; no pruning or reordering",
        "compatibility": {"profile": PROFILE, "default_enabled": False,
                          "source_sha256": fingerprints, "generator": meta.get("asset", {}).get("generator"),
                          "notes": ["Source SOG is already quantized; this cache adds no SH FP16 quantization.",
                                    "No unilateral axis flips: retain original SOG/PLY coordinates, SH and camera poses.",
                                    "Voxel files describe collisions, not Gaussian appearance, and are not needed for rendering.",
                                    "Compatibility mode uses pinhole projection and camera-depth ordering; the off comparison uses the existing radial order."]},
    }
    with tempfile.TemporaryDirectory(prefix=".insta-webgl2-", dir=directory) as temporary:
        stage = Path(temporary)
        print(f"Writing lossless cache: {len(geometry):,} Gaussians, SH {bands}, {len(camera_list)} cameras ...", flush=True)
        manifest["files"] = {
            "geometry": write_parts(stage, "insta-geometry", geometry, 48, 64 << 20, "gzip-shuffle"),
            "sh": write_parts(stage, "insta-sh", sh, 192, 64 << 20, "gzip-shuffle"),
        }
        set_sh_precision_metadata(manifest, "float32")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        if fingerprints != {path.name: sha256(path) for path in (sog, cameras)}:
            raise RuntimeError("Source changed during conversion; cache was not installed")
        new_parts = {part["path"] for file in manifest["files"].values() for part in file["parts"]}
        for name in sorted(new_parts):
            os.replace(stage / name, directory / name)
        os.replace(stage / "manifest.json", manifest_path)
        for name in old_parts - new_parts:
            (directory / name).unlink(missing_ok=True)
    print(f"Saved {manifest_path}; source files and alignment.json preserved", flush=True)
    return manifest


def attach_sog(directory, *, sog=None, cameras=None, source_dir=None, default_camera=None):
    """Add native SOG transport to an existing verified binary import.

    Source SOG bytes are preserved, with no second clustering/quantization.
    Keep binary files and alignment intact; back up the prior manifest before
    publishing the paired descriptor. This does not load a training model.
    """
    from export_both import combine_manifests

    directory = Path(directory).expanduser().resolve(strict=True)
    source_root = Path(source_dir).expanduser().resolve(strict=True) if source_dir else directory / "org" if (directory / "org").is_dir() else directory
    def select(explicit, pattern):
        if explicit is not None:
            return Path(explicit).expanduser().resolve(strict=True)
        matches = sorted(source_root.glob(pattern))
        if len(matches) != 1:
            raise ValueError(f"Expected one {pattern} in {source_root}; specify its path")
        return matches[0]
    source, cameras_path = select(sog, "*.sog"), select(cameras, "*cameras.json")
    manifest_path, target = directory / "manifest.json", directory / "model.sog"
    if manifest_path.is_symlink() or target.is_symlink():
        raise ValueError("Model outputs must not be symbolic links")
    original = manifest_path.read_bytes()
    binary = json.loads(original)
    if binary.get("schema") != SCHEMA or binary.get("compatibility", {}).get("profile") != PROFILE or binary.get("storage"):
        raise ValueError("Expected an existing Insta360 binary cache; import it first")
    fingerprint = {p.name: sha256(p) for p in (source, cameras_path)}
    if binary.get("compatibility", {}).get("source_sha256") != fingerprint:
        raise ValueError("Original SOG/cameras do not match this binary model")
    geometry, sh, bands, meta = decode_sog(source)
    count = len(geometry)
    expected_bounds = {"min": geometry[:, :3].min(0).tolist(), "max": geometry[:, :3].max(0).tolist()}
    expected_cameras = read_cameras(cameras_path)
    if (binary.get("source") != source.name or binary.get("gaussian_count") != count
            or binary.get("bounds") != expected_bounds or binary.get("cameras") != expected_cameras
            or binary.get("render", {}).get("sh_degree") != bands
            or binary.get("render", {}).get("antialiasing") != (meta.get("model") == "antialiased")):
        raise ValueError("Source model identity, SH or cameras differ from the existing cache")
    # The browser's native SOG reader handles numeric, lossless WebP textures.
    with zipfile.ZipFile(source) as archive:
        for section in ("means", "quats", "scales", "sh0", "shN"):
            for name in (meta.get(section) or {}).get("files", []):
                header = archive.read(name)[:12]
                if header[:4] != b"RIFF" or header[8:12] != b"WEBP":
                    raise ValueError("Native browser SOG transport requires WebP textures")
    if default_camera is not None:
        if type(default_camera) is not int or not 0 <= default_camera < len(expected_cameras):
            raise ValueError("default-camera must index the imported camera list")
        binary["default_camera"] = default_camera
    binary.pop("variants", None)
    binary["compatibility"]["default_enabled"] = False
    binary["compatibility"]["notes"] = [
        "Original SOG bytes and numeric coordinate/SH convention are retained.",
        "The current viewer keeps the compatibility control hidden and uses radial ordering by default.",
        "SOG is downloaded first; binary files are loaded only after the binary button is selected.",
    ]
    dtype = binary["files"]["sh"].get("dtype", "float32")
    if dtype not in ("float16", "float32"):
        raise ValueError("Unsupported SH dtype in the existing cache")
    for spec in binary["files"].values():
        for part in spec.get("parts", []):
            path = part["path"]
            if Path(path).name != path or not path.startswith("insta-") or (directory / path).is_symlink():
                raise ValueError("Unexpected binary cache path")
            if (directory / path).stat().st_size != part.get("downloadByteLength", part["byteLength"]):
                raise ValueError("Binary cache is missing or has an unexpected length")
    native = copy.deepcopy(binary)
    native["files"] = {"geometry": {"byteLength": count*48, "stride": 48},
                       "sh": {"byteLength": count*48*(2 if dtype == "float16" else 4),
                              "stride": 48*(2 if dtype == "float16" else 4), "dtype": dtype}}
    native["storage"] = {"format": "sog-v2", "path": "model.sog", "byteLength": source.stat().st_size,
                         "sha256": fingerprint[source.name], "generator": meta.get("asset", {}).get("generator")}
    native["precision"] = f"Original SOG v2 quantization; no re-encoding; geometry FP32, SH {dtype} after decode"
    paired = combine_manifests(binary, native)
    if target.exists() and sha256(target) != fingerprint[source.name]:
        raise FileExistsError("A different model.sog already exists; preserve it before attaching this source")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = directory / f"manifest.before-sog.{stamp}.json"
    with tempfile.TemporaryDirectory(prefix=".sog-attach-", dir=directory) as temporary:
        stage = Path(temporary)
        (stage / "model.sog").write_bytes(source.read_bytes())
        if sha256(stage / "model.sog") != fingerprint[source.name] or fingerprint != {p.name: sha256(p) for p in (source, cameras_path)}:
            raise RuntimeError("Source files changed during adaptation")
        (stage / "manifest.json").write_text(json.dumps(paired, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        if manifest_path.read_bytes() != original:
            raise RuntimeError("Existing manifest changed during adaptation")
        with backup.open("xb") as stream:
            stream.write(original)
        if not target.exists():
            os.replace(stage / "model.sog", target)
        os.replace(stage / "manifest.json", manifest_path)
    print(json.dumps({"manifest": str(manifest_path), "sog": str(target), "gaussian_count": count, "sh_degree": bands,
                      "sog_bytes": source.stat().st_size, "sh_dtype": dtype, "source_bytes_preserved": True,
                      "manifest_backup": str(backup)}, indent=2))
    return paired


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sog", type=Path)
    parser.add_argument("--cameras", type=Path)
    parser.add_argument("--source-dir", type=Path, help="Original SOG/cameras directory; automatically uses org/ when present")
    parser.add_argument("--attach-sog", action="store_true", help="Add native SOG loading to an existing binary cache; preserve source and binary files")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--default-camera", type=int, help="New imports default to 0; attach-sog preserves the existing default")
    args = parser.parse_args()
    if args.attach_sog:
        attach_sog(args.model_dir, sog=args.sog, cameras=args.cameras, source_dir=args.source_dir, default_camera=args.default_camera)
    else:
        import_model(args.model_dir, sog=args.sog, cameras=args.cameras, source_dir=args.source_dir,
                     overwrite=args.overwrite, default_camera=args.default_camera if args.default_camera is not None else 0)
