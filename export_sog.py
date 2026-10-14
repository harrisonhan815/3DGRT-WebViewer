"""SOG v2 transport for native-fisheye 3DGS checkpoints (not 3DGRT kernels).

Invoked by export_model.py --format sog. SOG quantizes geometry and SH; it
does not change the renderer, camera convention, training sort or alignment.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from export_model import (SCHEMA, SH_DTYPES, _cpu_float32, export_cameras,
                          prepare_gaussians, validate_config)


def write_ply(checkpoint, path, degree):
    """Standard 3DGS PLY: raw log scale/logit alpha, wxyz, channel-major SH."""
    count = len(checkpoint["positions"])
    coefficients = (degree + 1) ** 2 - 1
    maximum = (int(checkpoint["max_n_features"]) + 1) ** 2 - 1
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(coefficients * 3)]
    names += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = np.zeros((count, len(names)), dtype="<f4")
    data[:, :3] = _cpu_float32(checkpoint, "positions", (count, 3)).numpy()
    data[:, 6:9] = _cpu_float32(checkpoint, "features_albedo", (count, 3)).numpy()
    if coefficients:
        rest = _cpu_float32(checkpoint, "features_specular", (count, maximum * 3)).numpy()
        data[:, 9:9 + coefficients * 3] = rest.reshape(count, maximum, 3)[:, :coefficients].transpose(0, 2, 1).reshape(count, -1)
    data[:, -8:-7] = _cpu_float32(checkpoint, "density", (count, 1)).numpy()
    data[:, -7:-4] = _cpu_float32(checkpoint, "scale", (count, 3)).numpy()
    q = _cpu_float32(checkpoint, "rotation", (count, 4)).numpy().astype(np.float64)
    length = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(length < 1e-12):
        raise ValueError("SOG requires nonzero Gaussian quaternions")
    data[:, -4:] = q / length
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {count}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    with Path(path).open("wb") as stream:
        stream.write(header.encode("ascii"))
        data.tofile(stream)


def encoder_command():
    package = Path(__file__).resolve().parent / "tools/sog/node_modules/@playcanvas/splat-transform"
    if not (package / "bin/cli.mjs").is_file():
        raise RuntimeError("Install the pinned encoder first: npm ci --prefix 3DGRT-WebViewer/tools/sog --ignore-scripts")
    if json.loads((package / "package.json").read_text())["version"] != "3.10.1":
        raise RuntimeError("Expected splat-transform 3.10.1; restore tools/sog with npm ci")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("SOG export requires Node.js >= 22")
    major = int(subprocess.check_output([node, "--version"], text=True).strip().lstrip("v").split(".")[0])
    if major < 22:
        raise RuntimeError("SOG export requires Node.js >= 22")
    return [node, str(package / "bin/cli.mjs")]


def export_sog(checkpoint, output, *, source_name="checkpoint.pt", camera_json=None,
               data_path=None, fov_y=math.pi / 2, overwrite=False, sh_dtype="float16",
               alignment=None, gpu="0", iterations=10):
    from sog_codec import decode_sog
    from serve import validate_alignment

    output = Path(output).expanduser()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or not overwrite)):
        raise FileExistsError(f"Output exists: {output}; choose another directory or use --overwrite")
    if not math.isfinite(fov_y) or not 0 < fov_y < math.pi or sh_dtype not in SH_DTYPES:
        raise ValueError("Invalid FOV or SH dtype")
    if not str(gpu).isdigit() and gpu != "cpu":
        raise ValueError("sog-gpu must be an adapter index or cpu")
    if type(iterations) is not int or not 1 <= iterations <= 100:
        raise ValueError("sog-iterations must be between 1 and 100")
    command = encoder_command()
    render = validate_config(checkpoint)
    geometry, _ = prepare_gaussians(checkpoint, render)
    count = len(geometry)
    if count > 10_000_000:
        raise ValueError("SOG browser decoder supports at most 10 million Gaussians")
    positions = geometry[:, :3]
    cameras, camera_source = export_cameras(checkpoint, positions, camera_json, data_path, fov_y)
    navigation = np.quantile(positions, [.05, .95], axis=0)
    manifest = {
        "schema": SCHEMA, "source": Path(source_name).name, "gaussian_count": count,
        "bounds": {"min": positions.min(0).tolist(), "max": positions.max(0).tolist()},
        "bounds_definition": "checkpoint center bounds before SOG quantization; stable alignment identity",
        "navigation_bounds": {"min": navigation[0].tolist(), "max": navigation[1].tolist()},
        "render": render, "cameras": cameras, "default_camera": 0, "camera_source": camera_source,
        "byte_order": "little-endian", "gaussian_order": "SOG encoder Morton order; no pruning",
        "precision": f"SOG v2 quantized positions, rotations, scales, alpha and SH; decoded geometry FP32, SH {sh_dtype}",
        "files": {"geometry": {"byteLength": count * 48, "stride": 48},
                  "sh": {"byteLength": count * 48 * SH_DTYPES[sh_dtype].itemsize,
                         "stride": 48 * SH_DTYPES[sh_dtype].itemsize, "dtype": sh_dtype}},
        "known_differences": ["SOG is lossy quantization, including vector-quantized higher SH.",
                              "No coordinate transform, Gaussian pruning or SH band reduction is applied.",
                              "Morton reordering and quantization can change equal/near-equal depth ties.",
                              "Decoding restores the current WebGL2 layout; SOG alone does not reduce rendering work."],
    }
    if checkpoint.get("global_step") is not None:
        manifest["training_step"] = int(checkpoint["global_step"])
    if sh_dtype == "float16":
        manifest["files"]["sh"]["quantization_dtype"] = "float16"
    # Preserve the exact original sidecar and its identity. Do not silently
    # rewrite a calibration for a different checkpoint merely to make it fit.
    saved_alignment = output / "alignment.json"
    alignment_path = saved_alignment if saved_alignment.exists() else Path(alignment).expanduser() if alignment else None
    alignment_bytes = None
    if alignment_path:
        alignment_bytes = alignment_path.read_bytes()
        validate_alignment(json.loads(alignment_bytes), manifest)
    for name in ("manifest.json", "model.sog", "alignment.json"):
        if (output / name).is_symlink():
            raise ValueError(f"Refusing symlink output: {name}")
    previous_path = output / "manifest.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous.get("schema") != SCHEMA:
            raise ValueError("Refusing to replace a different manifest schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sog-export-", dir=output.parent) as temporary:
        stage = Path(temporary)
        print(f"Encoding {count:,} Gaussians as SOG (lossy, no pruning) ...", flush=True)
        write_ply(checkpoint, stage / "input.ply", render["sh_degree"])
        subprocess.run(command + ["--gpu", str(gpu), "--sh-iterations", str(iterations), "--no-tty",
                                  str(stage / "input.ply"), str(stage / "model.sog")], check=True)
        decoded, decoded_sh, bands, meta = decode_sog(stage / "model.sog")
        if len(decoded) != count or bands != render["sh_degree"]:
            raise ValueError("SOG encoder changed the Gaussian count or SH degree")
        if sh_dtype == "float16" and np.max(np.abs(decoded_sh)) > 65504:
            raise ValueError("Decoded SOG SH exceeds FP16; use --sh-dtype float32")
        manifest["decoded_bounds"] = {"min": decoded[:, :3].min(0).tolist(), "max": decoded[:, :3].max(0).tolist()}
        payload = (stage / "model.sog").read_bytes()
        manifest["storage"] = {"format": "sog-v2", "path": "model.sog", "byteLength": len(payload),
                               "sha256": hashlib.sha256(payload).hexdigest(),
                               "generator": meta.get("asset", {}).get("generator"), "sh_iterations": iterations}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        output.mkdir(exist_ok=overwrite)
        os.replace(stage / "model.sog", output / "model.sog")
        if alignment_bytes is not None and not saved_alignment.exists():
            saved_alignment.write_bytes(alignment_bytes)
        # Retain old binary files/sidecars when explicitly overwriting an
        # existing directory; the new manifest references only model.sog.
        os.replace(stage / "manifest.json", output / "manifest.json")
    return manifest
