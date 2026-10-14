"""Publish two encodings of one checkpoint with shared camera/alignment identity."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile

from export_model import SCHEMA, export_checkpoint
from export_sog import export_sog
from serve import validate_alignment

VARIANT_FIELDS = ("files", "storage", "precision", "gaussian_order", "decoded_bounds",
                  "known_differences", "native_support")


def variant_spec(manifest):
    return {key: copy.deepcopy(manifest[key]) for key in VARIANT_FIELDS if key in manifest}


def combine_manifests(binary, sog):
    for key in ("schema", "source", "gaussian_count", "bounds", "render", "cameras", "default_camera", "byte_order"):
        if binary.get(key) != sog.get(key):
            raise ValueError(f"Cannot pair different models/cameras: {key}")
    if binary.get("storage") is not None or sog.get("storage", {}).get("format") != "sog-v2":
        raise ValueError("Expected a binary export and a SOG v2 export")
    result = copy.deepcopy(binary)
    # Root stays a normal binary manifest for older clients/tools. New clients
    # select the encoding below and keep all shared identity/camera fields.
    result["variants"] = {"default": "sog", "bin": variant_spec(binary), "sog": variant_spec(sog)}
    return result


def export_both(checkpoint, output, *, source_name="checkpoint.pt", chunk_bytes=64 << 20,
                camera_json=None, data_path=None, fov_y=math.pi/2, overwrite=False,
                compression="gzip-shuffle", sh_dtype="float16", alignment=None, gpu="0", iterations=10):
    output = Path(output).expanduser()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or not overwrite)):
        raise FileExistsError(f"Output exists: {output}; use a new directory or --overwrite")
    for name in ("manifest.json", "alignment.json", "model.sog"):
        if (output / name).is_symlink():
            raise ValueError(f"Refusing symlink output: {name}")
    previous = None
    if (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text())
        if previous.get("schema") != SCHEMA:
            raise ValueError("Refusing to replace another manifest schema")
    saved_alignment = output / "alignment.json"
    alignment_path = saved_alignment if saved_alignment.exists() else Path(alignment).expanduser() if alignment else None
    calibration = alignment_path.read_bytes() if alignment_path else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dual-export-", dir=output.parent) as temporary:
        stage = Path(temporary)
        common = dict(source_name=source_name, camera_json=camera_json, data_path=data_path, fov_y=fov_y, sh_dtype=sh_dtype)
        binary = export_checkpoint(checkpoint, stage / "bin", chunk_bytes=chunk_bytes, compression=compression, **common)
        if calibration is not None:
            validate_alignment(json.loads(calibration), binary)
        sog = export_sog(checkpoint, stage / "sog", gpu=gpu, iterations=iterations, **common)
        manifest = combine_manifests(binary, sog)
        parts = {part["path"] for spec in binary["files"].values() for part in spec["parts"]}
        for name in parts:
            if Path(name).name != name or (output / name).is_symlink():
                raise ValueError(f"Invalid binary output target: {name}")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        if alignment_path and alignment_path.read_bytes() != calibration:
            raise RuntimeError("Alignment changed during export; retry with the current file")
        # Encode/check BOTH versions before touching an existing model. Keep
        # existing backups, diagnostics and the one shared alignment unchanged.
        output.mkdir(exist_ok=overwrite)
        for name in sorted(parts):
            os.replace(stage / "bin" / name, output / name)
        os.replace(stage / "sog/model.sog", output / "model.sog")
        if calibration is not None and not saved_alignment.exists():
            with saved_alignment.open("xb") as stream:
                stream.write(calibration)
        os.replace(stage / "manifest.json", output / "manifest.json")
    return manifest
