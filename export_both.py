"""Publish two encodings of one checkpoint with shared camera/alignment identity."""
import copy
import json
import math
from pathlib import Path
from export_files import ExportFiles

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
    with ExportFiles(output) as files:
        common = dict(source_name=source_name, camera_json=camera_json, data_path=data_path, fov_y=fov_y,
                      sh_dtype=sh_dtype, overwrite=True, _output_files=files)
        binary = export_checkpoint(checkpoint, output, chunk_bytes=chunk_bytes, compression=compression, **common)
        if calibration is not None:
            validate_alignment(json.loads(calibration), binary)
        sog = export_sog(checkpoint, output, gpu=gpu, iterations=iterations, **common)
        manifest = combine_manifests(binary, sog)
        if alignment_path and alignment_path.read_bytes() != calibration:
            raise RuntimeError("Alignment changed during export; retry with the current file")
        # Both encodings are written in the target folder and verified before
        # replacing existing files. Publish the one shared manifest last.
        if calibration is not None and not saved_alignment.exists():
            files.path('alignment.json').write_bytes(calibration)
        files.commit(manifest)
    return manifest
