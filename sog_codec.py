"""Shared SOG v2 decoding for model export validation and Insta360 import.

Requires NumPy and Pillow with WebP support. Produces the viewer's FP32
geometry and RGB-interleaved SH buffers, without changing source order or
coordinates. Does not depend on checkpoint exporters, import CLIs or servers.
"""

from io import BytesIO
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

MAX_COUNT = 10_000_000
MAX_MEMBER_BYTES = 128 * 1024 * 1024


def codebook(section):
    values = np.asarray(section.get("codebook"), dtype=np.float32)
    if values.shape != (256,) or not np.isfinite(values).all():
        raise ValueError("Expected a 256-entry finite SOG codebook")
    return values


def decode_quaternions(pixels):
    mode = pixels[:, 3].astype(np.int32) - 252
    if np.any((mode < 0) | (mode > 3)):
        raise ValueError("Invalid smallest-three quaternion mode")
    kept = (pixels[:, :3].astype(np.float64) / 255 - .5) * math.sqrt(2)
    omitted = np.sqrt(np.maximum(0, 1 - np.sum(kept**2, axis=1)))
    result = np.empty((len(pixels), 4), dtype=np.float64)
    for missing in range(4):
        selected = mode == missing
        result[np.ix_(selected, [i for i in range(4) if i != missing])] = kept[selected]
        result[selected, missing] = omitted[selected]
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def covariance(quaternion, scale):
    w, x, y, z = quaternion.T
    rows = [np.column_stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y))),
            np.column_stack((2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x))),
            np.column_stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))]
    rows = [row * scale for row in rows]
    return np.column_stack([np.sum(rows[a] * rows[b], axis=1)
                            for a, b in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))])


def decode_sog(path):
    """Decode ZIP/WebP integers without image gamma or alpha premultiplication."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 64 or len({item.filename for item in entries}) != len(entries):
            raise ValueError("SOG has too many or duplicate archive members")
        for item in entries:
            if (item.is_dir() or Path(item.filename).name != item.filename or "\\" in item.filename
                    or item.file_size > MAX_MEMBER_BYTES or item.flag_bits & 1):
                raise ValueError("Unsupported SOG archive member")
        if archive.getinfo("meta.json").file_size > 1 << 20:
            raise ValueError("SOG metadata is too large")
        meta = json.loads(archive.read("meta.json"))
        if meta.get("version") != 2:
            raise ValueError("Only SOG version 2 is supported")
        if meta.get("model") not in (None, "antialiased"):
            raise ValueError("Only 3DGS SOG models are supported (not 2DGS)")
        count = meta.get("count")
        if type(count) is not int or not 0 < count <= MAX_COUNT:
            raise ValueError("Invalid or oversized SOG Gaussian count")

        def filenames(section, length):
            files = meta[section].get("files")
            if not isinstance(files, list) or len(files) != length or any(not isinstance(name, str) for name in files):
                raise ValueError(f"Invalid {section} filenames")
            return files

        def pixels(name):
            if Path(name).name != name or "\\" in name:
                raise ValueError("SOG image must be an archive-root file")
            with Image.open(BytesIO(archive.read(name))) as image:
                if image.width * image.height > MAX_COUNT + 65536 or image.mode not in ("RGB", "RGBA"):
                    raise ValueError("SOG textures must be bounded RGB/RGBA byte images")
                return np.asarray(image.convert("RGBA"), dtype=np.uint8)

        low_name, high_name = filenames("means", 2)
        low, high = pixels(low_name), pixels(high_name)
        quats = pixels(filenames("quats", 1)[0])
        scales = pixels(filenames("scales", 1)[0])
        dc = pixels(filenames("sh0", 1)[0])
        if count > low.shape[0] * low.shape[1] or any(image.shape != low.shape for image in (high, quats, scales, dc)):
            raise ValueError("SOG property images have inconsistent dimensions")
        property_shape = low.shape
        low, high, quats, scales, dc = [image.reshape(-1, 4)[:count] for image in (low, high, quats, scales, dc)]
        lower = np.asarray(meta["means"].get("mins"), dtype=np.float64)
        upper = np.asarray(meta["means"].get("maxs"), dtype=np.float64)
        if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
                or np.any(upper < lower) or np.max(np.abs([lower, upper])) > 80):
            raise ValueError("Invalid SOG position ranges")
        scale_book, dc_book = codebook(meta["scales"]), codebook(meta["sh0"])
        if np.max(scale_book) > 40:
            raise ValueError("SOG covariance would overflow FP32")
        bands, palette, labels = 0, None, None
        if meta.get("shN") is not None:
            spec = meta["shN"]
            bands = spec.get("bands")
            palette_count = spec.get("count")
            if type(bands) is not int or bands not in (1, 2, 3) or type(palette_count) is not int or not 1 <= palette_count <= 65536:
                raise ValueError("Invalid SOG SH palette shape")
            coefficients = (3, 8, 15)[bands - 1]
            centre_name, labels_name = filenames("shN", 2)
            centroids, label_image = pixels(centre_name), pixels(labels_name)
            if (centroids.shape[1] != 64 * coefficients or centroids.shape[0] * 64 < palette_count
                    or label_image.shape != property_shape):
                raise ValueError("SOG SH images do not match the palette or Gaussian count")
            # Each palette row contains 64 entries, each with consecutive RGB
            # pixels for its 3/8/15 coefficients. Preserve RGB-interleaved SH.
            palette = codebook(spec)[centroids[:, :, :3]].reshape(-1, coefficients, 3)[:palette_count]
            raw_labels = label_image.reshape(-1, 4)[:count].astype(np.uint32)
            labels = raw_labels[:, 0] + (raw_labels[:, 1] << 8)
            if np.any(labels >= palette_count):
                raise ValueError("SOG SH label exceeds the palette")

        geometry = np.zeros((count, 12), dtype="<f4")
        sh = np.zeros((count, 16, 3), dtype="<f4")
        for start in range(0, count, 65536):
            end = min(start + 65536, count)
            codes = low[start:end, :3].astype(np.uint32) | (high[start:end, :3].astype(np.uint32) << 8)
            logarithm = lower + (upper - lower) * (codes / 65535.)
            geometry[start:end, :3] = np.sign(logarithm) * np.expm1(np.abs(logarithm))
            geometry[start:end, 3] = dc[start:end, 3].astype(np.float64) / 255
            linear_scale = np.exp(scale_book[scales[start:end, :3]].astype(np.float64))
            geometry[start:end, 4:10] = covariance(decode_quaternions(quats[start:end]), linear_scale)
            sh[start:end, 0] = dc_book[dc[start:end, :3]]
            if palette is not None:
                sh[start:end, 1:1 + palette.shape[1]] = palette[labels[start:end]]
        if not np.isfinite(geometry).all() or not np.isfinite(sh).all():
            raise ValueError("Decoded SOG contains non-finite GPU data")
        return geometry, sh.reshape(count, 48), bands, meta
