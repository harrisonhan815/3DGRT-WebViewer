#!/usr/bin/env python3
"""Serve the WebGL2 viewer on 0.0.0.0:55505 by default.

Remote clients can browse and download alignment corrections. Server-side
alignment writes retain the existing loopback-only, same-origin contract.
GitHub Pages remains static. Uses only the Python standard library.
"""

import argparse
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
from urllib.parse import quote, unquote, urlsplit


ALIGNMENT_SCHEMA_V1 = "fullcircle-viewer-alignment-v1"
ALIGNMENT_SCHEMA_V2 = "fullcircle-viewer-alignment-v2"
ALIGNMENT_SCHEMA = ALIGNMENT_SCHEMA_V2
ALIGNMENT_SCHEMAS = (ALIGNMENT_SCHEMA_V1, ALIGNMENT_SCHEMA_V2)
MODEL_SCHEMA = "fullcircle-webgpu-reference-v1"
MODEL_SCHEMAS = (MODEL_SCHEMA, "fullcircle-webgl2-ewa-v1")
MAX_BODY_BYTES = 16 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
DEFAULT_DIRECTORY = Path(__file__).resolve().parent


class RequestError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def strict_json(data):
    return json.loads(data, parse_constant=_invalid_constant, object_pairs_hook=_object_without_duplicates)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _vector(value, label):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three finite numbers")
    return [_number(component, label) for component in value]


def _normalize(vector):
    length = math.hypot(*vector)
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("Reference camera axes must be nonzero and finite")
    return [component / length for component in vector]


def _rotate(vector, axis, angle):
    axis = _normalize(axis)
    cosine, sine = math.cos(angle), math.sin(angle)
    dot = sum(a * b for a, b in zip(axis, vector))
    cross = [axis[1] * vector[2] - axis[2] * vector[1],
             axis[2] * vector[0] - axis[0] * vector[2],
             axis[0] * vector[1] - axis[1] * vector[0]]
    return [vector[i] * cosine + cross[i] * sine + axis[i] * dot * (1 - cosine) for i in range(3)]


def _validate_rigid_frame(alignment):
    """V2 columns are frame X/Y/Z axes in original model space; column Y is up."""
    origin = _vector(alignment["origin"], "origin")
    if any(abs(component) >= 1e12 for component in origin):
        raise ValueError("origin components must have absolute values below 1e12")
    rotation = alignment["rotation"]
    if not isinstance(rotation, list) or len(rotation) != 3:
        raise ValueError("rotation must be a proper 3x3 rotation matrix")
    rows = [_vector(row, "rotation row") for row in rotation]
    columns = [[rows[row][column] for row in range(3)] for column in range(3)]
    # Check lengths before dot products so finite-but-huge inputs cannot create
    # NaNs through overflowing multiplications and evade a comparison.
    for column in columns:
        length = math.hypot(*column)
        if not math.isfinite(length) or abs(length - 1.0) > 1e-4:
            raise ValueError("rotation columns must be unit vectors")
    for first, second in ((0, 1), (0, 2), (1, 2)):
        dot = sum(a * b for a, b in zip(columns[first], columns[second]))
        if abs(dot) > 1e-4:
            raise ValueError("rotation columns must be orthogonal")
    x, y, z = columns
    cross_xy = [x[1] * y[2] - x[2] * y[1],
                x[2] * y[0] - x[0] * y[2],
                x[0] * y[1] - x[1] * y[0]]
    determinant = sum(a * b for a, b in zip(cross_xy, z))
    if abs(determinant - 1.0) > 1e-4:
        raise ValueError("rotation must have determinant +1 (reflections are not supported)")


def validate_alignment(alignment, manifest):
    """Validate shared identity, then legacy scene-up or the V2 rigid frame."""
    if not isinstance(alignment, dict):
        raise ValueError("Alignment must contain exactly the supported schema fields")
    schema = alignment.get("schema")
    if schema not in ALIGNMENT_SCHEMAS:
        raise ValueError("Unsupported alignment schema")
    common = {"schema", "model_id", "source", "gaussian_count", "model_bounds", "enabled"}
    fields = {"up", "angles_deg", "reference_camera"} if schema == ALIGNMENT_SCHEMA_V1 else {"origin", "rotation"}
    if set(alignment) != common | fields:
        raise ValueError("Alignment must contain exactly the supported schema fields")
    if not isinstance(manifest, dict) or manifest.get("schema") not in MODEL_SCHEMAS:
        raise ValueError("Unsupported model manifest schema")
    model_id = alignment["model_id"]
    if not isinstance(model_id, str) or not model_id.strip() or len(model_id) > 200 or any(ord(c) < 32 for c in model_id):
        raise ValueError("model_id must be a nonempty display identifier")
    if not isinstance(alignment["source"], str) or alignment["source"] != manifest.get("source"):
        raise ValueError("Alignment source does not match this model")
    count = alignment["gaussian_count"]
    expected_count = manifest.get("gaussian_count")
    if type(count) is not int or count < 1 or type(expected_count) is not int or count != expected_count:
        raise ValueError("Alignment Gaussian count does not match this model")
    bounds = alignment["model_bounds"]
    expected_bounds = manifest.get("bounds")
    if not isinstance(bounds, dict) or set(bounds) != {"min", "max"} or not isinstance(expected_bounds, dict):
        raise ValueError("Alignment must contain model bounds")
    for key in ("min", "max"):
        actual = _vector(bounds[key], f"model_bounds.{key}")
        expected = _vector(expected_bounds.get(key), f"manifest.bounds.{key}")
        if any(abs(a - b) > 1e-6 * max(1.0, abs(b)) for a, b in zip(actual, expected)):
            raise ValueError("Alignment bounds do not match this model")
    if type(alignment["enabled"]) is not bool:
        raise ValueError("enabled must be a boolean")
    if schema == ALIGNMENT_SCHEMA_V2:
        _validate_rigid_frame(alignment)
        return alignment
    # Keep V1's fixed-reference roll/pitch relation unchanged. V2 deliberately
    # has no up/angles/reference-camera fields and needs no camera metadata.
    up = _vector(alignment["up"], "up")
    if abs(math.hypot(*up) - 1.0) > 1e-4:
        raise ValueError("up must be a unit vector")
    angles = alignment["angles_deg"]
    if not isinstance(angles, dict) or set(angles) != {"roll", "pitch"}:
        raise ValueError("angles_deg must contain roll and pitch")
    roll = _number(angles["roll"], "roll")
    pitch = _number(angles["pitch"], "pitch")
    if not -180 <= roll <= 180 or not -89 <= pitch <= 89:
        raise ValueError("Roll or pitch is outside its supported range")
    if type(alignment["reference_camera"]) is not int or alignment["reference_camera"] != 0:
        raise ValueError("reference_camera must be 0")
    cameras = manifest.get("cameras")
    if not isinstance(cameras, list) or not cameras or not isinstance(cameras[0], dict):
        raise ValueError("Model manifest has no reference camera")
    rotation = cameras[0].get("rotation")
    if not isinstance(rotation, list) or len(rotation) != 3:
        raise ValueError("Reference camera rotation must be a 3x3 matrix")
    rows = [_vector(row, "reference camera rotation") for row in rotation]
    right = [row[0] for row in rows]
    reference_up = _normalize([-row[1] for row in rows])
    forward = [row[2] for row in rows]
    expected_up = _normalize(_rotate(_rotate(reference_up, forward, math.radians(roll)), right, math.radians(pitch)))
    if any(abs(a - b) > 1e-4 for a, b in zip(up, expected_up)):
        raise ValueError("up does not match angles_deg and the reference camera")
    return alignment


def _relative_manifest(model):
    if not isinstance(model, str) or not model or len(model) > 2048:
        raise RequestError(400, "model must be a root-relative manifest path")
    split = urlsplit(model)
    if split.scheme or split.netloc or split.query or split.fragment:
        raise RequestError(400, "Model URLs, queries and fragments are not accepted")
    decoded = unquote(split.path, errors="strict")
    if decoded.startswith("/") or "\\" in decoded or "\x00" in decoded or "%" in decoded:
        raise RequestError(400, "model must be a root-relative manifest path")
    parts = decoded.split("/")
    if len(parts) < 2 or any(part in ("", ".", "..") for part in parts) or parts[-1] != "manifest.json":
        raise RequestError(400, "model must name an existing model directory's manifest.json")
    return Path(*parts)


def save_alignment(directory, model, alignment):
    relative = _relative_manifest(model)
    root = Path(directory).resolve(strict=True)
    try:
        manifest_path = (root / relative).resolve(strict=True)
        model_directory = (root / relative.parent).resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise RequestError(404, "Model manifest was not found") from error
    if not manifest_path.is_relative_to(root) or not model_directory.is_relative_to(root):
        raise RequestError(403, "Model path escapes the served directory")
    if not manifest_path.is_file() or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise RequestError(400, "Invalid or oversized model manifest")
    try:
        manifest = strict_json(manifest_path.read_text(encoding="utf-8"))
        validated = validate_alignment(alignment, manifest)
    except (UnicodeError, ValueError, KeyError, TypeError) as error:
        raise RequestError(400, str(error)) from error
    target = model_directory / "alignment.json"
    # Atomic replace does not follow a symlink, but explicitly refusing one
    # avoids overwriting a link the user may have intentionally created.
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise RequestError(403, "alignment.json must be a regular file")
    payload = (json.dumps(validated, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".alignment-", suffix=".tmp", dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(relative.parent / "alignment.json")


class ViewerHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript",
                      ".wgsl": "text/plain", ".json": "application/json",
                      ".bin": "application/octet-stream", ".wasm": "application/wasm"}

    def _json_response(self, status, payload):
        data = (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _host(self):
        values = self.headers.get_all("Host", [])
        if len(values) != 1:
            raise RequestError(403, "A single Host header is required")
        if any(ord(character) <= 32 or ord(character) == 127 for character in values[0]):
            raise RequestError(403, "Invalid Host header")
        try:
            host = urlsplit("//" + values[0])
            port = host.port if host.port is not None else 80
        except ValueError as error:
            raise RequestError(403, "Invalid Host header") from error
        if (not host.hostname or host.username is not None or host.password is not None
                or host.path or host.query or host.fragment or not 1 <= port <= 65535):
            raise RequestError(403, "Invalid Host header")
        if not self.server.public_read and (host.hostname not in self.server.allowed_hosts or port != self.server.server_port):
            raise RequestError(403, "Host must be the local viewer origin")
        return host.hostname, port

    def _client_is_loopback(self):
        address = ipaddress.ip_address(self.client_address[0])
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        return address.is_loopback

    def _can_write_alignment(self, host, port):
        # Check the actual socket peer, not forwarded headers. A public Host
        # also stays read-only even if a local reverse proxy connects to us.
        return (self._client_is_loopback() and host in self.server.allowed_hosts
                and port == self.server.server_port)

    def _same_origin(self):
        host, port = self._host()
        if not self._can_write_alignment(host, port):
            raise RequestError(403, "Remote viewers must download alignment.json; server writes require localhost")
        origins = self.headers.get_all("Origin", [])
        if len(origins) != 1:
            raise RequestError(403, "A same-origin Origin header is required")
        try:
            origin = urlsplit(origins[0])
            origin_port = origin.port if origin.port is not None else 80
        except ValueError as error:
            raise RequestError(403, "Invalid Origin header") from error
        if (origin.scheme != "http" or origin.hostname != host or origin_port != port
                or origin.username is not None or origin.password is not None
                or origin.path or origin.query or origin.fragment):
            raise RequestError(403, "Only same-origin saves are accepted")

    def _request_path(self):
        split = urlsplit(self.path)
        if split.scheme or split.netloc:
            raise RequestError(400, "Absolute request URLs are not accepted")
        return split

    def _serve_index(self, path):
        # Set local-service configuration in the served response, not the
        # checked-in HTML: GitHub Pages continues to use models.json. This
        # also works when a remote browser connects to this server's public IP.
        content = path.read_text(encoding="utf-8")
        setting = f'<meta name="viewer-model-catalog" content="{escape(self.server.model_catalog, quote=True)}">'
        content, inserted = re.subn(r'(<head\b[^>]*>)', lambda match: match[0] + '\n' + setting,
                                    content, count=1, flags=re.IGNORECASE)
        if not inserted:
            content += '\n' + setting
        data = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _get(self, head=False):
        try:
            host, port = self._host()
            split = self._request_path()
            if split.path == "/__viewer__/capabilities":
                self._json_response(200, {"alignment_write": self._can_write_alignment(host, port), "alignment_schema": ALIGNMENT_SCHEMA,
                                          "alignment_schemas": list(ALIGNMENT_SCHEMAS)})
                return
            if split.path.startswith("/__viewer__/"):
                raise RequestError(404, "Unknown viewer endpoint")
            decoded = unquote(split.path, errors="strict")
            if "\\" in decoded or "\x00" in decoded or ".." in decoded.split("/"):
                raise RequestError(403, "Path traversal is not accepted")
            if any(part.startswith(".") for part in decoded.split("/") if part):
                raise RequestError(403, "Hidden paths are not served")
            destination = (self.server.viewer_directory / decoded.lstrip("/")).resolve()
            if not destination.is_relative_to(self.server.viewer_directory):
                raise RequestError(403, "Path escapes the served directory")
            if destination.is_dir():
                has_index = False
                for name in ("index.html", "index.htm"):
                    index = destination / name
                    if index.is_file():
                        if not index.resolve().is_relative_to(self.server.viewer_directory):
                            raise RequestError(403, "Directory index escapes the served directory")
                        has_index = True
                        break
                if not has_index:
                    raise RequestError(404, "Directory listing is disabled")
                # Let SimpleHTTPRequestHandler keep its normal /dir -> /dir/
                # redirect, so relative model URLs retain the correct base.
                if split.path.endswith("/"):
                    self._serve_index(index)
                    return
            elif destination.name in ("index.html", "index.htm") and destination.is_file():
                self._serve_index(destination)
                return
            if destination.name == "alignment.json" and "If-Modified-Since" in self.headers:
                del self.headers["If-Modified-Since"]
            super().do_HEAD() if head else super().do_GET()
        except RequestError as error:
            self._json_response(error.status, {"error": str(error)})
        except (UnicodeError, ValueError, RuntimeError):
            self._json_response(400, {"error": "Invalid request path"})

    def do_GET(self):
        self._get()

    def do_HEAD(self):
        self._get(head=True)

    def end_headers(self):
        if self.path.split("?", 1)[0].endswith("/alignment.json"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        # Close rejected / completed POSTs rather than retaining unread body
        # bytes on a connection after a Content-Length or origin error.
        self.close_connection = True
        try:
            self._same_origin()
            split = self._request_path()
            if split.path != "/__viewer__/alignment" or split.query or split.fragment:
                raise RequestError(404, "Unknown viewer endpoint")
            types = self.headers.get_all("Content-Type", [])
            if len(types) != 1 or types[0].split(";", 1)[0].strip().lower() != "application/json":
                raise RequestError(415, "Content-Type must be application/json")
            if self.headers.get("Transfer-Encoding") is not None or self.headers.get("Content-Encoding") not in (None, "identity"):
                raise RequestError(400, "Encoded or chunked request bodies are not supported")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1:
                raise RequestError(411, "A single Content-Length header is required")
            if not lengths[0].isdigit() or len(lengths[0]) > 8:
                raise RequestError(400, "Invalid Content-Length")
            length = int(lengths[0])
            if length > MAX_BODY_BYTES:
                raise RequestError(413, "Alignment request exceeds 16 KiB")
            if length < 1:
                raise RequestError(400, "Empty alignment request")
            self.connection.settimeout(5)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise RequestError(400, "Incomplete alignment request")
            body = strict_json(raw.decode("utf-8"))
            if not isinstance(body, dict) or set(body) != {"model", "alignment"}:
                raise RequestError(400, "Request must contain model and alignment")
            with self.server.alignment_lock:
                saved = save_alignment(self.server.viewer_directory, body["model"], body["alignment"])
            self._json_response(200, {"saved": True, "path": saved})
        except RequestError as error:
            self._json_response(error.status, {"error": str(error)})
        except (UnicodeError, ValueError, TypeError, RuntimeError):
            self._json_response(400, {"error": "Invalid alignment JSON or path"})
        except TimeoutError:
            self._json_response(408, {"error": "Timed out reading the alignment request"})
        except PermissionError:
            self._json_response(403, {"error": "Model directory is not writable"})
        except OSError:
            self._json_response(500, {"error": "Could not save alignment.json"})


class ViewerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, directory, host="0.0.0.0", port=55505, catalog="models.local.json"):
        address = "127.0.0.1" if host == "localhost" else host
        try:
            parsed_address = ipaddress.ip_address(address)
        except ValueError as error:
            raise ValueError("--bind must be localhost or a numeric IP address") from error
        self.public_read = not parsed_address.is_loopback
        self.address_family = socket.AF_INET6 if parsed_address.version == 6 else socket.AF_INET
        self.viewer_directory = Path(directory).expanduser().resolve(strict=True)
        if not self.viewer_directory.is_dir():
            raise ValueError("--directory must name an existing viewer directory")
        catalog = str(catalog)
        parsed_catalog = urlsplit(catalog)
        catalog_path = Path(catalog)
        if (not catalog or parsed_catalog.scheme or parsed_catalog.netloc or parsed_catalog.query
                or parsed_catalog.fragment or catalog_path.is_absolute() or "\\" in catalog
                or any(part.startswith(".") for part in catalog_path.parts)
                or catalog_path.suffix.lower() != ".json"
                or not (self.viewer_directory / catalog_path).resolve().is_relative_to(self.viewer_directory)):
            raise ValueError("--catalog must be a JSON path inside --directory (for example models.local.json)")
        self.model_catalog = quote(catalog_path.as_posix(), safe="/")
        self.allowed_hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed_address.is_loopback:
            self.allowed_hosts.add(str(parsed_address))
        self.alignment_lock = threading.Lock()
        super().__init__((address, port), partial(ViewerHandler, directory=str(self.viewer_directory)))


def make_server(directory=DEFAULT_DIRECTORY, host="0.0.0.0", port=55505, catalog="models.local.json"):
    """Create a stopped server; callers own serve_forever/shutdown/close."""
    return ViewerHTTPServer(directory, host, port, catalog)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY, help="Existing static viewer directory (default: 3DGRT-WebViewer)")
    parser.add_argument("--bind", default="0.0.0.0", help="Listen address (default: 0.0.0.0, all IPv4 interfaces)")
    parser.add_argument("--port", type=int, default=55505, help="TCP port (default: 55505)")
    parser.add_argument("--catalog", default="models.local.json", help="Model JSON relative to --directory (default: models.local.json)")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        server = make_server(args.directory, args.bind, args.port, args.catalog)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    host = server.server_address[0]
    authority = f"[{host}]" if ":" in host else host
    print(f"Listening: {authority}:{server.server_port}", flush=True)
    if ipaddress.ip_address(host).is_unspecified:
        local = "[::1]" if ":" in host else "127.0.0.1"
        print(f"Local: http://{local}:{server.server_port}/", flush=True)
        print(f"Remote: http://<server-address>:{server.server_port}/", flush=True)
    else:
        print(f"Viewer: http://{authority}:{server.server_port}/", flush=True)
    print(f"Directory: {server.viewer_directory}", flush=True)
    print(f"Model list: {args.catalog}", flush=True)
    print("Local same-origin saves write alignment.json; remote saves download the file. Ctrl+C stops the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
