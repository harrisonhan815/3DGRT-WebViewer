"""Write export files inside the model directory, publishing the manifest last.

Temporary *files* protect existing models during encoding/validation. No sibling
staging directories or extra copies of complete export trees are created.
"""
import json
import os
from pathlib import Path
import tempfile


class ExportFiles:
    def __init__(self, output):
        self.output = Path(output)
        self.pending = {}
        self.scratch_files = []
        self.obsolete = set()
        self.created = False

    def __enter__(self):
        if self.output.is_symlink():
            raise ValueError(f"Refusing symlink output: {self.output}")
        self.created = not self.output.exists()
        self.output.mkdir(parents=True, exist_ok=True)
        return self

    def target(self, name):
        if not name or Path(name).name != name:
            raise ValueError(f"Invalid export filename: {name}")
        path = self.output/name
        if path.is_symlink() or path.is_dir():
            raise ValueError(f"Refusing non-file export target: {path}")
        return path

    def scratch(self, suffix=".tmp"):
        with tempfile.NamedTemporaryFile(prefix=".export-", suffix=suffix, dir=self.output, delete=False) as stream:
            path = Path(stream.name)
        self.scratch_files.append(path)
        return path

    def path(self, name):
        self.target(name)
        if name not in self.pending:
            # Keep .sog/.ply extensions for tools that infer the format from it.
            self.pending[name] = self.scratch(Path(name).suffix or ".tmp")
        return self.pending[name]

    def commit(self, manifest):
        self.path("manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
        # Validate every destination before replacing any existing files.
        for name in self.pending:
            self.target(name)
        for name, temporary in self.pending.items():
            if name != "manifest.json":
                os.replace(temporary, self.output/name)
        os.replace(self.pending["manifest.json"], self.output/"manifest.json")
        for name in self.obsolete-self.pending.keys():
            self.target(name).unlink(missing_ok=True)

    def __exit__(self, *exc):
        for path in self.scratch_files:
            path.unlink(missing_ok=True)
        if self.created and not any(self.output.iterdir()):
            self.output.rmdir()

