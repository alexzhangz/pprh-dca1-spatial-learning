"""Check the upload snapshot without requiring manuscript data."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use("Agg")
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SKIP = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv", "outputs"}
SOURCE_SUFFIXES = {".py", ".json", ".csv", ".md", ".yml", ".yaml", ".cff", ".txt", ".ipynb"}


def project_files():
    return [p for p in ROOT.rglob("*") if p.is_file()
            and not (set(p.relative_to(ROOT).parts) & SKIP)
            and p.relative_to(ROOT).parts[:2] != ("data", "input")]


def main():
    for required in ("LICENSE", "CITATION.cff", "README.md", "main.py", "requirements.txt",
                     "environment.yml", ".gitignore", ".github/workflows/ci.yml"):
        if not (ROOT / required).is_file():
            raise RuntimeError(f"Missing repository file: {required}")
    if "MIT License" not in (ROOT / "LICENSE").read_text(encoding="utf-8"):
        raise RuntimeError("Expected the selected MIT licence")
    paths = project_files()
    for path in paths:
        if path.suffix in SOURCE_SUFFIXES:
            content = path.read_text(encoding="utf-8")
            if path.suffix == ".ipynb":
                # Check decoded cell text. JSON-escaped code newlines can
                # otherwise resemble a Windows drive prefix.
                content = "\n".join(cell.source for cell in nbformat.read(path, as_version=4).cells)
            if re.search(r"(?<![A-Za-z])[A-Za-z]:[/\\]", content):
                raise RuntimeError(f"Machine-specific drive path in {path.relative_to(ROOT)}")
            if path.suffix == ".py":
                ast.parse(content, filename=str(path))
            if path.suffix == ".json":
                json.loads(content)
    notebooks = sorted((ROOT / "demos").glob("*.ipynb"))
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        defined_functions = 0
        for cell in notebook.cells:
            if cell.cell_type == "code":
                if cell.outputs or cell.execution_count is not None:
                    raise RuntimeError(f"Notebook contains saved outputs: {path.name}")
                tree = ast.parse(cell.source, filename=path.name)
                defined_functions += sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
                if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == "run_figure" for n in ast.walk(tree)):
                    raise RuntimeError(f"Notebook delegates its workflow instead of showing its steps: {path.name}")
        if not defined_functions:
            raise RuntimeError(f"Notebook contains no local functions: {path.name}")
    with (ROOT / "provenance/source_files.csv").open(encoding="utf-8", newline="") as f:
        modules = list(csv.DictReader(f))
    for row in modules:
        path = ROOT / row["release_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["release_file_sha256"]:
            raise RuntimeError(f"Collected module differs from its release hash: {row['module']}")
        importlib.import_module(row["module"])
    for module in ("paper", "paper_io"):
        importlib.import_module(module)
    with (ROOT / "data/required_files.csv").open(encoding="utf-8", newline="") as f:
        inputs = list(csv.DictReader(f))
    for row in inputs:
        p = PurePosixPath(row["path"])
        if p.is_absolute() or ".." in p.parts or "\\" in row["path"] or ":" in row["path"]:
            raise RuntimeError(f"Nonportable source-data path: {row['path']}")
    print(json.dumps({"python_syntax": "passed", "portable_paths": "passed",
                      "source_imports_and_hashes": len(modules),
                      "clean_notebooks": len(notebooks), "input_manifest_rows": len(inputs)}, indent=2))


if __name__ == "__main__":
    main()
