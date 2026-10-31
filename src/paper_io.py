"""Data loading and exports for the manuscript notebooks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def read_table(data_root, relative):
    path = Path(data_root) / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing manuscript input: {relative}. See data/README.md.")
    return pd.read_csv(path, low_memory=False)


def verify_data(data_root, manifest_file):
    manifest = pd.read_csv(manifest_file)
    results = []
    for row in manifest.itertuples():
        path = Path(data_root) / row.path
        present = path.is_file()
        matched = present and path.stat().st_size == row.bytes
        if matched:
            matched = hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256
        results.append({"path": row.path, "present": present, "sha256_matches": matched})
    return pd.DataFrame(results)


def save_panel(fig, output_root, panel, description, *, old_label=None, formats=("svg", "png"), dpi=300):
    if isinstance(fig, tuple):
        fig = fig[0]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    # Only panel identifiers change when the final manuscript renumbers figures.
    # Data, axes, point coordinates, scales and statistical annotations are retained.
    if old_label is not None:
        for text in fig.texts:
            if text.get_text().strip() in {str(old_label), str(old_label)[-1]}:
                text.set_text(panel.split("_")[-1].upper())
    stem = f"{panel}_{description}"
    paths = []
    for suffix in formats:
        target = output_root / f"{stem}.{suffix}"
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        paths.append(str(target))
    plt.close(fig)
    return paths


def write_tables(tables, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(directory / f"{name}.csv", index=False)


def p_symbol(p):
    if pd.isna(p):
        return "n.e."
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def write_run_record(path, *, figure, refit, exports):
    record = {"figure": figure, "statistics": "refitted from input tables" if refit else "frozen manuscript estimates", "exports": exports}
    Path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
