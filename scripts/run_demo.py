"""Exercise the included decoder on synthetic data, without manuscript inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from maze_analysis.bayesian_position import GaussianBinDecoder


def run(output_root: Path):
    """Show the original Gaussian-bin API on clearly labelled synthetic data."""
    rng = np.random.default_rng(20261031)
    bins = np.arange(98)
    centers = np.linspace(0, 97, 50)
    tuning = 3 * np.exp(-0.5 * ((bins[:, None] - centers[None, :]) / 6) ** 2)
    train_bins = np.tile(bins, 20)
    train = tuning[train_bins] + rng.normal(0, 0.35, (len(train_bins), 50))
    query = tuning + rng.normal(0, 0.35, tuning.shape)
    model = GaussianBinDecoder.fit(train, train_bins)
    posterior = model.predict_proba(query)
    np.testing.assert_allclose(posterior.sum(axis=1), 1, atol=1e-12)
    predicted = posterior.argmax(axis=1)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"synthetic_true_bin": bins, "predicted_bin": predicted}).to_csv(
        output_root / "synthetic_predictions.csv", index=False
    )
    np.savez_compressed(output_root / "synthetic_posterior.npz", posterior=posterior)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    artist = ax.imshow(posterior.T, origin="lower", aspect="auto", cmap="viridis")
    ax.set(xlabel="Synthetic true spatial bin", ylabel="Decoded spatial bin",
           title="Synthetic demo — not manuscript data")
    fig.colorbar(artist, ax=ax, label="Posterior probability")
    fig.savefig(output_root / "synthetic_posterior.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    record = {"data_type": "synthetic", "seed": 20261031,
              "manuscript_result": False, "posterior_shape": list(posterior.shape),
              "posterior_row_sums_verified": True}
    (output_root / "demo_run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Synthetic example passed. Outputs: {output_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/demo")
    run(parser.parse_args().output_root)
