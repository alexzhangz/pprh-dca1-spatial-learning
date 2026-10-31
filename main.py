"""Run manuscript figures from the same source-data inputs as the notebooks."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("3", "4", "5", "6", "7", "S4", "S7", "S4-S7", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("PPRH_DATA_DIR", ROOT / "data/input")))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--refit", action="store_true", help="Refit the manuscript statistical models from the input tables.")
    parser.add_argument("--formats", nargs="+", choices=("svg", "png", "pdf", "tiff"), default=("svg", "png"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--check-data", action="store_true", help="Verify required input hashes without running figures.")
    args = parser.parse_args()
    if args.check_data:
        from paper_io import verify_data
        result = verify_data(args.data_root, ROOT / "data/required_files.csv")
        print(result.sha256_matches.value_counts().to_string())
        if not result.sha256_matches.all():
            raise SystemExit("Input data are missing or do not match the manuscript source data.")
        return
    if not args.data_root.is_dir():
        parser.error(
            f"Manuscript input directory does not exist: {args.data_root}. "
            "Supply --data-root with the separate input dataset; see data/README.md. "
            "To check installation without manuscript data, run python scripts/run_demo.py."
        )
    from paper import WORKFLOWS, run_figure
    selected = list(WORKFLOWS) if args.figure == "all" else [args.figure]
    for figure in selected:
        print(f"Running {figure}", flush=True)
        exports = run_figure(figure, args.data_root, args.output_root, refit=args.refit, formats=args.formats, dpi=args.dpi)
        print(f"Completed {figure}: {len(exports)} exports", flush=True)


if __name__ == "__main__":
    main()
