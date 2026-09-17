"""Execute notebooks on the user's local data. No installs or downloads."""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["core", "all"], default="core",
                        help="core: notebooks 00-03; all: all six notebooks, including TensorFlow.")
    parser.add_argument("--kernel", default="ml510")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Per-cell timeout in seconds (default: 7200, allowing CPU bagging fits).")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = [
        "00_start_here.ipynb",
        "01_linear_regression.ipynb",
        "02_trees_and_ensembles.ipynb",
        "03_lagged_forecasting.ipynb",
        "04_sequence_forecasting.ipynb",
        "05_model_capacity.ipynb",
    ]
    if args.scope == "core":
        names = names[:4]
    files = [root / "notebooks" / "s4_regression" / name for name in names]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        parser.error(
            "Missing S4 notebooks: " + ", ".join(missing) + ". "
            "Download s4_regression from Moodle, extract it if zipped, and copy the folder "
            "into notebooks/ so notebooks/s4_regression/00_start_here.ipynb exists. "
            "See notebooks/README.md."
        )
    out = root / "notebooks" / "s4_corrections"
    for path in files:
        destination = out / path.name
        if destination.exists():
            existing = nbformat.read(destination, as_version=4)
            if existing.metadata.get("instructor_corrections") or any(
                cell.metadata.get("instructor_note") for cell in existing.cells
            ):
                parser.error(
                    f"Preserve or rename the annotated instructor copy before regenerating: {destination}"
                )
    out.mkdir(parents=True, exist_ok=True)
    for path in files:
        print("Executing", path.name, flush=True)
        book = nbformat.read(path, as_version=4)
        client = NotebookClient(book, kernel_name=args.kernel, timeout=args.timeout,
                                resources={"metadata": {"path": str(root)}})
        client.execute()
        nbformat.write(book, out / path.name)
        print("Saved", out / path.name, flush=True)


if __name__ == "__main__":
    main()
