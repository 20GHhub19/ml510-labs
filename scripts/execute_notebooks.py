"""Execute notebooks on the user's local data. No installs or downloads."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", choices=["s4", "s5"], default="s4")
    parser.add_argument("--scope", choices=["core", "all"], default=None,
                        help="S4 defaults to core (00-03); S5 runs all six stages.")
    parser.add_argument("--kernel", default="ml510")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Per-cell timeout in seconds (default: S4 7200; S5 43200 for full-data CPU training).")
    args = parser.parse_args()
    timeout = args.timeout if args.timeout is not None else (7200 if args.lab == "s4" else 43200)
    root = Path(__file__).resolve().parents[1]
    names = [
        "00_start_here.ipynb",
        "01_linear_regression.ipynb",
        "02_trees_and_ensembles.ipynb",
        "03_lagged_forecasting.ipynb",
        "04_sequence_forecasting.ipynb",
        "05_model_capacity.ipynb",
    ]
    scope = args.scope or ("core" if args.lab == "s4" else "all")
    if args.lab == "s5":
        if scope != "all":
            parser.error("S5 runs all six stages; use --lab s5 --scope all.")
        names = ["00_start_here.ipynb", "01_logistic_regression.ipynb", "02_trees_and_ensembles.ipynb",
                 "03_neural_models.ipynb", "04_probabilities.ipynb", "05_review_policy.ipynb"]
    elif scope == "core":
        names = names[:4]
    folder = "s4_regression" if args.lab == "s4" else "s5_classification"
    files = [root / "notebooks" / folder / name for name in names]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        parser.error(
            f"Missing {args.lab.upper()} notebooks: " + ", ".join(missing) + ". "
            f"Download {folder} from Moodle, extract it if zipped, and copy the folder "
            f"into notebooks/ so notebooks/{folder}/00_start_here.ipynb exists. "
            "See notebooks/README.md."
        )
    out = root / "notebooks" / f"{args.lab}_corrections"
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
    previous_run = None
    for phase, path in enumerate(files):
        print("Executing", path.name, flush=True)
        book = nbformat.read(path, as_version=4)
        environment = dict(os.environ)
        environment.pop("ML510_S5_SOURCE_RUN", None)
        if args.lab == "s5":
            if phase >= 3:
                environment["ML510_S5_SOURCE_RUN"] = previous_run
            # Read the run created by this kernel, never a directory named "latest".
            book.cells.append(nbformat.v4.new_code_cell("import json\nprint(json.dumps(str(run.directory)))"))
        started = perf_counter()
        def checkpoint(**kwargs):
            nbformat.write(book, out / path.name)
        client = NotebookClient(book, kernel_name=args.kernel, timeout=timeout,
                                resources={"metadata": {"path": str(root)}}, on_cell_executed=checkpoint)
        client.execute(env=environment)
        if args.lab == "s5":
            capture = book.cells.pop()
            text = "".join(o.get("text", "") for o in capture.outputs if o.output_type == "stream").strip()
            previous_run = json.loads(text)
            book.metadata["artifact_run"] = str(Path(previous_run).relative_to(root).as_posix())
            if phase >= 3:
                book.metadata["source_run"] = environment["ML510_S5_SOURCE_RUN"]
        book.metadata["execution_seconds"] = perf_counter() - started
        nbformat.write(book, out / path.name)
        print("Saved", out / path.name, "seconds:", round(book.metadata["execution_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
