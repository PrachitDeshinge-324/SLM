"""
Export: every number on the Phase 1 and Phase 2 web pages, from the run files in results/.

Re-scores every run with the current code (see src/experiment_d/web_export.py) and writes
web/assets/data/results.js. Run it again after any new or changed run file; the pages need no edits.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.web_export import build_web_data, write_web_data


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Export results for the project web pages")
    parser.add_argument("--results_dir", "--results-dir", dest="results_dir", type=Path, default=root / "results")
    parser.add_argument("--output", type=Path, default=root / "web" / "assets" / "data" / "results.js")
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        sys.exit(f"No results directory at {args.results_dir}")

    data = build_web_data(args.results_dir)
    if not data["generator"] and not data["verifier"]:
        sys.exit(f"No run files found under {args.results_dir}")

    path = write_web_data(data, args.output)
    print(f"{len(data['generator'])} generator runs and {len(data['verifier'])} verifier rows -> {path}")
    for run in data["verifier_left_out"]:
        print(f"Left out, {run['n_questions']} of {data['verifier_questions']} complete questions: {run['file']}")


if __name__ == "__main__":
    main()
