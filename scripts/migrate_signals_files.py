"""
Give every signals file written before names carried tags its tagged name (see scripts/run_signals.py).

Looks in results/*/signals/*/ for untagged files (signals_<run name>.jsonl) and in results/*/verifier_variants/*/
for signals files put there by hand, reads each file's header, and moves it to
results/<dataset>/signals/<model>/signals_<tag>_<run name>.jsonl. Never overwrites: a file whose target already
exists is reported and left in place. Prints the plan; moves only with --apply.

python scripts/migrate_signals_files.py --results_dir results [--apply]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.run_files import signals_path
from src.experiment_d.signals import tag_of_header


def plan(results_dir: Path) -> list:
    """(source, target or None, note) for every signals file that is not at its tagged name."""
    out = []
    candidates = sorted(results_dir.glob("*/signals/*/signals_*.jsonl")) + sorted(results_dir.glob("*/verifier_variants/*/signals_*.jsonl"))
    for path in candidates:
        with path.open() as handle:
            header = json.loads(handle.readline())
        if header.get("type") != "run_config" or "model" not in header:
            out.append((path, None, "no run_config header; left alone"))
            continue
        args = (results_dir, header["model"], header["precision"], header["dataset"], header["generator_n_samples"])
        target = signals_path(*args, tag=tag_of_header(header))
        if path.resolve() == target.resolve():
            continue
        if target.exists():
            out.append((path, None, f"target {target.name} already exists; left alone"))
        else:
            out.append((path, target, f"critics {header.get('critics')}, {len(header.get('question_ids', []))} questions"))
    return out


def main():
    parser = argparse.ArgumentParser(description="Move untagged signals files to their tagged names")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--apply", action="store_true", help="Move the files (default: only print the plan)")
    args = parser.parse_args()
    steps = plan(Path(args.results_dir))
    if not steps:
        print("Every signals file already has its tagged name.")
    claimed = set()
    for source, target, note in steps:
        if target is not None and target in claimed:
            print(f"SKIP  {source}: another file maps to {target.name}; left alone")
            continue
        if target is None:
            print(f"SKIP  {source}: {note}")
            continue
        claimed.add(target)
        print(f"{'MOVE' if args.apply else 'PLAN'}  {source}\n   -> {target}  ({note})")
        if args.apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
    if steps and not args.apply:
        print("Nothing moved; rerun with --apply.")


if __name__ == "__main__":
    main()
