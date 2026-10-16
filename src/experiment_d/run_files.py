"""
Where run files live and how they are named.

    results/<dataset>/generator/<model>/<model>_<precision>_<dataset>_n<N>.jsonl
    results/<dataset>/verifier_dataset/<model>/verifier_<same name>
    results/<dataset>/verifier_output/<model>/verifier_output_<same name>

<model> and <dataset> are the HuggingFace ids with '/' replaced by '_'.

Self-training keeps one such tree per round, plus that round's adapter:

    results/self_training/<experiment>/round<k>/<the layout above>
    results/self_training/<experiment>/round<k>/adapters/<model>_<precision>/
"""
import hashlib
import json
import re
from pathlib import Path

SELF_TRAINING_DIR = "self_training"

RUN_NAME_PATTERN = re.compile(
    r"^(?:verifier_output_|verifier_)?(?P<model>.+)_(?P<precision>16bit|8bit|4bit)_(?P<dataset>.+)_n(?P<n_samples>\d+)\.jsonl$"
)


def safe_name(value: str) -> str:
    """HuggingFace ids such as 'ChilleD/SVAMP' contain slashes; file and folder names cannot."""
    return value.replace("/", "_")


def parse_n_samples(value) -> int:
    """Accepts 8, '8' or the older 'n8' spelling."""
    return int(str(value).lstrip("n"))


def run_name(model: str, precision: str, dataset: str, n_samples) -> str:
    return f"{safe_name(model)}_{precision}_{safe_name(dataset)}_n{parse_n_samples(n_samples)}.jsonl"


def generator_run_path(results_dir, model: str, precision: str, dataset: str, n_samples) -> Path:
    return (Path(results_dir) / safe_name(dataset) / "generator" / safe_name(model)
            / run_name(model, precision, dataset, n_samples))


def verifier_dataset_path(results_dir, model: str, precision: str, dataset: str, n_samples) -> Path:
    return (Path(results_dir) / safe_name(dataset) / "verifier_dataset" / safe_name(model)
            / f"verifier_{run_name(model, precision, dataset, n_samples)}")


def verifier_output_path(results_dir, model: str, precision: str, dataset: str, n_samples) -> Path:
    return (Path(results_dir) / safe_name(dataset) / "verifier_output" / safe_name(model)
            / f"verifier_output_{run_name(model, precision, dataset, n_samples)}")


def generator_path_for_verifier_output(verifier_output_file) -> Path:
    """The generator run whose traces a verifier output file judged."""
    path = Path(verifier_output_file)
    return path.parent.parent.parent / "generator" / path.parent.name / path.name.replace("verifier_output_", "", 1)


def token_signals_path(results_dir, model: str, precision: str, dataset: str, n_samples, experiment=None, round_index=None) -> Path:
    """
    Token-level signals of a generator run's stored solutions (see token_signals.py), beside the run.
    With `experiment` and `round_index` the run is the one of that self-training round.
    """
    root = self_training_round_dir(results_dir, experiment, round_index) if experiment else Path(results_dir)
    run = generator_run_path(root, model, precision, dataset, n_samples)
    return run.parent.parent.parent / "token_signals" / run.parent.name / f"token_signals_{run.name}"


def parse_run_name(file_name: str):
    """
    {'model', 'precision', 'dataset', 'n_samples'} from a run file name, or None if it is not one.
    Model and dataset come back in their file-safe spelling.
    """
    match = RUN_NAME_PATTERN.match(Path(file_name).name)
    if not match:
        return None
    parts = match.groupdict()
    parts["n_samples"] = int(parts["n_samples"])
    return parts


def is_legacy_file(path) -> bool:
    """Backup copies of superseded runs are never analysed."""
    return "_backup" in Path(path).name


def self_training_round_dir(results_dir, experiment: str, round_index) -> Path:
    """Results root of one self-training round (round 0 is the untrained model)."""
    return Path(results_dir) / SELF_TRAINING_DIR / experiment / f"round{int(round_index)}"


def adapter_dir(results_dir, experiment: str, round_index, model: str, precision: str) -> Path:
    """Where the adapter trained in a round is saved, next to that round's run files."""
    return self_training_round_dir(results_dir, experiment, round_index) / "adapters" / f"{safe_name(model)}_{precision}"


def is_self_training_file(path) -> bool:
    """Self-training rounds reuse the run names of the main runs, so the main reports leave them out."""
    return SELF_TRAINING_DIR in Path(path).parts


def adapter_fingerprint(path) -> str:
    """Hash of an adapter's weights, so a run file records exactly which adapter produced it."""
    weights = [file for file in sorted(Path(path).glob("adapter_model.*")) if file.is_file()]
    if not weights:
        raise FileNotFoundError(f"No adapter weights found in {path}.")
    digest = hashlib.sha256()
    for file in weights:
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def read_question_records(path) -> list:
    """The per-question records of a generator run file, in file order (the header is skipped)."""
    records = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "raw_samples" in record:
                records.append(record)
    return records


def read_run_config(path) -> dict:
    """The header of a generator run file ({} when the file or its header is missing)."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open() as handle:
        for line in handle:
            if '"run_config"' not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "run_config":
                return obj
    return {}


def token_limit(run_config: dict, dataset: str):
    """Token limit of a generator run: one key in newer headers, one per dataset in older ones, absent in the oldest."""
    for key in ("max_new_tokens", f"max_new_tokens_{dataset}", f"max_new_tokens_{dataset.lower()}"):
        if run_config.get(key) is not None:
            return int(run_config[key])
    return None
