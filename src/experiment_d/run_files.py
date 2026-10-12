"""
Where run files live and how they are named.

    results/<dataset>/generator/<model>/<model>_<precision>_<dataset>_n<N>.jsonl
    results/<dataset>/verifier_dataset/<model>/verifier_<same name>
    results/<dataset>/verifier_output/<model>/verifier_output_<same name>

<model> and <dataset> are the HuggingFace ids with '/' replaced by '_'.
"""
import json
import re
from pathlib import Path

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
