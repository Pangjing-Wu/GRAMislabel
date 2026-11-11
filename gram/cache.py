from __future__ import annotations

import json
from pathlib import Path


def load_local_huggingface(name: str, root: Path, config: dict):
    from datasets import Dataset, DatasetDict, concatenate_datasets

    candidates = []
    for info_path in (root / "huggingface").glob("**/dataset_info.json"):
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sources = info.get("download_checksums", {})
        source = config["source"]
        prefixes = (
            f"hf://datasets/{source}@",
            f"https://huggingface.co/datasets/{source}/",
        )
        if not sources or not all((url.startswith(prefixes) for url in sources)):
            continue
        if "subset" in config and info.get("config_name") != config["subset"]:
            continue
        required_splits = {"train", "validation" if name == "qnli" else "test"}
        if not required_splits.issubset(info.get("splits", {})):
            continue
        candidates.append((info_path, info))
    candidates.sort(
        key=lambda item: (item[0].stat().st_mtime_ns, str(item[0])), reverse=True
    )
    for info_path, info in candidates:
        splits = {}
        for split, metadata in info["splits"].items():
            prefix = f"{info['dataset_name']}-{split}"
            single = info_path.parent / f"{prefix}.arrow"
            files = (
                [single]
                if single.is_file()
                else sorted(info_path.parent.glob(f"{prefix}-?????-of-?????.arrow"))
            )
            expected_shards = len(
                metadata.get("shard_lengths") or [metadata["num_examples"]]
            )
            if len(files) != expected_shards:
                break
            shards = [Dataset.from_file(str(path), split=split) for path in files]
            if sum((len(shard) for shard in shards)) != metadata["num_examples"]:
                break
            splits[split] = (
                shards[0] if len(shards) == 1 else concatenate_datasets(shards)
            )
        else:
            return DatasetDict(splits)
    raise FileNotFoundError(f"No local dataset found for {name} under {root}")


def local_openml_path(root: Path, config: dict) -> Path:
    _, name, version = config["source"].split(":", maxsplit=2)
    return root / "openml" / f"{name}-v{version}.joblib"


def load_raw_data(name: str, root: Path, config: dict):
    if config["source"].startswith("openml:"):
        import joblib
        from sklearn.datasets import fetch_openml

        path = local_openml_path(root, config)
        if path.is_file():
            return joblib.load(path)
        _, dataset_name, version = config["source"].split(":", maxsplit=2)
        bunch = fetch_openml(
            name=dataset_name,
            version=int(version),
            data_home=str(root / "openml"),
            as_frame=name == "adult",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bunch, path, compress=3)
        return bunch
    try:
        return load_local_huggingface(name, root, config)
    except FileNotFoundError:
        from datasets import load_dataset

        options = {"cache_dir": str(root / "huggingface")}
        if "data_files" in config:
            return load_dataset("parquet", data_files=config["data_files"], **options)
        if "subset" in config:
            return load_dataset(config["source"], config["subset"], **options)
        return load_dataset(config["source"], **options)
