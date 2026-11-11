from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy

from gram.config import (
    DATA_ROOT,
    DATASETS,
    DEFAULT_GRAM_VARIANT,
    GRAM,
    MODELS,
    RESULTS_ROOT,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GRAM mislabel detection with simulated expert verification."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--method", required=True, choices=("ours",))
    parser.add_argument(
        "--gram_variant", choices=(DEFAULT_GRAM_VARIANT,), default=DEFAULT_GRAM_VARIANT
    )
    parser.add_argument(
        "--noise_type",
        choices=("symmetric", "pairflip", "instance"),
        default="instance",
    )
    parser.add_argument("--rho", type=float, choices=(0.2, 0.4), default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output = (
        RESULTS_ROOT
        / args.dataset
        / args.noise_type
        / f"rho{args.rho:g}"
        / f"seed{args.seed}"
    )
    files = ("predictions.csv", "queries.csv", "config.json")
    if not args.overwrite and output.exists() and any(output.iterdir()):
        try:
            complete = (
                json.loads((output / "config.json").read_text()).get("status")
                == "complete"
            )
        except (OSError, ValueError):
            complete = False
        if complete and all((output / name).is_file() for name in files):
            print(f"Completed result already exists; skipping: {output}")
            return
        raise FileExistsError(
            f"Output already exists; use --overwrite to replace this run: {output}"
        )
    import numpy as np

    from gram.algorithm import GRAM as GramDetector
    from gram.data import load_data
    from gram.noise import inject_noise
    from gram.training import train_task_model

    output.mkdir(parents=True, exist_ok=True)
    settings = {
        "status": "running",
        "arguments": vars(args),
        "gram": deepcopy(GRAM),
        "model": deepcopy(MODELS[args.dataset]),
        "dataset": deepcopy(DATASETS[args.dataset]),
    }
    config_path = output / "config.json"
    config_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    data = load_data(
        args.dataset, DATA_ROOT, args.seed, settings["dataset"], settings["model"]
    )
    labels, _ = inject_noise(
        data.clean_train_y,
        data.noise_x,
        noise_type=args.noise_type,
        rho=args.rho,
        num_classes=data.num_classes,
        seed=args.seed,
    )
    artifacts = train_task_model(
        data, labels, settings["model"], args.seed, DATA_ROOT / "huggingface"
    )
    detector = GramDetector(artifacts, labels, settings["gram"], args.seed)
    budget = int(np.floor(args.rho * data.sample_count + 0.5))
    query_rows = []
    verified_count = 0
    while verified_count < budget:
        count = min(
            settings["gram"]["verification_batch_size"], budget - verified_count
        )
        selected, probability, informative, acquisition = detector.select(count)
        expert_labels = data.clean_train_y[selected]
        detector.observe(selected, expert_labels)
        for offset, sample_id in enumerate(selected):
            query_rows.append(
                {
                    "batch": detector.batch,
                    "query_order": verified_count + offset + 1,
                    "sample_id": int(sample_id),
                    "noisy_label": int(labels[sample_id]),
                    "verified_label": int(detector.corrected_labels[sample_id]),
                    "mislabel": int(detector.status[sample_id]),
                    "repair_value": float(probability[sample_id]),
                    "informative_value": float(informative[sample_id]),
                    "acquisition_score": float(acquisition[sample_id]),
                }
            )
        verified_count += count
        if detector.batch == 1 or detector.batch % 10 == 0 or verified_count == budget:
            print(f"Verified {verified_count}/{budget} samples", flush=True)
    scores, predicted, threshold = detector.predict()
    write_csv(
        output / "queries.csv",
        (
            "batch",
            "query_order",
            "sample_id",
            "noisy_label",
            "verified_label",
            "mislabel",
            "repair_value",
            "informative_value",
            "acquisition_score",
        ),
        query_rows,
    )
    write_csv(
        output / "predictions.csv",
        (
            "sample_id",
            "noisy_label",
            "score",
            "predicted_issue",
            "verified",
            "corrected_label",
        ),
        (
            {
                "sample_id": index,
                "noisy_label": int(labels[index]),
                "score": float(scores[index]),
                "predicted_issue": int(predicted[index]),
                "verified": int(detector.verified[index]),
                "corrected_label": int(detector.corrected_labels[index]),
            }
            for index in range(data.sample_count)
        ),
    )
    settings.update(
        status="complete",
        sample_count=data.sample_count,
        verification_budget=budget,
        verification_batches=detector.batch,
        kernel_weights=detector.weights.tolist(),
        otsu_threshold=threshold,
    )
    config_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"GRAM results saved to {output}", flush=True)


if __name__ == "__main__":
    main()
