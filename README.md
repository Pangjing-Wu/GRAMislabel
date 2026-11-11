# GRAM

Implementation of Global Risk-Aware Mislabel Detection for Training
Data Quality Diagnosis. It contains task-model training, trajectory graph kernels,
adaptive kernel weighting, Gaussian posterior inference, and risk-aware verification.

```text
.
├── run.py
├── gram/
│   ├── config.py
│   ├── data.py
│   ├── cache.py
│   ├── models.py
│   ├── training.py
│   ├── noise.py
│   ├── kernels.py
│   ├── inference.py
│   ├── algorithm.py
│   └── __init__.py
├── requirements.txt
└── README.md
```

Use Python 3.10 or 3.11. Run from the repository root:

```bash
pip install -r requirements.txt
python run.py --dataset letter --method ours --noise_type symmetric --rho 0.2 --seed 0
```

For image data:

```bash
python run.py --dataset cifar10 --method ours --gram_variant main_adaptive_mixture --noise_type instance --rho 0.4 --seed 0
```

| Argument | Meaning |
| --- | --- |
| `--dataset` | Required: `cifar10`, `cifar100`, `atis`, `qnli`, `adult`, or `letter`. |
| `--method` | Required: `ours` selects GRAM. |
| `--gram_variant` | Only `main_adaptive_mixture`, the default main algorithm. |
| `--noise_type` | `symmetric`: uniform different class; `pairflip`: next class cyclically; `instance`: feature-dependent corruption. Default: `instance`. |
| `--rho` | Fraction of labels corrupted and total verification budget: `0.2` (default) or `0.4`. |
| `--seed` | Non-negative random seed; default: `0`. |
| `--overwrite` | Replace this run's existing output files. Complete runs otherwise skip; incomplete outputs require this flag. |

Datasets and text-model weights are downloaded automatically into the repository's
`datasets/` directory on first use. CUDA is used when available; otherwise training
runs on CPU.

Algorithm settings are in `gram/config.py`: graph neighbors 50, rank 64 per
trajectory kernel, diffusion time 1, observation variance 0.1, weight
regularization 0.1, initial weights `(1/3, 1/3, 1/3)`, and acquisition
`p + beta * IV` with `beta=1`. Model settings follow the paper: ResNet-18 for
10 epochs, RoBERTa-base for 3 epochs, and an MLP for 50 epochs.
Kernel-weight learning requires at least 30 verified samples and both clean and
mislabeled verification outcomes. The fixed prior mean combines confidence deficit,
early loss, and embedding-neighbor label disagreement.

Verification uses batches of 32, as specified in the paper, with a smaller final
batch if necessary. Kernel weights and the posterior update between batches;
the task model and graphs remain fixed. Clean labels simulate expert responses
only for selected samples. The final binary decisions use an Otsu threshold over
all posterior scores; verified decisions use the expert feedback.

Outputs are saved under `results/<dataset>/<noise_type>/rho<rho>/seed<seed>/`:

- `predictions.csv`: sample ID, original noisy label, posterior mislabel score,
  binary decision, verification status, and label after expert repair. Unverified
  labels keep their original noisy values.
- `queries.csv`: verification order, expert labels, and acquisition values.
- `config.json`: arguments, effective settings, final kernel weights, and threshold.
