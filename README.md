# FL Client Selection — Capstone Project

Built on top of [PyramidFL](third_party/README.md) (MobiCom '22). The original framework is kept **untouched** inside `third_party/`. All capstone code lives in `src/`.

## Goal
Compare four federated learning client selection strategies on CIFAR-10 with non-IID data:

| Strategy | Description |
|---|---|
| **Random** | Uniform random sampling (baseline) |
| **Oort** | UCB bandit selector from PyramidFL |
| **DivFL** | Oort + diversity bonus (Jensen-Shannon divergence) |
| **FairFL** | Oort + fairness penalty (participation frequency) |

## Folder Structure
```
src/               # All original capstone code
experiments/       # YAML config files for each experiment run
reports/           # Output figures, result tables, report draft
docs/              # Design documents and experiment guide
third_party/       # Unmodified PyramidFL source (do not edit)
```

## Quick Start
```bash
conda activate base
pip install -r src/requirements.txt

# Run all four experiments
python src/fl_simulator.py --config experiments/cifar10_random.yml
python src/fl_simulator.py --config experiments/cifar10_oort.yml
python src/fl_simulator.py --config experiments/cifar10_divfl.yml
python src/fl_simulator.py --config experiments/cifar10_fairfl.yml

# Generate all comparison plots
python src/plot_results.py
```

## Results
See `reports/figures/` after running experiments.
