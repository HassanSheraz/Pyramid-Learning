# experiments/

YAML config files. One file per strategy per dataset.

```
cifar10_random.yml   — Random baseline
cifar10_oort.yml     — Oort (PyramidFL's bandit selector)
cifar10_divfl.yml    — DivFL (diversity-augmented Oort)
cifar10_fairfl.yml   — FairFL (fairness-penalized Oort)
base_config.yml      — Shared defaults inherited by all above
```

Run any experiment:
```bash
python src/fl_simulator.py --config experiments/cifar10_divfl.yml
```
