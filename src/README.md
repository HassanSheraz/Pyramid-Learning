# src/

All original capstone source code. Nothing here modifies `third_party/`.

| File / Folder | Purpose |
|---|---|
| `data_partition.py` | CIFAR-10 Dirichlet non-IID partitioner |
| `visualize_partition.py` | Plot class distribution per client |
| `client_model.py` | Simple CNN for CIFAR-10 + local training function |
| `fl_simulator.py` | Main FL simulation loop (strategy-agnostic) |
| `metrics_logger.py` | Saves training_perf.pkl and selection_log.pkl |
| `plot_results.py` | All comparison figures (accuracy, fairness, coverage) |
| `ablation.py` | Hyperparameter sweep for DivFL diversity_weight |
| `strategies/` | One file per selection strategy |
| `requirements.txt` | Python dependencies |
