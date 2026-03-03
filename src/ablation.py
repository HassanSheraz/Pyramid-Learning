"""
ablation.py

Ablation study: sweep the diversity_weight hyperparameter in DivFL to
find the best value. Runs DivFL with different lambda values and plots
the final accuracy after N rounds for each.

Usage:
    python src/ablation.py --rounds 30
    python src/ablation.py --rounds 30 --alphas 0.1 0.3 0.5 0.7 1.0

This helps me justify why I picked diversity_weight=0.3 in the main experiments.
"""

import argparse
import os
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fl_simulator import run_simulation, load_config


ABLATION_DIR = 'reports/ablation'


def run_ablation(diversity_weights, rounds, base_cfg_path):
    """Run DivFL for each diversity_weight value and record final accuracy."""
    cfg = load_config(base_cfg_path)
    cfg['strategy']   = 'divfl'
    cfg['num_rounds'] = rounds

    results = {}

    for lam in diversity_weights:
        print(f'\n--- diversity_weight = {lam} ---')
        cfg['diversity_weight'] = lam
        cfg['out_dir'] = os.path.join(ABLATION_DIR, f'lambda_{lam}')

        logger = run_simulation(cfg)

        # grab final accuracy
        max_round   = max(logger.training_perf.keys())
        final_acc   = logger.training_perf[max_round]['top_1']
        results[lam] = final_acc
        print(f'  Final accuracy @ round {max_round}: {final_acc:.2f}%')

    return results


def plot_ablation(results, out_path):
    lambdas  = sorted(results.keys())
    accs     = [results[l] for l in lambdas]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambdas, accs, 'o-', color='#E91E63', linewidth=2, markersize=7)

    # mark the best value
    best_lam = lambdas[int(np.argmax(accs))]
    best_acc = max(accs)
    ax.axvline(best_lam, color='gray', linestyle='--', alpha=0.5,
               label=f'Best: λ={best_lam} ({best_acc:.1f}%)')

    ax.set_xlabel('diversity_weight (λ)', fontsize=12)
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
    ax.set_title('DivFL Ablation: Effect of Diversity Weight', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'\nAblation plot saved to: {out_path}')
    print(f'Best diversity_weight: {best_lam}  (accuracy: {best_acc:.2f}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DivFL hyperparameter ablation')
    parser.add_argument('--rounds',  type=int,   default=30,
                        help='Rounds per ablation run (default: 30)')
    parser.add_argument('--weights', type=float, nargs='+',
                        default=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
                        help='diversity_weight values to test')
    parser.add_argument('--config',  type=str,
                        default='experiments/cifar10_divfl.yml',
                        help='Base config to use for ablation')
    args = parser.parse_args()

    print(f'Running DivFL ablation over diversity_weight = {args.weights}')
    print(f'Rounds per run: {args.rounds}\n')

    results = run_ablation(args.weights, args.rounds, args.config)

    out = os.path.join('reports', 'figures', 'ablation_diversity_weight.png')
    plot_ablation(results, out_path=out)
