"""
plot_results.py

Generates all comparison figures for the capstone report.
Reads the training_perf.pkl and selection_log.pkl files saved by fl_simulator.py
and produces 4 plots:

  1. Accuracy vs. Round          — did my strategies converge faster?
  2. Accuracy vs. Simulated Clock — accounting for system speed differences
  3. Selection frequency histogram — are certain clients dominating?
  4. Gini coefficient over rounds  — is participation becoming more fair over time?

Run this after all four experiments are done:
    python src/plot_results.py
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


STRATEGIES = ['random', 'oort', 'divfl', 'fairfl']
LABELS     = ['Random', 'Oort', 'DivFL (ours)', 'FairFL (ours)']
COLORS     = ['#888888', '#2196F3', '#E91E63', '#4CAF50']
LINESTYLES = ['--', '-.', '-', '-']

REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')


def load_results(strategy):
    """Load training_perf and selection_log for a strategy."""
    run_dir = os.path.join(REPORTS_DIR, f'run_{strategy}')
    perf_path = os.path.join(run_dir, 'training_perf.pkl')
    sel_path  = os.path.join(run_dir, 'selection_log.pkl')

    if not os.path.exists(perf_path):
        print(f"  [!] Missing results for '{strategy}' — run the experiment first.")
        return None, None

    with open(perf_path, 'rb') as f:
        perf = pickle.load(f)
    with open(sel_path, 'rb') as f:
        sel_log = pickle.load(f)

    return perf, sel_log


def gini(counts):
    """
    Gini coefficient for a list of counts.
    0 = perfectly equal, 1 = one client selected every time.
    Formula: G = (2 * sum(rank * value)) / (n * sum(value)) - (n+1)/n
    """
    counts = np.array(sorted(counts), dtype=float)
    n = len(counts)
    if counts.sum() == 0:
        return 0.0
    cumulative = np.cumsum(counts)
    # standard Gini formula
    return (2 * np.dot(np.arange(1, n+1), counts) / (n * counts.sum())) - (n + 1) / n


# ---- Plot 1: Accuracy vs. Round ----------------------------------------

def plot_accuracy_vs_round(all_results):
    fig, ax = plt.subplots(figsize=(8, 5))

    for (strategy, label, color, ls), (perf, _) in zip(
        zip(STRATEGIES, LABELS, COLORS, LINESTYLES), all_results
    ):
        if perf is None:
            continue
        rounds   = sorted(perf.keys())
        accuracy = [perf[r]['top_1'] for r in rounds]
        ax.plot(rounds, accuracy, label=label, color=color,
                linestyle=ls, linewidth=2)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy vs. Training Round', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = os.path.join(FIGURES_DIR, 'accuracy_vs_round.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


# ---- Plot 2: Accuracy vs. Simulated Clock --------------------------------

def plot_accuracy_vs_clock(all_results):
    fig, ax = plt.subplots(figsize=(8, 5))

    for (strategy, label, color, ls), (perf, _) in zip(
        zip(STRATEGIES, LABELS, COLORS, LINESTYLES), all_results
    ):
        if perf is None:
            continue
        rounds   = sorted(perf.keys())
        # convert clock from seconds to hours for readability
        clocks   = [perf[r]['clock'] / 3600.0 for r in rounds]
        accuracy = [perf[r]['top_1']  for r in rounds]
        ax.plot(clocks, accuracy, label=label, color=color,
                linestyle=ls, linewidth=2)

    ax.set_xlabel('Simulated Time (hours)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy vs. Simulated Wall-Clock Time', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = os.path.join(FIGURES_DIR, 'accuracy_vs_clock.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


# ---- Plot 3: Client selection frequency histogram -----------------------

def plot_selection_frequency(all_results, num_clients=100):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    for i, ((strategy, label, color, _), (_, sel_log)) in enumerate(zip(
        zip(STRATEGIES, LABELS, COLORS, LINESTYLES), all_results
    )):
        if sel_log is None:
            continue

        # count how many times each client was selected
        counts = np.zeros(num_clients)
        for selected in sel_log.values():
            for cid in selected:
                counts[int(cid)] += 1

        gini_val = gini(counts)
        axes[i].bar(range(num_clients), counts, color=color, alpha=0.7, width=1.0)
        axes[i].set_title(f'{label}  (Gini = {gini_val:.3f})', fontsize=11)
        axes[i].set_xlabel('Client ID', fontsize=9)
        axes[i].set_ylabel('Times selected', fontsize=9)
        axes[i].grid(True, alpha=0.2, axis='y')

    plt.suptitle('Client Selection Frequency (lower Gini = more fair)', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'selection_frequency.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


# ---- Plot 4: Gini coefficient over rounds --------------------------------

def plot_gini_over_rounds(all_results, num_clients=100):
    fig, ax = plt.subplots(figsize=(8, 5))

    for (strategy, label, color, ls), (_, sel_log) in zip(
        zip(STRATEGIES, LABELS, COLORS, LINESTYLES), all_results
    ):
        if sel_log is None:
            continue

        rounds = sorted(sel_log.keys())
        cumulative_counts = np.zeros(num_clients)
        gini_vals = []

        for r in rounds:
            for cid in sel_log[r]:
                cumulative_counts[int(cid)] += 1
            gini_vals.append(gini(cumulative_counts.copy()))

        ax.plot(rounds, gini_vals, label=label, color=color,
                linestyle=ls, linewidth=2)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Gini Coefficient', fontsize=12)
    ax.set_title('Participation Fairness over Training (lower = more equal)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    out = os.path.join(FIGURES_DIR, 'gini_over_rounds.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


# ---- Main ---------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print('Loading results...')
    all_results = [load_results(s) for s in STRATEGIES]

    available = sum(1 for perf, _ in all_results if perf is not None)
    print(f'Found results for {available}/{len(STRATEGIES)} strategies')
    print()

    print('Generating plots...')
    plot_accuracy_vs_round(all_results)
    plot_accuracy_vs_clock(all_results)
    plot_selection_frequency(all_results)
    plot_gini_over_rounds(all_results)

    print('\nAll plots saved to reports/figures/')
    print('Done.')
