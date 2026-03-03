"""
visualize_partition.py
----------------------
Plot the class distribution across all clients to visualise non-IID skew.

Usage:
    python src/visualize_partition.py
    python src/visualize_partition.py --alpha 0.1 --num_clients 50
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — saves file instead of opening a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Allow importing from src/ regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_partition import partition_cifar10, CIFAR10_CLASSES, NUM_CLASSES


def plot_partition(class_dists: dict, alpha: float, out_path: str):
    """
    Stacked bar chart: one bar per client, colour = class.
    Saves the figure to out_path.
    """
    num_clients = len(class_dists)
    # Matrix: rows=clients, cols=classes
    dist_matrix = np.stack(
        [class_dists[i] for i in range(num_clients)], axis=0
    )  # shape (num_clients, NUM_CLASSES)

    colours = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(num_clients)
    x = np.arange(num_clients)

    for c in range(NUM_CLASSES):
        ax.bar(x, dist_matrix[:, c], bottom=bottom, color=colours[c],
               width=1.0, linewidth=0)
        bottom += dist_matrix[:, c]

    ax.set_xlabel("Client ID", fontsize=12)
    ax.set_ylabel("Class proportion", fontsize=12)
    ax.set_title(
        f"CIFAR-10 class distribution per client  |  Dirichlet α = {alpha}  |  {num_clients} clients",
        fontsize=13
    )
    ax.set_xlim(-0.5, num_clients - 0.5)
    ax.set_ylim(0, 1)

    patches = [
        mpatches.Patch(color=colours[c], label=CIFAR10_CLASSES[c])
        for c in range(NUM_CLASSES)
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, title="Class", title_fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_alpha_comparison(alphas: list, num_clients: int, data_dir: str, out_dir: str):
    """Generate one partition plot per alpha value for comparison."""
    for alpha in alphas:
        _, dists, _, _ = partition_cifar10(
            num_clients=num_clients, alpha=alpha, data_dir=data_dir
        )
        out_path = os.path.join(out_dir, f"partition_alpha{alpha}.png")
        plot_partition(dists, alpha=alpha, out_path=out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise CIFAR-10 non-IID partition")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration (default: 0.5)")
    parser.add_argument("--num_clients", type=int, default=100,
                        help="Number of clients (default: 100)")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="CIFAR-10 download directory")
    parser.add_argument("--out_dir", type=str, default="reports/figures",
                        help="Output directory for plot(s)")
    parser.add_argument("--compare_alphas", action="store_true",
                        help="Generate plots for alpha in {0.1, 0.5, 1.0, 100}")
    args = parser.parse_args()

    if args.compare_alphas:
        plot_alpha_comparison(
            alphas=[0.1, 0.5, 1.0, 100],
            num_clients=args.num_clients,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
        )
    else:
        _, dists, _, _ = partition_cifar10(
            num_clients=args.num_clients,
            alpha=args.alpha,
            data_dir=args.data_dir,
        )
        out_path = os.path.join(args.out_dir, f"partition_alpha{args.alpha}.png")
        plot_partition(dists, alpha=args.alpha, out_path=out_path)
