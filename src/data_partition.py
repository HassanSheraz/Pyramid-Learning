"""
data_partition.py
-----------------
CIFAR-10 Dirichlet non-IID partitioner.

Usage:
    from src.data_partition import partition_cifar10
    partitions, class_dists = partition_cifar10(num_clients=100, alpha=0.5, data_dir='data/')

Returns:
    partitions   : dict {client_id (int): list of dataset indices}
    class_dists  : dict {client_id (int): np.ndarray of shape (num_classes,) — class probability vector}
"""

import numpy as np
import torchvision
import torchvision.transforms as transforms
import os


NUM_CLASSES = 10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_cifar10(data_dir: str = "data/"):
    """Download CIFAR-10 and return (train_dataset, test_dataset)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train, test


def partition_cifar10(
    num_clients: int = 100,
    alpha: float = 0.5,
    data_dir: str = "data/",
    seed: int = 42,
) -> tuple:
    """
    Partition CIFAR-10 training data across clients using a Dirichlet distribution.

    Args:
        num_clients : number of simulated FL clients
        alpha       : Dirichlet concentration parameter
                      - 0.1  -> severe non-IID (clients mostly have 1-2 classes)
                      - 0.5  -> moderate non-IID  (default)
                      - 100  -> near-IID
        data_dir    : where to download/cache CIFAR-10
        seed        : random seed for reproducibility

    Returns:
        partitions  : {client_id: [list of sample indices into train_dataset]}
        class_dists : {client_id: np.ndarray(shape=NUM_CLASSES) — class probability vector}
        train_dataset, test_dataset
    """
    np.random.seed(seed)
    train_dataset, test_dataset = load_cifar10(data_dir)

    # Group indices by class
    targets = np.array(train_dataset.targets)  # shape (50000,)
    class_indices = {c: np.where(targets == c)[0] for c in range(NUM_CLASSES)}

    # Sample Dirichlet proportions: shape (NUM_CLASSES, num_clients)
    proportions = np.random.dirichlet(
        alpha=[alpha] * num_clients, size=NUM_CLASSES
    )  # proportions[c, i] = fraction of class c's samples going to client i

    partitions = {i: [] for i in range(num_clients)}

    for c in range(NUM_CLASSES):
        idx = class_indices[c].copy()
        np.random.shuffle(idx)

        # Split class c's indices proportionally across clients
        splits = (proportions[c] * len(idx)).astype(int)
        # Fix rounding so total == len(idx)
        splits[-1] = len(idx) - splits[:-1].sum()

        start = 0
        for client_id, count in enumerate(splits):
            partitions[client_id].extend(idx[start: start + count].tolist())
            start += count

    # Compute each client's empirical class distribution
    class_dists = {}
    for client_id in range(num_clients):
        client_targets = targets[partitions[client_id]]
        hist = np.bincount(client_targets, minlength=NUM_CLASSES).astype(float)
        total = hist.sum()
        class_dists[client_id] = hist / total if total > 0 else hist

    return partitions, class_dists, train_dataset, test_dataset


def get_client_stats(partitions: dict, class_dists: dict) -> dict:
    """
    Return a summary dict per client used by the simulator.

    Returns:
        {client_id: {"size": int, "class_dist": np.ndarray}}
    """
    return {
        cid: {
            "size": len(partitions[cid]),
            "class_dist": class_dists[cid],
        }
        for cid in partitions
    }


if __name__ == "__main__":
    parts, dists, train_ds, test_ds = partition_cifar10(
        num_clients=100, alpha=0.5, data_dir="data/"
    )
    sizes = [len(v) for v in parts.values()]
    print(f"Total training samples : {sum(sizes)}")
    print(f"Clients                : {len(parts)}")
    print(f"Min samples per client : {min(sizes)}")
    print(f"Max samples per client : {max(sizes)}")
    print(f"Avg samples per client : {sum(sizes)/len(sizes):.1f}")
    print(f"\nClass dist of client 0 : {dists[0].round(3)}")
