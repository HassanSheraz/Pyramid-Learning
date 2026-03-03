"""
fl_simulator.py
---------------
Single-process federated learning simulation loop.

Supports four pluggable client selection strategies:
    random, oort, divfl, fairfl

Usage:
    python src/fl_simulator.py --config experiments/cifar10_random.yml
    python src/fl_simulator.py --config experiments/cifar10_oort.yml --rounds 50
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_partition import partition_cifar10, load_cifar10
from src.client_model import build_model, local_train
from src.metrics_logger import MetricsLogger


# ---------------------------------------------------------------------------
# Helper: load YAML config with base_config.yml defaults
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    base_path = os.path.join(
        os.path.dirname(config_path), "..", "experiments", "base_config.yml"
    )
    cfg = {}
    # Load base defaults if it exists
    if os.path.exists(base_path):
        with open(base_path) as f:
            cfg.update(yaml.safe_load(f) or {})
    # Override with strategy-specific config
    with open(config_path) as f:
        cfg.update(yaml.safe_load(f) or {})
    return cfg


# ---------------------------------------------------------------------------
# Helper: build the correct selector
# ---------------------------------------------------------------------------

def build_selector(strategy: str, cfg: dict, client_stats: dict):
    if strategy == "random":
        from src.strategies.random_selector import RandomSelector
        return RandomSelector(
            num_clients=cfg["num_clients"],
            clients_per_round=cfg["clients_per_round"],
            seed=cfg.get("seed", 42),
        )
    elif strategy == "oort":
        from src.strategies.oort_selector import OortSelector
        return OortSelector(
            num_clients=cfg["num_clients"],
            clients_per_round=cfg["clients_per_round"],
            client_stats=client_stats,
            cfg=cfg,
        )
    elif strategy == "divfl":
        from src.strategies.divfl_selector import DivFLSelector
        return DivFLSelector(
            num_clients=cfg["num_clients"],
            clients_per_round=cfg["clients_per_round"],
            client_stats=client_stats,
            cfg=cfg,
        )
    elif strategy == "fairfl":
        from src.strategies.fairfl_selector import FairFLSelector
        return FairFLSelector(
            num_clients=cfg["num_clients"],
            clients_per_round=cfg["clients_per_round"],
            client_stats=client_stats,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. "
                         "Choose from: random, oort, divfl, fairfl")


# ---------------------------------------------------------------------------
# Helper: simulate per-client device profiles (compute speed + bandwidth)
# ---------------------------------------------------------------------------

def generate_device_profiles(num_clients: int, seed: int = 42) -> dict:
    """
    Assign each client a simulated compute speed and bandwidth.

    Returns:
        {client_id: {"compute_speed": float,   # samples / second
                     "bandwidth_mbps": float}}  # MB / second
    """
    rng = np.random.default_rng(seed)
    profiles = {}
    for cid in range(num_clients):
        profiles[cid] = {
            # typical mobile CPU: 50-300 samples/sec at batch_size=1
            "compute_speed": float(rng.uniform(50, 300)),
            # WiFi / 4G: 1-20 MB/s
            "bandwidth_mbps": float(rng.uniform(1, 20)),
        }
    return profiles


def compute_round_time(
    selected: list,
    partitions: dict,
    profiles: dict,
    local_epochs: int,
    batch_size: int,
    model_size_mb: float,
) -> float:
    """
    Simulated round duration = max completion time across selected clients.
    local_time(i) = samples_i * epochs / compute_speed_i
    comm_time(i)  = model_size_mb / bandwidth_i
    """
    times = []
    for cid in selected:
        n = len(partitions[cid])
        local_t = (n * local_epochs) / profiles[cid]["compute_speed"]
        comm_t  = model_size_mb / profiles[cid]["bandwidth_mbps"]
        times.append(local_t + comm_t)
    return max(times) if times else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, test_dataset, batch_size: int = 256,
             device: str = "cpu") -> tuple:
    """Return (top1_accuracy_%, average_cross_entropy_loss)."""
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    total_correct = 0
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

    top1 = 100.0 * total_correct / total
    avg_loss = total_loss / total
    return top1, avg_loss


# ---------------------------------------------------------------------------
# FedAvg aggregation
# ---------------------------------------------------------------------------

def fedavg_aggregate(global_model: nn.Module, updates: list) -> nn.Module:
    """
    Weighted FedAvg: new_global = global + Σ (delta_i * ratio_i)
    where ratio_i = n_i / Σ n_j  (fraction of total samples this round).

    Args:
        global_model : current global model
        updates      : list of {"delta_weights": dict, "num_samples": int, ...}

    Returns:
        Updated global model (in place, also returned for convenience)
    """
    total_samples = sum(u["num_samples"] for u in updates)
    if total_samples == 0:
        return global_model

    global_state = global_model.state_dict()

    for key in global_state:
        weighted_delta = torch.zeros_like(global_state[key], dtype=torch.float32)
        for u in updates:
            if u["delta_weights"] is None:
                continue
            ratio = u["num_samples"] / total_samples
            weighted_delta += u["delta_weights"][key].float() * ratio
        global_state[key] = global_state[key].float() + weighted_delta

    global_model.load_state_dict(global_state)
    return global_model


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(cfg: dict, strategy_override: str = None):
    strategy = strategy_override or cfg.get("strategy", "random")
    num_rounds     = cfg.get("num_rounds", 100)
    num_clients    = cfg.get("num_clients", 100)
    local_epochs   = cfg.get("local_epochs", 5)
    batch_size     = cfg.get("batch_size", 32)
    lr             = cfg.get("learning_rate", 0.01)
    momentum       = cfg.get("momentum", 0.9)
    weight_decay   = cfg.get("weight_decay", 1e-4)
    alpha          = cfg.get("alpha", 0.5)
    seed           = cfg.get("seed", 42)
    data_dir       = cfg.get("data_dir", "data/")
    out_dir        = os.path.join(cfg.get("out_dir", "reports/"), f"run_{strategy}")
    device         = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  FL Simulation — strategy: {strategy.upper()}")
    print(f"  Rounds: {num_rounds}  |  Clients/round: {cfg['clients_per_round']}")
    print(f"  Non-IID alpha: {alpha}  |  Device: {device}")
    print(f"{'='*60}\n")

    # 1. Partition data
    print("Partitioning CIFAR-10...")
    partitions, class_dists, train_dataset, test_dataset = partition_cifar10(
        num_clients=num_clients, alpha=alpha, data_dir=data_dir, seed=seed
    )

    # 2. Device profiles (for simulated clock)
    profiles = generate_device_profiles(num_clients=num_clients, seed=seed)

    # 3. Build client_stats dict used by selectors
    client_stats = {
        cid: {
            "size": len(partitions[cid]),
            "class_dist": class_dists[cid],
            "compute_speed": profiles[cid]["compute_speed"],
            "bandwidth_mbps": profiles[cid]["bandwidth_mbps"],
        }
        for cid in range(num_clients)
    }

    # 4. Initialise selector
    selector = build_selector(strategy, cfg, client_stats)

    # 5. Initialise global model
    torch.manual_seed(seed)
    global_model = build_model(num_classes=cfg.get("num_classes", 10)).to(device)

    # Estimate model size in MB (for clock simulation)
    param_bytes = sum(p.numel() * 4 for p in global_model.parameters())
    model_size_mb = param_bytes / (1024 ** 2)
    print(f"Model size: {model_size_mb:.2f} MB  |  "
          f"Params: {sum(p.numel() for p in global_model.parameters()):,}\n")

    # 6. Logger
    logger = MetricsLogger(out_dir=out_dir)

    cumulative_clock = 0.0
    wall_start = time.time()

    # 7. FL rounds
    for rnd in range(1, num_rounds + 1):
        # --- Select clients ---
        selected = selector.select(round_num=rnd, client_stats=client_stats)

        # --- Local training ---
        updates = []
        for cid in selected:
            result = local_train(
                global_model=global_model,
                data_indices=partitions[cid],
                dataset=train_dataset,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                device=device,
            )
            result["client_id"] = cid
            updates.append(result)

        # --- Aggregate ---
        global_model = fedavg_aggregate(global_model, updates)

        # --- Evaluate ---
        top1, test_loss = evaluate(global_model, test_dataset,
                                   batch_size=256, device=device)

        # --- Simulated clock ---
        round_time = compute_round_time(
            selected, partitions, profiles,
            local_epochs, batch_size, model_size_mb
        )
        cumulative_clock += round_time

        # --- Feed results back to selector (for Oort/DivFL/FairFL) ---
        selector.update(
            round_num=rnd,
            results=[
                {"client_id": u["client_id"], "loss": u["loss"],
                 "num_samples": u["num_samples"]}
                for u in updates
            ]
        )

        # --- Log ---
        logger.log_round(
            round_num=rnd,
            top1=top1,
            loss=test_loss,
            clock=cumulative_clock,
            selected=selected,
        )

        elapsed = time.time() - wall_start
        print(f"  Round {rnd:3d}/{num_rounds}  |  "
              f"Acc: {top1:5.2f}%  |  "
              f"Loss: {test_loss:.4f}  |  "
              f"Clock: {cumulative_clock/3600:.2f}h  |  "
              f"Wall: {elapsed:.0f}s")

    logger.save()
    print(f"\nDone. Results saved to: {out_dir}\n")
    return logger


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Simulation")
    parser.add_argument("--config", required=True,
                        help="Path to experiment YAML config file")
    parser.add_argument("--strategy", default=None,
                        help="Override strategy from config (random/oort/divfl/fairfl)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override num_rounds from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.rounds:
        cfg["num_rounds"] = args.rounds

    run_simulation(cfg, strategy_override=args.strategy)
