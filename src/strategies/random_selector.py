"""
random_selector.py
------------------
Baseline strategy: uniform random sampling with no utility awareness.

API (shared by all selectors):
    select(round_num, client_stats) -> List[int]
    update(round_num, results)      -> None   (no-op for random)
"""

import random
from typing import List


class RandomSelector:
    """
    Selects K clients uniformly at random each round.

    Args:
        num_clients      : total number of simulated clients
        clients_per_round: K clients to select each round
        seed             : random seed for reproducibility
    """

    def __init__(self, num_clients: int, clients_per_round: int, seed: int = 42):
        self.num_clients       = num_clients
        self.clients_per_round = clients_per_round
        self.rng               = random.Random(seed)

    def select(self, round_num: int, client_stats: dict) -> List[int]:
        """Return K randomly chosen client IDs (without replacement)."""
        all_clients = list(client_stats.keys())
        k = min(self.clients_per_round, len(all_clients))
        return self.rng.sample(all_clients, k)

    def update(self, round_num: int, results: list):
        """No feedback needed for random selection."""
        pass
