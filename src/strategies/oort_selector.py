"""
oort_selector.py

Wraps the original Oort algorithm from third_party/oort/oort.py
so it works inside my simulation framework.

Oort is a UCB (Upper Confidence Bound) bandit algorithm for FL client selection.
It scores each client based on:
  - statistical utility: how much useful training that client can contribute
  - system utility: how fast the client is (we don't want to wait forever)
  - staleness bonus: clients not seen recently get a bonus (exploration)

Reference: Lai et al., "Oort: Efficient Federated Learning via Guided
Participant Selection", OSDI 2021.
"""

import sys
import os
import math
import random

# I need to add the third_party folder so Python can find the oort module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'oort'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'oort', 'utils'))

from oort import create_training_selector


class _OortArgs:
    """
    Oort needs an args object with specific attributes.
    I'm creating a simple class to hold those settings
    instead of using argparse (which the original code uses).
    """
    def __init__(self, cfg):
        self.exploration_factor  = cfg.get('exploration_factor', 0.9)
        self.exploration_decay   = cfg.get('exploration_decay', 0.95)
        self.exploration_min     = cfg.get('exploration_min', 0.2)
        self.exploration_alpha   = cfg.get('exploration_alpha', 0.3)
        self.pacer_step          = cfg.get('pacer_step', 20)
        self.pacer_delta         = cfg.get('pacer_delta', 5)
        self.round_threshold     = cfg.get('round_threshold', 30.0)
        self.round_penalty       = cfg.get('round_penalty', 2.0)
        self.cut_off_util        = cfg.get('cut_off_util', 0.7)
        self.sample_window       = cfg.get('sample_window', 5.0)
        self.blacklist_rounds    = cfg.get('blacklist_rounds', -1)
        self.blacklist_max_len   = cfg.get('blacklist_max_len', 0.3)
        self.clip_bound          = cfg.get('clip_bound', 0.98)


class OortSelector:
    """
    Client selector that uses the Oort UCB bandit algorithm.

    This is basically a thin wrapper around the original Oort code
    so it fits into my simulation loop cleanly.
    """

    def __init__(self, num_clients, clients_per_round, client_stats, cfg):
        self.num_clients       = num_clients
        self.clients_per_round = clients_per_round

        # Build the Oort trainer object
        oort_args = _OortArgs(cfg)
        self.oort = create_training_selector(oort_args)

        # Register all clients with Oort upfront so it knows they exist
        for cid, stats in client_stats.items():
            # initial reward = data size (bigger datasets are more useful initially)
            # duration = a rough estimate of how long this client takes per round
            estimated_duration = self._estimate_duration(stats, cfg)
            self.oort.register_client(cid, {
                'reward':   float(stats['size']),
                'duration': estimated_duration,
                'gradient': 0.0,
            })

    def _estimate_duration(self, stats, cfg):
        # rough time estimate: samples * epochs / compute_speed
        # this is used by oort to avoid picking really slow clients
        local_epochs = cfg.get('local_epochs', 5)
        batch_size   = cfg.get('batch_size', 32)
        speed        = stats.get('compute_speed', 100.0)
        bw           = stats.get('bandwidth_mbps', 5.0)
        # model is about 8MB, rough communication time
        comm_time    = 8.0 / bw
        local_time   = (stats['size'] * local_epochs) / speed
        return local_time + comm_time

    def select(self, round_num, client_stats):
        """
        Ask Oort to pick the best K clients for this round.

        Round 1 is always random — Oort needs at least one round of data
        before its UCB scores mean anything. This is standard practice in
        bandit-based FL methods ('warm-up round').
        """
        all_clients = list(client_stats.keys())
        k = self.clients_per_round

        # warm-up: first round is random since no utility scores exist yet
        if round_num == 1:
            return random.sample(all_clients, min(k, len(all_clients)))

        feasible = set(all_clients)
        try:
            selected = self.oort.select_participant(k, feasible)
            selected = [int(c) for c in selected]
        except Exception:
            # Oort can sometimes fail (e.g. all clients explored but not enough
            # scores computed). Fall back to random in that case.
            selected = random.sample(all_clients, min(k, len(all_clients)))

        # make sure we always return exactly k clients
        if len(selected) < k:
            remaining = [c for c in all_clients if c not in selected]
            selected += random.sample(remaining, min(k - len(selected), len(remaining)))

        return selected[:k]

    def update(self, round_num, results):
        """
        After each round, feed the training results back to Oort
        so it can update its score estimates for each client.

        The reward formula comes from the original Oort paper:
            utility = sqrt(loss) * num_samples
        Higher loss + more data = more useful client.
        """
        for r in results:
            cid  = r['client_id']
            loss = r['loss']
            n    = r['num_samples']

            # this is how PyramidFL computes the utility score (param_server.py ~line 420)
            utility = math.sqrt(loss) * n if loss > 0 else 0.0

            self.oort.update_client_util(cid, {
                'reward':     utility,
                'duration':   1.0,       # TODO: track real duration per client
                'time_stamp': round_num,
                'status':     True,
                'gradient':   0.0,
                'count':      1,
            })
