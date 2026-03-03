"""
fairfl_selector.py  —  FairFL: Fairness-Penalized Client Selection

This is my second original contribution.

Problem with Oort (and many FL methods): some clients get selected
over and over again because they consistently have high utility scores.
Other clients barely get selected at all. In a real FL deployment,
this is unfair — devices that rarely contribute lose the ability to
influence the model, and their local data is effectively ignored.

My approach: track how often each client has been selected, and apply
a soft penalty to clients that are already over-represented. This
encourages the server to spread participation more evenly without
completely ignoring high-utility clients.

The penalty formula:
    participation_rate(i) = count(i) / round_num
    adjusted_score(i) = oort_score(i) * (1 - beta * participation_rate(i))

Where beta (fairness_weight) controls how strongly we penalize.
    beta = 0.0  ->  pure Oort, no fairness
    beta = 0.5  ->  halves the score of a client selected every round
    beta = 1.0  ->  fully suppresses clients with 100% participation rate

I measure fairness improvement using the Gini coefficient of selection
counts across clients (lower Gini = more equal participation).
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'oort'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'oort', 'utils'))

from oort import create_training_selector
from src.strategies.oort_selector import _OortArgs


class FairFLSelector:
    """
    Fairness-penalized Oort selector (FairFL).

    Maintains a selection count for every client and applies a
    soft multiplicative penalty to over-selected clients.
    """

    def __init__(self, num_clients, clients_per_round, client_stats, cfg):
        self.num_clients       = num_clients
        self.clients_per_round = clients_per_round
        self.fairness_weight   = cfg.get('fairness_weight', 0.5)

        # Build Oort underneath
        oort_args = _OortArgs(cfg)
        self.oort = create_training_selector(oort_args)

        # Register all clients with Oort
        for cid, stats in client_stats.items():
            local_epochs = cfg.get('local_epochs', 5)
            speed = stats.get('compute_speed', 100.0)
            bw    = stats.get('bandwidth_mbps', 5.0)
            est_duration = (stats['size'] * local_epochs) / speed + 8.0 / bw
            self.oort.register_client(cid, {
                'reward':   float(stats['size']),
                'duration': est_duration,
                'gradient': 0.0,
            })

        # Track how many times each client has been selected
        self.selection_counts = {cid: 0 for cid in client_stats.keys()}

    def _fairness_penalty(self, cid, round_num):
        """
        Returns the participation rate of client cid so far.
        Ranges from 0 (never selected) to 1 (selected every round).
        """
        return self.selection_counts[cid] / max(round_num - 1, 1)

    def _get_adjusted_scores(self, all_clients, round_num):
        """
        Get each client's Oort base score and apply the fairness penalty.
        We use the reward stored in Oort's arm table as the base score.
        """
        arms = self.oort.getAllMetrics()
        rewards = [arms[c]['reward'] for c in all_clients if arms[c]['reward'] > 0]

        if not rewards:
            # early rounds — use data size as score
            adjusted = {c: arms[c]['reward'] for c in all_clients}
            return adjusted

        min_r   = min(rewards)
        max_r   = max(rewards)
        r_range = max(max_r - min_r, 1e-6)

        adjusted = {}
        for c in all_clients:
            arm = arms[c]
            norm_score = (arm['reward'] - min_r) / r_range

            # UCB staleness bonus (same as Oort)
            if arm['count'] > 0 and arm['time_stamp'] > 0:
                norm_score += math.sqrt(
                    0.1 * math.log(round_num) / max(arm['time_stamp'], 1)
                )

            # apply fairness penalty
            penalty = self._fairness_penalty(c, round_num)
            adjusted[c] = norm_score * (1.0 - self.fairness_weight * penalty)

        return adjusted

    def select(self, round_num, client_stats):
        """
        Select K clients using fairness-adjusted Oort scores.
        Clients that have been over-selected get a lower effective score.
        """
        all_clients = list(client_stats.keys())
        k = self.clients_per_round

        # warm-up round
        if round_num == 1:
            selected = random.sample(all_clients, min(k, len(all_clients)))
            for c in selected:
                self.selection_counts[c] += 1
            return selected

        # Get fairness-adjusted scores
        scores = self._get_adjusted_scores(all_clients, round_num)

        # Sort by adjusted score (highest first)
        ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

        # Take top-k
        selected = ranked[:k]

        # Update participation counts
        for c in selected:
            self.selection_counts[c] += 1

        return selected

    def update(self, round_num, results):
        """Feed results back into Oort (same as other selectors)."""
        for r in results:
            cid     = r['client_id']
            loss    = r['loss']
            n       = r['num_samples']
            utility = math.sqrt(loss) * n if loss > 0 else 0.0

            self.oort.update_client_util(cid, {
                'reward':     utility,
                'duration':   1.0,
                'time_stamp': round_num,
                'status':     True,
                'gradient':   0.0,
                'count':      1,
            })
