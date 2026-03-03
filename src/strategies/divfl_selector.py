"""
divfl_selector.py  —  DivFL: Diversity-Augmented Client Selection

This is my first original contribution to the project.

The idea: Oort selects clients based on utility (how useful their data is)
but it doesn't care whether the selected clients have similar or different
data distributions. In non-IID settings this can mean we pick 10 clients
that all have very similar class distributions, which wastes a round.

My fix: after computing Oort's score for each candidate, I add a
diversity bonus that rewards clients whose class distribution is different
from the clients already picked this round. I pick clients one-by-one
(greedy), and the bonus updates each time we pick someone.

The diversity bonus = Jensen-Shannon Divergence (JSD) between:
  - client i's class distribution  (p_i)
  - the average distribution of already-selected clients  (q)

JSD ranges from 0 (identical distributions) to 1 (completely different).
So higher JSD = more diverse = higher bonus.

Final score for each candidate:
  adjusted_score(i) = oort_score(i) + lambda * JSD(p_i, q_selected)

where lambda (diversity_weight) controls the tradeoff.
"""

import sys
import os
import math
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'oort'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'oort', 'utils'))

from oort import create_training_selector
from src.strategies.oort_selector import _OortArgs


def jsd(p, q, eps=1e-10):
    """
    Jensen-Shannon Divergence between two probability distributions p and q.
    Returns a value in [0, 1].
    Both p and q should be non-negative and sum to 1 (or close to it).
    """
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    # KL divergence: sum(p * log(p/m))
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


class DivFLSelector:
    """
    Diversity-augmented Oort selector (DivFL).

    Extends Oort by greedily selecting clients to maximize both
    utility AND distributional diversity in each round.
    """

    def __init__(self, num_clients, clients_per_round, client_stats, cfg):
        self.num_clients       = num_clients
        self.clients_per_round = clients_per_round
        self.diversity_weight  = cfg.get('diversity_weight', 0.3)

        # Build the underlying Oort model — same as the oort_selector
        oort_args = _OortArgs(cfg)
        self.oort = create_training_selector(oort_args)

        # Store class distributions for each client (needed for JSD)
        self.class_dists = {
            cid: stats['class_dist'] for cid, stats in client_stats.items()
        }

        # Register everyone with Oort
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

        # We'll track Oort scores ourselves so we can augment them
        # (no clean API to get the raw scores from Oort, so we recompute)
        self.last_oort_scores = {}

    def _get_oort_scores(self, round_num, all_clients):
        """
        Estimate Oort utility scores for all clients.
        We use the reward stored inside Oort's arm table as a proxy.
        Not perfect, but good enough for the diversity augmentation.
        """
        scores = {}
        arms = self.oort.getAllMetrics()
        rewards = [arms[c]['reward'] for c in all_clients if arms[c]['reward'] > 0]

        if not rewards:
            # no scores yet (early rounds) — just use data size as score
            for c in all_clients:
                scores[c] = arms[c]['reward']
            return scores

        min_r = min(rewards)
        max_r = max(rewards)
        r_range = max(max_r - min_r, 1e-6)

        for c in all_clients:
            arm = arms[c]
            norm_reward = (arm['reward'] - min_r) / r_range

            # UCB staleness bonus (same formula as Oort paper)
            staleness_bonus = 0.0
            if arm['count'] > 0 and arm['time_stamp'] > 0:
                staleness_bonus = math.sqrt(
                    0.1 * math.log(round_num) / max(arm['time_stamp'], 1)
                )

            scores[c] = norm_reward + staleness_bonus

        return scores

    def select(self, round_num, client_stats):
        """
        Greedy diversity-augmented selection:
          1. Get Oort utility scores for all clients
          2. Pick clients one-by-one, each time choosing the candidate
             with the highest (oort_score + lambda * JSD_bonus)
          3. After each pick, update the 'average selected distribution'
             so the JSD bonus for remaining candidates is recalculated
        """
        all_clients = list(client_stats.keys())
        k = self.clients_per_round

        # Round 1 warm-up — same reasoning as oort_selector
        if round_num == 1:
            return random.sample(all_clients, min(k, len(all_clients)))

        # Get Oort base scores
        oort_scores = self._get_oort_scores(round_num, all_clients)
        self.last_oort_scores = oort_scores

        selected = []
        remaining = list(all_clients)
        avg_selected_dist = None   # will be updated as we pick clients

        for _ in range(min(k, len(remaining))):
            best_client = None
            best_score  = -1.0

            for cid in remaining:
                oort_score = oort_scores.get(cid, 0.0)

                # diversity bonus: how different is this client from what's selected?
                if avg_selected_dist is None:
                    div_bonus = 0.0   # first pick: no diversity to compare against
                else:
                    div_bonus = jsd(self.class_dists[cid], avg_selected_dist)

                total_score = oort_score + self.diversity_weight * div_bonus

                if total_score > best_score:
                    best_score  = total_score
                    best_client = cid

            if best_client is None:
                break

            selected.append(best_client)
            remaining.remove(best_client)

            # update the running average distribution of selected clients
            selected_dists = [self.class_dists[c] for c in selected]
            avg_selected_dist = np.mean(selected_dists, axis=0)

        return selected

    def update(self, round_num, results):
        """Feed results back into Oort (same as oort_selector)."""
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
