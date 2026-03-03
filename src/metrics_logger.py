"""
metrics_logger.py
-----------------
Saves per-round training metrics and client selection logs to disk.

Two output files per experiment run:
  training_perf.pkl  — per-round {accuracy, loss, clock}  (matches PyramidFL format)
  selection_log.pkl  — per-round list of selected client IDs
"""

import os
import pickle
from typing import Dict, List


class MetricsLogger:
    """
    Accumulates metrics during a simulation run and writes them to disk.

    Usage:
        logger = MetricsLogger(out_dir="reports/run_random")
        logger.log_round(round_num=1, top1=55.3, loss=1.42,
                         clock=120.5, selected=[3, 17, 42, ...])
        logger.save()
    """

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # training_perf: matches PyramidFL's output format
        # key = round number (int), value = dict of metrics
        self.training_perf: Dict[int, dict] = {}

        # selection_log: records which clients were selected each round
        # key = round number (int), value = list of client IDs
        self.selection_log: Dict[int, List[int]] = {}

    def log_round(
        self,
        round_num: int,
        top1: float,
        loss: float,
        clock: float,
        selected: List[int],
        top5: float = 0.0,
    ):
        """Record metrics for one completed FL round."""
        self.training_perf[round_num] = {
            "round": round_num,
            "clock": clock,          # simulated wall time (seconds)
            "top_1": top1,           # global test accuracy (%)
            "top_5": top5,
            "loss": loss,            # global test loss
        }
        self.selection_log[round_num] = list(selected)

    def save(self):
        """Persist both pickle files to out_dir."""
        perf_path = os.path.join(self.out_dir, "training_perf.pkl")
        sel_path  = os.path.join(self.out_dir, "selection_log.pkl")

        with open(perf_path, "wb") as f:
            pickle.dump(self.training_perf, f)
        with open(sel_path, "wb") as f:
            pickle.dump(self.selection_log, f)

        print(f"  Saved: {perf_path}")
        print(f"  Saved: {sel_path}")

    def load(self, out_dir: str = None):
        """Load previously saved metrics (useful for plotting)."""
        out_dir = out_dir or self.out_dir
        perf_path = os.path.join(out_dir, "training_perf.pkl")
        sel_path  = os.path.join(out_dir, "selection_log.pkl")

        with open(perf_path, "rb") as f:
            self.training_perf = pickle.load(f)
        with open(sel_path, "rb") as f:
            self.selection_log = pickle.load(f)

        return self.training_perf, self.selection_log
