# Algorithm Design: DivFL and FairFL

## Background

In federated learning (FL), a central server selects K clients each round to perform local training.
PyramidFL's Oort selector scores each client using a UCB bandit formula:

```
score(i) = normalized_utility(i) + sqrt(0.1 * ln(t) / t_last_selected(i))
```

Where:
- `normalized_utility` = statistical utility (sqrt(loss) × data_size), normalized to [0,1]
- The second term is a UCB staleness bonus (rewards clients not seen recently)

Oort selects the top-K clients by this score (with an exploration/exploitation split).

**Problem:** Oort optimizes for speed + loss magnitude only.
- It can repeatedly select the same clients (fairness issue).
- Selected clients may have highly overlapping class distributions (diversity issue).

---

## Strategy 1: DivFL — Diversity-Augmented Oort

**Core idea:** Add a diversity bonus to Oort's score that rewards clients whose
data distribution covers underrepresented classes relative to already-selected clients.

**Per-client class distribution:** Each client `i` has a class histogram `p_i` (a probability
vector of length 10 for CIFAR-10, where `p_i[c]` = fraction of client i's data that is class c).

**Diversity bonus:** Computed greedily. We select clients one-at-a-time:

```
selected = []
for each pick k in 1..K:
    if selected is empty:
        diversity_bonus(i) = 0  for all i   # first pick: pure Oort
    else:
        q = average class distribution over already-selected clients
        diversity_bonus(i) = JSD(p_i, q)    # Jensen-Shannon Divergence

    adjusted_score(i) = oort_score(i) + lambda * diversity_bonus(i)
    pick i* = argmax adjusted_score over remaining candidates
    selected.append(i*)
```

**Jensen-Shannon Divergence:**

```
JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)    where m = 0.5*(p+q)
```

JSD is bounded in [0, 1] (when using log base 2), symmetric, and well-defined even when
distributions have zero entries — ideal for comparing sparse class histograms.

**Hyperparameter:** `diversity_weight` (λ) — controls tradeoff between utility and diversity.
- λ = 0: pure Oort
- λ = 0.3: recommended starting point (determined by ablation in `src/ablation.py`)
- λ = 1.0: diversity dominates

---

## Strategy 2: FairFL — Fairness-Penalized Oort

**Core idea:** Penalize clients that have been selected disproportionately often,
so over-represented clients are less likely to be selected again.

**Participation rate:** After round `t`, client `i` has been selected `count(i, t)` times.
Its participation rate is:

```
participation_rate(i, t) = count(i, t) / t
```

**Adjusted score:**

```
adjusted_score(i) = oort_score(i) * (1 - beta * participation_rate(i, t))
```

- Clients with zero selections so far: penalty = 0 (full Oort score, encouraged)
- Clients selected every round: penalty = beta (score reduced by fraction beta)
- β = 0: pure Oort
- β = 0.5: recommended (halves the score of a client selected every round)

**Why soft penalty instead of hard exclusion:**
- Hard exclusion (round-robin) ignores data quality and can hurt convergence.
- Soft penalty lets genuinely high-utility clients still be selected, just less often.

---

## Evaluation Metrics

| Metric | Measures | How computed |
|---|---|---|
| **Accuracy vs. Round** | Convergence speed | Global test accuracy after each FL round |
| **Accuracy vs. Clock** | Wall-time efficiency | X-axis = simulated time (sum of max client durations per round) |
| **Gini Coefficient** | Participation fairness | Gini of `selection_count` vector across all 100 clients |
| **Class Coverage** | Data diversity per round | Fraction of CIFAR-10 classes (out of 10) represented in selected clients each round |

**Gini coefficient:** 0 = perfectly equal participation, 1 = one client selected every round.

---

## Non-IID Data Setup

Dirichlet distribution controls heterogeneity:

```
p_i ~ Dirichlet(alpha)    for each client i
```

- `alpha = 0.1`: severe non-IID (clients have 1-2 dominant classes)
- `alpha = 0.5`: moderate non-IID (default for all experiments)
- `alpha = 100`: near-IID

All experiments use `alpha = 0.5`, `num_clients = 100`, `clients_per_round = 10`.
