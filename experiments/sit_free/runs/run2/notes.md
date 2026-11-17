# Run 2: free-bike balance from scratch

Hypothesis: compare learning without the fixed-bike policy's prior experience against
the matched pretrained run (run 1).

The actor, critic, optimizer and counters all start fresh. Both runs use
`LAAI-K1-Sit-Puky-Free`, 256 environments, seed 42 and 1,500 new PPO iterations, with the
same rewards, randomization, pushes and final 32-episode evaluation (seed 12345).
See [the experiment description](../../README.md) for physics and reward details.

Submitted on 2026-11-17 as Hex CPU job **54646**; started on `node03`.
W&B: [scratch run](https://forge.coreweave.com/wandb/K1Booster/laai-k1/runs/63cnaaev).
Comparison group: `free-bike-20261117T211126Z`.
Raw run: `logs/laai_sit_free/2026-11-17_22-11-41_free-bike-20261117T211126Z-scratch/`.
Manifest and source archive:
`logs/laai_sit_free/comparisons/free-bike-20261117T211126Z/`.

Full training and evaluation results are pending. Local scratch startup succeeded for
three PPO updates; the full local test suite passed (117 tests). No real-robot test has
been performed.
