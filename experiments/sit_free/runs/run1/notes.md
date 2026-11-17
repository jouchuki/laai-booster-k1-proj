# Run 1: pretrained seated policy on a free bike

Hypothesis: the fixed-bike policy's seated posture and smooth control provide a useful
initialization for unsupported balance compared with the matched scratch run (run 2).

The entire actor transfers, including observation normalization and exploration
distribution. The critic, optimizer, iteration counter and environment counter start
fresh. Both runs use `LAAI-K1-Sit-Puky-Free`, 256 environments, seed 42 and 1,500 new PPO
iterations. They share reward definitions and the final 32-episode evaluation (seed 12345).
See [the experiment description](../../README.md) for physics and reward details.

Submitted on 2026-11-17 as Hex CPU job **54645**; started on `node03`.
W&B: [pretrained run](https://forge.coreweave.com/wandb/K1Booster/laai-k1/runs/6tu6vq9j).
Comparison group: `free-bike-20261117T211126Z`.
Raw run: `logs/laai_sit_free/2026-11-17_22-11-46_free-bike-20261117T211126Z-pretrained/`.

Initialization: fixed-bike run `2026-11-17_17-32-52_cpu-stage0/model_900.pt`, frozen at
submission while that original run continued independently. SHA-256:
`42e0ff4f8d08e1e40cba032d8ac05f4a27c4296007b107053bf25d00318444c1`.
Manifest and source archive:
`logs/laai_sit_free/comparisons/free-bike-20261117T211126Z/`.

Full training and evaluation results are pending. Local checks: 117 tests passed; CPU
startup succeeded from the stage-0 model at iteration 800. A separate 256-environment,
three-update startup test peaked at 1.76 GiB RSS. Its final evaluation had 0/32 successes
and 0.89 s mean survival; this checks the pipeline, not trained performance.
No real-robot test has been performed.
