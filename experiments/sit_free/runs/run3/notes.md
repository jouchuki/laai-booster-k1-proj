# Run 3: push curriculum from run 2

Submitted on 2026-11-20 as Hex CPU job **55245**, run name `push-run3`.
Task: `LAAI-K1-Sit-Puky-Free-Push`, starting from free-bike run 2's complete actor.
Critic, optimizer and counters start fresh. Critic warm-up is not enabled.

Both comparison runs use 256 environments, seed 42, 3,000 PPO updates and 24 rollout
steps per update (18,432,000 transitions), with 4 CPUs, 4 GB RAM and a 24-hour limit
per run. Both run on `node03`. The GPU smoke job 55241 failed before its batch script
ran, so both use the documented CPU fallback. The earlier checkout's uncommitted
work was preserved; these jobs use a separate checkout of main commit
`4d10ca9d8bdd8b2e69e6a8f57ff7ddb80a8d5f26`.

[Submission commands, source and checkpoint hashes](../push-comparison/manifest.json).
Hex checkout: `/scratch/laoa/laai-booster-k1-push-20261120`.
W&B project: `laai-k1`, comparison group `push-comparison-20261120`.

## Acceptance

Watch `Curriculum/push/push_mps`; passing 1 m/s is a training signal, not the final
acceptance criterion. At the end use `evaluation.json` → `push_sweep` →
`beats_zero_action_at_mps`. Only listed speeds establish better survival than zero
actions in this 32-world, seed-12345 evaluation. One training seed per variant is
an initial comparison, not evidence across seeds.

Status at submission: running; no final evaluation or new policy is available yet.
A dependent CPU job archives the final evaluation, scalars, parameters, actor checkpoint
and verified deployment-format policy when training finishes. Raw output remains in
`logs/laai_sit_free/<date>_push-run3/` on Hex.

## Real robot

Not tested. Training retains welded hand contacts. Verify the actual clips and grip fit
before a robot balance test; the unclipped setup from 2026-11-19 does not match training.
Measure the final saddle dip height: about 39.4 cm was reported, but it remains unmeasured.
Use the next-visit procedure in [the comparison record](../push-comparison/README.md).
