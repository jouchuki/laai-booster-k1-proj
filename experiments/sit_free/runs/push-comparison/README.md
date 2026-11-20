# Push-curriculum comparison, 2026-11-20

| Run | Initialization | Hex job | Updates | Environments |
|---|---|---|---|---|
| [3](../run3/notes.md) | Free-bike run 2 actor; fresh critic | 55245 | 3,000 | 256 |
| [4](../run4/notes.md) | Scratch actor and critic | 55246 | 3,000 | 256 |

Both jobs use the student `batch` CPU partition, 4 CPUs and 4 GB RAM each, a 24-hour
limit, seed 42, identical physics and the same push curriculum. This is the CPU fallback:
GPU smoke job 55241 failed on node08 with SLURM exit `0:53` before producing script output.
The separate Hex checkout preserves the original checkout's uncommitted work.
The [manifest](manifest.json) records the exact commands, source revision/archive hash
and run-2 checkpoint hash. The source archive remains in the manifest's Hex directory.

Training and the final push sweep run inside their allocations. Watch W&B project
`laai-k1`, group `push-comparison-20261120`, metric `Curriculum/push/push_mps`.
The final criterion is a nonempty `push_sweep.beats_zero_action_at_mps` list in
`evaluation.json`. An empty list means the run has not beaten zero actions on any tested
push speed, even if its training reward rose. No final results are available at submission.

## Next lab visit

The new policies are not yet validated for deployment. Before testing:

1. Read each final push sweep and the deployment-format export check. Select a candidate
   that beats zero actions; record which pushes and seed were tested.
2. Verify both hand clips fit the modeled grip positions and test with them on. These
   training runs retain welded hands. If clips cannot be used, an unclipped-hand training
   model and another training/evaluation cycle are still needed.
3. Measure the final saddle dip height exactly. The reported approximately **39.4 cm** is
   an estimate, not a measurement. Record the floor-to-lowest-point measurement in the
   [2026-11-19 notes](../../../sit/robot_tests/2026-11-19/notes.md).
4. Follow the existing [robot procedure](../../../sit/README.md), with the bike held,
   both feet on the floor, a bike holder, catcher and operator, and the existing safety stops.
   Install the selected verified `policy.pt` with `robot/setup_robot.sh` as documented in
   [docs/commands.md](../../../../docs/commands.md#robot-test-the-trained-sit-policy-phase-3).
5. Use `laai_sit_ankle_kd15`. First let it sit calmly for 30 seconds, then apply a gentle
   push. Record whether it corrects the lean, any oscillation, and every stop reason.

The clip fit, saddle measurement and robot trial require a lab visit. They have not been
performed as part of cluster submission. Optional critic warm-up was left out so the
requested initialization comparison runs with the existing algorithm.
