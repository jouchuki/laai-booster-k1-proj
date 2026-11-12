# Booster K1 (robot model)

The robot model is Booster's own `robots/K1/K1_22dof.xml` with its meshes, from
[booster_assets](https://github.com/BoosterRobotics/booster_assets) at the commit pinned in
`scripts/setup_booster.sh`. It is not copied into this repository: setup clones it, and
`k1bike.paths.k1_xml()` finds it.

| Item | Value |
|---|---|
| Joints | 22 (head 2, arms 4 each, legs 6 each); the walk policy controls 20 (not the head) |
| Mass | about 19.7 kg |
| Hands | none: each forearm ends in a rounded tip (see [k1_on_bike](../k1_on_bike/README.md)) |
| Collisions | every collision geom touches every non-adjacent one (174 self-collision pairs) |

All code uses it through `src/k1bike/robots/k1.py`:

- `k1_spec()`: the model adapted for mjlab training (ground, lights, motors and old sensors
  removed; feet named; foot sites and the IMU sensors mjlab expects added).
- booster_deploy's `k1_walk` settings: joint order, default pose, PD gains, effort limits,
  policy joints. booster_deploy sends the same gains to the real robot.
- `name_collision_geoms()`: names the unnamed collision geoms so collision rules and
  contact sensors can select them.

Scenes built from the raw XML (viewer, seated scene) use its torque motors;
`hold_pose_torques()` holds a pose with the `k1_walk` gains.

To update the robot model, change the booster_assets commit in `scripts/setup_booster.sh`,
rerun it, and check that the tests pass.
