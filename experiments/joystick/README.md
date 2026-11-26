# Joystick path tracking

`LAAI-K1-Joystick-Puky` trains on the unsupported Puky. Each world samples a new
20-second reference at reset. Episodes contain three stick requests, with a mix of
straights, left and right arcs, and stops. Three quarters of moving requests
are forward, with reverse making up the remaining quarter.

The distance of the stick from center sets cruising speed. A request in the rear
half reverses the bike. Straight south rolls backward with the bars centered. The
planner latches the requested direction relative to the bike's reference heading
at the start of the segment and generates ramped speed and steering commands with
a planar 0.50 m wheelbase. It integrates those same commands to draw the reference.
The starting reference pose is anchored to the bike's reset pose.

The actor receives two targets per frame, signed drive speed in m/s and handlebar angle
in radians (positive is left). The controller adds a bounded steering correction
from the bike's deviation from the reference. PPO scores speed and handlebar
tracking, saddle load at a 78% target, knees more than 1 cm above their hips,
torque near the actuator limits, and falls. The `path_error_m` metric measures
distance to the scheduled reference pose.

The actor uses 680 observations and 20 joint actions. The archived straight-line
checkpoints have 690 inputs. This task starts from scratch with the default runner.

## HTML preview

The preview uses the same trajectory sampler as training. It shows the joystick
request, nominal policy inputs, and corresponding path for 18 sampled episodes.
The selector groups forward straights, left and right arcs, S-curves, and
stop-and-restart examples before the reverse and mixed-direction examples.
Use the timeline to inspect the signed commands.
See [commands](../../docs/commands.md#preview-and-train-joystick-paths)
to generate the HTML, run a CPU smoke test, or submit a training job on Hex.

## Shaped training and completion

`LAAI-K1-Joystick-Puky-Shaped` keeps the joystick task's tracking, saddle target,
knee penalty, torque limit and fall penalty. It adds the straight-line Shaped
task's gait, posture, balance, arm effort and action smoothing terms. Throttle
magnitude sets the gait reward's moving gate in both directions. The actor still
receives 680 observations and commands the same 20 joints.

The final checkpoint is evaluated on 32 seeded, held-out tracks. Completion
requires surviving the episode, reaching at least 80% of the commanded
distance in the correct direction on each moving segment, and finishing within
0.50 m of each segment endpoint and the overall endpoint. Forward and reverse
have separate segment completion, signed distance ratio, speed error and
correct-direction metrics. Fallen runs count against the planned segments
they did not reach. The report is written to `evaluation.json` and logged in W&B.

The saved Linux [4×4 viewer](../../scripts/play_joystick_grid.py) shows sixteen
live worlds with reference and actual traces, stick direction and target speed.
See [playback and evaluation commands](../../docs/commands.md#joystick-shaped-training-and-evaluation).
