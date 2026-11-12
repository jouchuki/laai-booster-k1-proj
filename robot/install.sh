#!/usr/bin/env bash
# Linux and macOS: install an experiment's robot-side files into the booster_deploy checkout.
#
# Copies experiments/<experiment>/deploy/*.py (booster_deploy task files), robot/dry_run.py, the
# solved seated poses (models/k1_on_bike/*.json -> models/seated_poses/) and a policy into
# booster_deploy, so its scripts/deploy.py can run them in MuJoCo and, after
# robot/setup_robot.sh, on the robot. The policy is installed as models/laai_<experiment>.pt;
# an experiment without trained runs (sit: it only holds a pose) installs without one.
#
# Usage (from anywhere):
#   bash robot/install.sh walk                                        # policy of the newest run
#   bash robot/install.sh walk experiments/walk/runs/run1/policy.pt   # a specific policy
#   bash robot/install.sh sit                                         # task files and poses only
set -euo pipefail

EXPERIMENT="${1:?Usage: bash robot/install.sh <experiment> [policy.pt]}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/booster_paths.sh"
DEPLOY_DIR="$(booster_checkout_path "$REPO_ROOT" booster_deploy)"
EXP_DIR="$REPO_ROOT/experiments/$EXPERIMENT"
K1_DIR="$DEPLOY_DIR/tasks/locomotion/robots/k1"

if [ ! -d "$EXP_DIR/deploy" ]; then
  echo "No robot-side files for experiment '$EXPERIMENT' (expected $EXP_DIR/deploy/)." >&2
  exit 1
fi
if [ ! -d "$K1_DIR" ]; then
  echo "booster_deploy not found at $DEPLOY_DIR. Run first: bash scripts/setup_booster.sh" >&2
  exit 1
fi

# Default policy: policy.pt of the highest-numbered run that has one.
if [ $# -ge 2 ]; then
  POLICY="$2"
else
  POLICY=""
  best=-1
  for p in "$EXP_DIR"/runs/run*/policy.pt; do
    [ -f "$p" ] || continue
    n="${p%/policy.pt}"
    n="${n##*/run}"
    if [ "$n" -gt "$best" ]; then best="$n"; POLICY="$p"; fi
  done
fi
if [ -n "$POLICY" ] && [ ! -f "$POLICY" ]; then
  echo "Policy file not found: $POLICY" >&2
  exit 1
fi
if [ -z "$POLICY" ] && compgen -G "$EXP_DIR/runs/run*" > /dev/null; then
  echo "No policy.pt in any of $EXP_DIR/runs/run*; give one as the second argument." >&2
  exit 1
fi

mkdir -p "$K1_DIR/models/seated_poses"
cp "$EXP_DIR"/deploy/*.py "$K1_DIR/"
cp "$REPO_ROOT"/models/k1_on_bike/*.json "$K1_DIR/models/seated_poses/"
if [ -n "$POLICY" ]; then cp "$POLICY" "$K1_DIR/models/laai_${EXPERIMENT}.pt"; fi
cp "$REPO_ROOT/robot/dry_run.py" "$DEPLOY_DIR/scripts/dry_run.py"

echo "Installed experiment '$EXPERIMENT' into $DEPLOY_DIR:"
for f in "$EXP_DIR"/deploy/*.py; do echo "  tasks/locomotion/robots/k1/$(basename "$f")"; done
for f in "$REPO_ROOT"/models/k1_on_bike/*.json; do echo "  tasks/locomotion/robots/k1/models/seated_poses/$(basename "$f")"; done
if [ -n "$POLICY" ]; then
  echo "  tasks/locomotion/robots/k1/models/laai_${EXPERIMENT}.pt  <- $POLICY"
else
  echo "  (no policy: experiment '$EXPERIMENT' has no trained runs)"
fi
echo "  scripts/dry_run.py"
