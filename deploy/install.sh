#!/usr/bin/env bash
# Linux and macOS: install a phase's robot-side files into the booster_deploy checkout.
#
# Copies phases/<phase>/deploy/*.py (booster_deploy task files), deploy/dry_run.py and a policy
# into booster_deploy, so its scripts/deploy.py can run them in MuJoCo and, after
# deploy/setup_robot.sh, on the robot. The policy is installed as models/laai_<phase>.pt.
#
# Usage (from anywhere):
#   bash deploy/install.sh walk                                    # policy of the newest run
#   bash deploy/install.sh walk phases/walk/runs/run1/policy.pt    # a specific policy
set -euo pipefail

PHASE="${1:?Usage: bash deploy/install.sh <phase> [policy.pt]}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/booster_paths.sh"
DEPLOY_DIR="$(booster_checkout_path "$REPO_ROOT" booster_deploy)"
PHASE_DIR="$REPO_ROOT/phases/$PHASE"
K1_DIR="$DEPLOY_DIR/tasks/locomotion/robots/k1"

if [ ! -d "$PHASE_DIR/deploy" ]; then
  echo "No robot-side files for phase '$PHASE' (expected $PHASE_DIR/deploy/)." >&2
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
  for p in "$PHASE_DIR"/runs/run*/policy.pt; do
    [ -f "$p" ] || continue
    n="${p%/policy.pt}"
    n="${n##*/run}"
    if [ "$n" -gt "$best" ]; then best="$n"; POLICY="$p"; fi
  done
fi
if [ -z "$POLICY" ] || [ ! -f "$POLICY" ]; then
  echo "Policy file not found: ${POLICY:-no runs/run*/policy.pt in $PHASE_DIR}" >&2
  exit 1
fi

mkdir -p "$K1_DIR/models"
cp "$PHASE_DIR"/deploy/*.py "$K1_DIR/"
cp "$POLICY" "$K1_DIR/models/laai_${PHASE}.pt"
cp "$REPO_ROOT/deploy/dry_run.py" "$DEPLOY_DIR/scripts/dry_run.py"

echo "Installed phase '$PHASE' into $DEPLOY_DIR:"
for f in "$PHASE_DIR"/deploy/*.py; do echo "  tasks/locomotion/robots/k1/$(basename "$f")"; done
echo "  tasks/locomotion/robots/k1/models/laai_${PHASE}.pt  <- $POLICY"
echo "  scripts/dry_run.py"
