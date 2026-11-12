#!/usr/bin/env bash
# Linux and macOS host: install an experiment into booster_deploy, copy it to the K1 and set it up there.
#
# Usage (from anywhere):
#   bash robot/setup_robot.sh <robot-ip> <experiment>                 # policy of the newest run
#   bash robot/setup_robot.sh <robot-ip> <experiment> <policy.pt>     # a specific policy
#
# You type the robot password once (ask a teammate; never commit it).
# The robot needs internet (lab Wi-Fi) for pip.
set -euo pipefail

USAGE="Usage: bash robot/setup_robot.sh <robot-ip> <experiment> [policy.pt]"
ROBOT_IP="${1:?$USAGE}"
EXPERIMENT="${2:?$USAGE}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/booster_paths.sh"
DEPLOY_DIR="$(booster_checkout_path "$REPO_ROOT" booster_deploy)"
TARGET="booster@${ROBOT_IP}"
# Reuse one SSH connection, so the password is asked only once.
SSH_OPTS=(-o ControlMaster=auto -o ControlPath=/tmp/k1-ssh-%r@%h -o ControlPersist=10m)

echo "== 1/4 Installing experiment '$EXPERIMENT' into booster_deploy"
bash "$REPO_ROOT/robot/install.sh" "$EXPERIMENT" "${@:3}"

echo "== 2/4 Connecting to the robot"
ssh "${SSH_OPTS[@]}" "$TARGET" \
  'echo "Connected to $(hostname)"; echo "Firmware: $(cat /opt/booster/version.txt)"; python3 --version'

echo "== 3/4 Copying booster_deploy to the robot (~/booster_deploy)"
COPYFILE_DISABLE=1 tar czf - -C "$(dirname "$DEPLOY_DIR")" \
  --exclude .git --exclude __pycache__ --exclude .venv --exclude logs "$(basename "$DEPLOY_DIR")" \
  | ssh "${SSH_OPTS[@]}" "$TARGET" 'tar xzf - -C ~'

echo "== 4/4 Installing on the robot (can take a few minutes)"
ssh "${SSH_OPTS[@]}" "$TARGET" 'bash -s' <<'REMOTE'
set -e
cd ~/booster_deploy
if [ ! -d .venv ]; then
  # --system-site-packages reuses the PyTorch that is already installed on the robot.
  python3 -m venv --system-site-packages .venv || {
    echo "Could not create a virtual environment. On the robot: sudo apt install -y python3-venv"
    exit 1
  }
fi
source .venv/bin/activate
python3 -m pip install --upgrade pip -q
python3 -m pip install -r requirements.txt -q || {
  echo "pip install failed. If it mentions evdev/gcc/Python.h, on the robot:"
  echo "  sudo apt install -y python3-dev gcc   and run this script again."
  exit 1
}
python3 -c "import torch, numpy, mujoco, scipy, evdev, yaml, onnxruntime; print('Python packages OK (torch', torch.__version__ + ')')"
source /opt/booster/BoosterRos2Interface/install/setup.bash
python3 scripts/deploy.py --list | grep laai_
REMOTE

echo
echo "Done. Next on the robot: dry run, then the robot test (docs/commands.md, section Robot test)."
