#!/bin/bash
# Copy booster_deploy (with your k1_my_walk policy) from this Mac to the Booster K1 and install it.
#
#   chmod +x setup_robot.sh
#   ./setup_robot.sh 10.0.0.239        <- use your robot's IP
#
# You type the robot password (123456) once. The robot needs internet (lab Wi-Fi) for pip.
set -e

ROBOT_IP="${1:?Usage: ./setup_robot.sh <robot-ip>}"
TARGET="booster@${ROBOT_IP}"
# Reuse one SSH connection, so the password is asked only once.
SSH_OPTS=(-o ControlMaster=auto -o ControlPath=/tmp/k1-ssh-%r@%h -o ControlPersist=10m)

cd ~/booster
if [ ! -f booster_deploy/tasks/locomotion/robots/k1/models/k1_my_walk.pt ]; then
  echo "k1_my_walk.pt not found in booster_deploy/tasks/locomotion/robots/k1/models/ - copy it there first."
  exit 1
fi

echo "== 1/3 Connecting to the robot"
ssh "${SSH_OPTS[@]}" "$TARGET" 'echo "Connected to $(hostname)"; echo "Firmware: $(cat /opt/booster/version.txt)"; python3 --version'

echo "== 2/3 Copying booster_deploy to the robot (~/booster_deploy)"
COPYFILE_DISABLE=1 tar czf - --exclude .git --exclude __pycache__ --exclude .venv booster_deploy \
  | ssh "${SSH_OPTS[@]}" "$TARGET" 'tar xzf - -C ~'

echo "== 3/3 Installing on the robot (can take a few minutes)"
ssh "${SSH_OPTS[@]}" "$TARGET" 'bash -s' <<'REMOTE'
set -e
cd ~/booster_deploy
if [ ! -d .venv ]; then
  # --system-site-packages reuses the PyTorch that is already installed on the robot.
  python3 -m venv --system-site-packages .venv || {
    echo "Could not create a virtual environment. Run on the robot: sudo apt install -y python3-venv  and try again."
    exit 1
  }
fi
source .venv/bin/activate
python3 -m pip install --upgrade pip -q
python3 -m pip install -r requirements.txt -q || {
  echo "pip install failed. If it mentions evdev/gcc/Python.h, run on the robot:"
  echo "  sudo apt install -y python3-dev gcc   and run this script again."
  exit 1
}
python3 -c "import torch, numpy, mujoco, scipy, evdev, yaml, onnxruntime; print('Python packages OK (torch', torch.__version__ + ')')"
source /opt/booster/BoosterRos2Interface/install/setup.bash
python3 scripts/deploy.py --list | grep -q k1_my_walk && echo "k1_my_walk found - robot is ready."
REMOTE

echo
echo "Done. To run your policy:"
echo "  ssh $TARGET"
echo "  cd ~/booster_deploy && source .venv/bin/activate && source /opt/booster/BoosterRos2Interface/install/setup.bash"
echo "  python3 scripts/deploy.py --task k1_my_walk"
