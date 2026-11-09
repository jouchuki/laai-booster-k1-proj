# Install MuJoCo
pip install mujoco
# Clone the K1 gym environment
git clone https://github.com/BoosterRobotics/booster_gym.git
cd booster_gym
pip install -e .
# Run the walking simulation
python examples/walk_sim.py --render