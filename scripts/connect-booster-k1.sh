git clone https://github.com/BoosterRobotics/booster_deploy.git 
git clone https://github.com/BoosterRobotics/booster_assets.git
git clone https://github.com/BoosterRobotics/booster_train.git
uv pip install torch numpy mujoco scipy pyyaml onnxruntime 
uv pip install -e booster_assets