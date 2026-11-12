# Booster K1 handover: setup progress

Nov 6, 2026 · @David Borensztajn

## Status at a glance

The Mac can run Booster's simulator, and the Mac can log in to the robot over Wi-Fi. The next step is training a simple walking policy, the first step toward teaching the robot to ride a bike.

| Item | Status |
| --- | --- |
| MuJoCo simulator with Booster's K1 model on the Mac | Works |
| Robot on home Wi-Fi via the Booster app | Works; the app is used again after each restart (see Part 3) |
| SSH login from Mac to robot | Works |
| Robot firmware version | v1.8.1.2 (updated 9 Nov 2026), new enough for booster\_deploy |
| Walking policy training | Not started; needs a Linux computer with an NVIDIA GPU (see Next steps) |

## Goal and background

The long-term goal is to teach the Booster K1 to ride a bike. The first milestone is training a simple walking policy in simulation and running it on the robot.

- **Booster's open-source toolchain:** [booster\_assets](https://github.com/BoosterRobotics/booster_assets) holds the K1 robot model, [booster\_train](https://github.com/BoosterRobotics/booster_train) trains policies in Isaac Lab, and [booster\_deploy](https://github.com/BoosterRobotics/booster_deploy) runs a trained policy in MuJoCo or on the real robot.
- **Training needs an NVIDIA GPU:** Isaac Lab and mjlab both train on Linux with an NVIDIA GPU. The Mac can view and test policies in MuJoCo, but is not suited for training.
- **Real robot:** running your own policy on the K1 needs firmware v1.7.2 or newer.

## Part 1: Mac setup (done)

The simulator runs in a conda environment called `booster` with Python 3.12 on an Apple Silicon (arm64) MacBook Pro.

1. **Downloaded Booster's code** into one folder:

   ```bash
   mkdir ~/booster && cd ~/booster
   git clone https://github.com/BoosterRobotics/booster_deploy.git
   git clone https://github.com/BoosterRobotics/booster_assets.git
   ```
2. **Hit an install error:** `Failed building wheel for mujoco`. Cause: the default Python was 3.9.12 from Anaconda, and MuJoCo needs Python 3.10 or newer.
3. **Fixed it with a new conda environment** on Python 3.12:

   ```bash
   cd ~/booster
   conda create -n booster python=3.12 -y
   conda activate booster
   pip install torch numpy mujoco scipy pyyaml onnxruntime
   pip install -e booster_assets
   ```

   `requirements.txt` is not used on the Mac, because it contains `evdev`, which only works on Linux.
4. **Checked that the simulator works:**

   ```bash
   cd ~/booster/booster_deploy
   python scripts/deploy.py --list
   mjpython scripts/deploy.py --task <task> --mujoco
   ```

   The list shows Booster's ready-trained example policies, including `k1_walk`, a walking policy for the K1. On a Mac the simulator window needs `mjpython` instead of `python`.

Folder layout on the Mac:

```
~/booster/
├── booster_assets/   (K1 robot model)
└── booster_deploy/   (runs policies in MuJoCo or on the robot)
```

Cleanup: an old `.venv` folder from the first attempt may still sit in `~/booster`. It is not used and can be removed with `rm -rf ~/booster/.venv`.

## Part 2: Connecting to the robot (done)

The robot joins the home Wi-Fi through the Booster app, and the Mac reaches it over SSH on the `10.0.0.x` network.

1. **Turn on the robot first** and wait about a minute for it to boot before connecting.
2. **Booster app:** on iPhone it is called "Booster OS" (Booster Robotics Technology Co., Ltd). The app talks to the robot over Bluetooth. In the app, choose Wi-Fi connect and join the robot to the lab's TP-Link network. The Mac must be on the same TP-Link network.
3. **Found the robot's IP** (the app did not show it) by scanning the network from the Mac for devices that accept SSH:

   ```bash
   SUBNET=$(ipconfig getifaddr en0 | cut -d. -f1-3); echo "Scanning $SUBNET.1-254 ..."
   ( for i in {1..254}; do (nc -z -G 1 $SUBNET.$i 22 2>/dev/null && echo "SSH open: $SUBNET.$i") & done; wait )
   ```

   If several addresses show up, find the robot by trying each one (type `yes` if asked). The robot is the one that prints `robot`:

   ```bash
   for ip in 165 172 182 238; do echo "--- 10.0.0.$ip"; ssh -o ConnectTimeout=3 booster@10.0.0.$ip hostname; done
   ```

   Replace the numbers after `for ip in` with the last part of the addresses the scan found.

   The robot is at `10.0.0.238`.
4. **Logged in:**

   ```bash
   ssh booster@10.0.0.238
   ```

   Password: ask a teammate (never commit it). The robot runs Ubuntu 22.04 (aarch64) with hostname `robot`; the prompt reads `booster@robot:~$`.

## Part 3: Wi-Fi persistence check (no changes made)

The Wi-Fi connection is already saved permanently on the robot, so nothing was changed. The cause of having to reconnect through the app after each restart is still unknown.

- `nmcli` shows the app-made connection `ble-provision-wifi` on `wlan0`, with autoconnect set to yes.
- It is stored in `/etc/NetworkManager/system-connections/`, which survives a restart (it is not temporary).
- Likely explanations: the app always needs a new Bluetooth link after a restart (separate from Wi-Fi), the robot gets a different IP after a restart, or Booster's own software resets the Wi-Fi on boot.

Test for next time: turn the robot on, wait about 2 minutes without using the app, then run the network scan from Part 2. If the robot shows up, Wi-Fi reconnects on its own and only the IP needs looking up.

## Next steps: training a walking policy

The main blocker is hardware: training needs a Linux computer with an NVIDIA GPU, and Booster has no ready-made K1 walking training task yet. `booster_train` currently only has motion-imitation tasks for the K1, and `booster_gym`'s walking setup is for the T1.

- [ ] Watch Booster's ready-trained K1 walk as a baseline: `mjpython scripts/deploy.py --task k1_walk --mujoco`, then type speed commands (x, y, yaw) in the terminal
- [ ] Get access to a Linux machine with an NVIDIA GPU (own PC, university lab or cloud)
- [ ] Pick a training setup for a K1 walking task (open decision): [mjlab](https://github.com/mujocolab/mjlab)'s velocity task adapted to the K1 model from `booster_assets`, the community K1 workspace [umn-cse-mnri/booster\_ws](https://github.com/umn-cse-mnri/booster_ws) (Isaac Lab), or a custom task in `booster_train`
- [ ] Train, export the policy (ONNX or TorchScript), and test it on the Mac in MuJoCo by adding it as a task in `booster_deploy`
- [ ] Update the robot firmware: now v1.5, needs v1.7.2 or newer (check with `cat /opt/booster/version.txt`)
- [ ] Run the Wi-Fi test from Part 3

Once a policy works in simulation, Booster's README gives these steps to run it on the robot:

1. Copy the code from the Mac: `scp -r ~/booster/booster_deploy booster@10.0.0.238:~/`
2. On the robot, once: `cd ~/booster_deploy`, `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt` (needs internet; `torch` is a large download)
3. Each run, on the robot: `source .venv/bin/activate`, then `source /opt/booster/BoosterRos2Interface/install/setup.bash`, then `python3 scripts/deploy.py --task <task>`
4. With the robot standing: press **X** (program takes over), then **A** (policy starts); **Ctrl+C** in the terminal stops it
5. Test first with `--task k1_walk`, and keep a second person ready to catch the robot (about 20 kg)

## Quick reference

| What | Value or command |
| --- | --- |
| Robot login | `ssh booster@10.0.0.238` |
| Robot IP | `10.0.0.238` |
| Find the robot IP | Network scan command in Part 2 |
| Start Mac environment | `conda activate booster` then `cd ~/booster/booster_deploy` |
| List example policies | `python scripts/deploy.py --list` |
| Run the example walk in the simulator | `mjpython scripts/deploy.py --task k1_walk --mujoco` |
| Robot firmware version | `cat /opt/booster/version.txt` |
| Robot Wi-Fi connection | `ble-provision-wifi` (NetworkManager) |

## Sources

- [BoosterRobotics/booster\_deploy](https://github.com/BoosterRobotics/booster_deploy) (running policies in MuJoCo and on the robot, remote buttons)
- [BoosterRobotics/booster\_assets](https://github.com/BoosterRobotics/booster_assets) (K1 robot model)
- [BoosterRobotics/booster\_train](https://github.com/BoosterRobotics/booster_train) (Isaac Lab training tasks)
- [BoosterRobotics/booster\_gym](https://github.com/BoosterRobotics/booster_gym) (walking training for the T1)
- [mjlab installation guide](https://mujocolab.github.io/mjlab/main/source/installation.html) (training needs Linux and an NVIDIA GPU)
- [umn-cse-mnri/booster\_ws](https://github.com/umn-cse-mnri/booster_ws) (community K1 training workspace)
- [K1 Instruction Manual V1.0 (PDF)](https://funduinoshop.com/media/pdf/66/1f/7a/Booster-K1-Instruction-Manual-V1-0-Feishu-Docs.pdf) (app, login, firmware check)
- [Booster OS app listing](https://appgoblin.info/apps/6746039279)
