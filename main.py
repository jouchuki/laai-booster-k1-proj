from boosteros.robots.booster import BoosterRobot

robot = BoosterRobot()

info = robot.robot_info
print(f"{info.manufacturer}, {info.model}, {info.serial_number}")
# Example output: Booster Robotics, Booster K1, rivt9

mode = robot.get_mode()
print(f"mode={mode}")
# Example output: mode=prepare

joints = robot.list_joints()
print(f"joints={len(joints)}")
# Example output: joints=22