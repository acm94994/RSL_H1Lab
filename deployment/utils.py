# from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ 
import numpy as np
import select
import tty
import termios
import sys

G1_NUM_MOTOR = 23


# default_angles_config = np.array([
#     0.0,    # LeftHipYaw
#     0.0,    # LeftHipRoll
#     -0.2,   # LeftHipPitch
#     0.42,   # LeftKnee
#     -0.23,  # LeftAnkle
#     0.0,    # RightHipYaw
#     0.0,    # RightHipRoll
#     -0.2,   # RightHipPitch
#     0.42,   # RightKnee
#     -0.23,  # RightAnkle
#     0.0,    # Torso
#     0.35,   # LeftShoulderPitch
#     0.16,   # LeftShoulderRoll
#     0.0,    # LeftShoulderYaw
#     0.87,   # LeftElbow
#     0.35,   # RightShoulderPitch
#     -0.16,  # RightShoulderRoll
#     0.0,    # RightShoulderYaw
#     0.87,   # RightElbow
#   ])

default_angles_config = np.array([
    0.0,    # LeftHipYaw
    0.0,    # LeftHipRoll
    0.0,   # LeftHipPitch
    0.0,   # LeftKnee
    0.0,  # LeftAnkle
    0.0,    # RightHipYaw
    0.0,    # RightHipRoll
    0.0,   # RightHipPitch
    0.0,   # RightKnee
    0.0,  # RightAnkle
    0.0,    # Torso
    0.0,   # LeftShoulderPitch
    0.0,   # LeftShoulderRoll
    0.0,    # LeftShoulderYaw
    0.0,   # LeftElbow
    0.0,   # RightShoulderPitch
    0.0,  # RightShoulderRoll
    0.0,    # RightShoulderYaw
    0.0,   # RightElbow
  ])



class H1JointIndex:
    """Joint indices based on the order in h1_description/scene_mjx.xml (23 DoF model)."""
    LeftHipYaw = 0
    LeftHipRoll = 1
    LeftHipPitch = 2
    LeftKnee = 3
    LeftAnkle = 4
    RightHipYaw = 5
    RightHipRoll = 6
    RightHipPitch = 7
    RightKnee = 8
    RightAnkle = 9
    Torso = 10
    LeftShoulderPitch = 11
    LeftShoulderRoll = 12
    LeftShoulderYaw = 13
    LeftElbow = 14
    RightShoulderPitch = 15
    RightShoulderRoll = 16
    RightShoulderYaw = 17
    RightElbow = 18

class H1PyTorchJointIndex:
    """Joint indices based on the order in h1_description/scene_mjx.xml (23 DoF model)."""
    LeftHipYaw = 0
    LeftHipRoll = 1
    LeftHipPitch = 2
    LeftKnee = 3
    LeftAnkle = 4
    RightHipYaw = 5
    RightHipRoll = 6
    RightHipPitch = 7
    RightKnee = 8
    RightAnkle = 9
    Torso = 10
    LeftShoulderPitch = 11
    LeftShoulderRoll = 12
    LeftShoulderYaw = 13
    LeftElbow = 14
    RightShoulderPitch = 15
    RightShoulderRoll = 16
    RightShoulderYaw = 17
    RightElbow = 18

pytorch2mujoco_idx = [
    # PyTorch idx -> MuJoCo idx
    H1JointIndex.LeftHipYaw,        # 5: left_hip_yaw_joint -> LeftHipYaw (2)
    H1JointIndex.LeftHipRoll,       # 3: left_hip_roll_joint -> LeftHipRoll (1)
    H1JointIndex.LeftHipPitch,      # 0: left_hip_pitch_joint -> LeftHipPitch (0)
    H1JointIndex.LeftKnee,          # 7: left_knee_joint -> LeftKnee (3)
    H1JointIndex.LeftAnkle,        # 11: left_ankle_joint -> LeftAnkle (4)
    H1JointIndex.RightHipYaw,       # 6: right_hip_yaw_joint -> RightHipYaw (8)
    H1JointIndex.RightHipRoll,      # 4: right_hip_roll_joint -> RightHipRoll (7)
    H1JointIndex.RightHipPitch,     # 1: right_hip_pitch_joint -> RightHipPitch (6)
    H1JointIndex.RightKnee,         # 8: right_knee_joint -> RightKnee (9)
    H1JointIndex.RightAnkle,   # 12: right_ankle_pitch_joint -> RightAnklePitch (10)
    H1JointIndex.Torso,          # 2: waist_yaw_joint -> WaistYaw (12)
    H1JointIndex.LeftShoulderPitch, # 9: left_shoulder_pitch_joint -> LeftShoulderPitch (13)
    H1JointIndex.LeftShoulderRoll,  # 13: left_shoulder_roll_joint -> LeftShoulderRoll (14)
    H1JointIndex.LeftShoulderYaw,   # 17: left_shoulder_yaw_joint -> LeftShoulderYaw (15)
    H1JointIndex.LeftElbow,         # 19: left_elbow_joint -> LeftElbow (16)
    H1JointIndex.RightShoulderPitch,# 10: right_shoulder_pitch_joint -> RightShoulderPitch (18)
    H1JointIndex.RightShoulderRoll, # 14: right_shoulder_roll_joint -> RightShoulderRoll (19)
    H1JointIndex.RightShoulderYaw,  # 18: right_shoulder_yaw_joint -> RightShoulderYaw (20)
    H1JointIndex.RightElbow,        # 20: right_elbow_joint -> RightElbow (21)
]

# This will be automatically generated in init_joint_mappings()
mujoco2pytorch_idx = [0] * 19


def init_joint_mappings():
    """Initialize the inverse mapping from MuJoCo to PyTorch indices."""
    global mujoco2pytorch_idx
    for pytorch_idx, mujoco_idx in enumerate(pytorch2mujoco_idx):
        mujoco2pytorch_idx[mujoco_idx] = pytorch_idx


def remap_pytorch_to_mujoco(pytorch_actions: np.ndarray) -> np.ndarray:
    """Remap actions from PyTorch model joint order to MuJoCo joint order."""
    mujoco_actions = np.zeros_like(pytorch_actions)
    # print(f"pytorch_actions shape: {pytorch_actions.shape}")
    # print(f"mujoco_actions shape: {mujoco_actions.shape}")
    for pytorch_idx, mujoco_idx in enumerate(pytorch2mujoco_idx):
        # print(f"Mapping PyTorch idx {pytorch_idx} to MuJoCo idx {mujoco_idx}")
        mujoco_actions[mujoco_idx] = pytorch_actions[pytorch_idx]
    return mujoco_actions


def remap_mujoco_to_pytorch(mujoco_data: np.ndarray) -> np.ndarray:
    """Remap data from MuJoCo joint order to PyTorch model joint order."""
    pytorch_data = np.zeros_like(mujoco_data)
    for pytorch_idx, mujoco_idx in enumerate(pytorch2mujoco_idx):
        pytorch_data[pytorch_idx] = mujoco_data[mujoco_idx]
    return pytorch_data
