import mujoco
import mujoco.viewer
import numpy as np
import torch
from pynput import keyboard

default_angles_config = np.array([
    0.0,    # LeftHipYaw
    0.0,    # LeftHipRoll
    -0.28,   # LeftHipPitch
    0.79,   # LeftKnee
    -0.52,   # LeftAnkle
    0.0,    # RightHipYaw
    0.0,    # RightHipRoll
    -0.28,   # RightHipPitch
    0.79,   # RightKnee
    -0.52,  # RightAnkle
    0.0,    # Torso
    0.28,   # LeftShoulderPitch
    0.0,   # LeftShoulderRoll
    0.0,    # LeftShoulderYaw
    0.52,   # LeftElbow
    0.28,   # RightShoulderPitch
    0.0,  # RightShoulderRoll
    0.0,    # RightShoulderYaw
    0.52,   # RightElbow
  ])

pytorch_to_mujoco_idx = [
    0,
    3,
    7,
    11,
    15,
    1,
    4,
    8,
    12,
    16,
    2,
    5,
    9,
    13,
    17,
    6,
    10,
    14,
    18
]

mujoco_to_pytorch_idx = [
    0,
    5,
    10,
    1,
    6,
    11,
    15,
    2,
    7,
    12,
    16,
    3,
    8,
    13,
    17,
    4,
    9,
    14,
    18
]

