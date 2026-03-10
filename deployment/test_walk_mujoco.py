"""H1 Humanoid Robot Walk Simulation with MuJoCo.

Deploys a trained PyTorch policy for H1 robot locomotion with keyboard or automated control.
Handles joint remapping, PD control, and body-frame velocity transformations.
"""

import argparse
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch

from keyboard_reader import KeyboardController

# Default joint angles for standing pose (MuJoCo order)
DEFAULT_ANGLES = np.array([
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

# Joint index mappings between PyTorch policy order and MuJoCo simulation order
MUJOCO_TO_PYTORCH_IDX = [
    0, 3, 7, 11, 15, 1, 4, 8, 12, 16, 
    2, 5, 9, 13, 17, 6, 10, 14, 18
]
PYTORCH_TO_MUJOCO_IDX = [
    0, 5, 10, 1, 6, 11, 15, 2, 7, 12,
    16, 3, 8, 13, 17, 4, 9, 14, 18
]

# PD control gains
KP_GAINS = np.array([
    150.0, 150.0, 200.0, 200.0, 20.0,   # Left leg
    150.0, 150.0, 200.0, 200.0, 20.0,   # Right leg
    200.0,                               # Torso
    40.0, 40.0, 40.0, 40.0,             # Left arm
    40.0, 40.0, 40.0, 40.0              # Right arm
], dtype=np.float32) / 4

KD_GAINS = np.array([
    5.0, 5.0, 5.0, 5.0, 4.0,    # Left leg
    5.0, 5.0, 5.0, 5.0, 4.0,    # Right leg
    5.0,                         # Torso
    10.0, 10.0, 10.0, 10.0,     # Left arm
    10.0, 10.0, 10.0, 10.0      # Right arm
], dtype=np.float32) / 4



def remap_pytorch_to_mujoco(pytorch_actions: np.ndarray) -> np.ndarray:
    """Remap actions from PyTorch model joint order to MuJoCo joint order.
    
    Args:
        pytorch_actions: Joint actions in PyTorch policy order (19,)
        
    Returns:
        Joint actions in MuJoCo simulation order (19,)
    """
    mujoco_actions = np.zeros_like(pytorch_actions)
    for pytorch_idx, mujoco_idx in enumerate(PYTORCH_TO_MUJOCO_IDX):
        mujoco_actions[mujoco_idx] = pytorch_actions[pytorch_idx]
    return mujoco_actions


def remap_mujoco_to_pytorch(mujoco_data: np.ndarray) -> np.ndarray:
    """Remap data from MuJoCo joint order to PyTorch model joint order.
    
    Args:
        mujoco_data: Joint data in MuJoCo simulation order (19,)
        
    Returns:
        Joint data in PyTorch policy order (19,)
    """
    pytorch_data = np.zeros_like(mujoco_data)
    for pytorch_idx, mujoco_idx in enumerate(PYTORCH_TO_MUJOCO_IDX):
        pytorch_data[pytorch_idx] = mujoco_data[mujoco_idx]
    return pytorch_data




# Default file paths
POLICY_PATH = "assets/IsaacH1FlatPlayRSL_policy_5499_iterations.pt"
SCENE_PATH = "./assets/h1_description/mjcf/scene.xml"
IMU_SITE_NAME = "imu"


@dataclass
class SimulationConfig:
    """Configuration parameters for H1 robot simulation."""
    policy_path: str = POLICY_PATH
    n_substeps: int = 4
    action_scale: float = 0.5
    vel_scale_x: float = 0.5
    vel_scale_y: float = 0.5
    vel_scale_rot: float = 1.0
    keyboard: bool = True
    time_walk: int = 10


class TorchController:
    """Controls H1 robot using a trained PyTorch policy with PD control.
    
    Handles observation generation, policy inference, joint remapping,
    and PD torque computation for MuJoCo simulation.
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        default_angles: np.ndarray,
    ) -> None:
        """Initialize the controller.
        
        Args:
            config: Simulation configuration parameters
            default_angles: Default joint angles in MuJoCo order (19,)
        """
        self._policy = torch.jit.load(config.policy_path)
        self._policy.eval()

        self._action_scale = config.action_scale
        self._default_angles = default_angles
        self._last_action = default_angles.copy()

        self._counter = 0
        self._n_substeps = config.n_substeps
        self._keyboard = config.keyboard
        self._start_time = None
        self._vel_scale_x = config.vel_scale_x
        self.time_walk = config.time_walk

        if self._keyboard:
            self._controller = KeyboardController(
                vel_scale_x=config.vel_scale_x,
                vel_scale_y=config.vel_scale_y,
                vel_scale_rot=config.vel_scale_rot,
            )
        else:
            self._controller = None

    def _get_obs(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """Generate observation vector for policy.
        
        Constructs 69-dimensional observation: base velocities (6), projected gravity (3),
        velocity commands (3), joint positions (19), joint velocities (19), last actions (19).
        
        Args:
            model: MuJoCo model
            data: MuJoCo simulation data
            
        Returns:
            Observation array (69,) in float32
        """

        world_gravity = model.opt.gravity
        world_gravity /= np.linalg.norm(world_gravity)
        imu_xmat = data.site_xmat[model.site(IMU_SITE_NAME).id].reshape(3, 3)
        projected_gravity = imu_xmat.T @ world_gravity
        
        # Get velocity commands and transform to robot's body frame
        # The policy was trained with commands in the yaw frame (robot's local coordinates)
        
        # Get robot's yaw angle from quaternion
        quat = data.qpos[3:7]  # [w, x, y, z]
        yaw = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                        1.0 - 2.0 * (quat[2]**2 + quat[3]**2))
        
        # Get velocity commands based on mode
        if self._keyboard:
            # Pass yaw to controller so it can transform commands in body frame BEFORE smoothing
            velocity_commands = self._controller.get_command(yaw)  # Already in body frame
        else:
            # Automated mode: 1s wait, W for 5s, 1s wait, Q for 3s, 1s wait, E for 3s, 1s wait, W for 6s
            if self._start_time is None:
                self._start_time = time.time()
            
            elapsed = time.time() - self._start_time
            
            vx_world = 0.0
            vy_world = 0.0
            vyaw_world = 0.0
            
            if elapsed < 1.0:
                # Wait 1 second
                pass
            if elapsed < self.time_walk + 1.0:
                vx_world = self._vel_scale_x

            # if elapsed < 1.0:
            #     # Wait 1 second
            #     pass
            # elif elapsed < 6.0:
            #     # W for 5 seconds (1s to 6s) - forward
            #     vx_world = self._vel_scale_x
            # elif elapsed < 7.0:
            #     # Wait 1 second (6s to 7s)
            #     pass
            # elif elapsed < 8.5:
            #     # Q for 3 seconds (7s to 10s) - rotate left
            #     vyaw_world = self._vel_scale_x  # Positive yaw
            # elif elapsed < 12.0:
            #     # Wait 1 second (10s to 11s)
            #     pass
            # elif elapsed < 15.5:
            #     # E for 3 seconds (11s to 14s) - rotate right
            #     vyaw_world = -self._vel_scale_x  # Negative yaw
            # elif elapsed < 18.0:
            #     # Wait 1 second (14s to 15s)
            #     pass
            # elif elapsed < 22.0:
            #     # W for 3 seconds (15s to 18s) - forward
            #     vx_world = self._vel_scale_x
            
            # Transform to body frame
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            vx_body = cos_yaw * vx_world - sin_yaw * vy_world
            vy_body = sin_yaw * vx_world + cos_yaw * vy_world
            
            velocity_commands = np.array([vx_body, vy_body, vyaw_world], dtype=np.float32)

        joint_pos_mujoco = data.qpos[7:]  - self._default_angles# Exclude global position and yaw (23,)
        joint_vel_mujoco = data.qvel[6:]  # Exclude global vel (23,)

        joint_pos_pytorch = remap_mujoco_to_pytorch(joint_pos_mujoco)
        joint_vel_pytorch = remap_mujoco_to_pytorch(joint_vel_mujoco)
        last_action_pytorch = remap_mujoco_to_pytorch(self._last_action)
        base_ang_vel = data.qvel[3:6]
        base_lin_vel = data.qvel[0:3]

        obs = np.hstack([
                base_lin_vel, #3
                base_ang_vel, #3
                projected_gravity, #3
                velocity_commands, #3
                joint_pos_pytorch, #19
                joint_vel_pytorch, #19
                last_action_pytorch, #19
        ])

        return obs.astype(np.float32)
    
    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """MuJoCo control callback - computes and applies joint torques.
        
        Called every simulation step. Runs policy inference every n_substeps.
        
        Args:
            model: MuJoCo model
            data: MuJoCo simulation data (modified in-place)
        """
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self._get_obs(model, data)
            
            obs_tensor = torch.from_numpy(obs).float()
            
            with torch.no_grad():
                action_tensor = self._policy(obs_tensor)
                pytorch_pred = action_tensor.numpy()

            mujoco_pred = remap_pytorch_to_mujoco(pytorch_pred)
            self._last_action = mujoco_pred.copy()
            
            desired_pos = mujoco_pred * self._action_scale + self._default_angles

            n_joints = desired_pos.shape[0]
            qpos_joints = np.array(data.qpos[7:7+n_joints], dtype=np.float32)
            qvel_joints = np.array(data.qvel[6:6+n_joints], dtype=np.float32)

            torques = KP_GAINS * (desired_pos - qpos_joints) + KD_GAINS * (0.0 - qvel_joints)

            ctrl_min = model.actuator_ctrlrange[:n_joints, 0]
            ctrl_max = model.actuator_ctrlrange[:n_joints, 1]
            torques = np.clip(torques, ctrl_min, ctrl_max)

            data.ctrl[:] = torques


class Walk:
    """Main simulation orchestrator for H1 robot walk.
    
    Initializes MuJoCo simulation and policy controller, then launches interactive viewer.
    """
    
    def __init__(self, config: SimulationConfig = None) -> None:
        """Initialize walk simulation.
        
        Args:
            config: Simulation configuration. If None, uses default configuration.
        """
        self._config = config or SimulationConfig()
        self._load_model()
        self._controller = TorchController(
            config=self._config,
            default_angles=DEFAULT_ANGLES,
        )
    
    def _load_model(self) -> None:
        """Load MuJoCo model and initialize simulation data.
        
        Sets up physics parameters, initial pose, and forward kinematics.
        Initializes self._model and self._data.
        """
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)

        model.opt.timestep = 0.0005
        model.opt.iterations = 1
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        data.qpos[:3] = np.array([0, 0, 1.05])
        data.qpos[3:7] = np.array([1, 0, 0, 0])
        data.qpos[7:] = DEFAULT_ANGLES.copy()

        mujoco.mj_forward(model, data)

        print(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
        print(f"Initial gravity: {model.opt.gravity}")

        self._model = model
        self._data = data
    
    def execute(self) -> None:
        """Launch MuJoCo viewer and start simulation."""
        mujoco.set_mjcb_control(self._controller.get_control)
        viewer.launch(self._model, self._data)


def main() -> None:
    """CLI entry point for H1 robot walk simulation."""
    parser = argparse.ArgumentParser(description="H1 Robot Walk Simulation")
    parser.add_argument(
        "--policy_path",
        type=str,
        default=POLICY_PATH,
        help="Path to the PyTorch policy model.",
    )
    parser.add_argument(
        "--n_substeps",
        type=int,
        default=4,
        help="Number of simulation substeps per control step.",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=0.5,
        help="Scaling factor for the actions output by the policy.",
    )
    parser.add_argument(
        "--vel_scale_x",
        type=float,
        default=0.5,
        help="Scaling factor for forward/backward velocity commands.",
    )
    parser.add_argument(
        "--vel_scale_y",
        type=float,
        default=0.5,
        help="Scaling factor for lateral velocity commands.",
    )
    parser.add_argument(
        "--vel_scale_rot",
        type=float,
        default=1.0,
        help="Scaling factor for rotational velocity commands.",
    )
    parser.add_argument(
        "--no-keyboard",
        dest="keyboard",
        action="store_false",
        default=True,
        help="Disable keyboard control and use automated mode (1s wait, 4s forward).",
    )
    parser.add_argument(
        "--time_walk",
        type=int,
        default=10,
        help="Number of seconds to run locomotion.",
    )
    

    args = parser.parse_args()

    config = SimulationConfig(
        policy_path=args.policy_path,
        n_substeps=args.n_substeps,
        action_scale=args.action_scale,
        vel_scale_x=args.vel_scale_x,
        vel_scale_y=args.vel_scale_y,
        vel_scale_rot=args.vel_scale_rot,
        keyboard=args.keyboard,
        time_walk=args.time_walk,
    )

    walk_simulation = Walk(config=config)
    walk_simulation.execute()


if __name__ == "__main__":
    main()
    
