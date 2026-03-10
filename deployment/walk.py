"""H1 Humanoid Robot Walk Simulation with MuJoCo.

Deploys a trained PyTorch policy for H1 robot locomotion with keyboard or automated control.
Handles joint remapping, PD control, and body-frame velocity transformations.
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional

import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch
import rerun as rr
import rerun_loader_mjcf

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
    time_walk: int = 10
    trajectory_save_path: str = "h1_walk_trajectory.npz"

@dataclass
class SimulationTrajectory:
    """Recorded simulation trajectory data.
    
    Contains full state history from a simulation run, suitable for
    analysis, replay, or conversion to various formats.
    """
    timestamps: np.ndarray           # (T,) simulation times
    base_positions: np.ndarray       # (T, 3) xyz positions
    base_quaternions: np.ndarray     # (T, 4) orientations [w, x, y, z]
    base_linear_velocities: np.ndarray   # (T, 3) linear velocities
    base_angular_velocities: np.ndarray  # (T, 3) angular velocities
    joint_positions: np.ndarray      # (T, 19) joint angles
    joint_velocities: np.ndarray     # (T, 19) joint velocities
    joint_torques: np.ndarray        # (T, 19) applied torques
    
    @classmethod
    def from_lists(
        cls,
        timestamps: List[float],
        base_positions: List[np.ndarray],
        base_quaternions: List[np.ndarray],
        base_linear_velocities: List[np.ndarray],
        base_angular_velocities: List[np.ndarray],
        joint_positions: List[np.ndarray],
        joint_velocities: List[np.ndarray],
        joint_torques: List[np.ndarray],
    ) -> "SimulationTrajectory":
        """Create trajectory from lists (used during recording)."""
        return cls(
            timestamps=np.array(timestamps, dtype=np.float64),
            base_positions=np.stack(base_positions, axis=0),
            base_quaternions=np.stack(base_quaternions, axis=0),
            base_linear_velocities=np.stack(base_linear_velocities, axis=0),
            base_angular_velocities=np.stack(base_angular_velocities, axis=0),
            joint_positions=np.stack(joint_positions, axis=0),
            joint_velocities=np.stack(joint_velocities, axis=0),
            joint_torques=np.stack(joint_torques, axis=0),
        )
    
    def __len__(self) -> int:
        return len(self.timestamps)
    
    def __repr__(self) -> str:
        duration = self.timestamps[-1] - self.timestamps[0] if len(self) > 0 else 0
        return f"SimulationTrajectory(steps={len(self)}, duration={duration:.2f}s)"
    
    def save_npz(self, path: str) -> str:
        """Save trajectory to compressed numpy file."""
        np.savez_compressed(
            path,
            timestamps=self.timestamps,
            base_positions=self.base_positions,
            base_quaternions=self.base_quaternions,
            base_linear_velocities=self.base_linear_velocities,
            base_angular_velocities=self.base_angular_velocities,
            joint_positions=self.joint_positions,
            joint_velocities=self.joint_velocities,
            joint_torques=self.joint_torques,
        )
        return path
    
    @classmethod
    def load_npz(cls, path: str) -> "SimulationTrajectory":
        """Load trajectory from numpy file."""
        data = np.load(path)
        return cls(
            timestamps=data["timestamps"],
            base_positions=data["base_positions"],
            base_quaternions=data["base_quaternions"],
            base_linear_velocities=data["base_linear_velocities"],
            base_angular_velocities=data["base_angular_velocities"],
            joint_positions=data["joint_positions"],
            joint_velocities=data["joint_velocities"],
            joint_torques=data["joint_torques"],
        )
    
    def to_rerun(self, model: mujoco.MjModel, spawn: bool = True, save_path: str = None) -> Optional[str]:
        """Convert trajectory to Rerun visualization.
        
        Args:
            model: MuJoCo model for MJCF logging
            spawn: If True, spawn Rerun viewer
            save_path: If provided, save to .rrd file
            
        Returns:
            Path to saved file if save_path provided, else None
        """
        rr.init("h1_trajectory_replay", spawn=False)
        logger = rerun_loader_mjcf.MJCFLogger(model)
        logger.log_model()
        
        # Create temporary MjData to set poses for logging
        data = mujoco.MjData(model)
        
        for i, t in enumerate(self.timestamps):
            rr.set_time("sim_time", timestamp=t)
            
            # Set state in data for MJCF logger
            data.time = t
            data.qpos[:3] = self.base_positions[i]
            data.qpos[3:7] = self.base_quaternions[i]
            data.qpos[7:] = self.joint_positions[i]
            data.qvel[:3] = self.base_linear_velocities[i]
            data.qvel[3:6] = self.base_angular_velocities[i]
            data.qvel[6:] = self.joint_velocities[i]
            data.ctrl[:] = self.joint_torques[i]
            
            mujoco.mj_forward(model, data)
            logger.log_data(data)
        
        if save_path:
            rr.save(save_path)
            return save_path
        elif spawn:
            rr.spawn()
        return None



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
        self._keyboard = None
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
    """Main class to run the H1 robot walk simulation."""
    
    def __init__(self, config: SimulationConfig) -> None:
        """Initialize the simulation environment and controller.
        
        Args:
            config: Simulation configuration parameters
        """
        self._config = config or SimulationConfig()
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        
        # Instantiating the model and data before the controller to ensure they are available for observation generation
        model.opt.timestep = 0.0005
        model.opt.iterations = 1
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        data.qpos[:3] = np.array([0.0, 0.0, 1.05])  # Start slightly above the ground to avoid initial penetration
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # No initial rotation (quaternion)
        data.qpos[7:] = DEFAULT_ANGLES.copy()  # Set initial joint angles to default pose
        
        mujoco.mj_forward(model, data)
        print(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
        print(f"Initial gravity: {model.opt.gravity}")


        self._model = model
        self._data = data
        self._controller = TorchController(
            config=self._config,
            default_angles=DEFAULT_ANGLES,
        )




    def _record_trajectory(self, log_hz: float = 30.0, save_path: str = None) -> SimulationTrajectory:
        """Run simulation and return structured trajectory data.
        
        Runs the complete simulation without any viewers, recording
        all state data at the specified frequency.
        
        Args:
            log_hz: Logging frequency in Hz (default 30)
            save_path: Path to save the recorded trajectory (optional)
        Returns:
            SimulationTrajectory containing all recorded state data
        """
        log_dt = 1.0 / log_hz
        last_logged_time = -1.0
        
        # Recording buffers
        timestamps = []
        base_positions = []
        base_quaternions = []
        base_linear_velocities = []
        base_angular_velocities = []
        joint_positions = []
        joint_velocities = []
        joint_torques = []
        
        duration = self._config.time_walk + 2.0
        print(f"Recording trajectory for {duration:.1f}s at {log_hz}Hz...")
        
        while self._data.time < duration:
            # Run controller
            self._controller.get_control(self._model, self._data)
            
            # Log at fixed intervals
            current_time = self._data.time
            if current_time - last_logged_time >= log_dt:
                timestamps.append(current_time)
                base_positions.append(self._data.qpos[:3].copy())
                base_quaternions.append(self._data.qpos[3:7].copy())
                base_linear_velocities.append(self._data.qvel[:3].copy())
                base_angular_velocities.append(self._data.qvel[3:6].copy())
                joint_positions.append(self._data.qpos[7:].copy())
                joint_velocities.append(self._data.qvel[6:].copy())
                joint_torques.append(self._data.ctrl[:].copy())
                last_logged_time = current_time
            
            mujoco.mj_step(self._model, self._data)
        
        trajectory = SimulationTrajectory.from_lists(
            timestamps=timestamps,
            base_positions=base_positions,
            base_quaternions=base_quaternions,
            base_linear_velocities=base_linear_velocities,
            base_angular_velocities=base_angular_velocities,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
        )
        
        print(f"Recording complete: {trajectory}")
        trajectory.save_npz(save_path)
        return trajectory
    
    def execute(self) -> SimulationTrajectory:
        """Run the simulation with real-time visualization and logging."""
        trajectory = self._record_trajectory(log_hz=30.0, save_path=self._config.trajectory_save_path)
        # # trajectory.to_rerun(self._model, spawn=True)
        

        # # Print summary
        # print(f"\nTrajectory summary:")
        # print(f"  Duration: {trajectory.timestamps[-1]:.2f}s")
        # print(f"  Steps: {len(trajectory)}")
        # print(f"  Base position range: {trajectory.base_positions.min(axis=0)} to {trajectory.base_positions.max(axis=0)}")

        return trajectory


