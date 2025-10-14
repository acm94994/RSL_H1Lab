import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch
from keyboard_reader import KeyboardController

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
    RightHipYaw = 1
    Torso = 2
    LeftHipRoll = 3
    RightHipRoll = 4
    LeftShoulderPitch = 5
    RightShoulderPitch = 6
    LeftHipPitch = 7
    RightHipPitch = 8
    LeftShoulderRoll = 9
    RightShoulderRoll = 10
    LeftKnee = 11
    RightKnee = 12
    LeftShoulderYaw = 13
    RightShoulderYaw = 14
    LeftAnkle = 15
    RightAnkle = 16
    LeftElbow = 17
    RightElbow = 18

pytorch2mujoco_idx = [
    # PyTorch idx -> MuJoCo idx
    H1JointIndex.LeftHipYaw,        # 5: left_hip_yaw_joint -> LeftHipYaw (2)
    H1JointIndex.RightHipYaw,       # 6: right_hip_yaw_joint -> RightHipYaw (8)
    H1JointIndex.Torso,          # 2: waist_yaw_joint -> WaistYaw (12)
    H1JointIndex.LeftHipRoll,       # 3: left_hip_roll_joint -> LeftHipRoll (1)
    H1JointIndex.RightHipRoll,      # 4: right_hip_roll_joint -> RightHipRoll (7)
    H1JointIndex.LeftShoulderPitch, # 9: left_shoulder_pitch_joint -> LeftShoulderPitch (13)
    H1JointIndex.RightShoulderPitch,# 10: right_shoulder_pitch_joint -> RightShoulderPitch (18)
    H1JointIndex.LeftHipPitch,      # 0: left_hip_pitch_joint -> LeftHipPitch (0)
    H1JointIndex.RightHipPitch,     # 1: right_hip_pitch_joint -> RightHipPitch (6)
    H1JointIndex.LeftShoulderRoll,  # 13: left_shoulder_roll_joint -> LeftShoulderRoll (14)
    H1JointIndex.RightShoulderRoll, # 14: right_shoulder_roll_joint -> RightShoulderRoll (19)
    H1JointIndex.LeftKnee,          # 7: left_knee_joint -> LeftKnee (3)
    H1JointIndex.RightKnee,         # 8: right_knee_joint -> RightKnee (9)
    H1JointIndex.LeftShoulderYaw,   # 17: left_shoulder_yaw_joint -> LeftShoulderYaw (15)
    H1JointIndex.RightShoulderYaw,  # 18: right_shoulder_yaw_joint -> RightShoulderYaw (20)
    H1JointIndex.LeftAnkle,        # 11: left_ankle_joint -> LeftAnkle (4)
    H1JointIndex.RightAnkle,   # 12: right_ankle_pitch_joint -> RightAnklePitch (10)
    H1JointIndex.LeftElbow,         # 19: left_elbow_joint -> LeftElbow (16)
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




POLICY_PATH = "assets/policy.pt"


class TorchController:
    def __init__(
        self,
        policy_path: str,
        default_angles: np.ndarray,
        n_substeps: int,
        action_scale: float = 0.5,
        vel_scale_x: float = 1.0,
        vel_scale_y: float = 1.0,
        vel_scale_rot: float = 1.0,
    ):
        # self._policy = torch.load(policy_path, weights_only=False)
        self._policy = torch.jit.load(policy_path)
        self._policy.eval()  # Set to evaluation mode

        self._action_scale = action_scale
        self._default_angles = default_angles
        # self._last_action = np.zeros_like(default_angles, dtype=np.float32)  # In MuJoCo order
        self._last_action = default_angles.copy()  # In MuJoCo order

        self._counter = 0
        self._n_substeps = n_substeps

        self._controller = KeyboardController(
            vel_scale_x=vel_scale_x,
            vel_scale_y=vel_scale_y,
            vel_scale_rot=vel_scale_rot,
        )

    # self.obs_encoder = torch.nn.Linear(63, 256) # Observation space = 63

    # Initialize joint mappings
    init_joint_mappings()

    def get_obs(self, model, data) -> np.ndarray:
        #69 + 187 = 256
        """Get the observation for the policy."""

        world_gravity = model.opt.gravity
        # print(world_gravity)
        world_gravity /= np.linalg.norm(world_gravity) # Normalize gravity vector   
        # world_gravity = np.array([0, 0, 0])  # Override to ensure consistent gravity direction
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        projected_gravity = imu_xmat.T @ world_gravity  # Project gravity into IMU frame
        # print(imu_xmat)
        # print(world_gravity, projected_gravity)
        # print(projected_gravity.shape) # (3,)
        velocity_commands = self._controller.get_command()  # (3,)

        joint_pos_mujoco = data.qpos[7:]  - self._default_angles# Exclude global position and yaw (23,)
        joint_vel_mujoco = data.qvel[6:]  # Exclude global vel (23,)

        joint_pos_pytorch = remap_mujoco_to_pytorch(joint_pos_mujoco)
        joint_vel_pytorch = remap_mujoco_to_pytorch(joint_vel_mujoco)
        last_action_pytorch = remap_mujoco_to_pytorch(self._last_action)
        base_ang_vel = data.qvel[3:6]
        base_lin_vel = data.qvel[0:3]

        # print("Base lin vel:", base_lin_vel
        #       , "Base ang vel:", base_ang_vel
        #       , "Projected gravity:", projected_gravity
        #       , "Velocity commands:", velocity_commands
        #       )
        # print(velocity_commands)

        # boilerplate for height_scan
        height_scan = np.zeros(187, dtype=np.float32)

        obs = np.hstack([
                base_lin_vel, #3
                base_ang_vel, #3
                projected_gravity, #3
                velocity_commands, #3
                joint_pos_pytorch, #23
                joint_vel_pytorch, #23
                last_action_pytorch, #23
                height_scan, #187
        ])

        # assert obs.shape == (256,)
        return obs.astype(np.float32)
    
    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            
            # Convert to torch tensor and run inference
            obs_tensor = torch.from_numpy(obs).float()
            # obs_encoded = self.obs_encoder(obs_tensor)
            # print(f"Encoded obs: {obs_encoded}")
            # print(f"Raw obs: {obs_tensor}")
            
            with torch.no_grad():
                action_tensor = self._policy(obs_tensor)
                pytorch_pred = action_tensor.numpy()  # Actions in PyTorch model joint order

            # Zero out arm control similar to training logic (in PyTorch order)
            # In PyTorch order, arm joints are:
            # 9-10: shoulders, 13-14: shoulder rolls, 17-18: shoulder yaws, 19-20: elbows, 21-22: wrists
            # ZERO_ARM_CONTROL = True  # Set this flag as needed
            # if ZERO_ARM_CONTROL:
            #   # Arm joint indices in PyTorch order
            #   arm_indices = [9, 10, 13, 14, 17, 18, 19, 20, 21, 22]  # All arm joints
            #   pytorch_pred[arm_indices] = 0.0

            # Convert actions from PyTorch order to MuJoCo order
            mujoco_pred = remap_pytorch_to_mujoco(pytorch_pred)
            # print(mujoco_pred)

            self._last_action = mujoco_pred.copy()  # Store in MuJoCo order
            # data.ctrl[:] =  self._default_angles
            # print(mujoco_pred * self._action_scale + self._default_angles)

            data.ctrl[:] = mujoco_pred * self._action_scale #+ self._default_angles
            # return mujoco_pred * self._action_scale + self._default_angles
            # print(mujoco_pred * self._action_scale + self._default_angles) #Prints the output control
            # print(model.opt.integrator)
            # print(data.qpos)



def load_model():
    # Path to your H1 MJCF
    model = mujoco.MjModel.from_xml_path("./assets/h1_description/mjcf/scene.xml")

    # ======= Basic physics setup =======
    model.opt.timestep = 0.005            # small, stable integration step
    # model.opt.gravity[:] = [0, 0, -9.81]  # normal Earth gravity
    # model.opt.gravity[:] = [0, 0, 0]  # zero gravity for testing

    model.opt.iterations = 50
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER  # Euler or RK4
    
    # Add damping to prevent jitter / free base blowup
    # model.dof_damping[:] = 0.2

    # ======= Initialize data =======
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)      # clean reset
    mujoco.mj_forward(model, data)

    # ======= Ensure valid base pose =======
    # qpos[:3] = base position (x, y, z)
    data.qpos[:3] = np.array([0, 0, 1.04])   # start 1m above ground
    # qpos[3:7] = base quaternion (w, x, y, z)
    data.qpos[3:7] = np.array([1, 0, 0, 0]) # identity orientation
    # data.qpos[3:7] = np.array([0.7071, 0, 0, -0.7071]) # 90 deg rotation around z-axis

    data.qpos[7:] = default_angles_config.copy()

    mujoco.mj_forward(model, data)  # recompute transforms

    # print(default_angles_config.copy())

    data.ctrl[:] = default_angles_config.copy()
    print(data.qpos)
    print(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    print(f"Initial gravity: {model.opt.gravity}")
    # data.ctrl[:] = np.zeros_like(default_angles_config)
    # print(data.qpos)
    

    return model, data


def loaded():
    model, data = load_model()

    policy = TorchController(
        policy_path=POLICY_PATH,
        default_angles=np.array(default_angles_config),
        n_substeps=1,
        action_scale=0.5,
        vel_scale_x=1.0,
        vel_scale_y=1.0,
        vel_scale_rot=1.0,
    )

    # with mujoco.viewer.launch_passive(model, data) as v:
    #     while v.is_running():
    #         # Step the physics
    #         mujoco.mj_step(model, data)

    #         # Optionally freeze control for debugging
    #         data.ctrl[:] = policy.get_control(model, data)
    #         # print(data.qpos)

    #         # Sync viewer
    #         v.sync()

    mujoco.set_mjcb_control(policy.get_control)
    # viewer.launch(model, data)
    return model, data

if __name__ == "__main__":
    viewer.launch(loader=loaded)